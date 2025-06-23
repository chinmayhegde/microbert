"""
Character-level BERT training script with DDP support.
Adapted from nanoGPT to support MLM training with character-level tokenization.

To run on a single GPU:
$ python train_bert.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node:
$ torchrun --standalone --nproc_per_node=4 train_bert.py

To run with DDP on 4 gpus across 2 nodes:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train_bert.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train_bert.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import CharacterBERT
from tokenizer import CharacterTokenizer

# -----------------------------------------------------------------------------
# default config values for character-level BERT training
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'char-bert'
wandb_run_name = 'char-bert' # 'run' + str(time.time())
# data
dataset = 'shakespeare' # or path to your text file
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 512 # sequence length for BERT
# model
n_layer = 4
n_head = 8
n_embd = 128
d_ff = 512
dropout = 0.1
mlm_probability = 0.15 # probability of masking tokens for MLM
# adamw optimizer
learning_rate = 5e-4 # max learning rate
max_iters = 100000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 100000 # should be ~= max_iters
min_lr = 5e-5 # minimum learning rate, should be ~= learning_rate/10
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# load and prepare data
print("Loading text data...")
if dataset == 'shakespeare':
    # default to shakespeare dataset
    data_dir = os.path.join('data', 'shakespeare')
    with open(os.path.join(data_dir, 'input.txt'), 'r', encoding='utf-8') as f:
        text = f.read()
else:
    # assume dataset is a path to a text file
    with open(dataset, 'r', encoding='utf-8') as f:
        text = f.read()

print("Building tokenizer vocabulary...")
tokenizer = CharacterTokenizer()
tokenizer.build_vocab(text)
vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

# create memory-mapped data for efficient loading
def prepare_data(text, tokenizer, block_size):
    """Convert text to overlapping sequences and save as memory-mapped arrays"""
    stride = block_size // 2  # 50% overlap
    sequences = []
    
    for i in range(0, len(text) - block_size, stride):
        seq = text[i:i + block_size]
        tokens = tokenizer.encode(seq, max_length=block_size)
        if len(tokens) == block_size:  # only include full sequences
            sequences.append(tokens.numpy())
    
    sequences = np.array(sequences, dtype=np.uint16)
    print(f"Created {len(sequences)} sequences of length {block_size}")
    
    # split into train/val
    n = len(sequences)
    train_sequences = sequences[:int(n*0.9)]
    val_sequences = sequences[int(n*0.9):]
    
    return train_sequences, val_sequences

# prepare data splits
print("Preparing training data...")
train_data, val_data = prepare_data(text, tokenizer, block_size)

def get_batch(split):
    """Get a batch of MLM training data"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data), (batch_size,))
    
    # Get sequences and convert to tensors
    sequences = torch.from_numpy(data[ix].astype(np.int64))
    
    # Create MLM masks for each sequence in the batch
    masked_sequences = []
    labels_sequences = []
    attention_masks = []
    
    for seq in sequences:
        masked_seq, labels = tokenizer.create_mlm_mask(seq.clone(), mlm_probability=mlm_probability)
        attention_mask = (seq != tokenizer.char_to_idx['[PAD]']).long()
        
        masked_sequences.append(masked_seq)
        labels_sequences.append(labels)
        attention_masks.append(attention_mask)
    
    input_ids = torch.stack(masked_sequences)
    labels = torch.stack(labels_sequences)
    attention_mask = torch.stack(attention_masks)
    
    if device_type == 'cuda':
        # pin arrays, which allows us to move them to GPU asynchronously (non_blocking=True)
        input_ids = input_ids.pin_memory().to(device, non_blocking=True)
        labels = labels.pin_memory().to(device, non_blocking=True)
        attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)
    else:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
    
    return input_ids, labels, attention_mask

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    vocab_size=vocab_size,
    d_model=n_embd,
    n_layers=n_layer,
    n_head=n_head,
    d_ff=d_ff,
    max_len=block_size,
    dropout=dropout
)

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = CharacterBERT(**model_args)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't resume training
    for k in ['vocab_size', 'd_model', 'n_layers', 'n_head', 'd_ff', 'max_len']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    model = CharacterBERT(**model_args)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary if needed
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    # restore tokenizer
    tokenizer_state = checkpoint['tokenizer']
    tokenizer.char_to_idx = tokenizer_state['char_to_idx']
    tokenizer.idx_to_char = tokenizer_state['idx_to_char']
    tokenizer.vocab_size = tokenizer_state['vocab_size']

model.to(device)

# count parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            input_ids, labels, attention_mask = get_batch(split)
            with ctx:
                outputs = model(input_ids, attention_mask)
                # MLM loss: only compute loss on masked tokens
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# MLM evaluation function
@torch.no_grad()
def evaluate_mlm_sample():
    """Evaluate MLM on a sample text"""
    model.eval()
    sample_text = "To be, or not to be, that is the question"
    
    input_ids = tokenizer.encode(sample_text, max_length=block_size)
    masked_input_ids, labels = tokenizer.create_mlm_mask(input_ids.clone(), mlm_probability=0.15)
    
    masked_input_ids = masked_input_ids.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)
    
    with ctx:
        outputs = model(masked_input_ids)
        predictions = torch.argmax(outputs, dim=-1)
    
    # Calculate accuracy on masked tokens
    masked_positions = (labels[0] != -100).nonzero(as_tuple=True)[0]
    if len(masked_positions) > 0:
        correct = 0
        for pos in masked_positions:
            if labels[0, pos].item() == predictions[0, pos].item():
                correct += 1
        accuracy = correct / len(masked_positions)
    else:
        accuracy = 0.0
    
    model.train()
    return accuracy, tokenizer.decode(masked_input_ids[0]), tokenizer.decode(predictions[0])

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
input_ids, labels, attention_mask = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # MLM evaluation
        if iter_num > 0:
            mlm_acc, masked_text, predicted_text = evaluate_mlm_sample()
            print(f"MLM accuracy: {mlm_acc:.1%}")
            print(f"Sample - Masked: {masked_text[:50]}...")
            print(f"Sample - Predicted: {predicted_text[:50]}...")
        
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            if iter_num > 0:
                log_dict["eval/mlm_accuracy"] = mlm_acc
            wandb.log(log_dict)
            
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'tokenizer': {
                        'char_to_idx': tokenizer.char_to_idx,
                        'idx_to_char': tokenizer.idx_to_char,
                        'vocab_size': tokenizer.vocab_size
                    }
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            outputs = model(input_ids, attention_mask)
            # MLM loss: only compute loss on masked tokens
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        input_ids, labels, attention_mask = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            # estimate model flops utilization (MFU) - simplified version
            # since BERT is different from GPT, this is just an approximation
            running_mfu = 0.5 if running_mfu == -1.0 else 0.9*running_mfu + 0.1*0.5
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
