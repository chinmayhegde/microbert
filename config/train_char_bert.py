# config for training character-level BERT
# example usage: python train_bert.py config/train_char_bert.py

# I/O
out_dir = 'out-char-bert'
eval_interval = 500
eval_iters = 100
log_interval = 10

# data
dataset = 'data/shakespeare/input.txt'  # path to your text file
gradient_accumulation_steps = 1
batch_size = 16
block_size = 256  # shorter sequences for character-level

# model - smaller model for character-level
n_layer = 6
n_head = 8
n_embd = 256
d_ff = 1024
dropout = 0.1
mlm_probability = 0.15

# optimizer
learning_rate = 1e-3
max_iters = 20000
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 20000
min_lr = 1e-4

# system
device = 'cuda'
dtype = 'bfloat16'
compile = False  # set to True if you have PyTorch 2.0+

# logging
wandb_log = False  # set to True if you want to use wandb
wandb_project = 'char-bert'
wandb_run_name = 'char-bert-run-1'
