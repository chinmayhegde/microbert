import torch
import torch.nn as nn
from tokenizer import CharacterTokenizer
from model import CharacterBERT
import math
import numpy as np

# Load data and create tokenizer
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = CharacterTokenizer()
tokenizer.build_vocab(text)
print(f"Vocabulary size: {tokenizer.vocab_size}")

# Analyze masking strategy
sample_text = text[:1000]
input_ids = tokenizer.encode(sample_text, max_length=512)
masked_ids, labels = tokenizer.create_mlm_mask(input_ids.clone())

# Count how many tokens are masked
mask_positions = (labels != -100).sum().item()
total_positions = len(input_ids)
actual_mask_rate = mask_positions / total_positions
print(f"\nMasking statistics:")
print(f"Total positions: {total_positions}")
print(f"Masked positions: {mask_positions}")
print(f"Actual mask rate: {actual_mask_rate:.3f}")

# Check class distribution in labels
unique_labels = labels[labels != -100]
if len(unique_labels) > 0:
    print(f"\nLabel distribution:")
    print(f"Min label: {unique_labels.min().item()}")
    print(f"Max label: {unique_labels.max().item()}")
    print(f"Unique labels: {len(unique_labels.unique())}")

# Calculate theoretical minimum loss
print(f"\nTheoretical loss bounds:")
print(f"Random guessing loss: {math.log(tokenizer.vocab_size):.3f}")
print(f"With 15% masking, effective loss ≈ {0.15 * math.log(tokenizer.vocab_size):.3f}")

# Test model predictions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CharacterBERT(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_layers=4,
    n_head=8,
    d_ff=512,
    max_len=512,
    dropout=0.1
).to(device)

# Load checkpoint if exists
import os
if os.path.exists('microbert_checkpoint.pt'):
    checkpoint = torch.load('microbert_checkpoint.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("\nLoaded trained model checkpoint")

# Test prediction quality
model.eval()
with torch.no_grad():
    test_batch = masked_ids.unsqueeze(0).to(device)
    outputs = model(test_batch)
    
    # Calculate loss only on masked positions
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.to(device).view(-1))
    print(f"\nTest loss on sample: {loss.item():.3f}")
    
    # Check prediction distribution
    probs = torch.softmax(outputs[0], dim=-1)
    masked_probs = probs[labels != -100]
    
    if len(masked_probs) > 0:
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        max_entropy = math.log(tokenizer.vocab_size)
        print(f"\nPrediction entropy: {entropy.item():.3f} (max: {max_entropy:.3f})")
        
        # Top-k accuracy
        predictions = outputs[0].argmax(dim=-1)
        correct = (predictions[labels != -100] == labels[labels != -100].to(device)).float().mean()
        print(f"Accuracy on masked tokens: {correct.item():.3f}")