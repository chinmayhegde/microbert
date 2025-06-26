import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random
import os

from tokenizer import CharacterTokenizer
from model import CharacterBERT


class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer: CharacterTokenizer, seq_length: int = 512, stride: int = 256):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.sequences = []
        
        for i in range(0, len(text) - seq_length, stride):
            seq = text[i:i + seq_length]
            self.sequences.append(seq)
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        text = self.sequences[idx]
        input_ids = self.tokenizer.encode(text, max_length=self.seq_length)
        
        masked_input_ids, labels = self.tokenizer.create_mlm_mask(input_ids.clone())
        
        return {
            'input_ids': masked_input_ids,
            'labels': labels,
            'attention_mask': (input_ids != self.tokenizer.char_to_idx['[PAD]']).long()
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def evaluate_mlm(model, tokenizer, sample_text, device):
    model.eval()
    
    input_ids = tokenizer.encode(sample_text, max_length=512)
    
    # Find the actual length by looking for the first padding token
    actual_length = len(input_ids)
    if tokenizer.char_to_idx['[PAD]'] in input_ids:
        actual_length = input_ids.tolist().index(tokenizer.char_to_idx['[PAD]'])
    
    masked_input_ids, labels = tokenizer.create_mlm_mask(input_ids.clone(), mlm_probability=0.15)
    
    masked_input_ids = masked_input_ids.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(masked_input_ids)
        predictions = torch.argmax(outputs, dim=-1)
    
    masked_positions = (labels[0] != -100).nonzero(as_tuple=True)[0]
    
    results = []
    correct_count = 0
    for pos in masked_positions:
        if pos >= actual_length:  # Skip positions beyond actual text
            continue
        original_char = tokenizer.idx_to_char[labels[0, pos].item()]
        predicted_char = tokenizer.idx_to_char[predictions[0, pos].item()]
        is_correct = original_char == predicted_char
        if is_correct:
            correct_count += 1
        results.append({
            'position': pos.item(),
            'original': original_char,
            'predicted': predicted_char,
            'correct': is_correct
        })
    
    accuracy = correct_count / len(results) if len(results) > 0 else 0
    
    # Create reconstructed text: use original for unmasked, predictions for masked
    reconstructed_ids = input_ids.clone()
    for pos in masked_positions:
        if pos < actual_length:
            reconstructed_ids[pos] = predictions[0, pos]
    
    # Only decode up to the actual text length
    masked_text_display = tokenizer.decode(masked_input_ids[0][:actual_length])
    predicted_text_display = tokenizer.decode(reconstructed_ids[:actual_length])
    
    return results, masked_text_display, predicted_text_display, accuracy




def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading text data...")
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("Building tokenizer vocabulary...")
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    print("Creating dataset...")
    dataset = TextDataset(text, tokenizer, seq_length=256, stride=128)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    print("Initializing model...")
    model = CharacterBERT(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_head=8,
        d_ff=512,
        max_len=512,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    print("\nStarting training...")
    num_epochs = 20
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Average loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 2 == 0:
            print("\nEvaluating MLM predictions...")
            sample = "To be, or not to be, that is the question"
            results, masked_text, predicted_text, accuracy = evaluate_mlm(model, tokenizer, sample, device)
            
            print(f"Original: {sample}")
            print(f"Masked: {masked_text}")
            print(f"Predicted: {predicted_text}")
            print(f"\n {accuracy:.1%} accuracy):")

            for r in results[:5]:
                status = "✓" if r['correct'] else "✗"
                print(f"  Position {r['position']}: '{r['original']}' -> '{r['predicted']}' {status}")
    
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_vocab': {
            'char_to_idx': tokenizer.char_to_idx,
            'idx_to_char': tokenizer.idx_to_char,
            'vocab_size': tokenizer.vocab_size
        }
    }, 'microbert_checkpoint.pt')
    
    print("Training complete!")


if __name__ == "__main__":
    main()