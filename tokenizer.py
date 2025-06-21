import torch
from typing import List, Dict, Tuple
import numpy as np


class CharacterTokenizer:
    def __init__(self):
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = len(self.special_tokens)
        
    def build_vocab(self, text: str):
        unique_chars = sorted(set(text))
        
        self.idx_to_char = {idx: token for token, idx in self.special_tokens.items()}
        self.char_to_idx = self.special_tokens.copy()
        
        for i, char in enumerate(unique_chars):
            idx = i + len(self.special_tokens)
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
            
        self.vocab_size = len(self.char_to_idx)
        
    def encode(self, text: str, max_length: int = 512) -> torch.Tensor:
        tokens = [self.char_to_idx.get(char, self.char_to_idx['[UNK]']) for char in text]
        
        tokens = [self.char_to_idx['[CLS]']] + tokens + [self.char_to_idx['[SEP]']]
        
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.char_to_idx['[SEP]']]
        else:
            tokens = tokens + [self.char_to_idx['[PAD]']] * (max_length - len(tokens))
            
        return torch.tensor(tokens, dtype=torch.long)
    
    def encode_batch(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        batch = []
        for text in texts:
            batch.append(self.encode(text, max_length))
        return torch.stack(batch)
    
    def decode(self, tokens: torch.Tensor) -> str:
        if len(tokens.shape) > 1:
            tokens = tokens.squeeze()
        
        chars = []
        for idx in tokens.tolist():
            if idx in self.idx_to_char:
                token = self.idx_to_char[idx]
                if token not in self.special_tokens:
                    chars.append(token)
        return ''.join(chars)
    
    def create_mlm_mask(self, input_ids: torch.Tensor, mlm_probability: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = input_ids.clone()
        
        probability_matrix = torch.full(labels.shape, mlm_probability)
        
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
        for token_id in self.special_tokens.values():
            special_tokens_mask |= (labels == token_id)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100
        
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.char_to_idx['[MASK]']
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_chars = torch.randint(len(self.special_tokens), self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_chars[indices_random]
        
        return input_ids, labels