import torch
import random
from tokenizer import CharacterTokenizer
from model import CharacterBERT


def load_model_and_tokenizer(checkpoint_path='microbert_checkpoint.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    tokenizer = CharacterTokenizer()
    tokenizer.char_to_idx = checkpoint['tokenizer_vocab']['char_to_idx']
    tokenizer.idx_to_char = checkpoint['tokenizer_vocab']['idx_to_char']
    tokenizer.vocab_size = checkpoint['tokenizer_vocab']['vocab_size']
    
    model = CharacterBERT(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_head=8,
        d_ff=512,
        max_len=512,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer, device


def test_mlm_prediction(model, tokenizer, text, num_masks=5, device='cpu'):
    print(f"\nOriginal text: {text}")
    print("-" * 50)
    
    chars = list(text)
    
    valid_positions = [i for i in range(len(chars)) if chars[i] not in [' ', '\n', '\t']]
    
    if len(valid_positions) < num_masks:
        num_masks = len(valid_positions)
    
    mask_positions = sorted(random.sample(valid_positions, num_masks))
    
    original_chars = {}
    for pos in mask_positions:
        original_chars[pos] = chars[pos]
        chars[pos] = '[MASK]'
    
    masked_text = ''.join(chars)
    print(f"Masked text: {masked_text}")
    
    input_text = masked_text.replace('[MASK]', tokenizer.idx_to_char[tokenizer.char_to_idx['[MASK]']])
    input_ids = tokenizer.encode(input_text, max_length=512).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = torch.argmax(outputs, dim=-1)
    
    predicted_text = tokenizer.decode(predictions[0])
    
    cls_sep_offset = 1
    print("\nPredictions:")
    correct = 0
    for i, pos in enumerate(mask_positions):
        token_pos = pos + cls_sep_offset
        if token_pos < len(predictions[0]):
            pred_idx = predictions[0, token_pos].item()
            if pred_idx in tokenizer.idx_to_char:
                predicted_char = tokenizer.idx_to_char[pred_idx]
            else:
                predicted_char = '[UNK]'
            
            original = original_chars[pos]
            is_correct = predicted_char == original
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  Position {pos}: '{original}' -> '{predicted_char}' {status}")
    
    accuracy = correct / len(mask_positions) if mask_positions else 0
    print(f"\nAccuracy: {correct}/{len(mask_positions)} ({accuracy:.1%})")
    
    reconstructed = list(text)
    for i, pos in enumerate(mask_positions):
        token_pos = pos + cls_sep_offset
        if token_pos < len(predictions[0]):
            pred_idx = predictions[0, token_pos].item()
            if pred_idx in tokenizer.idx_to_char:
                reconstructed[pos] = tokenizer.idx_to_char[pred_idx]
    
    print(f"\nReconstructed: {''.join(reconstructed)}")


def main():
    print("Loading model and tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer()
    
    print("\nReading sample text from input.txt...")
    with open('input.txt', 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    test_samples = [
        "First Citizen:\nBefore we proceed any further, hear me speak.",
        "What authority surfeits on would relieve us",
        "Let us kill him, and we'll have corn at our own price.",
        "The gods know I speak this in hunger for bread",
        full_text[1000:1100],
        full_text[5000:5100]
    ]
    
    print("\nTesting MLM predictions on various samples...")
    print("=" * 70)
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\nTest {i}:")
        test_mlm_prediction(model, tokenizer, sample.strip(), num_masks=5, device=device)
        print("=" * 70)


if __name__ == "__main__":
    main()