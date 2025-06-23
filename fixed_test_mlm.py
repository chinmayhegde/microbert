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
    
    # Use the same masking approach as training
    input_ids = tokenizer.encode(text, max_length=512)
    
    # Find actual length (excluding padding)
    actual_length = len(input_ids)
    if tokenizer.char_to_idx['[PAD]'] in input_ids:
        actual_length = input_ids.tolist().index(tokenizer.char_to_idx['[PAD]'])
    
    # Create masks using the tokenizer's method (same as training)
    masked_input_ids, labels = tokenizer.create_mlm_mask(input_ids.clone(), mlm_probability=0.15)
    
    # Find which positions were actually masked
    masked_positions = (labels != -100).nonzero(as_tuple=True)[0]
    # Only keep positions within actual text length
    masked_positions = [pos for pos in masked_positions if pos < actual_length]
    
    if len(masked_positions) == 0:
        print("No positions were masked in this text. Trying again with higher probability...")
        masked_input_ids, labels = tokenizer.create_mlm_mask(input_ids.clone(), mlm_probability=0.3)
        masked_positions = (labels != -100).nonzero(as_tuple=True)[0]
        masked_positions = [pos for pos in masked_positions if pos < actual_length]
    
    # Show masked text
    masked_text_display = tokenizer.decode(masked_input_ids[:actual_length])
    print(f"Masked text: {masked_text_display}")
    
    # Get model predictions
    masked_input_ids = masked_input_ids.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(masked_input_ids)
        predictions = torch.argmax(outputs, dim=-1)
    
    # Evaluate predictions
    print("\nPredictions:")
    correct = 0
    results = []
    
    for pos in masked_positions:
        if pos >= actual_length:
            continue
            
        original_char = tokenizer.idx_to_char[labels[pos].item()]
        predicted_char = tokenizer.idx_to_char[predictions[0, pos].item()]
        is_correct = original_char == predicted_char
        
        if is_correct:
            correct += 1
            
        status = "✓" if is_correct else "✗"
        print(f"  Position {pos}: '{original_char}' -> '{predicted_char}' {status}")
        
        results.append({
            'position': pos,
            'original': original_char,
            'predicted': predicted_char,
            'correct': is_correct
        })
    
    accuracy = correct / len(results) if results else 0
    print(f"\nAccuracy: {correct}/{len(results)} ({accuracy:.1%})")
    
    # Reconstruct text properly
    reconstructed_ids = input_ids.clone()
    for pos in masked_positions:
        if pos < actual_length:
            reconstructed_ids[pos] = predictions[0, pos]
    
    reconstructed_text = tokenizer.decode(reconstructed_ids[:actual_length])
    print(f"\nReconstructed: {reconstructed_text}")
    
    return results, accuracy


def manual_mask_test(model, tokenizer, text, positions_to_mask, device='cpu'):
    """Test with manually specified positions to mask"""
    print(f"\nManual mask test - Original: {text}")
    print("-" * 50)
    
    # Encode the text
    input_ids = tokenizer.encode(text, max_length=512)
    
    # Find actual length
    actual_length = len(input_ids)
    if tokenizer.char_to_idx['[PAD]'] in input_ids:
        actual_length = input_ids.tolist().index(tokenizer.char_to_idx['[PAD]'])
    
    # Manually create masks
    masked_input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    
    valid_positions = [pos for pos in positions_to_mask if pos < actual_length]
    
    for pos in valid_positions:
        labels[pos] = input_ids[pos]  # Store original
        masked_input_ids[pos] = tokenizer.char_to_idx['[MASK]']
    
    # Show masked text
    masked_text_display = tokenizer.decode(masked_input_ids[:actual_length])
    print(f"Masked text: {masked_text_display}")
    
    # Get predictions
    masked_input_ids = masked_input_ids.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(masked_input_ids)
        predictions = torch.argmax(outputs, dim=-1)
    
    # Evaluate
    print("Predictions:")
    correct = 0
    for pos in valid_positions:
        original_char = tokenizer.idx_to_char[labels[pos].item()]
        predicted_char = tokenizer.idx_to_char[predictions[0, pos].item()]
        is_correct = original_char == predicted_char
        
        if is_correct:
            correct += 1
            
        status = "✓" if is_correct else "✗"
        print(f"  Position {pos}: '{original_char}' -> '{predicted_char}' {status}")
    
    accuracy = correct / len(valid_positions) if valid_positions else 0
    print(f"\nAccuracy: {correct}/{len(valid_positions)} ({accuracy:.1%})")


def main():
    print("Loading model and tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer()
    
    print("\nReading sample text from input.txt...")
    with open('input.txt', 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    test_samples = [
        "To be, or not to be, that is the question",
        "First Citizen:\nBefore we proceed any further, hear me speak.",
        "What authority surfeits on would relieve us",
        "Let us kill him, and we'll have corn at our own price.",
        "The gods know I speak this in hunger for bread",
        full_text[1000:1200].strip() if len(full_text) > 1200 else "The quick brown fox jumps over the lazy dog",
    ]
    
    print("\nTesting MLM predictions on various samples...")
    print("=" * 70)
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\nTest {i}:")
        try:
            test_mlm_prediction(model, tokenizer, sample.strip(), device=device)
        except Exception as e:
            print(f"Error in test {i}: {e}")
        print("=" * 70)
    
    # Test with manual masking for more control
    print("\nManual masking tests:")
    print("=" * 70)
    
    manual_test_text = "Hello world, this is a test"
    manual_mask_test(model, tokenizer, manual_test_text, [0, 6, 12, 17], device=device)


if __name__ == "__main__":
    main()