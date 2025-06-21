import torch
from tokenizer import CharacterTokenizer
from model import CharacterBERT


def interactive_mlm_demo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading model and tokenizer...")
    checkpoint = torch.load('microbert_checkpoint.pt', map_location=device)
    
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
    
    print("\nCharacter-level BERT MLM Demo")
    print("Enter text with [MASK] tokens to see predictions")
    print("Example: 'To [MASK]e or not to [MASK]e'")
    print("Type 'quit' to exit\n")
    
    while True:
        text = input("Enter text: ").strip()
        
        if text.lower() == 'quit':
            break
            
        if '[MASK]' not in text:
            print("Please include at least one [MASK] token in your text\n")
            continue
        
        mask_count = text.count('[MASK]')
        text_for_model = text.replace('[MASK]', tokenizer.idx_to_char[tokenizer.char_to_idx['[MASK]']])
        
        input_ids = tokenizer.encode(text_for_model, max_length=512).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            predictions = torch.argmax(outputs, dim=-1)
        
        mask_positions = []
        current_pos = 0
        for _ in range(mask_count):
            pos = text.find('[MASK]', current_pos)
            if pos != -1:
                mask_positions.append(pos)
                current_pos = pos + 6
        
        result = list(text)
        cls_offset = 1
        
        print("\nPredictions:")
        for i, text_pos in enumerate(mask_positions):
            token_pos = text_pos - (i * 5) + cls_offset
            
            if token_pos < len(predictions[0]):
                pred_idx = predictions[0, token_pos].item()
                if pred_idx in tokenizer.idx_to_char:
                    predicted_char = tokenizer.idx_to_char[pred_idx]
                    print(f"  [MASK] {i+1} -> '{predicted_char}'")
                    
                    mask_start = result.index('[MASK]')
                    result[mask_start:mask_start+6] = list(predicted_char)
        
        print(f"\nFilled text: {''.join(result)}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    print("Note: Make sure to run train.py first to create the model checkpoint!")
    try:
        interactive_mlm_demo()
    except FileNotFoundError:
        print("\nError: Model checkpoint not found. Please run train.py first.")
    except KeyboardInterrupt:
        print("\n\nGoodbye!")