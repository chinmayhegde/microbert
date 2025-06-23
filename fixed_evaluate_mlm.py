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