import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import argparse

def load_model(model_path):
    """Load the trained model and tokenizer"""
    model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device

def predict_toxicity(text, model, tokenizer, device):
    """Predict toxicity labels for a given text"""
    # Tokenize text
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.sigmoid(outputs.logits)
    
    # Convert to probabilities
    probabilities = predictions[0].cpu().numpy()
    
    # Labels for toxicity types
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Create results dictionary
    results = {}
    for label, prob in zip(labels, probabilities):
        results[label] = float(prob)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Predict toxicity of text using trained model')
    parser.add_argument('--model_path', type=str, default='weights/toxic_classifier_xlm-roberta-large',
                      help='Path to the trained model')
    parser.add_argument('--text', type=str, required=True,
                      help='Text to classify')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Probability threshold for toxic classification')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model(args.model_path)
    
    # Make prediction
    print("\nAnalyzing text...")
    predictions = predict_toxicity(args.text, model, tokenizer, device)
    
    # Print results
    print("\nResults:")
    print("-" * 50)
    print(f"Text: {args.text}")
    print("\nToxicity Probabilities:")
    for label, prob in predictions.items():
        if prob > args.threshold:
            print(f"- {label}: {prob:.2%}")
    
    # Overall assessment
    if any(prob > args.threshold for prob in predictions.values()):
        print("\nOverall: ⚠️ This text contains toxic content")
    else:
        print("\nOverall: ✅ This text appears to be non-toxic")

if __name__ == "__main__":
    main() 