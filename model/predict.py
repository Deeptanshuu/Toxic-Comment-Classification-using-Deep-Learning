import torch
from model.language_aware_transformer import LanguageAwareTransformer
from transformers import XLMRobertaTokenizer
import argparse
import os

def load_model(model_path):
    """Load the trained model and tokenizer"""
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"Warning: Model path {model_path} not found.")
        print("Please make sure you have trained the model first.")
        return None, None, None
        
    try:
        # Initialize the custom model architecture
        model = LanguageAwareTransformer(
            num_labels=6,
            hidden_size=1024,
            num_attention_heads=16,
            model_name='xlm-roberta-large',
            dropout=0.1
        )
        
        # Load the trained weights
        state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'))
        model.load_state_dict(state_dict)
        
        # For tokenizer, first try to load from model path, if fails, load base model tokenizer
        try:
            tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        except:
            print("Loading base XLM-RoBERTa tokenizer...")
            tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("\nPlease ensure that:")
        print("1. You have trained the model first using train.py")
        print("2. The model weights are saved in the correct location")
        print("3. You have sufficient permissions to access the model files")
        return None, None, None

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
        predictions = outputs['probabilities']
    
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
    parser.add_argument('--threshold', type=float, default=0.3,
                      help='Probability threshold for toxic classification')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model(args.model_path)
    
    if model is None or tokenizer is None:
        return
    
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