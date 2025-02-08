import torch
from model.language_aware_transformer import LanguageAwareTransformer
from transformers import XLMRobertaTokenizer
import json
import os
from langdetect import detect

SUPPORTED_LANGUAGES = {
    'en': 0, 'ru': 1, 'tr': 2, 'es': 3,
    'fr': 4, 'it': 5, 'pt': 6
}

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
            model_name='xlm-roberta-large'
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

def load_thresholds(thresholds_path='evaluation_results/eval_20250208_161149/thresholds.json'):
    """Load language-specific classification thresholds"""
    try:
        with open(thresholds_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load thresholds from {thresholds_path}: {str(e)}")
        return None

def detect_language(text):
    """Detect language of input text and map to supported language ID"""
    try:
        detected = detect(text)
        # Map detected language to our supported languages
        if detected in SUPPORTED_LANGUAGES:
            return SUPPORTED_LANGUAGES[detected]
        else:
            print(f"Warning: Detected language '{detected}' not supported. Using English (en) as default.")
            return SUPPORTED_LANGUAGES['en']
    except:
        print("Warning: Could not detect language. Using English (en) as default.")
        return SUPPORTED_LANGUAGES['en']

def predict_toxicity(text, model, tokenizer, device, thresholds=None):
    """Predict toxicity labels for a given text"""
    # Detect language
    lang_id = detect_language(text)
    
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
    
    # Get language-specific thresholds if available
    lang_thresholds = thresholds.get(str(lang_id)) if thresholds else None
    
    for label, prob in zip(labels, probabilities):
        threshold = lang_thresholds.get(label) if lang_thresholds else 0.3
        results[label] = {
            'probability': float(prob),
            'is_toxic': prob > threshold,
            'threshold': threshold
        }
    
    return results, lang_id

def main():
    # Load model
    print("Loading model...")
    model_path = 'weights/toxic_classifier_xlm-roberta-large'
    model, tokenizer, device = load_model(model_path)
    
    if model is None or tokenizer is None:
        return
    
    # Load thresholds
    thresholds = load_thresholds()
    
    while True:
        # Get input text
        print("\nEnter text to analyze (or 'q' to quit):")
        text = input().strip()
        
        if text.lower() == 'q':
            break
        
        if not text:
            print("Please enter some text to analyze.")
            continue
        
        # Make prediction
        print("\nAnalyzing text...")
        predictions, lang_id = predict_toxicity(text, model, tokenizer, device, thresholds)
        
        # Get language name
        lang_name = [k for k, v in SUPPORTED_LANGUAGES.items() if v == lang_id][0]
        
        # Print results
        print("\nResults:")
        print("-" * 50)
        print(f"Text: {text}")
        print(f"Detected Language: {lang_name}")
        print("\nToxicity Analysis:")
        
        any_toxic = False
        for label, result in predictions.items():
            if result['is_toxic']:
                any_toxic = True
                print(f"- {label}: {result['probability']:.2%} (threshold: {result['threshold']:.2%}) ⚠️")
        
        # Print non-toxic results with lower emphasis
        print("\nOther categories:")
        for label, result in predictions.items():
            if not result['is_toxic']:
                print(f"- {label}: {result['probability']:.2%} (threshold: {result['threshold']:.2%}) ✓")
        
        # Overall assessment
        print("\nOverall Assessment:")
        if any_toxic:
            print("⚠️  This text contains toxic content")
        else:
            print("✅  This text appears to be non-toxic")

if __name__ == "__main__":
    main() 