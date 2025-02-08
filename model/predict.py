import torch
from model.language_aware_transformer import LanguageAwareTransformer
from transformers import XLMRobertaTokenizer
import json
import os
import re
from collections import Counter

SUPPORTED_LANGUAGES = {
    'en': 0, 'ru': 1, 'tr': 2, 'es': 3,
    'fr': 4, 'it': 5, 'pt': 6
}

# Threshold adjustments to reduce overflagging while maintaining sensitivity for rare classes
THRESHOLD_ADJUSTMENTS = {
    'toxic': 0.75,         # Increased from ~46% to reduce overflagging
    'insult': 0.70,        # Increased from ~26% to reduce overflagging
    'threat': 0.30,        # Kept low due to rare class importance
    'identity_hate': 0.30  # Kept low due to rare class importance
}

# Unicode ranges for different scripts
UNICODE_RANGES = {
    'ru': [
        (0x0400, 0x04FF),  # Cyrillic
        (0x0500, 0x052F),  # Cyrillic Supplement
    ],
    'tr': [
        (0x011E, 0x011F),  # Ğ ğ
        (0x0130, 0x0131),  # İ ı
        (0x015E, 0x015F),  # Ş ş
    ],
    'es': [
        (0x00C1, 0x00C1),  # Á
        (0x00C9, 0x00C9),  # É
        (0x00CD, 0x00CD),  # Í
        (0x00D1, 0x00D1),  # Ñ
        (0x00D3, 0x00D3),  # Ó
        (0x00DA, 0x00DA),  # Ú
        (0x00DC, 0x00DC),  # Ü
    ],
    'fr': [
        (0x00C0, 0x00C6),  # À-Æ
        (0x00C8, 0x00CB),  # È-Ë
        (0x00CC, 0x00CF),  # Ì-Ï
        (0x00D2, 0x00D6),  # Ò-Ö
        (0x0152, 0x0153),  # Œ œ
    ],
    'it': [
        (0x00C0, 0x00C0),  # À
        (0x00C8, 0x00C8),  # È
        (0x00C9, 0x00C9),  # É
        (0x00CC, 0x00CC),  # Ì
        (0x00D2, 0x00D2),  # Ò
        (0x00D9, 0x00D9),  # Ù
    ],
    'pt': [
        (0x00C0, 0x00C3),  # À-Ã
        (0x00C7, 0x00C7),  # Ç
        (0x00C9, 0x00CA),  # É-Ê
        (0x00D3, 0x00D5),  # Ó-Õ
    ]
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

def adjust_thresholds(thresholds):
    """
    Adjust thresholds based on recommendations to reduce overflagging
    """
    if not thresholds:
        return thresholds
        
    adjusted = thresholds.copy()
    # Adjust thresholds for each language
    for lang_id in adjusted:
        for category, recommended in THRESHOLD_ADJUSTMENTS.items():
            if category in adjusted[lang_id]:
                # Only increase threshold if recommended is higher
                adjusted[lang_id][category] = max(adjusted[lang_id][category], recommended)
    
    return adjusted

def load_thresholds(thresholds_path='evaluation_results/eval_20250208_161149/thresholds.json'):
    """Load language-specific classification thresholds"""
    try:
        with open(thresholds_path, 'r') as f:
            thresholds = json.load(f)
            # Apply threshold adjustments
            return adjust_thresholds(thresholds)
    except Exception as e:
        print(f"Warning: Could not load thresholds from {thresholds_path}: {str(e)}")
        return None

def analyze_unicode_ranges(text):
    """Analyze text for characters in language-specific Unicode ranges"""
    scores = {lang: 0 for lang in SUPPORTED_LANGUAGES.keys()}
    
    for char in text:
        code = ord(char)
        for lang, ranges in UNICODE_RANGES.items():
            for start, end in ranges:
                if start <= code <= end:
                    scores[lang] += 1
    
    return scores

def analyze_tokenizer_stats(text, tokenizer):
    """Analyze tokenizer statistics for language detection"""
    # Get tokenizer output
    tokens = tokenizer.tokenize(text)
    
    # Count language-specific token patterns
    scores = {lang: 0 for lang in SUPPORTED_LANGUAGES.keys()}
    
    # Analyze token patterns
    for token in tokens:
        token = token.lower()
        # Check for language-specific subwords
        if 'en' in token or '_en' in token:
            scores['en'] += 1
        elif 'ru' in token or '_ru' in token:
            scores['ru'] += 1
        elif 'tr' in token or '_tr' in token:
            scores['tr'] += 1
        elif 'es' in token or '_es' in token:
            scores['es'] += 1
        elif 'fr' in token or '_fr' in token:
            scores['fr'] += 1
        elif 'it' in token or '_it' in token:
            scores['it'] += 1
        elif 'pt' in token or '_pt' in token:
            scores['pt'] += 1
    
    return scores

def detect_language(text, tokenizer):
    """
    Enhanced language detection using multiple methods:
    1. Unicode range analysis
    2. Tokenizer statistics
    3. ASCII analysis for English
    """
    try:
        # Clean text
        text = text.strip()
        
        # If empty or just punctuation
        if not text or not re.search(r'\w', text):
            return SUPPORTED_LANGUAGES['en']
            
        # If text is ASCII only, likely English
        if all(ord(c) < 128 for c in text):
            return SUPPORTED_LANGUAGES['en']
        
        # Get scores from different methods
        unicode_scores = analyze_unicode_ranges(text)
        tokenizer_scores = analyze_tokenizer_stats(text, tokenizer)
        
        # Combine scores with weights
        final_scores = {lang: 0 for lang in SUPPORTED_LANGUAGES.keys()}
        for lang in SUPPORTED_LANGUAGES.keys():
            final_scores[lang] = (
                unicode_scores[lang] * 2 +  # Unicode ranges have higher weight
                tokenizer_scores[lang]
            )
        
        # Get language with highest score
        if any(score > 0 for score in final_scores.values()):
            detected_lang = max(final_scores.items(), key=lambda x: x[1])[0]
            return SUPPORTED_LANGUAGES[detected_lang]
        
        # Default to English if no clear match
        return SUPPORTED_LANGUAGES['en']
        
    except Exception as e:
        print(f"Note: Language detection failed ({str(e)}). Using English.")
        return SUPPORTED_LANGUAGES['en']

def predict_toxicity(text, model, tokenizer, device, thresholds=None):
    """Predict toxicity labels for a given text"""
    # Detect language
    lang_id = detect_language(text, tokenizer)
    
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
        text = input().strip().lower()
        
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