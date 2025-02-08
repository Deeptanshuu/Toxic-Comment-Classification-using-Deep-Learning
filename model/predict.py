import torch
from model.language_aware_transformer import LanguageAwareTransformer
from transformers import XLMRobertaTokenizer
import json
import os
import re

SUPPORTED_LANGUAGES = {
    'en': 0, 'ru': 1, 'tr': 2, 'es': 3,
    'fr': 4, 'it': 5, 'pt': 6
}

# Threshold adjustments to reduce overflagging while maintaining sensitivity for rare classes
THRESHOLD_ADJUSTMENTS = {
    'toxic': 0.70,         # Increased from ~46% to reduce overflagging
    'insult': 0.70,        # Increased from ~26% to reduce overflagging
    'threat': 0.30,        # Kept low due to rare class importance
    'identity_hate': 0.30  # Kept low due to rare class importance
}

# Language detection patterns
LANG_PATTERNS = {
    'ru': [
        r'[а-яА-Я]',  # Cyrillic characters
        r'\b(и|в|не|что|на|я|быть|он|с|а|это|к|по|они|мы|как|все|она|так|его|но|да|ты|у|же|вы|за|тот|от|меня|еще|нет|вот)\b',  # Common words
        r'[ёЁ]',  # Specific Russian character
        r'\b(привет|спасибо|пожалуйста|здравствуйте|хорошо|да|нет|конечно)\b'  # Common expressions
    ],
    'tr': [
        r'[şŞıİğĞüÜöÖçÇ]',  # Turkish-specific characters
        r'\b(ve|bu|bir|için|ben|sen|o|biz|siz|de|da|ki|ne|kim|ile|ama|fakat|çünkü|nasıl|nerede|ne zaman)\b',  # Common words
        r'\b(merhaba|teşekkürler|lütfen|iyi günler|evet|hayır|tabii)\b',  # Common expressions
        r'\b[a-zA-Z]*[iıİI][a-zA-Z]*\b'  # Words with Turkish i/ı
    ],
    'es': [
        r'\b(el|la|los|las|un|una|unos|unas|y|o|pero|porque|si|cuando|donde|como|que|esto|eso|esta|ese|muy)\b',  # Articles and common words
        r'\b(de|en|para|por|con|sin|sobre|entre|detrás|después|antes|durante|hacia|hasta|desde)\b',  # Prepositions
        r'[¿¡]',  # Spanish punctuation
        r'\b(hola|gracias|por favor|buenos días|sí|no|claro|vale|bien)\b',  # Common expressions
        r'\b(estar|ser|tener|hacer|decir|ir|ver|dar|saber|querer|llegar|pasar|deber|poner|parecer|quedar|creer)\b'  # Common verbs
    ],
    'fr': [
        r'\b(le|la|les|un|une|des|et|ou|mais|donc|car|ni|que|qui|quoi|où|comment|pourquoi|ce|cette|ces|mon|ton|son)\b',  # Articles and pronouns
        r'\b(de|à|dans|par|pour|en|vers|avec|sans|sous|sur|chez|avant|après|pendant|depuis|vers)\b',  # Prepositions
        r'\b(être|avoir|faire|dire|aller|voir|savoir|pouvoir|falloir|vouloir)\b',  # Common verbs
        r'\b(bonjour|merci|s\'il vous plaît|au revoir|oui|non|bien sûr|d\'accord)\b',  # Common expressions
        r'[àâæçéèêëîïôœùûüÿ]'  # French-specific characters
    ],
    'it': [
        r'\b(il|lo|la|i|gli|le|un|uno|una|dei|degli|delle|e|o|ma|perché|quando|dove|come|che|questo|quello|questa|quella)\b',  # Articles and pronouns
        r'\b(di|a|da|in|con|su|per|tra|fra|dentro|fuori|sopra|sotto|prima|dopo)\b',  # Prepositions
        r'\b(essere|avere|fare|dire|andare|vedere|sapere|potere|dovere|volere)\b',  # Common verbs
        r'\b(ciao|grazie|per favore|arrivederci|sì|no|certo|va bene)\b',  # Common expressions
        r'[àèéìíîòóùú]'  # Italian-specific characters
    ],
    'pt': [
        r'\b(o|a|os|as|um|uma|uns|umas|e|ou|mas|porque|se|quando|onde|como|que|isto|isso|esta|esse|muito)\b',  # Articles and common words
        r'\b(de|em|para|por|com|sem|sobre|entre|atrás|depois|antes|durante)\b',  # Prepositions
        r'\b(ser|estar|ter|fazer|dizer|ir|ver|dar|saber|querer)\b',  # Common verbs
        r'\b(olá|obrigado|obrigada|por favor|bom dia|sim|não|claro|tudo bem)\b',  # Common expressions
        r'[áâãàçéêíóôõúü]'  # Portuguese-specific characters
    ],
    'en': [
        r'\b(the|a|an|and|or|but|if|when|where|how|what|this|that|these|those|my|your|his|her|its|our|their)\b',  # Articles and pronouns
        r'\b(in|on|at|to|for|with|by|from|about|into|through|after|before|under|over)\b',  # Prepositions
        r'\b(be|have|do|say|get|make|go|know|take|see|come|think|look|want|give|use)\b',  # Common verbs
        r'\b(hello|thanks|thank you|please|goodbye|yes|no|maybe|sure|okay)\b',  # Common expressions
        r'\b(i|you|he|she|it|we|they|me|him|her|us|them)\b'  # Personal pronouns
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

def detect_language_regex(text):
    """
    Detect language using regex patterns of common words and special characters.
    Returns language code of best match.
    """
    # Normalize text
    text = ' ' + text.lower() + ' '  # Add spaces to help with word boundary detection
    
    # Count matches for each language
    scores = {}
    
    # Check each language's patterns
    for lang, patterns in LANG_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, text))
            # Weight matches based on pattern type
            if '[' in pattern:  # Special characters pattern
                score += matches * 2  # Give more weight to special characters
            else:  # Word pattern
                score += matches
        scores[lang] = score
    
    # Special case: if no matches or only English matches, check for ASCII-only content
    if all(score == 0 for lang, score in scores.items()) or (sum(scores.values()) == scores.get('en', 0)):
        # If text is ASCII-only and has words, likely English
        if all(ord(c) < 128 for c in text) and bool(re.search(r'\w', text)):
            return SUPPORTED_LANGUAGES['en']
    
    # Get language with highest score
    max_score = max(scores.values())
    if max_score > 0:
        # Get the language with the highest score
        detected = max(scores.items(), key=lambda x: x[1])[0]
        return SUPPORTED_LANGUAGES[detected]
    
    # Default to English if no clear match
    return SUPPORTED_LANGUAGES['en']

def detect_language(text):
    """Detect language of input text and map to supported language ID"""
    try:
        # Clean and normalize text
        cleaned_text = text.strip()
        
        # If text is empty or just whitespace/punctuation
        if not re.search(r'\w', cleaned_text):
            return SUPPORTED_LANGUAGES['en']
        
        # Detect language using regex patterns
        lang_id = detect_language_regex(cleaned_text)
        return lang_id
        
    except Exception as e:
        print(f"Note: Could not detect language ({str(e)}). Using English.")
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