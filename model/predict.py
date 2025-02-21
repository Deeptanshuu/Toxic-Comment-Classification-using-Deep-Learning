import torch
from model.language_aware_transformer import LanguageAwareTransformer
from transformers import XLMRobertaTokenizer
import os
import re
import json
from pathlib import Path
import logging
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import sys
import locale
import io

# Force UTF-8 encoding for stdin/stdout
if sys.platform == 'win32':
    # Windows-specific handling
    import msvcrt
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    # Set console to UTF-8 mode
    os.system('chcp 65001')
else:
    # Unix-like systems
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if sys.stdin.encoding != 'utf-8':
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# Set up logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure reproducibility with langdetect
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {
    'en': 0, 'ru': 1, 'tr': 2, 'es': 3,
    'fr': 4, 'it': 5, 'pt': 6
}

# Default thresholds optimized on validation set
DEFAULT_THRESHOLDS = {
    'toxic': 0.80,         # Optimized for general toxicity
    'severe_toxic': 0.45,  # Lower to catch serious cases
    'obscene': 0.48,      # Balanced for precision/recall
    'threat': 0.42,       # Lower to catch potential threats
    'insult': 0.70,       # Balanced for common cases
    'identity_hate': 0.43  # Lower to catch hate speech
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
    try:
        # Convert to absolute Path object
        model_dir = Path(model_path).absolute()
        
        if model_dir.is_dir():
            # Check for 'latest' symlink first
            latest_link = model_dir / 'latest'
            if latest_link.exists() and latest_link.is_symlink():
                # Get the target of the symlink
                target = latest_link.readlink()
                # If target is absolute, use it directly
                if target.is_absolute():
                    model_dir = target
                else:
                    # If target is relative, resolve it relative to the symlink's directory
                    model_dir = (latest_link.parent / target).resolve()
                logger.info(f"Using latest checkpoint: {model_dir}")
            else:
                # Find most recent checkpoint
                checkpoints = sorted([
                    d for d in model_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('checkpoint_epoch')
                ])
                if checkpoints:
                    model_dir = checkpoints[-1]
                    logger.info(f"Using most recent checkpoint: {model_dir}")
                else:
                    logger.info("No checkpoints found, using base directory")
        
        logger.info(f"Loading model from: {model_dir}")
        
        # Verify the directory exists
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Initialize the custom model architecture
        model = LanguageAwareTransformer(
            num_labels=6,
            hidden_size=1024,
            num_attention_heads=16,
            model_name='xlm-roberta-large'
        )
        
        # Load the trained weights
        weights_path = model_dir / 'pytorch_model.bin'
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
            
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully")
        
        # Load base XLM-RoBERTa tokenizer directly
        logger.info("Loading XLM-RoBERTa tokenizer...")
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        
        # Load training metadata if available
        metadata_path = model_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.info(f"Loaded checkpoint metadata: Epoch {metadata.get('epoch', 'unknown')}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error("\nPlease ensure that:")
        logger.error("1. You have trained the model first using train.py")
        logger.error("2. The model weights are saved in the correct location")
        logger.error("3. You have sufficient permissions to access the model files")
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
        for category, recommended in DEFAULT_THRESHOLDS.items():
            if category in adjusted[lang_id]:
                # Only increase threshold if recommended is higher
                adjusted[lang_id][category] = max(adjusted[lang_id][category], recommended)
    
    return adjusted

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
    Enhanced language detection using langdetect with multiple fallback methods:
    1. Primary: langdetect library
    2. Fallback 1: ASCII analysis for English
    3. Fallback 2: Unicode range analysis
    4. Fallback 3: Tokenizer statistics
    """
    try:
        # Clean text
        text = text.strip()
        
        # If empty or just punctuation, default to English
        if not text or not re.search(r'\w', text):
            return SUPPORTED_LANGUAGES['en']
            
        # Primary method: Use langdetect
        try:
            detected_code = detect(text)
            # Map some common language codes that might differ
            lang_mapping = {
                'eng': 'en',
                'rus': 'ru',
                'tur': 'tr',
                'spa': 'es',
                'fra': 'fr',
                'ita': 'it',
                'por': 'pt'
            }
            detected_code = lang_mapping.get(detected_code, detected_code)
            
            if detected_code in SUPPORTED_LANGUAGES:
                return SUPPORTED_LANGUAGES[detected_code]
        except LangDetectException:
            pass  # Continue to fallback methods
            
        # Fallback 1: If text is ASCII only, likely English
        if all(ord(c) < 128 for c in text):
            return SUPPORTED_LANGUAGES['en']
        
        # Fallback 2 & 3: Combine Unicode analysis and tokenizer statistics
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
        logger.warning(f"Language detection failed ({str(e)}). Using English.")
        return SUPPORTED_LANGUAGES['en']

def predict_toxicity(text, model, tokenizer, device):
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
    
    # Create results dictionary using optimized thresholds
    results = {}
    for label, prob in zip(labels, probabilities):
        threshold = DEFAULT_THRESHOLDS.get(label, 0.5)  # Use optimized defaults
        results[label] = {
            'probability': float(prob),
            'is_toxic': prob > threshold,
            'threshold': threshold
        }
    
    return results, lang_id

def main():
    # Load model
    print("Loading model...")
    model_path = 'weights/toxic_classifier_xlm-roberta-large/latest'
    model, tokenizer, device = load_model(model_path)
    
    if model is None or tokenizer is None:
        return
    
    while True:
        try:
            # Get input text with proper Unicode handling
            print("\nEnter text to analyze (or 'q' to quit):")
            try:
                if sys.platform == 'win32':
                    # Windows-specific input handling
                    text = sys.stdin.buffer.readline().decode('utf-8').strip()
                else:
                    text = input().strip()
            except UnicodeDecodeError:
                # Fallback to latin-1 if UTF-8 fails
                if sys.platform == 'win32':
                    text = sys.stdin.buffer.readline().decode('latin-1').strip()
                else:
                    text = sys.stdin.buffer.readline().decode('latin-1').strip()
            
            if text.lower() == 'q':
                break
            
            if not text:
                print("Please enter some text to analyze.")
                continue
            
            # Make prediction
            print("\nAnalyzing text...")
            predictions, lang_id = predict_toxicity(text, model, tokenizer, device)
            
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
                
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            print("\nAn unexpected error occurred. Please try again.")
            continue

if __name__ == "__main__":
    main() 