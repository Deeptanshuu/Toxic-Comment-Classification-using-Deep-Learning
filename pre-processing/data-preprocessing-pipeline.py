import re
import pandas as pd
import numpy as np
from transformers import XLMRobertaTokenizer
from sklearn.utils.class_weight import compute_class_weight
from concurrent.futures import ThreadPoolExecutor
from nltk.stem import SnowballStemmer
from TurkishStemmer import TurkishStemmer
from langdetect import detect, DetectorFactory, detect_langs
from alive_progress import alive_bar, config_handler
import logging
import os
from time import time
from tqdm.auto import tqdm
import multiprocessing

# Configure alive-progress for better visuals
config_handler.set_global(
    spinner='dots_waves',
    bar='smooth',
    unknown='brackets',
    force_tty=True
)

# Set seed for reproducible language detection
DetectorFactory.seed = 0

class MultilingualPreprocessor:
    def __init__(self, max_length=128):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.max_length = max_length
        self.num_workers = min(multiprocessing.cpu_count() - 1, 8)  # Optimal worker count
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize stemmers
        self._initialize_stemmers()
        
        # Ensure data directories exist
        os.makedirs('dataset/processed', exist_ok=True)
        
    def _initialize_stemmers(self):
        """Initialize stemmers with progress bar"""
        self.logger.info("Initializing stemmers...")
        with alive_bar(6, title='Loading Stemmers', force_tty=True) as bar:
            self.turkish_stemmer = TurkishStemmer()
            bar()
            self.lang_stemmers = {}
            for lang, name in [
                ('en', 'english'), ('es', 'spanish'), 
                ('it', 'italian'), ('ru', 'russian'),
                ('pt', 'portuguese')
            ]:
                self.lang_stemmers[lang] = SnowballStemmer(name)
                bar()
        
    def detect_language(self, text):
        """Enhanced language detection with better preprocessing"""
        try:
            if not text or len(text.split()) < 3:
                return 'en'
            
            # Preprocess text for better detection
            text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
            text = re.sub(r'@\w+', '', text)  # Remove mentions
            text = re.sub(r'#\w+', '', text)  # Remove hashtags
            text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
            
            if not text:
                return 'en'
            
            # First check for specific patterns
            patterns = {
                'ru': r'[а-яА-Я]{3,}',  # Russian characters
                'tr': r'[çğıöşüÇĞİÖŞÜ]{2,}',  # Turkish specific characters
                'es': r'\b(el|la|los|las|esto|esta|estos|estas|que|porque|como)\b',  # Spanish common words
                'pt': r'\b(isso|esta|estes|estas|que|porque|como|não|sim)\b',  # Portuguese common words
                'it': r'\b(il|lo|la|gli|questo|questa|questi|queste|che|perché|come)\b'  # Italian common words
            }
            
            for lang, pattern in patterns.items():
                if re.search(pattern, text):
                    return lang
            
            # If no specific patterns found, use langdetect
            langs = detect_langs(text)
            if not langs:
                return 'en'
            
            # Get top 2 languages and their probabilities
            top_langs = langs[:2]
            
            # If top language has high confidence
            if top_langs[0].prob > 0.3:  # Lowered threshold
                lang_mapping = {
                    'en': 'en',
                    'es': 'es',
                    'tr': 'tr',
                    'it': 'it',
                    'ru': 'ru',
                    'pt': 'pt'
                }
                return lang_mapping.get(top_langs[0].lang, 'en')
            
            # If close call between two languages, check for specific patterns
            if len(top_langs) > 1 and (top_langs[0].prob - top_langs[1].prob) < 0.1:
                for lang in top_langs:
                    if lang.lang in patterns and re.search(patterns[lang.lang], text):
                        return lang.lang
            
            return 'en'
            
        except Exception as e:
            if 'No features' not in str(e):
                self.logger.warning(f"Language detection error for text '{text[:100]}...': {str(e)}")
            return 'en'
            
    def detect_language_batch(self, texts, batch_size=100):
        """Batch language detection with progress tracking"""
        results = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.detect_language(text) for text in batch]
            results.extend(batch_results)
            
            if (i + batch_size) % 1000 == 0:
                self.logger.info(f"Processed {i + len(batch)}/{total} texts")
                
        return results

    def clean_text(self, text, lang='en'):
        """Language-specific text cleaning"""
        # Common cleaning
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Language-specific rules
        if lang == 'tr':
            text = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]', '', text)
        elif lang == 'ru':
            text = re.sub(r'[^а-яА-Я\s]', '', text)
        else:
            text = re.sub(r'[^\w\s]', '', text)
            
        return text.lower()

    def stem_text(self, text, lang='en'):
        """Apply language-specific stemming"""
        words = text.split()
        if lang == 'tr':
            # Use TurkishStemmer for Turkish
            return ' '.join([self.turkish_stemmer.stem(word) for word in words])
        else:
            # Use SnowballStemmer for other languages, default to English if language not supported
            stemmer = self.lang_stemmers.get(lang, self.lang_stemmers['en'])
            return ' '.join([stemmer.stem(word) for word in words])

    def tokenize_batch(self, batch):
        """Parallel tokenization with language detection"""
        processed = []
        for text in batch:
            lang = self.detect_language(text)
            cleaned = self.clean_text(text, lang)
            stemmed = self.stem_text(cleaned, lang)
            encoded = self.tokenizer.encode_plus(
                stemmed,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=False
            )
            processed.append({
                'input_word_ids': encoded['input_ids'],
                'input_mask': encoded['attention_mask']
            })
        return processed

    def create_class_weights(self, df):
        """Dynamic class weighting per language"""
        weights = {}
        for lang in df['lang'].unique():
            lang_df = df[df['lang'] == lang]
            for col in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']:
                if col in lang_df.columns:  # Check if column exists
                    # Convert to numpy array
                    classes = np.array([0, 1])
                    y = lang_df[col].values
                    if len(np.unique(y)) > 1:  # Check if we have both classes
                        class_weights = compute_class_weight('balanced', classes=classes, y=y)
                        weights[f'{lang}_{col}'] = class_weights[1]
                    else:
                        weights[f'{lang}_{col}'] = 1.0  # Default weight if only one class present
        return weights

    def preprocess_pipeline(self, df_path, output_path):
        """Complete preprocessing pipeline with progress tracking"""
        start_time = time()
        
        # Load data with progress bar
        self.logger.info("\n1. Loading dataset...")
        chunksize = 10000
        chunks = []
        
        try:
            # Count total rows with proper encoding
            with open(df_path, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f) - 1  # Count rows minus header
            
            with alive_bar(total_rows, title='Loading Data', force_tty=True) as bar:
                # Read CSV in chunks with proper encoding
                for chunk in pd.read_csv(df_path, chunksize=chunksize, encoding='utf-8', on_bad_lines='warn'):
                    chunks.append(chunk)
                    bar(len(chunk))
            
            df = pd.concat(chunks, ignore_index=True)
            
        except UnicodeDecodeError:
            self.logger.warning("UTF-8 encoding failed, trying with 'latin-1'...")
            # Fallback to latin-1 encoding if UTF-8 fails
            with open(df_path, 'r', encoding='latin-1') as f:
                total_rows = sum(1 for _ in f) - 1
            
            with alive_bar(total_rows, title='Loading Data', force_tty=True) as bar:
                for chunk in pd.read_csv(df_path, chunksize=chunksize, encoding='latin-1', on_bad_lines='warn'):
                    chunks.append(chunk)
                    bar(len(chunk))
            
            df = pd.concat(chunks, ignore_index=True)
        
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise
        
        # Language detection with parallel processing
        self.logger.info("\n2. Detecting languages...")
        with alive_bar(len(df), title='Language Detection', force_tty=True) as bar:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for i in range(0, len(df), chunksize):
                    batch = df['comment_text'].iloc[i:i+chunksize]
                    future = executor.submit(self.detect_language_batch, batch)
                    futures.append((i, future))
                
                lang_results = []
                for i, future in futures:
                    lang_results.extend(future.result())
                    bar(len(future.result()))
        
        df['lang'] = lang_results
        
        # Log language distribution with percentages
        self._log_language_distribution(df)
        
        # Process text in parallel with progress tracking
        self.logger.info("\n3. Processing text...")
        batch_size = 1000
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        with alive_bar(total_batches, title='Text Processing', force_tty=True) as bar:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    future = executor.submit(self._process_batch, batch)
                    futures.append(future)
                
                processed_batches = []
                for future in futures:
                    processed_batches.append(future.result())
                    bar()
        
        # Combine processed batches
        processed_df = pd.concat(processed_batches, ignore_index=True)
        
        # Calculate class weights
        self.logger.info("\n4. Calculating class weights...")
        class_weights = self.create_class_weights(processed_df)
        
        # Save results
        self.logger.info("\n5. Saving processed data...")
        processed_df.to_csv(output_path, index=False)
        
        # Log completion statistics
        total_time = time() - start_time
        self._log_completion_stats(processed_df, total_time)
        
        return processed_df, class_weights
    
    def _process_batch(self, batch):
        """Process a batch of texts with all preprocessing steps"""
        processed_batch = batch.copy()
        
        # Tokenize texts
        tokenized = self.tokenize_batch(batch['comment_text'].tolist())
                
        # Add processed columns
        for key in ['input_word_ids', 'input_mask']:
            processed_batch[key] = [item[key] for item in tokenized]
        
        # Generate segment IDs
        processed_batch['all_segment_id'] = processed_batch['input_word_ids'].apply(
            lambda x: [0]*len(x)
        )
        
        return processed_batch
    
    def _log_language_distribution(self, df):
        """Log detailed language distribution statistics"""
        lang_dist = df['lang'].value_counts()
        total = len(df)
        
        self.logger.info("\nLanguage Distribution:")
        self.logger.info("-" * 50)
        for lang, count in lang_dist.items():
            percentage = (count / total) * 100
            self.logger.info(f"{lang:2s}: {count:8,d} texts ({percentage:6.2f}%)")
    
    def _log_completion_stats(self, df, total_time):
        """Log completion statistics"""
        self.logger.info("\nPreprocessing Complete!")
        self.logger.info("-" * 50)
        self.logger.info(f"Total texts processed: {len(df):,}")
        self.logger.info(f"Total time: {total_time/60:.2f} minutes")
        self.logger.info(f"Processing speed: {len(df)/total_time:.2f} texts/second")
        self.logger.info(f"Output file size: {os.path.getsize(processed_path)/1024/1024:.2f} MB")

# Usage Example
if __name__ == "__main__":
    processor = MultilingualPreprocessor()
    
    # Process main dataset
    raw_path = os.path.join('dataset', 'raw', 'jigsaw-toxic-comment-train.csv')
    processed_path = os.path.join('dataset', 'processed', 'jigsaw-toxic-comment-train-processed.csv')
    
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Input file not found: {raw_path}")
        
    df, weights = processor.preprocess_pipeline(raw_path, processed_path)
    
    print(f"\nProcessed data saved to {processed_path}")
    print("\nLanguage distribution:")
    print(df['lang'].value_counts(normalize=True).mul(100).round(2).to_string())
    print("\nSample class weights:", {k: v for k,v in weights.items() if 'toxic' in k})

# Validation Checks
def validate_preprocessing(input_path):
    """Ensure processed data matches expected format"""
    df = pd.read_csv(input_path, nrows=5)
    
    # Check required columns
    assert {'input_word_ids', 'input_mask', 'all_segment_id'}.issubset(df.columns)
    
    # Validate sequence lengths
    seq_lengths = df['input_word_ids'].apply(lambda x: len(eval(x)))
    assert all(seq_lengths == 128)
    
    # Check mask values
    masks = df['input_mask'].apply(lambda x: eval(x))
    assert all(sum(m) >= 5 for m in masks)  # At least 5 meaningful tokens
    
    print("Validation passed!")

validate_preprocessing(processed_path)
