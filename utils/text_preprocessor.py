import re
import nltk
import logging
from typing import List, Set, Dict, Optional
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from TurkishStemmer import TurkishStemmer
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import unicodedata
import warnings

# Suppress BeautifulSoup warning about markup resembling a filename
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {str(e)}")

# Configure logging
logging.basicConfig(level=logging.WARNING)

class TextPreprocessor:
    """
    A comprehensive text preprocessor for multilingual text cleaning and normalization.
    Supports multiple languages and provides various text cleaning operations.
    """
    
    SUPPORTED_LANGUAGES = {'en', 'es', 'fr', 'it', 'pt', 'ru', 'tr'}
    
    # Common contractions mapping (can be extended)
    CONTRACTIONS = {
        "ain't": "is not", "aren't": "are not", "can't": "cannot", 
        "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
        "don't": "do not", "hadn't": "had not", "hasn't": "has not",
        "haven't": "have not", "he'd": "he would", "he'll": "he will",
        "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am",
        "i've": "i have", "isn't": "is not", "it's": "it is",
        "let's": "let us", "shouldn't": "should not", "that's": "that is",
        "there's": "there is", "they'd": "they would", "they'll": "they will",
        "they're": "they are", "they've": "they have", "wasn't": "was not",
        "we'd": "we would", "we're": "we are", "we've": "we have",
        "weren't": "were not", "what's": "what is", "where's": "where is",
        "who's": "who is", "won't": "will not", "wouldn't": "would not",
        "you'd": "you would", "you'll": "you will", "you're": "you are",
        "you've": "you have"
    }
    
    def __init__(self, languages: Optional[Set[str]] = None):
        """
        Initialize the text preprocessor with specified languages.
        
        Args:
            languages: Set of language codes to support. If None, all supported languages are used.
        """
        self.languages = languages or self.SUPPORTED_LANGUAGES
        self._initialize_resources()
        
    def _initialize_resources(self):
        """Initialize language-specific resources like stop words and stemmers."""
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize stop words for each language
        self.stop_words = {}
        nltk_langs = {
            'en': 'english', 'es': 'spanish', 'fr': 'french',
            'it': 'italian', 'pt': 'portuguese', 'ru': 'russian'
        }
        
        for lang, nltk_name in nltk_langs.items():
            if lang in self.languages:
                try:
                    self.stop_words[lang] = set(stopwords.words(nltk_name))
                except Exception as e:
                    self.logger.warning(f"Could not load stop words for {lang}: {str(e)}")
                    self.stop_words[lang] = set()
        
        # Add Turkish stop words manually
        if 'tr' in self.languages:
            self.stop_words['tr'] = {
                'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 
                'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 
                'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 
                'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 
                'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 
                'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani'
            }
        
        # Initialize stemmers
        self.stemmers = {}
        for lang, name in [
            ('en', 'english'), ('es', 'spanish'), ('fr', 'french'),
            ('it', 'italian'), ('pt', 'portuguese'), ('ru', 'russian')
        ]:
            if lang in self.languages:
                self.stemmers[lang] = SnowballStemmer(name)
        
        # Initialize Turkish stemmer separately
        if 'tr' in self.languages:
            self.stemmers['tr'] = TurkishStemmer()
    
    def remove_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        return BeautifulSoup(text, "html.parser").get_text()
    
    def expand_contractions(self, text: str) -> str:
        """Expand contractions in English text."""
        for contraction, expansion in self.CONTRACTIONS.items():
            text = re.sub(rf'\b{contraction}\b', expansion, text, flags=re.IGNORECASE)
        return text
    
    def remove_accents(self, text: str) -> str:
        """Remove accents from text while preserving base characters."""
        return ''.join(c for c in unicodedata.normalize('NFKD', text)
                      if not unicodedata.combining(c))
    
    def clean_text(self, text: str, lang: str = 'en', 
                  remove_stops: bool = True, 
                  remove_numbers: bool = True,
                  remove_urls: bool = True,
                  remove_emails: bool = True,
                  remove_mentions: bool = True,
                  remove_hashtags: bool = True,
                  expand_contractions: bool = True,
                  remove_accents: bool = False,
                  min_word_length: int = 2) -> str:
        """
        Clean and normalize text with configurable options.
        
        Args:
            text: Input text to clean
            lang: Language code of the text
            remove_stops: Whether to remove stop words
            remove_numbers: Whether to remove numbers
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            remove_mentions: Whether to remove social media mentions
            remove_hashtags: Whether to remove hashtags
            expand_contractions: Whether to expand contractions (English only)
            remove_accents: Whether to remove accents from characters
            min_word_length: Minimum length of words to keep
            
        Returns:
            Cleaned text string
        """
        try:
            # Convert to string and lowercase
            text = str(text).lower().strip()
            
            # Remove HTML tags if any HTML-like content is detected
            if '<' in text and '>' in text:
                text = self.remove_html(text)
            
            # Remove URLs if requested
            if remove_urls:
                text = re.sub(r'http\S+|www\S+', '', text)
            
            # Remove email addresses if requested
            if remove_emails:
                text = re.sub(r'\S+@\S+', '', text)
            
            # Remove mentions if requested
            if remove_mentions:
                text = re.sub(r'@\w+', '', text)
            
            # Remove hashtags if requested
            if remove_hashtags:
                text = re.sub(r'#\w+', '', text)
            
            # Remove numbers if requested
            if remove_numbers:
                text = re.sub(r'\d+', '', text)
            
            # Expand contractions for English text
            if lang == 'en' and expand_contractions:
                text = self.expand_contractions(text)
            
            # Remove accents if requested
            if remove_accents:
                text = self.remove_accents(text)
            
            # Language-specific character cleaning
            if lang == 'tr':
                text = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]', '', text)
            elif lang == 'ru':
                text = re.sub(r'[^а-яА-Я\s]', '', text)
            else:
                text = re.sub(r'[^\w\s]', '', text)
            
            # Simple word splitting as fallback if tokenization fails
            try:
                words = word_tokenize(text)
            except Exception as e:
                self.logger.debug(f"Word tokenization failed, falling back to simple split: {str(e)}")
                words = text.split()
            
            # Remove stop words if requested
            if remove_stops and lang in self.stop_words:
                words = [w for w in words if w not in self.stop_words[lang]]
            
            # Remove short words
            words = [w for w in words if len(w) > min_word_length]
            
            # Rejoin words
            return ' '.join(words)
            
        except Exception as e:
            self.logger.warning(f"Error in text cleaning: {str(e)}")
            return text
    
    def stem_text(self, text: str, lang: str = 'en') -> str:
        """
        Apply language-specific stemming to text.
        
        Args:
            text: Input text to stem
            lang: Language code of the text
            
        Returns:
            Stemmed text string
        """
        try:
            if lang not in self.stemmers:
                return text
                
            words = text.split()
            stemmed_words = [self.stemmers[lang].stem(word) for word in words]
            return ' '.join(stemmed_words)
            
        except Exception as e:
            self.logger.warning(f"Error in text stemming: {str(e)}")
            return text
    
    def preprocess_text(self, text: str, lang: str = 'en', 
                       clean_options: Dict = None, 
                       do_stemming: bool = True) -> str:
        """
        Complete preprocessing pipeline combining cleaning and stemming.
        
        Args:
            text: Input text to preprocess
            lang: Language code of the text
            clean_options: Dictionary of options to pass to clean_text
            do_stemming: Whether to apply stemming
            
        Returns:
            Preprocessed text string
        """
        # Use default cleaning options if none provided
        clean_options = clean_options or {}
        
        # Clean text
        cleaned_text = self.clean_text(text, lang, **clean_options)
        
        # Apply stemming if requested
        if do_stemming:
            cleaned_text = self.stem_text(cleaned_text, lang)
        
        return cleaned_text.strip()

# Usage example
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Example texts in different languages
    examples = {
        'en': "Here's an example! This is a test text with @mentions and #hashtags http://example.com",
        'es': "¡Hola! Este es un ejemplo de texto en español con números 12345",
        'fr': "Voici un exemple de texte en français avec des accents é è à",
        'tr': "Bu bir Türkçe örnek metindir ve bazı özel karakterler içerir."
    }
    
    # Process each example
    for lang, text in examples.items():
        print(f"\nProcessing {lang} text:")
        print("Original:", text)
        processed = preprocessor.preprocess_text(text, lang)
        print("Processed:", processed) 