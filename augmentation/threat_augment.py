import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from langdetect import detect
from presidio_analyzer import AnalyzerEngine
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import gc
import logging
from typing import List, Dict, Optional, Tuple
import random
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('augmentation.log'),
        logging.StreamHandler()
    ]
)

class ThreatAugmenter:
    def __init__(
        self, 
        seed_samples_path: str = "dataset/split/train.csv",
        toxicity_model_path: Optional[str] = None,
        batch_size: int = 8
    ):
        # Check dependencies
        try:
            from presidio_analyzer import AnalyzerEngine
            from langdetect import detect
        except ImportError:
            raise ImportError("Please install required packages: pip install presidio-analyzer langdetect")

        # Validate dataset
        if not Path(seed_samples_path).exists():
            raise FileNotFoundError(f"Dataset not found: {seed_samples_path}")

        # Memory optimization
        torch.cuda.empty_cache()
        gc.collect()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        logging.info("Initializing models...")
        
        # Initialize FLAN-T5-base with memory optimizations
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base",
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        
        # Initialize XLM-RoBERTa for validation
        try:
            self.validator = AutoModelForSequenceClassification.from_pretrained(
                "xlm-roberta-large",
                num_labels=6,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.validator_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
            logging.info("Loaded XLM-RoBERTa for toxicity validation")
        except Exception as e:
            raise ValueError(f"Failed to load toxicity model: {str(e)}")
        
        # Label mapping for toxicity classification
        self.label_map = {
            'toxic': 0,
            'severe_toxic': 1,
            'obscene': 2,
            'threat': 3,
            'insult': 4,
            'identity_hate': 5
        }
        
        # Initialize PII analyzer
        self.pii_analyzer = AnalyzerEngine()
        
        # Load and validate seed samples
        self.seed_df = pd.read_csv(seed_samples_path)
        required_columns = ['comment_text', 'lang', 'threat']
        missing_columns = [col for col in required_columns if col not in self.seed_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.en_threat_samples = self.seed_df[
            (self.seed_df['lang'] == 'en') & 
            (self.seed_df['threat'] == 1)
        ]
        logging.info(f"Found {len(self.en_threat_samples)} EN threat samples")
        
        if len(self.en_threat_samples) == 0:
            raise ValueError("No English threat samples found in dataset")

    def generate_prompt(self, seed_text: str) -> str:
        """Generate a prompt for threat content generation"""
        return f"""Generate a threatening comment similar to this example, but make it unique:
Example: {seed_text}
Requirements:
- Make it threatening but avoid explicit profanity
- Keep it under 50 words
- Use different wording than the example
- Make it sound natural"""

    @torch.inference_mode()
    def generate_samples(
        self, 
        prompts: List[str], 
        temperature: float = 0.9,
        max_attempts: int = 3
    ) -> List[str]:
        """Generate samples using FLAN-T5 with retry logic"""
        for attempt in range(max_attempts):
            try:
                inputs = self.llm_tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=128
                ).to(self.device)
                
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=1,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
                
                generated_texts = self.llm_tokenizer.batch_decode(
                    outputs, 
                    skip_special_tokens=True
                )
                return [text.strip() for text in generated_texts]
                
            except Exception as e:
                logging.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_attempts - 1:
                    logging.error("All generation attempts failed")
                    return []
                torch.cuda.empty_cache()
                gc.collect()
    
    @torch.inference_mode()
    def validate_toxicity(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate generated texts using XLM-RoBERTa"""
        if not texts:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        
        # Truncate long texts
        texts = [text[:512] for text in texts]
        
        try:
            inputs = self.validator_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            outputs = self.validator(**inputs)
            predictions = torch.sigmoid(outputs.logits)
            
            # Get specific scores
            threat_scores = predictions[:, self.label_map['threat']]
            other_toxicity = torch.mean(torch.stack([
                predictions[:, self.label_map[label]]
                for label in ['toxic', 'severe_toxic', 'obscene', 'insult', 'identity_hate']
            ]), dim=0)
            
            # Validation criteria
            valid_mask = (threat_scores > 0.7) & (other_toxicity < 0.8)
            return valid_mask, threat_scores
            
        except Exception as e:
            logging.error(f"Toxicity validation error: {str(e)}")
            return torch.zeros(len(texts), dtype=torch.bool, device=self.device), None
    
    def validate_language_and_pii(self, texts: List[str]) -> List[bool]:
        """Validate language consistency and check for PII"""
        valid_samples = []
        for text in texts:
            try:
                # Check length
                if len(text.split()) > 50:
                    valid_samples.append(False)
                    continue
                    
                # Check language
                if detect(text) != 'en':
                    valid_samples.append(False)
                    continue
                
                # Check PII
                entities = self.pii_analyzer.analyze(text, language='en')
                if len(entities) > 0:
                    valid_samples.append(False)
                    continue
                
                valid_samples.append(True)
                
            except Exception as e:
                logging.warning(f"Validation error for text '{text[:50]}...': {str(e)}")
                valid_samples.append(False)
        
        return valid_samples
    
    def save_checkpoint(self, samples: List[str], output_dir: Path):
        """Save intermediate results"""
        checkpoint_path = output_dir / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump({'generated_samples': samples}, f)
    
    def augment_dataset(
        self, 
        target_samples: int = 3000,
        output_dir: str = "dataset/augmented",
        checkpoint_frequency: int = 100
    ):
        """Main augmentation loop with checkpointing"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Starting augmentation to generate {target_samples} samples")
        
        generated_samples = []
        pbar = tqdm(total=target_samples, desc="Generating samples")
        
        try:
            while len(generated_samples) < target_samples:
                # Sample seed texts
                seed_texts = self.en_threat_samples['comment_text'].sample(
                    min(self.batch_size, target_samples - len(generated_samples))
                ).tolist()
                
                prompts = [self.generate_prompt(text) for text in seed_texts]
                
                # Generate new samples
                new_samples = self.generate_samples(prompts)
                if not new_samples:
                    continue
                    
                # Validate toxicity
                toxicity_mask, threat_scores = self.validate_toxicity(new_samples)
                valid_samples = [s for i, s in enumerate(new_samples) if toxicity_mask[i]]
                
                if valid_samples:
                    # Validate language and PII
                    lang_pii_mask = self.validate_language_and_pii(valid_samples)
                    final_samples = [s for i, s in enumerate(valid_samples) if lang_pii_mask[i]]
                    
                    # Add to collection
                    generated_samples.extend(final_samples)
                    pbar.update(len(final_samples))
                    
                    # Log batch stats
                    logging.info(f"Batch success rate: {len(final_samples)}/{self.batch_size}")
                    
                    # Checkpoint
                    if len(generated_samples) % checkpoint_frequency == 0:
                        self.save_checkpoint(generated_samples, output_path)
                
                # Memory cleanup
                if len(generated_samples) % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        except KeyboardInterrupt:
            logging.info("Augmentation interrupted by user")
        except Exception as e:
            logging.error(f"Augmentation error: {str(e)}")
        finally:
            pbar.close()
        
        # Create final dataset
        aug_df = pd.DataFrame({
            'comment_text': generated_samples[:target_samples],
            'toxic': 1,
            'severe_toxic': 0,
            'obscene': 0,
            'threat': 1,
            'insult': 0,
            'identity_hate': 0,
            'lang': 'en'
        })
        
        # Save final results
        aug_df.to_csv(output_path / "en_threat_augmented.csv", index=False)
        logging.info(f"Successfully generated {len(aug_df)} samples")
        return aug_df

if __name__ == "__main__":
    try:
        # Set random seed for reproducibility
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize augmenter with default model
        augmenter = ThreatAugmenter(
            seed_samples_path="dataset/split/train.csv"  # Removed toxicity_model_path
        )
        
        # Run augmentation
        augmented_df = augmenter.augment_dataset(
            target_samples=3000,
            output_dir="dataset/augmented",
            checkpoint_frequency=100
        )
        
        print(f"Generated {len(augmented_df)} new EN threat samples")
        
    except Exception as e:
        logging.error(f"Augmentation failed: {str(e)}")