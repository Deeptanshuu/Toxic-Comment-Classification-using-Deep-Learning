import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from langdetect import detect
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
import gc
from typing import List
import json
from datetime import datetime
import time
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Create log directories
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Get timestamp for log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"generation_{timestamp}.log"

# Configure logging once at the start
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Starting new run. Log file: {log_file}")

def log_separator(message: str = ""):
    """Print a separator line with optional message"""
    if message:
        logger.info("\n" + "="*40 + f" {message} " + "="*40)
    else:
        logger.info("\n" + "="*100)

class FastThreatValidator:
    """Fast threat validation using logistic regression"""
    def __init__(self, model_path: str = "weights/threat_validator.joblib"):
        self.model_path = model_path
        if Path(model_path).exists():
            logger.info("Loading fast threat validator...")
            model_data = joblib.load(model_path)
            self.vectorizer = model_data['vectorizer']
            self.model = model_data['model']
            logger.info("✓ Fast validator loaded")
        else:
            logger.info("Training fast threat validator...")
            self._train_validator()
            logger.info("✓ Fast validator trained and saved")
    
    def _train_validator(self):
        """Train a simple logistic regression model for threat detection"""
        # Load training data
        train_df = pd.read_csv("dataset/split/train.csv")
        
        # Prepare data
        X = train_df['comment_text'].fillna('')
        y = train_df['threat']
        
        # Create and fit vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            strip_accents='unicode',
            min_df=2
        )
        X_vec = self.vectorizer.fit_transform(X)
        
        # Train model
        self.model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=200,
            n_jobs=-1
        )
        self.model.fit(X_vec, y)
        
        # Save model
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.model
        }, self.model_path)
    
    def validate(self, texts: List[str], threshold: float = 0.6) -> List[bool]:
        """Validate texts using the fast model"""
        # Vectorize texts
        X = self.vectorizer.transform(texts)
        
        # Get probabilities
        probs = self.model.predict_proba(X)[:, 1]
        
        # Return boolean mask
        return probs >= threshold

class ThreatAugmenter:
    def __init__(self, seed_samples_path: str = "dataset/split/train.csv"):
        log_separator("INITIALIZATION")
        
        # Use global log file
        self.log_file = log_file
        
        # Initialize generation buffer
        self.generation_buffer = []
        self.buffer_size = 100  # Flush buffer every 100 entries
        
        # Multi-GPU setup
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 0:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info(f"Found {self.num_gpus} GPUs:")
            for i in range(self.num_gpus):
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f}GB)")
        
        # Load models
        log_separator("LOADING MODELS")
        logger.info("Loading Mistral-7B...")
        
        # Configure model for multi-GPU
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            device_map="balanced",  # Ensures proper dual GPU usage
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            max_memory={0: "22GB", 1: "22GB"}  # Explicitly set memory limits for each GPU
        )
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            padding_side="left",
            use_fast=True
        )
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        logger.info("✓ Mistral-7B loaded")
        
        # Initialize fast validator
        self.validator = FastThreatValidator()
        logger.info("✓ Fast validator initialized")
        
        # Load and preprocess data
        log_separator("LOADING DATA")
        logger.info(f"Loading samples from {seed_samples_path}")
        self.seed_df = pd.read_csv(seed_samples_path)
        self.en_threat_samples = self.seed_df[
            (self.seed_df['lang'] == 'en') & 
            (self.seed_df['threat'] == 1)
        ]
        logger.info(f"Found {len(self.en_threat_samples)} EN threat samples")
        
        # Optimize batch processing
        self.max_batch_size = 48  # Increased batch size
        self.prefetch_factor = 4
        self.num_workers = 8
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        log_separator()

    def generate_prompt(self, seed_text: str) -> str:
        """Generate a prompt for realistic threatening content"""
        return f"""<s>[INST] Generate a threatening social media comment that is realistic and intimidating.

Reference example (generate something different): {seed_text}

Requirements:
- Make it genuinely threatening
- Use authentic social media language
- Keep it under 50 words
- Must be different from example

Generate ONLY the comment: [/INST]"""

    def flush_buffer(self):
        """Flush the generation buffer to disk"""
        if self.generation_buffer:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    for entry in self.generation_buffer:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                self.generation_buffer = []
            except Exception as e:
                logger.error(f"Failed to flush buffer: {str(e)}")

    def log_generation(self, seed_text: str, prompt: str, generated_text: str, is_valid: bool):
        """Buffer log generation details"""
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed_text": seed_text,
            "prompt": prompt,
            "generated_text": generated_text,
            "is_valid": is_valid
        }
        
        self.generation_buffer.append(log_entry)
        
        # Flush buffer if it reaches the size limit
        if len(self.generation_buffer) >= self.buffer_size:
            self.flush_buffer()

    def generate_samples(self, prompts: List[str], seed_texts: List[str]) -> List[str]:
        """Generate samples using Mistral-7B-Instruct with optimized batching"""
        try:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                # Process in optimal batch sizes
                batch_size = len(prompts)
                inputs = self.llm_tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                ).to(self.llm.device)
                
                # Optimize generation parameters
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=100,  # Increased to ensure we get complete responses
                    temperature=0.95,
                    do_sample=True,
                    top_p=0.92,
                    top_k=50,
                    num_return_sequences=1,
                    repetition_penalty=1.15,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1  # Disable beam search for speed
                )
                
                texts = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=False)
                cleaned_texts = []
                
                # Process responses
                for idx, text in enumerate(texts):
                    # Debug output
                    logger.debug(f"Raw output {idx+1}: {text}")
                    
                    # Extract response between [/INST] and </s>
                    if "[/INST]" in text:
                        response = text.split("[/INST]")[-1].split("</s>")[0].strip()
                        response = response.strip().strip('"').strip("'")
                        
                        # Debug output
                        logger.debug(f"Extracted response {idx+1}: {response}")
                        
                        word_count = len(response.split())
                        if (word_count >= 3 and word_count <= 50 and
                            not any(x in response.lower() for x in [
                                "generate",
                                "requirements:",
                                "reference",
                                "[inst]",
                                "example"
                            ])):
                            cleaned_texts.append(response)
                            logger.info(f"✓ [{idx+1}] {response}")
                        else:
                            logger.info(f"✗ [{idx+1}] Failed validation (words: {word_count})")
                    else:
                        logger.info(f"✗ [{idx+1}] Invalid format - No [/INST] tag found")
                        logger.debug(f"Invalid text: {text}")
                
                success_rate = len(cleaned_texts) / len(texts) * 100
                logger.info(f"\nGeneration Success: {len(cleaned_texts)}/{len(texts)} ({success_rate:.1f}%)")
                return cleaned_texts
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return []

    def validate_toxicity(self, texts: List[str]) -> torch.Tensor:
        """Validate texts using fast logistic regression"""
        if not texts:
            return torch.zeros(0, dtype=torch.bool)
        
        # Get validation mask from fast validator
        validation_mask = self.validator.validate(texts)
        
        # Convert to torch tensor
        return torch.tensor(validation_mask, dtype=torch.bool, device=self.llm.device)
    
    def validate_language(self, texts: List[str]) -> List[bool]:
        """Simple language validation"""
        return [detect(text) == 'en' for text in texts]
    
    def augment_dataset(self, target_samples: int = 3000, batch_size: int = 16):
        """Main augmentation loop with optimized batch processing"""
        try:
            log_separator("STARTING GENERATION")
            logger.info(f"Target: {target_samples} samples | Batch size: {batch_size}")
            
            generated_samples = []
            pbar = tqdm(total=target_samples, desc="Progress")
            
            stats = {
                "total_attempts": 0,
                "valid_samples": 0,
                "invalid_toxicity": 0,
                "invalid_language": 0,
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            while len(generated_samples) < target_samples:
                # Sample and generate
                seed_texts = self.en_threat_samples['comment_text'].sample(batch_size).tolist()
                prompts = [self.generate_prompt(text) for text in seed_texts]
                new_samples = self.generate_samples(prompts, seed_texts)
                
                if not new_samples:
                    continue
                    
                stats["total_attempts"] += len(new_samples)
                
                # Validate samples
                toxicity_mask = self.validate_toxicity(new_samples)
                valid_samples = [s for i, s in enumerate(new_samples) if toxicity_mask[i]]
                stats["invalid_toxicity"] += len(new_samples) - len(valid_samples)
                
                lang_mask = self.validate_language(valid_samples)
                final_samples = [s for i, s in enumerate(valid_samples) if lang_mask[i]]
                stats["invalid_language"] += len(valid_samples) - len(final_samples)
                
                # Buffer log generations
                for seed, prompt, generated in zip(seed_texts, prompts, new_samples):
                    is_valid = generated in final_samples
                    self.log_generation(seed, prompt, generated, is_valid)
                
                # Update progress
                generated_samples.extend(final_samples)
                stats["valid_samples"] = len(generated_samples)
                pbar.update(len(final_samples))
                
                # Print simplified batch stats
                if final_samples:
                    logger.info(
                        f"\n=== Batch Summary ===\n"
                        f"Success Rate: {len(final_samples)}/{batch_size} ({len(final_samples)/batch_size*100:.1f}%)\n"
                        f"Total Progress: {len(generated_samples)}/{target_samples} ({len(generated_samples)/target_samples*100:.1f}%)\n"
                        f"Overall Success: {(stats['valid_samples']/stats['total_attempts']*100):.1f}%"
                    )
                
                # Cleanup
                if len(generated_samples) % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            pbar.close()
            
            # Final buffer flush
            self.flush_buffer()
            
            # Update stats with timing info
            stats["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stats["total_time"] = str(datetime.strptime(stats["end_time"], "%Y-%m-%d %H:%M:%S") - 
                                    datetime.strptime(stats["start_time"], "%Y-%m-%d %H:%M:%S"))
            
            # Save results
            log_separator("SAVING RESULTS")
            output_path = Path("dataset/augmented")
            output_path.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create dataset
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
            
            # Save files
            output_file = output_path / f"threat_augmented_{timestamp}.csv"
            stats_file = log_dir / f"stats_{timestamp}.json"
            
            aug_df.to_csv(output_file, index=False)
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(
                f"Files saved:\n"
                f"- Dataset: {output_file}\n"
                f"- Stats: {stats_file}\n"
                f"- Samples: {self.log_file}\n"
                f"Total time: {stats['total_time']}"
            )
            
            return aug_df
            
        except Exception as e:
            # Ensure buffer is flushed even on error
            self.flush_buffer()
            logger.error(f"Generation failed: {str(e)}")
            raise
        finally:
            # Final cleanup
            self.flush_buffer()
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    
    augmenter = ThreatAugmenter()
    augmented_df = augmenter.augment_dataset()