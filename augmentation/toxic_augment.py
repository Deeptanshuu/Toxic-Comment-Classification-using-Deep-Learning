import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
import gc
from typing import List, Dict
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

# Configure logging
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

class FastToxicValidator:
    """Fast toxicity validation using logistic regression"""
    def __init__(self, model_path: str = "weights/toxic_validator.joblib"):
        self.model_path = model_path
        if Path(model_path).exists():
            logger.info("Loading fast toxic validator...")
            model_data = joblib.load(model_path)
            self.vectorizers = model_data['vectorizers']
            self.models = model_data['models']
            logger.info("✓ Fast validator loaded")
        else:
            logger.info("Training fast toxic validator...")
            self._train_validator()
            logger.info("✓ Fast validator trained and saved")
    
    def _train_validator(self):
        """Train logistic regression models for each toxicity type"""
        # Load training data
        train_df = pd.read_csv("dataset/split/train.csv")
        
        # Labels to validate
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        self.vectorizers = {}
        self.models = {}
        
        # Train a model for each label
        for label in labels:
            # Create and fit vectorizer
            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                strip_accents='unicode',
                min_df=2
            )
            X = vectorizer.fit_transform(train_df['comment_text'].fillna(''))
            y = train_df[label]
            
            # Train model
            model = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=200,
                n_jobs=-1
            )
            model.fit(X, y)
            
            self.vectorizers[label] = vectorizer
            self.models[label] = model
        
        # Save models
        joblib.dump({
            'vectorizers': self.vectorizers,
            'models': self.models
        }, self.model_path)

    def get_probabilities(self, texts: List[str], label: str) -> np.ndarray:
        """Get raw probabilities for a specific label"""
        X = self.vectorizers[label].transform(texts)
        return self.models[label].predict_proba(X)[:, 1]

class ToxicAugmenter:
    def __init__(self):
        logger.info("Initializing ToxicAugmenter...")
        
        # Initialize generation buffer
        self.generation_buffer = []
        self.buffer_size = 100
        
        # Multi-GPU setup
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 0:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info(f"Found {self.num_gpus} GPUs:")
            for i in range(self.num_gpus):
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f}GB)")
        
        # Load models with optimized settings
        logger.info("Loading Mistral-7B...")
        
        # Configure model for multi-GPU with optimized settings
        quantization_config = BitsAndBytesConfig(
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            device_map="balanced",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            max_memory={0: "22GB", 1: "22GB"},
            use_cache=True  # Enable KV cache for faster generation
        )
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            padding_side="left",
            use_fast=True,
            model_max_length=512  # Limit context size for faster processing
        )
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        logger.info("✓ Mistral-7B loaded")
        
        # Initialize validator
        self.validator = FastToxicValidator()
        logger.info("✓ Fast validator initialized")

    def generate_prompt(self, seed_text: str, label_combo: Dict[str, int]) -> str:
        """Generate a prompt based on the target label combination"""
        # Convert label combination to description
        toxicity_types = []
        if label_combo['toxic']:
            toxicity_types.append('toxic')
        if label_combo['severe_toxic']:
            toxicity_types.append('severely toxic')
        if label_combo['obscene']:
            toxicity_types.append('obscene')
        if label_combo['threat']:
            toxicity_types.append('threatening')
        if label_combo['insult']:
            toxicity_types.append('insulting')
        if label_combo['identity_hate']:
            toxicity_types.append('identity-based hate')
        
        toxicity_desc = ', '.join(toxicity_types)
        
        return f"""<s>[INST] Generate a social media comment that is {toxicity_desc}.

Reference example (generate something different): {seed_text}

Requirements:
- Make it genuinely {toxicity_desc}
- Use authentic social media language
- Keep it under 50 words
- Must be different from example

Generate ONLY the comment: [/INST]"""

    def flush_buffer(self):
        """Flush the generation buffer to disk"""
        if self.generation_buffer:
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    for entry in self.generation_buffer:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                self.generation_buffer = []
            except Exception as e:
                logger.error(f"Failed to flush buffer: {str(e)}")

    def log_generation(self, seed_text: str, prompt: str, generated_text: str, validation_results: Dict[str, bool]):
        """Buffer log generation details with proper JSON serialization"""
        # Convert numpy/torch boolean values to Python booleans
        serializable_results = {
            k: bool(v) for k, v in validation_results.items()
        }
        
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed_text": seed_text,
            "prompt": prompt,
            "generated_text": generated_text,
            "validation_results": serializable_results
        }
        
        self.generation_buffer.append(log_entry)
        
        # Flush buffer if it reaches the size limit
        if len(self.generation_buffer) >= self.buffer_size:
            self.flush_buffer()

    def generate_samples(self, prompts: List[str], seed_texts: List[str], target_labels: Dict[str, int], start_time: float, timeout_seconds: int) -> pd.DataFrame:
        """Generate samples with optimized batch processing and dynamic validation"""
        try:
            # First get toxicity scores for seed texts to use as reference
            seed_scores = {}
            for label in target_labels.keys():
                seed_scores[label] = self.validator.get_probabilities(seed_texts, label)
            
            # Set dynamic thresholds with adaptive tolerance
            dynamic_thresholds = {}
            for label, scores in seed_scores.items():
                mean_score = scores.mean()
                std_score = scores.std()
                
                # Adjust tolerance based on score variability
                tolerance = max(0.2, min(0.4, 2 * std_score))  # Allow 20-40% deviation based on std
                
                if target_labels[label]:  # For toxic labels
                    min_threshold = max(0.5, mean_score - tolerance)
                    max_threshold = min(0.99, mean_score + tolerance)
                    logger.info(f"Dynamic threshold for {label}: {min_threshold:.2f} - {max_threshold:.2f} (seed mean: {mean_score:.2f})")
                else:  # For non-toxic labels
                    max_threshold = min(0.3, mean_score + tolerance/2)
                    min_threshold = max(0.0, mean_score - tolerance/2)
                    logger.info(f"Dynamic threshold for {label}: {min_threshold:.2f} - {max_threshold:.2f} (seed mean: {mean_score:.2f})")
                
                dynamic_thresholds[label] = (min_threshold, max_threshold)
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                # Ensure we have matching lengths
                if len(prompts) != len(seed_texts):
                    logger.error("Mismatch between prompts and seed texts length")
                    return None
                
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    return None
                
                inputs = self.llm_tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.llm.device)
                
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=32,
                    temperature=0.95,
                    do_sample=True,
                    top_p=0.92,
                    top_k=50,
                    num_return_sequences=1,
                    repetition_penalty=1.15,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    use_cache=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0
                )
                
                texts = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=False)
                cleaned_texts = []
                
                for text in texts:
                    if "[/INST]" in text and "</s>" in text:
                        response = text.split("[/INST]")[1].split("</s>")[0].strip()
                        response = response.strip().strip('"').strip("'")
                        
                        word_count = len(response.split())
                        if (word_count >= 3 and word_count <= 50 and
                            not any(x in response.lower() for x in [
                                "generate", "requirements:", "reference",
                                "[inst]", "example"
                            ])):
                            cleaned_texts.append(response)
                
                if cleaned_texts:
                    # Get toxicity scores for generated texts
                    validation_results = {}
                    for label in target_labels:
                        probs = self.validator.get_probabilities(cleaned_texts, label)
                        min_thresh, max_thresh = dynamic_thresholds[label]
                        validation_results[label] = (probs >= min_thresh) & (probs <= max_thresh)
                    
                    # Create DataFrame with valid samples
                    valid_samples = []
                    for i, text in enumerate(cleaned_texts):
                        matches_target = all(
                            bool(validation_results[label][i]) == bool(target_labels[label])
                            for label in target_labels
                        )
                        
                        if matches_target:
                            sample = {
                                'comment_text': text,
                                **target_labels
                            }
                            valid_samples.append(sample)
                            
                            self.log_generation(
                                seed_texts[i],
                                prompts[i],
                                text,
                                {label: bool(validation_results[label][i]) for label in target_labels}
                            )
                    
                    if valid_samples:
                        return pd.DataFrame(valid_samples)
                
                return None
                
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return None

    def augment_dataset(self, target_samples: int, label_combo: Dict[str, int], seed_texts: List[str], timeout_minutes: int = 5) -> pd.DataFrame:
        """Generate a specific number of samples with given label combination"""
        logger.info(f"\nGenerating {target_samples} samples with labels: {label_combo}")
        
        generated_samples = []
        batch_size = min(32, target_samples)
        start_time = time.time()
        timeout_seconds = min(timeout_minutes * 60, 300)  # Hard limit of 5 minutes
        total_generated = 0
        pbar = None
        
        try:
            # Create progress bar
            pbar = tqdm(
                total=target_samples,
                desc="Generating",
                unit="samples",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
            while total_generated < target_samples:
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout_seconds:
                    logger.warning(f"Time limit reached after {elapsed_time/60:.1f} minutes")
                    break
                
                # Calculate remaining samples needed
                remaining = target_samples - total_generated
                current_batch_size = min(batch_size, remaining)
                
                # Select batch of seed texts
                batch_seeds = np.random.choice(seed_texts, size=current_batch_size)
                prompts = [self.generate_prompt(seed, label_combo) for seed in batch_seeds]
                
                # Generate and validate samples
                batch_start = time.time()
                new_samples = self.generate_samples(prompts, batch_seeds, label_combo, start_time, timeout_seconds)
                
                if new_samples is not None and not new_samples.empty:
                    if len(new_samples) > remaining:
                        new_samples = new_samples.head(remaining)
                    
                    generated_samples.append(new_samples)
                    num_new = len(new_samples)
                    total_generated += num_new
                    
                    # Update progress bar
                    pbar.update(num_new)
                    
                    # Calculate and display metrics
                    elapsed_minutes = elapsed_time / 60
                    rate = total_generated / elapsed_minutes if elapsed_minutes > 0 else 0
                    batch_time = time.time() - batch_start
                    time_remaining = max(0, timeout_seconds - elapsed_time)
                    
                    pbar.set_postfix({
                        'rate': f'{rate:.1f}/min',
                        'batch': f'{batch_time:.1f}s',
                        'remain': f'{time_remaining:.0f}s'
                    }, refresh=True)
                
                # Memory management every few batches
                if total_generated % (batch_size * 4) == 0:
                    torch.cuda.empty_cache()
            
            # Combine all generated samples
            if generated_samples:
                final_df = pd.concat(generated_samples, ignore_index=True)
                if len(final_df) > target_samples:
                    final_df = final_df.head(target_samples)
                logger.info(f"Successfully generated {len(final_df)} samples in {elapsed_time/60:.1f} minutes")
                return final_df
            
            return None
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return None
        finally:
            if pbar is not None:
                pbar.close()
            # Final cleanup
            self.flush_buffer()
            torch.cuda.empty_cache() 