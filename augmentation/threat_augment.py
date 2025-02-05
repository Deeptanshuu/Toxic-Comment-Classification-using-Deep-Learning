import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification
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

# Create log directories
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / f'generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

def log_separator(message: str = ""):
    """Print a separator line with optional message"""
    if message:
        logger.info("\n" + "="*40 + f" {message} " + "="*40)
    else:
        logger.info("\n" + "="*100)

class ThreatAugmenter:
    def __init__(self, seed_samples_path: str = "dataset/split/train.csv"):
        log_separator("INITIALIZATION")
        
        # Initialize logging
        self.log_file = log_dir / f"samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        logger.info(f"Sample log file: {self.log_file}")
        
        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Load models
        log_separator("LOADING MODELS")
        logger.info("Loading Mistral-7B...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            padding_side="left",
            use_fast=True
        )
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        logger.info("✓ Mistral-7B loaded")
        
        logger.info("Loading XLM-RoBERTa...")
        self.validator = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-large",
            num_labels=6,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.device
        )
        self.validator_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        logger.info("✓ XLM-RoBERTa loaded")
        
        # Load data
        log_separator("LOADING DATA")
        logger.info(f"Loading samples from {seed_samples_path}")
        self.seed_df = pd.read_csv(seed_samples_path)
        self.en_threat_samples = self.seed_df[
            (self.seed_df['lang'] == 'en') & 
            (self.seed_df['threat'] == 1)
        ]
        logger.info(f"Found {len(self.en_threat_samples)} EN threat samples")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        log_separator()

    def generate_prompt(self, seed_text: str) -> str:
        """Generate a prompt for realistic threatening content"""
        return f"""<s>[INST] You are tasked with generating a threatening social media comment. The threat should be realistic and intimidating.

Reference example for tone (generate something different): {seed_text}

Rules:
- Make it genuinely threatening and hostile
- Use authentic social media language and slang
- Include aggressive language that fits naturally
- Maximum 50 words
- Must be different from the example
- Focus on personal threats or intimidation

Generate only the threatening comment, nothing else: [/INST]"""

    def log_generation(self, seed_text: str, prompt: str, generated_text: str, is_valid: bool):
        """Log generation details to JSONL file"""
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed_text": seed_text,
            "prompt": prompt,
            "generated_text": generated_text,
            "is_valid": is_valid
        }
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to log file: {str(e)}")

    def generate_samples(self, prompts: List[str], seed_texts: List[str]) -> List[str]:
        """Generate samples using Mistral-7B-Instruct"""
        try:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                inputs = self.llm_tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.9,
                    do_sample=True,
                    top_p=0.85,
                    top_k=40,
                    num_return_sequences=1,
                    repetition_penalty=1.2,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    use_cache=True
                )
                
                texts = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                cleaned_texts = []
                
                for idx, text in enumerate(texts):
                    if "[/INST]" in text:
                        response = text.split("[/INST]")[-1].strip()
                        
                        if '"' in response:
                            try:
                                response = response[response.find('"')+1:response.rfind('"')]
                            except:
                                pass
                        
                        response = response.strip('"').strip("'").strip()
                        
                        # Remove artifacts from comparison text
                        comparison_text = ' '.join([word for word in response.split() 
                                                  if not word.startswith('#') 
                                                  and not any(char in word for char in ['😈', '🔥', '👀', '💀', '⚡️', '🔫', '🚫', '💣'])])
                        
                        if (len(comparison_text.split()) >= 5 and
                            not any(x in response.lower() for x in [
                                "make it genuinely",
                                "generate only",
                                "you are tasked",
                                "rules:",
                                "reference example",
                                "for tone"
                            ]) and
                            not any(seed.lower() in response.lower() for seed in seed_texts)):
                            
                            cleaned_texts.append(response)
                            logger.info(f"\nGenerated ({idx+1}/{len(texts)}):\n{response}\n")
                
                logger.info(f"Generated {len(cleaned_texts)} valid responses from {len(texts)} attempts")
                return cleaned_texts
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return []

    def validate_toxicity(self, texts: List[str]) -> torch.Tensor:
        """Validate generated texts using XLM-RoBERTa with adjusted thresholds"""
        if not texts:
            return torch.zeros(0, dtype=torch.bool)
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            inputs = self.validator_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            outputs = self.validator(**inputs)
            predictions = torch.sigmoid(outputs.logits)
            
            # Adjusted thresholds for more realistic content
            threat_scores = predictions[:, 3]  # Threat is index 3
            other_toxicity = predictions[:, [0,1,2,4,5]].mean(dim=1)
            
            # More permissive validation criteria
            valid_mask = (threat_scores > 0.6) & (other_toxicity < 0.9)
            return valid_mask
    
    def validate_language(self, texts: List[str]) -> List[bool]:
        """Simple language validation"""
        return [detect(text) == 'en' for text in texts]
    
    def augment_dataset(self, target_samples: int = 3000, batch_size: int = 8):
        """Main augmentation loop"""
        log_separator("STARTING GENERATION")
        logger.info(f"Target samples: {target_samples}")
        logger.info(f"Batch size: {batch_size}")
        
        generated_samples = []
        pbar = tqdm(total=target_samples, desc="Generating")
        
        stats = {
            "total_attempts": 0,
            "valid_samples": 0,
            "invalid_toxicity": 0,
            "invalid_language": 0
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
            
            # Log generations
            for seed, prompt, generated in zip(seed_texts, prompts, new_samples):
                is_valid = generated in final_samples
                self.log_generation(seed, prompt, generated, is_valid)
            
            # Update progress
            generated_samples.extend(final_samples)
            stats["valid_samples"] = len(generated_samples)
            pbar.update(len(final_samples))
            
            # Print batch stats
            if final_samples:
                log_separator("BATCH STATS")
                success_rate = len(final_samples) / batch_size * 100
                logger.info(
                    f"Batch Success: {len(final_samples)}/{batch_size} ({success_rate:.1f}%)\n"
                    f"Total Attempts: {stats['total_attempts']}\n"
                    f"Valid Samples: {stats['valid_samples']}\n"
                    f"Failed Toxicity: {stats['invalid_toxicity']}\n"
                    f"Failed Language: {stats['invalid_language']}\n"
                    f"Overall Success: {(stats['valid_samples']/stats['total_attempts']*100):.1f}%"
                )
            
            # Cleanup
            if len(generated_samples) % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        pbar.close()
        
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
            f"- Samples: {self.log_file}"
        )
        
        return aug_df

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    
    augmenter = ThreatAugmenter()
    augmented_df = augmenter.augment_dataset()