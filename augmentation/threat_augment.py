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

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def print_with_delay(message: str, delay: float = 0.5):
    """Print message with a delay and clear formatting"""
    print("\n" + "="*80)
    print(message)
    print("="*80)
    time.sleep(delay)

# Create a log directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

class ThreatAugmenter:
    def __init__(self, seed_samples_path: str = "dataset/split/train.csv"):
        print_with_delay("Initializing ThreatAugmenter...")
        
        # Initialize logging
        self.log_file = log_dir / f"generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        print_with_delay(f"Logging details to: {self.log_file}")
        
        # GPU setup and optimization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print_with_delay(f"Using GPU: {torch.cuda.get_device_name()}")
            print_with_delay(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        print_with_delay("Loading Mistral-7B model...")
        # Initialize Mistral-7B-Instruct with better quality settings
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
        print_with_delay("✓ Loaded Mistral-7B with 4-bit quantization")
        
        print_with_delay("Loading XLM-RoBERTa validator...")
        # Initialize XLM-RoBERTa with optimizations
        self.validator = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-large",
            num_labels=6,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.device
        )
        self.validator_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        print_with_delay("✓ Loaded XLM-RoBERTa validator")
        
        # Load seed samples
        print_with_delay("Loading seed samples...")
        self.seed_df = pd.read_csv(seed_samples_path)
        self.en_threat_samples = self.seed_df[
            (self.seed_df['lang'] == 'en') & 
            (self.seed_df['threat'] == 1)
        ]
        print_with_delay(f"✓ Found {len(self.en_threat_samples)} EN threat samples")
        
        # Clear GPU cache after initialization
        torch.cuda.empty_cache()
        gc.collect()

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

    @torch.inference_mode()
    def generate_samples(self, prompts: List[str]) -> List[str]:
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
                    temperature=0.95,  # Increased for more variety
                    do_sample=True,
                    top_p=0.92,  # Slightly increased
                    top_k=50,
                    num_return_sequences=1,
                    repetition_penalty=1.15,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    use_cache=True
                )
                
                # Clean up generated texts
                texts = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                cleaned_texts = []
                for text in texts:
                    # Remove the instruction part and clean up
                    text = text.split("[/INST]")[-1].strip()
                    # Remove any remaining prompt artifacts
                    text = text.replace("[INST]", "").replace("</s>", "").strip()
                    text = text.replace("Generate only the threatening comment, nothing else:", "").strip()
                    # Remove any leading/trailing quotes and cleanup
                    text = text.strip('"').strip("'").strip()
                    cleaned_texts.append(text)
                
                return cleaned_texts
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return []
    
    @torch.inference_mode()
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
    
    def log_generation(self, seed_text: str, prompt: str, generated_text: str, is_valid: bool):
        """Log the generation details"""
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed_text": seed_text,
            "prompt": prompt,
            "generated_text": generated_text,
            "is_valid": is_valid
        }
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def augment_dataset(self, target_samples: int = 3000, batch_size: int = 8):
        """Main augmentation loop with detailed logging"""
        print_with_delay(f"Starting augmentation to generate {target_samples} samples", delay=1)
        print_with_delay(f"Logging details to: {self.log_file}", delay=1)
        
        generated_samples = []
        pbar = tqdm(total=target_samples, desc="Generating samples")
        
        generation_stats = {
            "total_attempts": 0,
            "valid_samples": 0,
            "invalid_toxicity": 0,
            "invalid_language": 0
        }
        
        while len(generated_samples) < target_samples:
            # Sample seed texts
            seed_texts = self.en_threat_samples['comment_text'].sample(batch_size).tolist()
            prompts = [self.generate_prompt(text) for text in seed_texts]
            
            # Print sample prompt occasionally with clear formatting
            if generation_stats["total_attempts"] % 50 == 0:
                print_with_delay("\nExample Prompt:", delay=1)
                print_with_delay(prompts[0], delay=2)
            
            # Generate new samples
            new_samples = self.generate_samples(prompts)
            if not new_samples:
                continue
            
            generation_stats["total_attempts"] += len(new_samples)
            
            # Validate toxicity
            toxicity_mask = self.validate_toxicity(new_samples)
            valid_samples = [s for i, s in enumerate(new_samples) if toxicity_mask[i]]
            generation_stats["invalid_toxicity"] += len(new_samples) - len(valid_samples)
            
            # Validate language
            lang_mask = self.validate_language(valid_samples)
            final_samples = [s for i, s in enumerate(valid_samples) if lang_mask[i]]
            generation_stats["invalid_language"] += len(valid_samples) - len(final_samples)
            
            # Log generations with clear formatting
            for i, (seed, prompt, generated) in enumerate(zip(seed_texts, prompts, new_samples)):
                is_valid = i < len(final_samples)
                self.log_generation(seed, prompt, generated, is_valid)
                
                if is_valid:
                    print_with_delay("\nGenerated Valid Sample:", delay=1)
                    print_with_delay(f"Seed: {seed[:100]}...", delay=1)
                    print_with_delay(f"Generated: {generated}", delay=2)
            
            # Add to collection
            generated_samples.extend(final_samples)
            generation_stats["valid_samples"] = len(generated_samples)
            pbar.update(len(final_samples))
            
            # Print batch stats with clear formatting
            if final_samples:
                success_rate = len(final_samples) / batch_size * 100
                print_with_delay("\nBatch Statistics:", delay=1)
                print_with_delay(
                    f"Success Rate: {len(final_samples)}/{batch_size} ({success_rate:.1f}%)\n"
                    f"Total Attempts: {generation_stats['total_attempts']}\n"
                    f"Valid Samples: {generation_stats['valid_samples']}\n"
                    f"Failed Toxicity: {generation_stats['invalid_toxicity']}\n"
                    f"Failed Language: {generation_stats['invalid_language']}\n"
                    f"Overall Success: {(generation_stats['valid_samples']/generation_stats['total_attempts']*100):.1f}%",
                    delay=2
                )
            
            # Memory cleanup every 5 batches
            if len(generated_samples) % (batch_size * 5) == 0:
                print_with_delay("Cleaning up memory...", delay=0.5)
                torch.cuda.empty_cache()
                gc.collect()
        
        pbar.close()
        
        # Final statistics with clear formatting
        print_with_delay("\nGeneration Complete!", delay=1)
        print_with_delay(
            f"Final Statistics:\n"
            f"Total Attempts: {generation_stats['total_attempts']}\n"
            f"Valid Samples: {generation_stats['valid_samples']}\n"
            f"Failed Toxicity: {generation_stats['invalid_toxicity']}\n"
            f"Failed Language: {generation_stats['invalid_language']}\n"
            f"Overall Success: {(generation_stats['valid_samples']/generation_stats['total_attempts']*100):.1f}%",
            delay=2
        )
        
        # Save results
        output_path = Path("dataset/augmented")
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create and save augmented dataset
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
        
        output_file = output_path / f"en_threat_augmented_{timestamp}.csv"
        stats_file = log_dir / f"generation_stats_{timestamp}.json"
        
        print_with_delay("Saving outputs...", delay=1)
        aug_df.to_csv(output_file, index=False)
        with open(stats_file, 'w') as f:
            json.dump(generation_stats, f, indent=2)
        
        print_with_delay(
            f"Outputs saved to:\n"
            f"- Dataset: {output_file}\n"
            f"- Logs: {self.log_file}\n"
            f"- Stats: {stats_file}",
            delay=2
        )
        
        return aug_df

if __name__ == "__main__":
    # Set memory growth
    torch.cuda.empty_cache()
    gc.collect()
    
    augmenter = ThreatAugmenter()
    augmented_df = augmenter.augment_dataset()