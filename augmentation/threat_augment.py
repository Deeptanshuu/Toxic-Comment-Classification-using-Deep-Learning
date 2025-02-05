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

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

# Create a log directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

class ThreatAugmenter:
    def __init__(self, seed_samples_path: str = "dataset/split/train.csv"):
        # Initialize logging
        self.log_file = log_dir / f"generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        print(f"Logging details to: {self.log_file}")
        
        # GPU setup and optimization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        print("Loading models...")
        # Initialize Mistral-7B-Instruct with better quality settings
        self.llm = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,  # Using 4-bit quantization for better quality
            use_flash_attention_2=True,  # Enable flash attention for better performance
            attn_implementation="flash_attention_2"
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            padding_side="left",
            use_fast=True
        )
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        print("Loaded Mistral-7B with 4-bit quantization and flash attention")
        
        # Initialize XLM-RoBERTa with optimizations
        self.validator = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-large",
            num_labels=6,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.device
        )
        self.validator_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        
        # Load seed samples
        self.seed_df = pd.read_csv(seed_samples_path)
        self.en_threat_samples = self.seed_df[
            (self.seed_df['lang'] == 'en') & 
            (self.seed_df['threat'] == 1)
        ]
        print(f"Found {len(self.en_threat_samples)} EN threat samples")
        
        # Clear GPU cache after initialization
        torch.cuda.empty_cache()
        gc.collect()

    def generate_prompt(self, seed_text: str) -> str:
        """Generate a prompt for threat content generation using Mistral's format"""
        return f"""<s>[INST] You are tasked with generating a threatening comment similar to the example below. The comment should be threatening but avoid explicit profanity. Keep it under 50 words and make it sound natural.

Example threat: {seed_text}

Requirements:
- Must be threatening in nature
- No explicit profanity
- Different wording than the example
- Under 50 words
- Social media style language
- Make it sound natural and believable

Generate a single threatening comment: [/INST]"""

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
                    temperature=0.82,  # Slightly lower for more focused outputs
                    do_sample=True,
                    top_p=0.88,  # More focused sampling
                    top_k=40,  # More selective
                    num_return_sequences=1,
                    repetition_penalty=1.18,  # Increased slightly
                    presence_penalty=0.1,  # Add presence penalty
                    frequency_penalty=0.1,  # Add frequency penalty
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    use_cache=True
                )
                
                # Clean up generated texts with improved cleaning
                texts = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                cleaned_texts = []
                for text in texts:
                    # Remove the instruction part and clean up
                    text = text.split("[/INST]")[-1].strip()
                    # Remove any remaining prompt artifacts and clean up
                    text = text.replace("[INST]", "").replace("</s>", "").strip()
                    text = text.replace("Generate a single threatening comment:", "").strip()
                    # Remove any leading/trailing quotes
                    text = text.strip('"').strip("'").strip()
                    cleaned_texts.append(text)
                
                return cleaned_texts
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return []
    
    @torch.inference_mode()
    def validate_toxicity(self, texts: List[str]) -> torch.Tensor:
        """Validate generated texts using XLM-RoBERTa"""
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
            
            # We want high threat score but controlled other toxicity
            threat_scores = predictions[:, 3]  # Threat is index 3
            other_toxicity = predictions[:, [0,1,2,4,5]].mean(dim=1)
            
            # Relaxed validation criteria
            valid_mask = (threat_scores > 0.7) & (other_toxicity < 0.8)
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
        print(f"Starting augmentation to generate {target_samples} samples")
        print(f"Logging details to: {self.log_file}")
        
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
            
            # Print sample prompt occasionally
            if generation_stats["total_attempts"] % 50 == 0:
                print("\nSample prompt:")
                print("-" * 50)
                print(prompts[0])
                print("-" * 50)
            
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
            
            # Log generations
            for i, (seed, prompt, generated) in enumerate(zip(seed_texts, prompts, new_samples)):
                is_valid = i < len(final_samples)
                self.log_generation(seed, prompt, generated, is_valid)
                
                if is_valid:
                    print(f"\nGenerated (Valid):")
                    print(f"Seed: {seed[:100]}...")
                    print(f"Generated: {generated}")
            
            # Add to collection
            generated_samples.extend(final_samples)
            generation_stats["valid_samples"] = len(generated_samples)
            pbar.update(len(final_samples))
            
            # Print batch stats
            if final_samples:
                success_rate = len(final_samples) / batch_size * 100
                print(f"\nBatch success rate: {len(final_samples)}/{batch_size} ({success_rate:.1f}%)")
                print(f"Overall stats:")
                print(f"- Total attempts: {generation_stats['total_attempts']}")
                print(f"- Valid samples: {generation_stats['valid_samples']}")
                print(f"- Failed toxicity: {generation_stats['invalid_toxicity']}")
                print(f"- Failed language: {generation_stats['invalid_language']}")
                print(f"- Overall success rate: {(generation_stats['valid_samples']/generation_stats['total_attempts']*100):.1f}%")
            
            # Memory cleanup every 5 batches
            if len(generated_samples) % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        pbar.close()
        
        # Create augmented dataset
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
        
        # Save augmented samples
        output_path = Path("dataset/augmented")
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"en_threat_augmented_{timestamp}.csv"
        aug_df.to_csv(output_file, index=False)
        
        # Save final stats
        stats_file = log_dir / f"generation_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(generation_stats, f, indent=2)
        
        print(f"\nFinal Statistics:")
        print(f"- Total attempts: {generation_stats['total_attempts']}")
        print(f"- Valid samples: {generation_stats['valid_samples']}")
        print(f"- Failed toxicity: {generation_stats['invalid_toxicity']}")
        print(f"- Failed language: {generation_stats['invalid_language']}")
        print(f"- Overall success rate: {(generation_stats['valid_samples']/generation_stats['total_attempts']*100):.1f}%")
        print(f"\nOutputs saved to:")
        print(f"- Dataset: {output_file}")
        print(f"- Logs: {self.log_file}")
        print(f"- Stats: {stats_file}")
        
        return aug_df

if __name__ == "__main__":
    # Set memory growth
    torch.cuda.empty_cache()
    gc.collect()
    
    augmenter = ThreatAugmenter()
    augmented_df = augmenter.augment_dataset()