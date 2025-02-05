import torch
from transformers import (
    AutoModelForSeq2SeqLM,  # Changed for T5
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from huggingface_hub import login
from langdetect import detect
from presidio_analyzer import AnalyzerEngine
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
import json
import gc
import logging
from typing import List, Dict, Tuple
import random
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/threat_augmentation.log'),
        logging.StreamHandler()
    ]
)

class ThreatAugmenter:
    def __init__(self, seed_samples_path: str = "dataset/split/train.csv"):
        self.device_llm = torch.device("cuda:0")
        self.device_validator = torch.device("cuda:1")
        
        # HuggingFace authentication
        if 'HUGGINGFACE_TOKEN' in os.environ:
            login(token=os.environ['HUGGINGFACE_TOKEN'])
            logging.info("Authenticated with HuggingFace")
        else:
            logging.warning("HUGGINGFACE_TOKEN not found in environment variables")
        
        logging.info("Initializing models...")
        # Initialize FLAN-T5-XL on GPU 0 (fully open source)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-xl",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True  # Use 8-bit to save memory
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        
        # Initialize XLM-RoBERTa on GPU 1
        self.validator = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-large",
            num_labels=6,
            device_map={"": 1}
        )
        self.validator_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        
        # Initialize PII analyzer
        self.pii_analyzer = AnalyzerEngine()
        
        # Load seed samples
        self.seed_df = pd.read_csv(seed_samples_path)
        self.en_threat_samples = self.seed_df[
            (self.seed_df['lang'] == 'en') & 
            (self.seed_df['threat'] == 1)
        ]
        logging.info(f"Found {len(self.en_threat_samples)} EN threat samples")
        
        # Initialize wandb for tracking
        wandb.init(
            project="toxic-comment-augmentation",
            config={
                "target_samples": 3000,
                "source_samples": len(self.en_threat_samples),
                "temperature": 0.9,  # Increased for T5
                "max_length": 100,
                "batch_size": 32,
                "model": "google/flan-t5-xl"
            }
        )
    
    def generate_prompt(self, seed_text: str) -> str:
        """Generate a prompt for threat content generation using T5 format"""
        return f"""Task: Generate a threatening comment in English similar to the example below.
Requirements: 
- Must be threatening
- Avoid explicit profanity
- Keep it under 100 words
- No personal information
- Use social media style

Example threat: {seed_text}

Generate new threat:"""

    @torch.inference_mode()
    def generate_samples(self, prompts: List[str], temperature: float = 0.9) -> List[str]:
        """Generate samples using FLAN-T5"""
        try:
            inputs = self.llm_tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=256  # T5 input length
            ).to(self.device_llm)
            
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=100,
                temperature=temperature,
                do_sample=True,
                top_p=0.92,
                top_k=50,
                num_return_sequences=1,
                repetition_penalty=1.3,
                length_penalty=0.8,
                early_stopping=True
            )
            
            generated_texts = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Clean up the generated texts
            cleaned_texts = []
            for text in generated_texts:
                # Remove any prompt remnants and clean up
                text = text.replace("Generate new threat:", "").strip()
                text = text.split("Example threat:")[-1].strip()
                cleaned_texts.append(text)
            
            return cleaned_texts
            
        except Exception as e:
            logging.error(f"Error in generate_samples: {str(e)}")
            return []
    
    @torch.inference_mode()
    def validate_toxicity(self, texts: List[str]) -> torch.Tensor:
        """Validate generated texts using XLM-RoBERTa"""
        inputs = self.validator_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device_validator)
        
        outputs = self.validator(**inputs)
        predictions = torch.sigmoid(outputs.logits)
        
        # We want high threat score but controlled other toxicity
        threat_scores = predictions[:, 3]  # Threat is index 3
        other_toxicity = predictions[:, [0,1,2,4,5]].mean(dim=1)
        
        # Valid samples have high threat (>0.8) and moderate other toxicity (<0.7)
        valid_mask = (threat_scores > 0.8) & (other_toxicity < 0.7)
        return valid_mask
    
    def validate_language_and_pii(self, texts: List[str]) -> List[bool]:
        """Validate language consistency and check for PII"""
        valid_samples = []
        for text in texts:
            try:
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
                logging.warning(f"Validation error: {str(e)}")
                valid_samples.append(False)
        
        return valid_samples
    
    def augment_dataset(self, target_samples: int = 3000, batch_size: int = 32):
        """Main augmentation loop"""
        logging.info(f"Starting augmentation to generate {target_samples} samples")
        
        generated_samples = []
        pbar = tqdm(total=target_samples, desc="Generating samples")
        
        while len(generated_samples) < target_samples:
            # Sample seed texts
            seed_texts = self.en_threat_samples['comment_text'].sample(batch_size).tolist()
            prompts = [self.generate_prompt(text) for text in seed_texts]
            
            # Generate new samples
            new_samples = self.generate_samples(prompts)
            
            # Validate toxicity
            toxicity_mask = self.validate_toxicity(new_samples)
            valid_samples = [s for i, s in enumerate(new_samples) if toxicity_mask[i]]
            
            # Validate language and PII
            lang_pii_mask = self.validate_language_and_pii(valid_samples)
            final_samples = [s for i, s in enumerate(valid_samples) if lang_pii_mask[i]]
            
            # Add to collection
            generated_samples.extend(final_samples)
            
            # Log progress
            wandb.log({
                "generated_samples": len(generated_samples),
                "batch_toxicity_valid": toxicity_mask.float().mean().item(),
                "batch_lang_pii_valid": sum(lang_pii_mask) / len(lang_pii_mask) if lang_pii_mask else 0
            })
            
            pbar.update(len(final_samples))
            
            # Memory cleanup
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
        aug_df.to_csv(output_path / "en_threat_augmented.csv", index=False)
        
        logging.info(f"Successfully generated {len(aug_df)} samples")
        wandb.finish()
        
        return aug_df

if __name__ == "__main__":
    # Check for HuggingFace token
    if 'HUGGINGFACE_TOKEN' not in os.environ:
        print("Please set your HUGGINGFACE_TOKEN environment variable")
        print("You can get it from https://huggingface.co/settings/tokens")
        exit(1)
    
    augmenter = ThreatAugmenter()
    augmented_df = augmenter.augment_dataset()
    print(f"Generated {len(augmented_df)} new EN threat samples") 