import torch
from transformers import (
    AutoModelForSeq2SeqLM,
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

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

class ThreatAugmenter:
    def __init__(self, seed_samples_path: str = "dataset/split/train.csv"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading models...")
        # Initialize FLAN-T5-base
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base",
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        
        # Initialize XLM-RoBERTa for validation
        self.validator = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-large",
            num_labels=6
        ).to(self.device)
        self.validator_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        
        # Load seed samples
        self.seed_df = pd.read_csv(seed_samples_path)
        self.en_threat_samples = self.seed_df[
            (self.seed_df['lang'] == 'en') & 
            (self.seed_df['threat'] == 1)
        ]
        print(f"Found {len(self.en_threat_samples)} EN threat samples")

    def generate_prompt(self, seed_text: str) -> str:
        """Generate a prompt for threat content generation"""
        return f"""Generate a threatening comment similar to this example:
{seed_text}
Make it threatening but avoid explicit profanity. Keep it under 50 words."""

    @torch.inference_mode()
    def generate_samples(self, prompts: List[str]) -> List[str]:
        """Generate samples using FLAN-T5"""
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
                temperature=0.9,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1,
                repetition_penalty=1.2
            )
            
            return [text.strip() for text in self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)]
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return []
    
    @torch.inference_mode()
    def validate_toxicity(self, texts: List[str]) -> torch.Tensor:
        """Validate generated texts using XLM-RoBERTa"""
        if not texts:
            return torch.zeros(0, dtype=torch.bool)
        
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
    
    def augment_dataset(self, target_samples: int = 3000, batch_size: int = 16):
        """Main augmentation loop"""
        print(f"Starting augmentation to generate {target_samples} samples")
        
        generated_samples = []
        pbar = tqdm(total=target_samples, desc="Generating samples")
        
        while len(generated_samples) < target_samples:
            # Sample seed texts
            seed_texts = self.en_threat_samples['comment_text'].sample(batch_size).tolist()
            prompts = [self.generate_prompt(text) for text in seed_texts]
            
            # Generate new samples
            new_samples = self.generate_samples(prompts)
            if not new_samples:
                continue
                
            # Validate toxicity
            toxicity_mask = self.validate_toxicity(new_samples)
            valid_samples = [s for i, s in enumerate(new_samples) if toxicity_mask[i]]
            
            # Validate language
            lang_mask = self.validate_language(valid_samples)
            final_samples = [s for i, s in enumerate(valid_samples) if lang_mask[i]]
            
            # Add to collection
            generated_samples.extend(final_samples)
            pbar.update(len(final_samples))
            
            # Memory cleanup
            if len(generated_samples) % 100 == 0:
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
        
        print(f"Successfully generated {len(aug_df)} samples")
        return aug_df

if __name__ == "__main__":
    augmenter = ThreatAugmenter()
    augmented_df = augmenter.augment_dataset()