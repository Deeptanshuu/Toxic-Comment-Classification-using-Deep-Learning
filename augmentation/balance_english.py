import os
import torch

# Configure CPU and thread settings FIRST, before any other imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPU_ENABLE_AVX2'] = '1'
os.environ['TF_CPU_ENABLE_AVX512F'] = '1'
os.environ['TF_CPU_ENABLE_AVX512_VNNI'] = '1'
os.environ['TF_CPU_ENABLE_FMA'] = '1'
os.environ['MKL_NUM_THREADS'] = '80'
os.environ['OMP_NUM_THREADS'] = '80'

# Set PyTorch thread configurations once
torch.set_num_threads(80)
torch.set_num_interop_threads(10)

# Now import everything else
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys
from toxic_augment import ToxicAugmenter
import json

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_file = log_dir / f"balance_english_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)

logger = logging.getLogger(__name__)

def analyze_label_distribution(df, lang='en'):
    """Analyze label distribution for a specific language"""
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    lang_df = df[df['lang'] == lang]
    total = len(lang_df)
    
    if total == 0:
        logger.warning(f"No samples found for language {lang.upper()}.")
        return {}
    
    logger.info(f"\nLabel Distribution for {lang.upper()}:")
    logger.info("-" * 50)
    dist = {}
    for label in labels:
        count = lang_df[label].sum()
        percentage = (count / total) * 100
        dist[label] = {'count': int(count), 'percentage': percentage}
        logger.info(f"{label}: {count:,} ({percentage:.2f}%)")
    return dist

def analyze_language_distribution(df):
    """Analyze current language distribution"""
    lang_dist = df['lang'].value_counts()
    logger.info("\nCurrent Language Distribution:")
    logger.info("-" * 50)
    for lang, count in lang_dist.items():
        logger.info(f"{lang}: {count:,} comments ({count/len(df)*100:.2f}%)")
    return lang_dist

def calculate_required_samples(df):
    """Calculate how many English samples we need to generate"""
    lang_counts = df['lang'].value_counts()
    target_count = lang_counts.max()  # Use the largest language count as target
    en_count = lang_counts.get('en', 0)
    required_samples = target_count - en_count
    
    logger.info(f"\nTarget count per language: {target_count:,}")
    logger.info(f"Current English count: {en_count:,}")
    logger.info(f"Required additional English samples: {required_samples:,}")
    
    return required_samples

def generate_balanced_samples(df, required_samples):
    """Generate samples maintaining original class distribution ratios"""
    logger.info("\nGenerating balanced samples...")
    
    # Get English samples
    en_df = df[df['lang'] == 'en']
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Calculate target counts for each label
    target_counts = {}
    for label in labels:
        count = en_df[label].sum()
        ratio = count / len(en_df)
        target_count = int(ratio * required_samples)
        target_counts[label] = target_count
        logger.info(f"Target count for {label}: {target_count:,}")
    
    augmented_samples = []
    augmenter = ToxicAugmenter()
    total_generated = 0
    
    # Generate samples for each label
    for label, target_count in target_counts.items():
        if target_count == 0:
            continue
            
        logger.info(f"\nGenerating {target_count:,} samples for {label}")
        
        # Get seed texts with this label
        seed_texts = en_df[en_df[label] == 1]['comment_text'].tolist()
        
        if not seed_texts:
            logger.warning(f"No seed texts found for {label}, skipping...")
            continue
        
        # Generate samples with 5-minute timeout
        new_samples = augmenter.augment_dataset(
            target_samples=target_count,
            label=label,  # Using single label instead of label_combo
            seed_texts=seed_texts,
            timeout_minutes=5
        )
        
        if new_samples is not None and not new_samples.empty:
            augmented_samples.append(new_samples)
            total_generated += len(new_samples)
            
            # Log progress
            logger.info(f"✓ Generated {len(new_samples):,} samples")
            logger.info(f"Progress: {total_generated:,}/{required_samples:,}")
            
            # Check if we have reached our global required samples
            if total_generated >= required_samples:
                logger.info("Reached required sample count, stopping generation")
                break
    
    # Combine all generated samples
    if augmented_samples:
        augmented_df = pd.concat(augmented_samples, ignore_index=True)
        augmented_df['lang'] = 'en'
        
        # Ensure we don't exceed the required sample count
        if len(augmented_df) > required_samples:
            logger.info(f"Trimming excess samples from {len(augmented_df):,} to {required_samples:,}")
            augmented_df = augmented_df.head(required_samples)
        
        # Log final class distribution
        logger.info("\nFinal class distribution in generated samples:")
        for label in labels:
            count = augmented_df[label].sum()
            percentage = (count / len(augmented_df)) * 100
            logger.info(f"{label}: {count:,} ({percentage:.2f}%)")
        
        # Also log clean samples
        clean_count = len(augmented_df[augmented_df[labels].sum(axis=1) == 0])
        clean_percentage = (clean_count / len(augmented_df)) * 100
        logger.info(f"Clean samples: {clean_count:,} ({clean_percentage:.2f}%)")
        
        return augmented_df
    else:
        raise Exception("Failed to generate any valid samples")

def balance_english_data():
    """Main function to balance English data with other languages"""
    try:
        # Load dataset
        input_file = 'dataset/processed/MULTILINGUAL_TOXIC_DATASET_360K_7LANG_FINAL.csv'
        logger.info(f"Loading dataset from {input_file}")
        df = pd.read_csv(input_file)
        
        # Analyze current distribution
        logger.info("\nAnalyzing current distribution...")
        initial_dist = analyze_language_distribution(df)
        initial_label_dist = analyze_label_distribution(df, 'en')
        
        # Calculate required samples
        required_samples = calculate_required_samples(df)
        
        if required_samples <= 0:
            logger.info("English data is already balanced. No augmentation needed.")
            return
        
        # Generate balanced samples
        augmented_df = generate_balanced_samples(df, required_samples)
        
        # Merge with original dataset
        logger.info("\nMerging datasets...")
        output_file = 'dataset/processed/MULTILINGUAL_TOXIC_DATASET_BALANCED.csv'
        
        # Combine datasets
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        
        # Save balanced dataset
        combined_df.to_csv(output_file, index=False)
        logger.info(f"\nSaved balanced dataset to {output_file}")
        
        # Final distribution check
        logger.info("\nFinal distribution after balancing:")
        final_dist = analyze_language_distribution(combined_df)
        final_label_dist = analyze_label_distribution(combined_df, 'en')
        
        # Save distribution statistics
        stats = {
            'timestamp': timestamp,
            'initial_distribution': {
                'languages': initial_dist.to_dict(),
                'english_labels': initial_label_dist
            },
            'final_distribution': {
                'languages': final_dist.to_dict(),
                'english_labels': final_label_dist
            },
            'samples_generated': len(augmented_df),
            'total_samples': len(combined_df)
        }
        
        stats_file = f'logs/balance_stats_{timestamp}.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"\nSaved balancing statistics to {stats_file}")
        
    except Exception as e:
        logger.error(f"Error during balancing: {str(e)}")
        raise

def main():
    balance_english_data()

if __name__ == "__main__":
    logger.info("Starting English data balancing process...")
    main() 