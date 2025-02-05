import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys
from threat_augment import ThreatAugmenter
import json

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        
        # Calculate required samples
        required_samples = calculate_required_samples(df)
        
        if required_samples <= 0:
            logger.info("English data is already balanced. No augmentation needed.")
            return
        
        # Initialize augmenter
        logger.info("\nInitializing augmenter...")
        augmenter = ThreatAugmenter()
        
        # Generate additional English samples
        logger.info(f"\nGenerating {required_samples:,} English samples...")
        augmented_df = augmenter.augment_dataset(target_samples=required_samples)
        
        # Add language tag
        augmented_df['lang'] = 'en'
        
        # Merge with original dataset
        logger.info("\nMerging datasets...")
        output_file = f'dataset/processed/MULTILINGUAL_TOXIC_DATASET_BALANCED_{timestamp}.csv'
        
        # Combine datasets
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        
        # Save balanced dataset
        combined_df.to_csv(output_file, index=False)
        logger.info(f"\nSaved balanced dataset to {output_file}")
        
        # Final distribution check
        logger.info("\nFinal distribution after balancing:")
        final_dist = analyze_language_distribution(combined_df)
        
        # Save distribution statistics
        stats = {
            'timestamp': timestamp,
            'initial_distribution': initial_dist.to_dict(),
            'final_distribution': final_dist.to_dict(),
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

if __name__ == "__main__":
    logger.info("Starting English data balancing process...")
    balance_english_data() 