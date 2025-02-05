import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)
logger = logging.getLogger(__name__)

def merge_datasets():
    """Merge augmented threat dataset with main dataset"""
    try:
        # Load main dataset
        logger.info("Loading main dataset...")
        main_df = pd.read_csv("dataset/processed/MULTILINGUAL_TOXIC_DATASET_360K_7LANG_FINAL.csv")
        logger.info(f"Main dataset: {len(main_df):,} rows")
        
        # Load augmented dataset
        augmented_path = Path("dataset/augmented")
        latest_augmented = max(augmented_path.glob("threat_augmented_*.csv"))
        logger.info(f"Loading augmented dataset: {latest_augmented.name}")
        aug_df = pd.read_csv(latest_augmented)
        logger.info(f"Augmented dataset: {len(aug_df):,} rows")
        
        # Standardize columns for augmented data
        logger.info("Standardizing columns...")
        aug_df_standardized = pd.DataFrame({
            'comment_text': aug_df['text'],
            'toxic': 1,
            'severe_toxic': 0,
            'obscene': 0,
            'threat': 1,
            'insult': 0,
            'identity_hate': 0,
            'lang': 'en'
        })
        
        # Check for duplicates between datasets
        logger.info("Checking for duplicates...")
        combined_texts = pd.concat([main_df['comment_text'], aug_df_standardized['comment_text']])
        duplicates = combined_texts.duplicated(keep='first')
        duplicate_count = duplicates[len(main_df):].sum()
        logger.info(f"Found {duplicate_count} duplicates in augmented data")
        
        # Remove duplicates from augmented data
        aug_df_standardized = aug_df_standardized[~duplicates[len(main_df):].values]
        logger.info(f"Augmented dataset after duplicate removal: {len(aug_df_standardized):,} rows")
        
        # Merge datasets
        merged_df = pd.concat([main_df, aug_df_standardized], ignore_index=True)
        logger.info(f"Final merged dataset: {len(merged_df):,} rows")
        
        # Save merged dataset
        output_path = f"dataset/processed/MULTILINGUAL_TOXIC_DATASET_AUGMENTED.csv"
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Saved merged dataset to: {output_path}")
        
        # Print statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Original samples: {len(main_df):,}")
        logger.info(f"Added threat samples: {len(aug_df_standardized):,}")
        logger.info(f"Total samples: {len(merged_df):,}")
        logger.info(f"Threat samples in final dataset: {merged_df['threat'].sum():,}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging datasets: {str(e)}")
        raise

if __name__ == "__main__":
    merged_df = merge_datasets() 