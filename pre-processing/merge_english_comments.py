import pandas as pd
from tqdm import tqdm
import logging
import os

# Configure logging
logging.basicConfig(
    filename='dataset_merge.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_and_process_dataset(file_path, expected_rows=None):
    """Load and validate dataset with progress tracking"""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {file_path}: {len(df):,} rows")
        if expected_rows and len(df) != expected_rows:
            raise ValueError(f"Row count mismatch: Expected {expected_rows}, got {len(df)}")
        return df
    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        raise

def main():
    # Path configuration
    base_path = "dataset/raw"
    output_path = "dataset/processed"
    os.makedirs(output_path, exist_ok=True)

    # Load datasets with progress and validation
    logging.info("Loading English dataset")
    english_df = load_and_process_dataset(
        os.path.join(base_path, "english_raw.csv")
    ).drop(columns=['id'])  # Remove unnecessary column
    
    initial_english_count = len(english_df)
    logging.info(f"Initial English comments: {initial_english_count:,}")

    logging.info("Loading non-English dataset")
    non_english_df = load_and_process_dataset(
        os.path.join(base_path, "TOXIC_DATASET_367k_non_english.csv")
    )
    
    initial_non_english_count = len(non_english_df)
    logging.info(f"Initial non-English comments: {initial_non_english_count:,}")

    # Add language column to English dataset
    english_df['lang'] = 'en'

    # Column alignment check
    required_columns = ['comment_text', 'toxic', 'severe_toxic', 'obscene',
                       'threat', 'insult', 'identity_hate', 'lang']
    
    if not all(col in english_df.columns for col in required_columns):
        missing = set(required_columns) - set(english_df.columns)
        raise ValueError(f"Missing columns in English data: {missing}")

    # Merge datasets
    logging.info("Merging datasets")
    merged_df = pd.concat([english_df, non_english_df], ignore_index=True)
    logging.info(f"After merge: {len(merged_df):,} rows")

    # Clean and validate with detailed logging
    logging.info("Cleaning data")
    
    # Check for duplicates
    duplicates = merged_df.duplicated(subset=['comment_text'], keep=False)
    duplicate_count = duplicates.sum()
    logging.info(f"Found {duplicate_count:,} duplicate comments")
    
    # Remove duplicates
    merged_df = merged_df.drop_duplicates(subset=['comment_text'], keep='first')
    logging.info(f"After removing duplicates: {len(merged_df):,} rows")
    
    # Check for NA values
    na_counts = merged_df[['toxic', 'comment_text']].isna().sum()
    logging.info(f"NA counts before cleaning:\n{na_counts}")
    
    # Remove NA values
    merged_df = merged_df.dropna(subset=['toxic', 'comment_text'])
    logging.info(f"After removing NA values: {len(merged_df):,} rows")

    # Language distribution check
    lang_dist = merged_df['lang'].value_counts()
    logging.info(f"Language distribution:\n{lang_dist}")

    # Save merged dataset
    output_file = os.path.join(output_path, "FINAL_TOXIC_DATASET_merged.parquet")
    merged_df.to_parquet(
        output_file,
        engine='pyarrow',
        compression='brotli'
    )
    logging.info(f"Saved merged dataset to {output_file}")

    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Initial English comments: {initial_english_count:,}")
    print(f"Initial non-English comments: {initial_non_english_count:,}")
    print(f"Total after merge: {len(merged_df):,}")
    print(f"English comments in final dataset: {merged_df['lang'].value_counts().get('en', 0):,}")
    print(f"Number of languages: {merged_df['lang'].nunique()}")
    print(f"\nDataset saved to: {output_file}")

if __name__ == "__main__":
    tqdm.pandas()
    main()
