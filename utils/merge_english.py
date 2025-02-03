import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_dataset(file_path, encoding='utf-8'):
    """Load dataset with fallback encodings"""
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    if encoding != 'utf-8':
        encodings.insert(0, encoding)  # Try specified encoding first
    
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with {enc}: {str(e)}")
            continue
    
    raise ValueError(f"Could not read {file_path} with any encoding")

def merge_english_comments(output_file=None):
    """Merge English comments from multiple datasets"""
    
    # Define input files
    multilingual_file = 'dataset/raw/MULTILINGUAL_TOXIC_DATASET_347K_7LANG.csv'
    english_file = 'dataset/raw/english-comments-cleaned.csv'
    
    print("\nProcessing multilingual dataset...")
    multi_df = load_dataset(multilingual_file)
    # Extract English comments
    multi_df = multi_df[multi_df['lang'] == 'en'].copy()
    print(f"Found {len(multi_df):,} English comments in multilingual dataset")
    
    print("\nProcessing English cleaned dataset...")
    eng_df = load_dataset(english_file)
    print(f"Found {len(eng_df):,} comments in English dataset")
    
    # Ensure both dataframes have the same columns
    required_cols = ['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Handle multilingual dataset
    if 'comment_text' not in multi_df.columns and 'text' in multi_df.columns:
        multi_df['comment_text'] = multi_df['text']
    
    # Add missing toxicity columns with 0s if they don't exist
    for col in required_cols[1:]:  # Skip comment_text
        if col not in multi_df.columns:
            multi_df[col] = 0
        if col not in eng_df.columns:
            eng_df[col] = 0
    
    # Keep only required columns
    multi_df = multi_df[required_cols]
    eng_df = eng_df[required_cols]
    
    # Merge datasets
    print("\nMerging datasets...")
    merged_df = pd.concat([multi_df, eng_df], ignore_index=True)
    initial_count = len(merged_df)
    print(f"Initial merged size: {initial_count:,} comments")
    
    # Remove exact duplicates
    merged_df = merged_df.drop_duplicates(subset=['comment_text'], keep='first')
    final_count = len(merged_df)
    print(f"After removing duplicates: {final_count:,} comments")
    print(f"Removed {initial_count - final_count:,} duplicates")
    
    # Print toxicity distribution
    print("\nToxicity distribution in final dataset:")
    for col in required_cols[1:]:
        toxic_count = (merged_df[col] > 0).sum()
        print(f"{col.replace('_', ' ').title()}: {toxic_count:,} ({toxic_count/final_count*100:.1f}%)")
    
    # Save merged dataset
    if output_file is None:
        output_file = "dataset/processed/english_merged.csv"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"\nSaving merged dataset to: {output_file}")
    merged_df.to_csv(output_file, index=False)
    print(f"File size: {Path(output_file).stat().st_size / (1024*1024):.1f} MB")
    
    return merged_df

if __name__ == "__main__":
    output_file = "dataset/processed/english_merged.csv"
    merged_df = merge_english_comments(output_file) 