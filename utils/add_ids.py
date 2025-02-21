import pandas as pd
import numpy as np
from pathlib import Path
import os
import hashlib

def generate_comment_id(row, toxicity_cols):
    """Generate a unique ID encoding language and toxicity information"""
    # Get toxicity type codes
    tox_code = ''.join(['1' if row[col] > 0 else '0' for col in toxicity_cols])
    
    # Create a hash of the comment text for uniqueness
    text_hash = hashlib.md5(row['comment_text'].encode()).hexdigest()[:6]
    
    # Combine language, toxicity code, and hash
    # Format: {lang}_{toxicity_code}_{hash}
    # Example: en_100010_a1b2c3 (English comment with toxic and insult flags)
    return f"{row['lang']}_{tox_code}_{text_hash}"

def add_dataset_ids(input_file, output_file=None):
    """Add meaningful IDs to the dataset"""
    print(f"\nReading dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    # Initial stats
    total_rows = len(df)
    print(f"\nInitial dataset size: {total_rows:,} comments")
    
    # Toxicity columns in order
    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    print("\nGenerating IDs...")
    # Generate IDs for each row
    df['id'] = df.apply(lambda row: generate_comment_id(row, toxicity_cols), axis=1)
    
    # Verify ID uniqueness
    unique_ids = df['id'].nunique()
    print(f"\nGenerated {unique_ids:,} unique IDs")
    
    if unique_ids < total_rows:
        print(f"Warning: {total_rows - unique_ids:,} duplicate IDs found")
        # Handle duplicates by adding a suffix
        df['id'] = df.groupby('id').cumcount().astype(str) + '_' + df['id']
        print("Added suffixes to make IDs unique")
        
    # Print sample IDs for each language
    print("\nSample IDs by language:")
    print("-" * 50)
    for lang in df['lang'].unique():
        lang_sample = df[df['lang'] == lang].sample(n=min(3, len(df[df['lang'] == lang])), random_state=42)
        print(f"\n{lang.upper()}:")
        for _, row in lang_sample.iterrows():
            tox_types = [col for col in toxicity_cols if row[col] > 0]
            print(f"ID: {row['id']}")
            print(f"Toxicity: {', '.join(tox_types) if tox_types else 'None'}")
            print(f"Text: {row['comment_text'][:100]}...")
    
    # Move ID column to first position
    cols = ['id'] + [col for col in df.columns if col != 'id']
    df = df[cols]
    
    # Save dataset with IDs
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_with_ids{ext}"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"\nSaving dataset with IDs to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"File size: {Path(output_file).stat().st_size / (1024*1024):.1f} MB")
    
    return df

if __name__ == "__main__":
    input_file = "dataset/raw/MULTILINGUAL_TOXIC_DATASET_360K_7LANG_binary.csv"
    output_file = "dataset/processed/MULTILINGUAL_TOXIC_DATASET_360K_7LANG_binary_with_ids.csv"
    
    df_with_ids = add_dataset_ids(input_file, output_file) 