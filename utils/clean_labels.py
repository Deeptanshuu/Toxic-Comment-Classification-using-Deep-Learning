import pandas as pd
import numpy as np
from pathlib import Path
import os

def clean_toxicity_labels(input_file, output_file=None):
    """Clean toxicity labels by converting fractional values to binary using ceiling"""
    print(f"\nReading dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    # Initial stats
    total_rows = len(df)
    print(f"\nInitial dataset size: {total_rows:,} comments")
    
    # Toxicity columns to clean
    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Print initial value distribution
    print("\nInitial value distribution:")
    print("-" * 50)
    for col in toxicity_cols:
        unique_vals = df[col].value_counts().sort_index()
        print(f"\n{col.replace('_', ' ').title()}:")
        for val, count in unique_vals.items():
            print(f"  {val}: {count:,} comments")
    
    # Clean each toxicity column
    print("\nCleaning labels...")
    for col in toxicity_cols:
        # Get unique values before cleaning
        unique_before = df[col].nunique()
        non_binary = df[~df[col].isin([0, 1])][col].unique()
        
        if len(non_binary) > 0:
            print(f"\n{col.replace('_', ' ').title()}:")
            print(f"  Found {len(non_binary)} non-binary values: {sorted(non_binary)}")
            
            # Convert to binary using ceiling (any value > 0 becomes 1)
            df[col] = np.ceil(df[col]).clip(0, 1).astype(int)
            
            # Print conversion results
            unique_after = df[col].nunique()
            print(f"  Unique values before: {unique_before}")
            print(f"  Unique values after: {unique_after}")
    
    # Print final value distribution
    print("\nFinal value distribution:")
    print("-" * 50)
    for col in toxicity_cols:
        value_counts = df[col].value_counts().sort_index()
        total = len(df)
        print(f"\n{col.replace('_', ' ').title()}:")
        for val, count in value_counts.items():
            percentage = (count / total) * 100
            print(f"  {val}: {count:,} comments ({percentage:.2f}%)")
    
    # Save cleaned dataset
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_cleaned{ext}"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"\nSaving cleaned dataset to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"File size: {Path(output_file).stat().st_size / (1024*1024):.1f} MB")
    
    return df

if __name__ == "__main__":
    input_file = "dataset/raw/MULTILINGUAL_TOXIC_DATASET_360K_7LANG.csv"
    output_file = "dataset/processed/MULTILINGUAL_TOXIC_DATASET_360K_7LANG_binary.csv"
    
    cleaned_df = clean_toxicity_labels(input_file, output_file) 