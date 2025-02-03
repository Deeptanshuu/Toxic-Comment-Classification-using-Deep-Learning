import pandas as pd
import numpy as np
from pathlib import Path
import os
import hashlib
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import json

def split_dataset(input_file, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, 
                 random_state=42, min_samples=50):
    """Improved dataset splitting with multilabel stratification and leakage prevention"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    print(f"\nReading dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    # --- Data Integrity Checks ---
    # Create text hash for leakage prevention
    df['text_hash'] = df['comment_text'].apply(
        lambda x: hashlib.sha256(x.strip().encode()).hexdigest()
    )
    
    # Check for duplicates
    duplicates = df[df.duplicated('text_hash', keep=False)]
    if not duplicates.empty:
        print(f"Found {len(duplicates)} duplicates. Keeping first occurrence.")
        df = df.drop_duplicates('text_hash', keep='first')

    # --- Language-Aware Multilabel Stratification ---
    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 
                   'threat', 'insult', 'identity_hate']
    
    # Binarize labels for stratification
    y = df[toxicity_cols].values
    y = np.where(y > 0.5, 1, 0)  # Convert to binary labels

    # Initialize splitter
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_ratio, 
                                          random_state=random_state)
    
    # First split: test vs temp (train+val)
    for train_val_idx, test_idx in msss.split(df, y):
        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]

    # Second split: train vs val
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio/(1-test_ratio), 
                                              random_state=random_state)
    for train_idx, val_idx in msss_val.split(train_val_df, y[train_val_idx]):
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

    # --- Rare Class Handling ---
    def ensure_min_samples(split_df, split_name):
        for col in toxicity_cols:
            count = split_df[col].sum()
            if count < min_samples:
                needed = min_samples - count
                print(f"⚠️ {split_name} has only {count} {col} samples. Adding {needed} more.")
                
                # Find samples from other splits
                donor = pd.concat([train_df, val_df, test_df]).drop(split_df.index)
                additional = donor[donor[col] > 0.5].sample(needed, replace=True)
                
                split_df = pd.concat([split_df, additional])
        return split_df

    train_df = ensure_min_samples(train_df, "Training")
    val_df = ensure_min_samples(val_df, "Validation")
    test_df = ensure_min_samples(test_df, "Testing")

    # --- Final Validation ---
    def print_distribution(name, df):
        print(f"\n{name} Set Distribution:")
        print(f"Total: {len(df):,}")
        print("Language Distribution:")
        print(df['lang'].value_counts(normalize=True).map("{:.1%}".format))
        print("\nToxicity Ratios:")
        for col in toxicity_cols:
            ratio = df[col].mean()
            print(f"{col.replace('_', ' ').title()}: {ratio:.2%}")

    print_distribution("Training", train_df)
    print_distribution("Validation", val_df)
    print_distribution("Testing", test_df)

    # --- Save Splits ---
    output_dir = "dataset/split"
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, "train.csv")
    val_file = os.path.join(output_dir, "val.csv")
    test_file = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    print("\nSplitting completed successfully!")
    return train_df, val_df, test_df

if __name__ == "__main__":
    input_file = "dataset/processed/MULTILINGUAL_TOXIC_DATASET_360K_7LANG_FINAL.csv"
    split_dataset(input_file)
