import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.model_selection import train_test_split

def split_dataset(input_file, train_ratio=0.8, random_state=42):
    """Split dataset into training and testing sets while maintaining language and toxicity balance"""
    print(f"\nReading dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    # Initial stats
    total_rows = len(df)
    print(f"\nInitial dataset size: {total_rows:,} comments")
    
    # Toxicity columns
    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Create output directory
    output_dir = "dataset/split"
    os.makedirs(output_dir, exist_ok=True)
    
    # Split for each language separately to maintain distribution
    train_dfs = []
    test_dfs = []
    
    languages = df['lang'].unique()
    print(f"\nProcessing {len(languages)} languages...")
    
    for lang in languages:
        lang_df = df[df['lang'] == lang]
        print(f"\n{lang.upper()} (Total: {len(lang_df):,} comments)")
        
        # Calculate toxic and non-toxic masks
        toxic_mask = lang_df[toxicity_cols].any(axis=1)
        toxic_df = lang_df[toxic_mask]
        non_toxic_df = lang_df[~toxic_mask]
        
        print(f"  Toxic comments: {len(toxic_df):,}")
        print(f"  Non-toxic comments: {len(non_toxic_df):,}")
        
        # Split toxic and non-toxic separately to maintain balance
        toxic_train, toxic_test = train_test_split(
            toxic_df, train_size=train_ratio, random_state=random_state
        )
        non_toxic_train, non_toxic_test = train_test_split(
            non_toxic_df, train_size=train_ratio, random_state=random_state
        )
        
        # Combine toxic and non-toxic for this language
        train_dfs.append(pd.concat([toxic_train, non_toxic_train]))
        test_dfs.append(pd.concat([toxic_test, non_toxic_test]))
        
        print(f"  Training set: {len(toxic_train) + len(non_toxic_train):,}")
        print(f"  Testing set: {len(toxic_test) + len(non_toxic_test):,}")
    
    # Combine all languages
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Print final statistics
    print("\nFinal Dataset Statistics:")
    print("-" * 50)
    print(f"Total rows: {total_rows:,}")
    print(f"Training set: {len(train_df):,} ({len(train_df)/total_rows*100:.1f}%)")
    print(f"Testing set: {len(test_df):,} ({len(test_df)/total_rows*100:.1f}%)")
    
    print("\nLanguage Distribution:")
    print("-" * 50)
    print("\nTraining Set:")
    train_lang_dist = train_df['lang'].value_counts()
    for lang, count in train_lang_dist.items():
        print(f"  {lang}: {count:,} ({count/len(train_df)*100:.1f}%)")
    
    print("\nTesting Set:")
    test_lang_dist = test_df['lang'].value_counts()
    for lang, count in test_lang_dist.items():
        print(f"  {lang}: {count:,} ({count/len(test_df)*100:.1f}%)")
    
    print("\nToxicity Distribution:")
    print("-" * 50)
    print("\nTraining Set:")
    for col in toxicity_cols:
        train_toxic = (train_df[col] > 0).sum()
        print(f"  {col.replace('_', ' ').title()}: {train_toxic:,} ({train_toxic/len(train_df)*100:.1f}%)")
    
    print("\nTesting Set:")
    for col in toxicity_cols:
        test_toxic = (test_df[col] > 0).sum()
        print(f"  {col.replace('_', ' ').title()}: {test_toxic:,} ({test_toxic/len(test_df)*100:.1f}%)")
    
    # Save splits
    train_file = os.path.join(output_dir, "train.csv")
    test_file = os.path.join(output_dir, "test.csv")
    
    print(f"\nSaving splits...")
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Training set saved to: {train_file}")
    print(f"Testing set saved to: {test_file}")
    print(f"Training file size: {Path(train_file).stat().st_size / (1024*1024):.1f} MB")
    print(f"Testing file size: {Path(test_file).stat().st_size / (1024*1024):.1f} MB")
    
    return train_df, test_df

if __name__ == "__main__":
    input_file = "dataset/processed/MULTILINGUAL_TOXIC_DATASET_360K_7LANG_FINAL.csv"
    train_df, test_df = split_dataset(input_file) 