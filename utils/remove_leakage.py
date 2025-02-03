import pandas as pd
import hashlib
import os
from collections import defaultdict
from pathlib import Path

def text_hash(text):
    """Create a hash of the text after basic normalization"""
    # Convert to string and normalize
    text = str(text).strip().lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Create hash
    return hashlib.sha256(text.encode()).hexdigest()

def remove_leaked_samples(train_path, val_path, test_path, output_dir='dataset/clean'):
    """Remove overlapping samples between dataset splits"""
    print("\n=== Removing Data Leakage ===\n")
    
    # Create hash registry
    hash_registry = defaultdict(set)
    splits = {}
    original_sizes = {}
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    splits = {
        'train': pd.read_csv(train_path),
        'val': pd.read_csv(val_path),
        'test': pd.read_csv(test_path)
    }
    
    # Store original sizes
    for split_name, df in splits.items():
        original_sizes[split_name] = len(df)
        print(f"Original {split_name} size: {len(df):,} samples")
    
    # Process each split
    print("\nChecking for overlaps...")
    removed_counts = defaultdict(int)
    
    for split_name, df in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Calculate hashes for current split
        current_hashes = set(df['comment_text'].apply(text_hash))
        hash_registry[split_name] = current_hashes
        
        # Check overlaps with other splits
        for other_split in splits:
            if other_split != split_name:
                if hash_registry[other_split]:  # Only check if other split is processed
                    overlaps = current_hashes & hash_registry[other_split]
                    if overlaps:
                        print(f"  Found {len(overlaps):,} overlaps with {other_split}")
                        # Remove overlapping samples
                        df = df[~df['comment_text'].apply(text_hash).isin(overlaps)]
                        removed_counts[f"{split_name}_from_{other_split}"] = len(overlaps)
        
        # Update splits dictionary with cleaned dataframe
        splits[split_name] = df
    
    # Save cleaned splits
    print("\nSaving cleaned datasets...")
    for split_name, df in splits.items():
        output_path = os.path.join(output_dir, f"{split_name}_clean.csv")
        df.to_csv(output_path, index=False)
        reduction = ((original_sizes[split_name] - len(df)) / original_sizes[split_name]) * 100
        print(f"Cleaned {split_name}: {len(df):,} samples (-{reduction:.2f}%)")
    
    # Print detailed overlap statistics
    print("\nDetailed Overlap Statistics:")
    print("-" * 50)
    for overlap_type, count in removed_counts.items():
        split_name, other_split = overlap_type.split('_from_')
        print(f"{split_name} → {other_split}: {count:,} overlapping samples removed")
    
    return splits

def validate_cleaning(splits):
    """Validate that no overlaps remain between splits"""
    print("\nValidating Cleaning...")
    print("-" * 50)
    
    all_clean = True
    for split1 in splits:
        for split2 in splits:
            if split1 < split2:  # Check each pair only once
                hashes1 = set(splits[split1]['comment_text'].apply(text_hash))
                hashes2 = set(splits[split2]['comment_text'].apply(text_hash))
                overlaps = hashes1 & hashes2
                if overlaps:
                    print(f"⚠️ Warning: Found {len(overlaps)} overlaps between {split1} and {split2}")
                    all_clean = False
                else:
                    print(f"✅ No overlaps between {split1} and {split2}")
    
    if all_clean:
        print("\n✅ All splits are now clean with no overlaps!")
    else:
        print("\n⚠️ Some overlaps still remain. Consider additional cleaning.")

if __name__ == "__main__":
    # Define paths
    train_path = "dataset/split/train.csv"
    val_path = "dataset/split/val.csv"
    test_path = "dataset/split/test.csv"
    
    # Remove leaked samples
    cleaned_splits = remove_leaked_samples(train_path, val_path, test_path)
    
    # Validate cleaning
    validate_cleaning(cleaned_splits) 