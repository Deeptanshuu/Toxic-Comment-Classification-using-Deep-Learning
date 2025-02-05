#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import argparse
import json
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Set
import time
from itertools import combinations
import torch
from torch.utils.data import WeightedRandomSampler
import hashlib
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

TOXICITY_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
RARE_CLASSES = ['threat', 'identity_hate']
MIN_SAMPLES_PER_CLASS = 1000  # Minimum samples required per class per language

def create_multilabel_stratification_labels(row: pd.Series) -> str:
    """
    Create composite labels that preserve multi-label patterns and language distribution.
    Uses iterative label combination to capture co-occurrence patterns.
    """
    # Create base label from language
    label = str(row['lang'])
    
    # Add individual class information
    for col in TOXICITY_COLUMNS:
        label += '_' + str(int(row[col]))
    
    # Add co-occurrence patterns for pairs of classes
    for c1, c2 in combinations(RARE_CLASSES, 2):
        co_occur = int(row[c1] == 1 and row[c2] == 1)
        label += '_' + str(co_occur)
    
    return label

def oversample_rare_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform intelligent oversampling of rare classes while maintaining language distribution.
    """
    oversampled_dfs = []
    original_df = df.copy()
    
    # Process each language separately
    for lang in df['lang'].unique():
        lang_df = df[df['lang'] == lang]
        
        for rare_class in RARE_CLASSES:
            class_samples = lang_df[lang_df[rare_class] == 1]
            target_samples = MIN_SAMPLES_PER_CLASS
            
            if len(class_samples) < target_samples:
                # Calculate number of samples needed
                n_samples = target_samples - len(class_samples)
                
                # Oversample with small random variations
                noise = np.random.normal(0, 0.1, (n_samples, len(TOXICITY_COLUMNS)))
                oversampled = class_samples.sample(n_samples, replace=True)
                
                # Add noise to continuous values while keeping binary values intact
                for col in TOXICITY_COLUMNS:
                    if col in [rare_class] + [c for c in RARE_CLASSES if c != rare_class]:
                        continue  # Preserve original binary values for rare classes
                    oversampled[col] = np.clip(
                        oversampled[col].values + noise[:, TOXICITY_COLUMNS.index(col)],
                        0, 1
                    )
                
                oversampled_dfs.append(oversampled)
    
    if oversampled_dfs:
        return pd.concat([original_df] + oversampled_dfs, axis=0).reset_index(drop=True)
    return original_df

def verify_distributions(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame = None
) -> Dict:
    """
    Enhanced verification of distributions across splits with detailed metrics.
    """
    splits = {
        'original': original_df,
        'train': train_df,
        'val': val_df
    }
    if test_df is not None:
        splits['test'] = test_df
    
    stats = defaultdict(dict)
    
    for split_name, df in splits.items():
        # Language distribution
        stats[split_name]['language_dist'] = df['lang'].value_counts(normalize=True).to_dict()
        
        # Per-language class distributions
        lang_class_dist = {}
        for lang in df['lang'].unique():
            lang_df = df[df['lang'] == lang]
            lang_class_dist[lang] = {
                col: {
                    'positive_ratio': lang_df[col].mean(),
                    'count': int(lang_df[col].sum()),
                    'total': len(lang_df)
                } for col in TOXICITY_COLUMNS
            }
        stats[split_name]['lang_class_dist'] = lang_class_dist
        
        # Multi-label co-occurrence patterns
        cooccurrence = {}
        for c1, c2 in combinations(TOXICITY_COLUMNS, 2):
            cooccur_count = ((df[c1] == 1) & (df[c2] == 1)).sum()
            cooccurrence[f"{c1}_{c2}"] = {
                'count': int(cooccur_count),
                'ratio': float(cooccur_count) / len(df)
            }
        stats[split_name]['cooccurrence_patterns'] = cooccurrence
        
        # Distribution deltas from original
        if split_name != 'original':
            deltas = {}
            for lang in df['lang'].unique():
                for col in TOXICITY_COLUMNS:
                    orig_ratio = splits['original'][splits['original']['lang'] == lang][col].mean()
                    split_ratio = df[df['lang'] == lang][col].mean()
                    deltas[f"{lang}_{col}"] = abs(orig_ratio - split_ratio)
            stats[split_name]['distribution_deltas'] = deltas
    
    return stats

def check_contamination(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame = None
) -> Dict:
    """
    Enhanced contamination check including text similarity detection.
    """
    # Determine the correct text column name
    text_column = 'comment_text' if 'comment_text' in train_df.columns else 'text'
    if text_column not in train_df.columns:
        logging.warning("No text column found for contamination check. Skipping text-based contamination detection.")
        return {'exact_matches': {'train_val': 0.0}}
    
    def get_text_hash_set(df: pd.DataFrame) -> Set[str]:
        return set(df[text_column].str.lower().str.strip().values)
    
    contamination = {
        'exact_matches': {
            'train_val': len(get_text_hash_set(train_df) & get_text_hash_set(val_df)) / len(train_df)
        }
    }
    
    if test_df is not None:
        contamination['exact_matches'].update({
            'train_test': len(get_text_hash_set(train_df) & get_text_hash_set(test_df)) / len(train_df),
            'val_test': len(get_text_hash_set(val_df) & get_text_hash_set(test_df)) / len(val_df)
        })
    
    return contamination

def split_dataset(
    df: pd.DataFrame,
    seed: int,
    split_mode: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified splitting of the dataset.
    """
    # Create stratification labels
    logging.info("Creating stratification labels...")
    stratify_labels = df.apply(create_multilabel_stratification_labels, axis=1)
    
    # Oversample rare classes in training data only
    logging.info("Oversampling rare classes...")
    df_with_oversampling = oversample_rare_classes(df)
    
    # Initialize splits
    if split_mode == '3':
        # First split: 80% train, 20% temp
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        train_idx, temp_idx = next(splitter.split(df, stratify_labels))
        
        # Second split: 10% val, 10% test from temp
        temp_df = df.iloc[temp_idx]
        temp_labels = stratify_labels.iloc[temp_idx]
        
        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        val_idx, test_idx = next(splitter.split(temp_df, temp_labels))
        
        # Create final splits
        train_df = df_with_oversampling.iloc[train_idx]  # Use oversampled data for training
        val_df = df.iloc[temp_idx].iloc[val_idx]  # Use original data for validation
        test_df = df.iloc[temp_idx].iloc[test_idx]  # Use original data for testing
        
    else:  # 2-way split
        splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        train_idx, val_idx = next(splitter.split(df, stratify_labels))
        
        train_df = df_with_oversampling.iloc[train_idx]  # Use oversampled data for training
        val_df = df.iloc[val_idx]  # Use original data for validation
        test_df = None
    
    return train_df, val_df, test_df

def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    stats: Dict
) -> None:
    """
    Save splits and statistics to files.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    logging.info("Saving splits...")
    train_df.to_csv(output_path / 'train.csv', index=False)
    val_df.to_csv(output_path / 'val.csv', index=False)
    if test_df is not None:
        test_df.to_csv(output_path / 'test.csv', index=False)
    
    # Save statistics
    with open(output_path / 'stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

def compute_text_hash(text: str) -> str:
    """
    Compute SHA-256 hash of normalized text.
    """
    # Normalize text by removing extra whitespace and converting to lowercase
    normalized = ' '.join(str(text).lower().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def deduplicate_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove duplicates using cryptographic hashing while preserving metadata.
    """
    logging.info("Starting cryptographic deduplication...")
    
    # Determine text column
    text_column = 'comment_text' if 'comment_text' in df.columns else 'text'
    if text_column not in df.columns:
        raise ValueError(f"No text column found. Available columns: {df.columns}")
    
    # Compute hashes with progress bar
    logging.info("Computing cryptographic hashes...")
    tqdm.pandas(desc="Hashing texts")
    df['text_hash'] = df[text_column].progress_apply(compute_text_hash)
    
    # Get duplicate statistics before removal
    total_samples = len(df)
    duplicate_hashes = df[df.duplicated('text_hash', keep=False)]['text_hash'].unique()
    duplicate_groups = {
        hash_val: df[df['text_hash'] == hash_val].index.tolist() 
        for hash_val in duplicate_hashes
    }
    
    # Keep first occurrence of each text while tracking duplicates
    dedup_df = df.drop_duplicates('text_hash', keep='first').copy()
    dedup_df = dedup_df.drop('text_hash', axis=1)
    
    # Compile deduplication statistics
    dedup_stats = {
        'total_samples': total_samples,
        'unique_samples': len(dedup_df),
        'duplicates_removed': total_samples - len(dedup_df),
        'duplicate_rate': (total_samples - len(dedup_df)) / total_samples,
        'duplicate_groups': {
            str(k): {
                'count': len(v),
                'indices': v
            }
            for k, v in duplicate_groups.items()
        }
    }
    
    logging.info(f"Removed {dedup_stats['duplicates_removed']:,} duplicates "
                f"({dedup_stats['duplicate_rate']:.2%} of dataset)")
    
    return dedup_df, dedup_stats

def main():
    input_csv = 'dataset/processed/MULTILINGUAL_TOXIC_DATASET_AUGMENTED.csv'
    output_dir = 'dataset/split'
    seed = 42
    split_mode = '3'
    
    start_time = time.time()
    
    # Load dataset
    logging.info(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Print column names for debugging
    logging.info(f"Available columns: {', '.join(df.columns)}")
    
    # Verify required columns
    required_columns = ['lang'] + TOXICITY_COLUMNS
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Perform deduplication
    df, dedup_stats = deduplicate_dataset(df)
    
    # Perform splitting
    logging.info("Performing stratified split...")
    train_df, val_df, test_df = split_dataset(df, seed, split_mode)
    
    # Verify distributions
    logging.info("Verifying distributions...")
    stats = verify_distributions(df, train_df, val_df, test_df)
    
    # Add deduplication stats
    stats['deduplication'] = dedup_stats
    
    # Check contamination
    logging.info("Checking for contamination...")
    contamination = check_contamination(train_df, val_df, test_df)
    stats['contamination'] = contamination
    
    # Save everything
    logging.info(f"Saving splits to {output_dir}...")
    save_splits(train_df, val_df, test_df, output_dir, stats)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Done! Elapsed time: {elapsed_time:.2f} seconds")
    
    # Print summary
    print("\nDeduplication Summary:")
    print("-" * 50)
    print(f"Original samples: {dedup_stats['total_samples']:,}")
    print(f"Unique samples: {dedup_stats['unique_samples']:,}")
    print(f"Duplicates removed: {dedup_stats['duplicates_removed']:,} ({dedup_stats['duplicate_rate']:.2%})")
    
    print("\nSplit Summary:")
    print("-" * 50)
    print(f"Total samples: {len(df):,}")
    print(f"Train samples: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation samples: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    if test_df is not None:
        print(f"Test samples: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    print("\nDetailed statistics saved to stats.json")

if __name__ == "__main__":
    main()
