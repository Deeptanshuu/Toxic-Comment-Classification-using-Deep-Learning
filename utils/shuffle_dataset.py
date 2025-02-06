#!/usr/bin/env python3
"""
Thoroughly shuffle the dataset while maintaining class distributions and data integrity.
This script implements stratified shuffling to ensure balanced representation of classes
and languages in the shuffled data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import logging
import json
from typing import List, Dict, Tuple
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/shuffle_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def create_stratification_label(row: pd.Series, toxicity_labels: List[str]) -> str:
    """
    Create a composite label for stratification that captures the combination of 
    toxicity labels and language.
    """
    # Convert toxicity values to binary string
    toxicity_str = ''.join(['1' if row[label] == 1 else '0' for label in toxicity_labels])
    # Combine with language
    return f"{row['lang']}_{toxicity_str}"

def validate_data(df: pd.DataFrame, toxicity_labels: List[str]) -> bool:
    """
    Validate the dataset for required columns and data integrity.
    """
    try:
        # Check required columns
        required_columns = ['comment_text', 'lang'] + toxicity_labels
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for null values in critical columns
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values:\n{null_counts[null_counts > 0]}")
        
        # Validate label values are binary
        for label in toxicity_labels:
            invalid_values = df[label][~df[label].isin([0, 1, np.nan])]
            if not invalid_values.empty:
                raise ValueError(f"Found non-binary values in {label}: {invalid_values.unique()}")
        
        # Validate text content
        if df['comment_text'].str.len().min() == 0:
            logger.warning("Found empty comments in dataset")
        
        return True
    
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        return False

def analyze_distribution(df: pd.DataFrame, toxicity_labels: List[str]) -> Dict:
    """
    Analyze the class distribution and language distribution in the dataset.
    """
    stats = {
        'total_samples': len(df),
        'language_distribution': df['lang'].value_counts().to_dict(),
        'class_distribution': {
            label: {
                'positive': int(df[label].sum()),
                'negative': int(len(df) - df[label].sum()),
                'ratio': float(df[label].mean())
            }
            for label in toxicity_labels
        },
        'language_class_distribution': defaultdict(dict)
    }
    
    # Calculate per-language class distributions
    for lang in df['lang'].unique():
        lang_df = df[df['lang'] == lang]
        stats['language_class_distribution'][lang] = {
            label: {
                'positive': int(lang_df[label].sum()),
                'negative': int(len(lang_df) - lang_df[label].sum()),
                'ratio': float(lang_df[label].mean())
            }
            for label in toxicity_labels
        }
    
    return stats

def shuffle_dataset(
    input_file: str,
    output_file: str,
    toxicity_labels: List[str],
    n_splits: int = 10,
    random_state: int = 42
) -> Tuple[bool, Dict]:
    """
    Thoroughly shuffle the dataset while maintaining class distributions.
    Uses stratified k-fold splitting for balanced shuffling.
    """
    try:
        logger.info(f"Loading dataset from {input_file}")
        df = pd.read_csv(input_file)
        
        # Validate data
        if not validate_data(df, toxicity_labels):
            return False, {}
        
        # Analyze initial distribution
        initial_stats = analyze_distribution(df, toxicity_labels)
        logger.info("Initial distribution stats:")
        logger.info(json.dumps(initial_stats, indent=2))
        
        # Create stratification labels
        logger.info("Creating stratification labels")
        df['strat_label'] = df.apply(
            lambda row: create_stratification_label(row, toxicity_labels), 
            axis=1
        )
        
        # Initialize stratified k-fold
        skf = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=random_state
        )
        
        # Get shuffled indices using stratified split
        logger.info(f"Performing stratified shuffling with {n_splits} splits")
        all_indices = []
        for _, fold_indices in skf.split(df, df['strat_label']):
            all_indices.extend(fold_indices)
        
        # Create shuffled dataframe
        shuffled_df = df.iloc[all_indices].copy()
        shuffled_df = shuffled_df.drop('strat_label', axis=1)
        
        # Analyze final distribution
        final_stats = analyze_distribution(shuffled_df, toxicity_labels)
        
        # Save shuffled dataset
        logger.info(f"Saving shuffled dataset to {output_file}")
        shuffled_df.to_csv(output_file, index=False)
        
        # Save distribution statistics
        stats_file = Path(output_file).parent / 'shuffle_stats.json'
        stats = {
            'initial': initial_stats,
            'final': final_stats,
            'shuffle_params': {
                'n_splits': n_splits,
                'random_state': random_state
            }
        }
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Shuffling complete. Statistics saved to {stats_file}")
        return True, stats
    
    except Exception as e:
        logger.error(f"Error shuffling dataset: {str(e)}")
        return False, {}

def main():
    parser = argparse.ArgumentParser(description='Thoroughly shuffle the dataset.')
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--splits', 
        type=int, 
        default=10,
        help='Number of splits for stratified shuffling (default: 10)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed (default: 42)'
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Define toxicity labels
    toxicity_labels = [
        'toxic', 'severe_toxic', 'obscene', 'threat', 
        'insult', 'identity_hate'
    ]
    
    # Shuffle dataset
    success, stats = shuffle_dataset(
        args.input,
        args.output,
        toxicity_labels,
        args.splits,
        args.seed
    )
    
    if success:
        logger.info("Dataset shuffling completed successfully")
        # Print final class distribution
        for label, dist in stats['final']['class_distribution'].items():
            logger.info(f"{label}: {dist['ratio']:.3f} "
                       f"(+:{dist['positive']}, -:{dist['negative']})")
    else:
        logger.error("Dataset shuffling failed")
        sys.exit(1)

if __name__ == '__main__':
    main() 