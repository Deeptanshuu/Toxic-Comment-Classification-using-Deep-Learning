import pandas as pd
import numpy as np
from pathlib import Path
import json
import os

def get_threat_stats(df, lang='pt'):
    """Calculate threat statistics for a given language"""
    lang_df = df[df['lang'] == lang]
    total = int(len(lang_df))  # Convert to native Python int
    threat_count = int(lang_df['threat'].sum())  # Convert to native Python int
    return {
        'total': total,
        'threat_count': threat_count,
        'threat_ratio': float(threat_count / total if total > 0 else 0)  # Convert to native Python float
    }

def fix_pt_threat_distribution(input_dir='dataset/split', output_dir='dataset/balanced'):
    """Fix Portuguese threat class overrepresentation while maintaining dataset balance"""
    print("\n=== Fixing Portuguese Threat Distribution ===\n")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(input_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(input_dir, 'test.csv'))
    
    print("\nInitial Portuguese Threat Distribution:")
    print("-" * 50)
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        stats = get_threat_stats(df)
        print(f"{name}: {stats['threat_count']}/{stats['total']} ({stats['threat_ratio']:.2%})")
    
    # Calculate target ratio based on train set
    target_ratio = float(get_threat_stats(train_df)['threat_ratio'])  # Convert to native Python float
    print(f"\nTarget threat ratio (from train): {target_ratio:.2%}")
    
    # Fix test set distribution
    pt_test = test_df[test_df['lang'] == 'pt']
    current_ratio = float(get_threat_stats(test_df)['threat_ratio'])  # Convert to native Python float
    
    if current_ratio > target_ratio:
        # Calculate how many samples to remove
        current_threats = int(pt_test['threat'].sum())  # Convert to native Python int
        target_threats = int(len(pt_test) * target_ratio)
        samples_to_remove = int(current_threats - target_threats)
        
        print(f"\nRemoving {samples_to_remove} Portuguese threat samples from test set...")
        
        # Identify samples to remove
        pt_threat_samples = test_df[
            (test_df['lang'] == 'pt') & 
            (test_df['threat'] > 0)
        ]
        
        # Randomly select samples to remove
        np.random.seed(42)  # For reproducibility
        remove_idx = np.random.choice(
            pt_threat_samples.index,
            size=samples_to_remove,
            replace=False
        ).tolist()  # Convert to native Python list
        
        # Remove selected samples
        test_df = test_df.drop(remove_idx)
        
        # Verify new distribution
        new_ratio = float(get_threat_stats(test_df)['threat_ratio'])  # Convert to native Python float
        print(f"New Portuguese threat ratio: {new_ratio:.2%}")
        
        # Save statistics
        stats = {
            'original_distribution': {
                'train': get_threat_stats(train_df),
                'val': get_threat_stats(val_df),
                'test': get_threat_stats(test_df)
            },
            'samples_removed': samples_to_remove,
            'target_ratio': target_ratio,
            'achieved_ratio': new_ratio
        }
        
        with open(os.path.join(output_dir, 'pt_threat_fix_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save balanced datasets
        print("\nSaving balanced datasets...")
        train_df.to_csv(os.path.join(output_dir, 'train_balanced.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val_balanced.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_balanced.csv'), index=False)
        
        print("\nFinal Portuguese Threat Distribution:")
        print("-" * 50)
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            stats = get_threat_stats(df)
            print(f"{name}: {stats['threat_count']}/{stats['total']} ({stats['threat_ratio']:.2%})")
    else:
        print("\nNo fix needed - test set threat ratio is not higher than train")
    
    return train_df, val_df, test_df

def validate_distributions(train_df, val_df, test_df):
    """Validate the threat distributions across all languages"""
    print("\nValidating Threat Distributions Across Languages:")
    print("-" * 50)
    
    for lang in sorted(train_df['lang'].unique()):
        print(f"\n{lang.upper()}:")
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            stats = get_threat_stats(df, lang)
            print(f"  {name}: {stats['threat_count']}/{stats['total']} ({stats['threat_ratio']:.2%})")

if __name__ == "__main__":
    # Fix Portuguese threat distribution
    train_df, val_df, test_df = fix_pt_threat_distribution()
    
    # Validate distributions across all languages
    validate_distributions(train_df, val_df, test_df) 