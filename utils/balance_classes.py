import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
from googletrans import Translator
from tqdm import tqdm
import time

def get_class_stats(df, lang, column):
    """Calculate statistics for a specific class and language"""
    lang_df = df[df['lang'] == lang]
    total = int(len(lang_df))
    positive_count = int(lang_df[column].sum())
    return {
        'total': total,
        'positive_count': positive_count,
        'positive_ratio': float(positive_count / total if total > 0 else 0)
    }

def backtranslate_text(text, translator, intermediate_lang='fr'):
    """Backtranslate text using an intermediate language"""
    try:
        # Add delay to avoid rate limiting
        time.sleep(1)
        # Translate to intermediate language
        intermediate = translator.translate(text, dest=intermediate_lang).text
        # Translate back to English
        time.sleep(1)
        back_to_en = translator.translate(intermediate, dest='en').text
        return back_to_en
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

def balance_dataset_distributions(input_dir='dataset/balanced', output_dir='dataset/final_balanced'):
    """Balance Turkish toxic class and augment English identity hate samples"""
    print("\n=== Balancing Dataset Distributions ===\n")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(input_dir, 'train_balanced.csv'))
    val_df = pd.read_csv(os.path.join(input_dir, 'val_balanced.csv'))
    test_df = pd.read_csv(os.path.join(input_dir, 'test_balanced.csv'))
    
    # 1. Fix Turkish Toxic Class Balance
    print("\nInitial Turkish Toxic Distribution:")
    print("-" * 50)
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        stats = get_class_stats(df, 'tr', 'toxic')
        print(f"{name}: {stats['positive_count']}/{stats['total']} ({stats['positive_ratio']:.2%})")
    
    # Remove excess Turkish toxic samples from test
    tr_test = test_df[test_df['lang'] == 'tr']
    target_ratio = get_class_stats(train_df, 'tr', 'toxic')['positive_ratio']
    current_ratio = get_class_stats(test_df, 'tr', 'toxic')['positive_ratio']
    
    if current_ratio > target_ratio:
        samples_to_remove = 150  # As specified
        print(f"\nRemoving {samples_to_remove} Turkish toxic samples from test set...")
        
        # Identify and remove samples
        np.random.seed(42)
        tr_toxic_samples = test_df[
            (test_df['lang'] == 'tr') & 
            (test_df['toxic'] > 0)
        ]
        remove_idx = tr_toxic_samples.sample(n=samples_to_remove).index
        test_df = test_df.drop(remove_idx)
    
    # 2. Augment English Identity Hate in Validation
    print("\nInitial English Identity Hate Distribution:")
    print("-" * 50)
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        stats = get_class_stats(df, 'en', 'identity_hate')
        print(f"{name}: {stats['positive_count']}/{stats['total']} ({stats['positive_ratio']:.2%})")
    
    # Select samples for backtranslation
    print("\nAugmenting English identity hate samples in validation set...")
    en_train_hate = train_df[
        (train_df['lang'] == 'en') & 
        (train_df['identity_hate'] > 0)
    ]
    samples = en_train_hate.sample(n=50, replace=True, random_state=42)
    
    # Initialize translator
    translator = Translator()
    
    # Perform backtranslation
    print("Performing backtranslation (this may take a few minutes)...")
    augmented_samples = []
    for _, row in tqdm(samples.iterrows(), total=len(samples)):
        # Create new sample with backtranslated text
        new_sample = row.copy()
        new_sample['comment_text'] = backtranslate_text(row['comment_text'], translator)
        augmented_samples.append(new_sample)
    
    # Add augmented samples to validation set
    val_df = pd.concat([val_df, pd.DataFrame(augmented_samples)], ignore_index=True)
    
    # Save balanced datasets
    print("\nSaving final balanced datasets...")
    train_df.to_csv(os.path.join(output_dir, 'train_final.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_final.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_final.csv'), index=False)
    
    # Save balancing statistics
    stats = {
        'turkish_toxic': {
            'original_distribution': {
                'train': get_class_stats(train_df, 'tr', 'toxic'),
                'val': get_class_stats(val_df, 'tr', 'toxic'),
                'test': get_class_stats(test_df, 'tr', 'toxic')
            },
            'samples_removed': 150
        },
        'english_identity_hate': {
            'original_distribution': {
                'train': get_class_stats(train_df, 'en', 'identity_hate'),
                'val': get_class_stats(val_df, 'en', 'identity_hate'),
                'test': get_class_stats(test_df, 'en', 'identity_hate')
            },
            'samples_added': 50
        }
    }
    
    with open(os.path.join(output_dir, 'balancing_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    return train_df, val_df, test_df

def validate_final_distributions(train_df, val_df, test_df):
    """Validate the final distributions of all classes across languages"""
    print("\nFinal Distribution Validation:")
    print("-" * 50)
    
    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    languages = sorted(train_df['lang'].unique())
    
    for lang in languages:
        print(f"\n{lang.upper()}:")
        for class_name in classes:
            print(f"\n  {class_name.upper()}:")
            for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
                stats = get_class_stats(df, lang, class_name)
                print(f"    {name}: {stats['positive_count']}/{stats['total']} ({stats['positive_ratio']:.2%})")

if __name__ == "__main__":
    # First install required package if not already installed
    # !pip install googletrans==4.0.0-rc1
    
    # Balance datasets
    train_df, val_df, test_df = balance_dataset_distributions()
    
    # Validate final distributions
    validate_final_distributions(train_df, val_df, test_df) 