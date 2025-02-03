import pandas as pd
import numpy as np
import json
from pathlib import Path
import os

def calculate_class_weights(df, toxicity_cols):
    """Calculate class weights using inverse frequency scaling"""
    total_samples = len(df)
    weights = {}
    
    # Calculate weights for each toxicity type
    for col in toxicity_cols:
        positive_count = (df[col] > 0).sum()
        negative_count = total_samples - positive_count
        
        # Use balanced weights formula: n_samples / (n_classes * n_samples_for_class)
        pos_weight = total_samples / (2 * positive_count) if positive_count > 0 else 0
        neg_weight = total_samples / (2 * negative_count) if negative_count > 0 else 0
        
        weights[col] = {
            'positive_weight': pos_weight,
            'negative_weight': neg_weight,
            'positive_count': int(positive_count),
            'negative_count': int(negative_count),
            'positive_ratio': float(positive_count/total_samples),
            'negative_ratio': float(negative_count/total_samples)
        }
    
    return weights

def calculate_language_weights(df, toxicity_cols):
    """Calculate class weights for each language"""
    languages = df['lang'].unique()
    language_weights = {}
    
    for lang in languages:
        lang_df = df[df['lang'] == lang]
        lang_weights = calculate_class_weights(lang_df, toxicity_cols)
        language_weights[lang] = lang_weights
    
    return language_weights

def normalize_weights(weights_dict, baseline_class='obscene'):
    """Normalize weights relative to a baseline class"""
    # Get the positive weight of baseline class
    baseline_weight = None
    for lang, lang_weights in weights_dict.items():
        if baseline_weight is None:
            baseline_weight = lang_weights[baseline_class]['positive_weight']
    
    normalized_weights = {}
    for lang, lang_weights in weights_dict.items():
        normalized_weights[lang] = {}
        for col, weights in lang_weights.items():
            normalized_weights[lang][col] = {
                'positive_weight': weights['positive_weight'] / baseline_weight,
                'negative_weight': weights['negative_weight'] / baseline_weight,
                'positive_count': weights['positive_count'],
                'negative_count': weights['negative_count'],
                'positive_ratio': weights['positive_ratio'],
                'negative_ratio': weights['negative_ratio']
            }
    
    return normalized_weights

def generate_weights(input_file):
    """Generate and save class weights for the dataset"""
    print(f"\nReading dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    # Initial stats
    total_rows = len(df)
    print(f"\nTotal samples: {total_rows:,}")
    
    # Toxicity columns
    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Calculate overall weights
    print("\nCalculating overall weights...")
    overall_weights = calculate_class_weights(df, toxicity_cols)
    
    # Calculate language-specific weights
    print("\nCalculating language-specific weights...")
    language_weights = calculate_language_weights(df, toxicity_cols)
    
    # Normalize weights
    print("\nNormalizing weights...")
    normalized_overall = normalize_weights({'overall': overall_weights})['overall']
    normalized_language = normalize_weights(language_weights)
    
    # Prepare weights dictionary
    weights_dict = {
        'dataset_info': {
            'total_samples': total_rows,
            'n_languages': len(df['lang'].unique()),
            'languages': list(df['lang'].unique())
        },
        'overall_weights': overall_weights,
        'normalized_overall_weights': normalized_overall,
        'language_weights': language_weights,
        'normalized_language_weights': normalized_language
    }
    
    # Save weights
    output_dir = "weights"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "class_weights.json")
    
    print(f"\nSaving weights to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    
    # Print summary
    print("\nWeight Summary (Normalized Overall):")
    print("-" * 50)
    for col in toxicity_cols:
        pos_weight = normalized_overall[col]['positive_weight']
        pos_count = normalized_overall[col]['positive_count']
        pos_ratio = normalized_overall[col]['positive_ratio']
        print(f"\n{col.replace('_', ' ').title()}:")
        print(f"  Positive samples: {pos_count:,} ({pos_ratio*100:.2f}%)")
        print(f"  Weight: {pos_weight:.2f}x")
    
    return weights_dict

if __name__ == "__main__":
    input_file = "dataset/processed/MULTILINGUAL_TOXIC_DATASET_360K_7LANG_FINAL.csv"
    weights = generate_weights(input_file) 