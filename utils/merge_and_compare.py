import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_dataset(file_path, encoding='utf-8'):
    """Load dataset with fallback encodings"""
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with {enc}: {str(e)}")
            continue
    
    raise ValueError(f"Could not read {file_path} with any encoding")

def print_dataset_stats(df, name="Dataset"):
    """Print detailed statistics about a dataset"""
    print(f"\n{name} Statistics:")
    print(f"Total comments: {len(df):,}")
    
    if 'lang' in df.columns:
        print("\nLanguage distribution:")
        lang_dist = df['lang'].value_counts()
        for lang, count in lang_dist.items():
            print(f"{lang}: {count:,} ({count/len(df)*100:.1f}%)")
    
    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    print("\nToxicity distribution:")
    for col in toxicity_cols:
        if col in df.columns:
            toxic_count = (df[col] > 0).sum()
            print(f"{col.replace('_', ' ').title()}: {toxic_count:,} ({toxic_count/len(df)*100:.1f}%)")
    
    if all(col in df.columns for col in toxicity_cols):
        toxic_mask = df[toxicity_cols].any(axis=1)
        total_toxic = toxic_mask.sum()
        print(f"\nTotal Toxic Comments: {total_toxic:,} ({total_toxic/len(df)*100:.1f}%)")
        print(f"Total Non-Toxic Comments: {len(df)-total_toxic:,} ({(len(df)-total_toxic)/len(df)*100:.1f}%)")

def merge_and_compare_datasets():
    """Merge filtered English with non-English data and compare with original"""
    
    # Define file paths
    english_filtered = "dataset/raw/english_filtered.csv"
    non_english = "dataset/raw/MULTILINGUAL_TOXIC_DATASET_347k_7LANG_non_english.csv"
    original = "dataset/raw/MULTILINGUAL_TOXIC_DATASET_347K_7LANG.csv"
    output_file = "dataset/processed/final_merged_dataset.csv"
    
    print("Loading datasets...")
    
    # Load English filtered dataset
    print("\nLoading filtered English dataset...")
    eng_df = load_dataset(english_filtered)
    eng_df['lang'] = 'en'  # Ensure language column exists
    print_dataset_stats(eng_df, "Filtered English Dataset")
    
    # Load non-English dataset
    print("\nLoading non-English dataset...")
    non_eng_df = load_dataset(non_english)
    print_dataset_stats(non_eng_df, "Non-English Dataset")
    
    # Merge datasets
    print("\nMerging datasets...")
    merged_df = pd.concat([eng_df, non_eng_df], ignore_index=True)
    print_dataset_stats(merged_df, "Merged Dataset")
    
    # Load original dataset for comparison
    print("\nLoading original dataset for comparison...")
    original_df = load_dataset(original)
    print_dataset_stats(original_df, "Original Dataset")
    
    # Compare datasets
    print("\nComparison Summary:")
    print(f"Original dataset size: {len(original_df):,}")
    print(f"Merged dataset size: {len(merged_df):,}")
    print(f"Difference: {len(merged_df) - len(original_df):,} comments")
    
    if 'lang' in merged_df.columns and 'lang' in original_df.columns:
        print("\nLanguage Distribution Comparison:")
        orig_lang = original_df['lang'].value_counts()
        new_lang = merged_df['lang'].value_counts()
        
        all_langs = sorted(set(orig_lang.index) | set(new_lang.index))
        for lang in all_langs:
            orig_count = orig_lang.get(lang, 0)
            new_count = new_lang.get(lang, 0)
            diff = new_count - orig_count
            print(f"{lang}:")
            print(f"  Original: {orig_count:,}")
            print(f"  New: {new_count:,}")
            print(f"  Difference: {diff:,} ({diff/orig_count*100:.1f}% change)")
    
    # Save merged dataset
    print(f"\nSaving merged dataset to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    print(f"File size: {Path(output_file).stat().st_size / (1024*1024):.1f} MB")
    
    return merged_df

if __name__ == "__main__":
    merged_df = merge_and_compare_datasets() 