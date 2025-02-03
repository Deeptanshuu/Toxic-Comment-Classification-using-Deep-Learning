import pandas as pd
import os
import numpy as np

def filter_and_balance_comments(input_file, output_file=None):
    """Filter and balance dataset by maximizing toxic comments and matching with non-toxic"""
    print(f"\nReading dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    # Initial stats
    total_rows = len(df)
    print(f"\nInitial dataset size: {total_rows:,} comments")
    
    # Toxicity columns
    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Print initial toxicity distribution
    print("\nInitial toxicity distribution:")
    for col in toxicity_cols:
        toxic_count = (df[col] > 0).sum()
        print(f"{col.replace('_', ' ').title()}: {toxic_count:,} ({toxic_count/total_rows*100:.1f}%)")
    
    # Create mask for any toxicity
    toxic_mask = df[toxicity_cols].any(axis=1)
    
    # Process each language separately to maintain balance
    languages = df['lang'].unique() if 'lang' in df.columns else ['en']
    balanced_dfs = []
    
    print("\nProcessing each language:")
    for lang in languages:
        print(f"\n{lang}:")
        # If no lang column, use entire dataset
        if 'lang' in df.columns:
            lang_df = df[df['lang'] == lang]
        else:
            lang_df = df
        
        # Split into toxic and non-toxic
        lang_toxic_df = lang_df[toxic_mask] if 'lang' in df.columns else lang_df[toxic_mask]
        lang_non_toxic_df = lang_df[~toxic_mask] if 'lang' in df.columns else lang_df[~toxic_mask]
        
        toxic_count = len(lang_toxic_df)
        non_toxic_count = len(lang_non_toxic_df)
        
        print(f"Total comments: {len(lang_df):,}")
        print(f"Toxic comments available: {toxic_count:,}")
        print(f"Non-toxic comments available: {non_toxic_count:,}")
        
        # Keep all toxic comments
        sampled_toxic = lang_toxic_df
        print(f"Kept all {toxic_count:,} toxic comments")
            
        # Sample equal number of non-toxic comments
        if non_toxic_count >= toxic_count:
            sampled_non_toxic = lang_non_toxic_df.sample(n=toxic_count, random_state=42)
            print(f"Sampled {toxic_count:,} non-toxic comments to match")
        else:
            # If we have fewer non-toxic than toxic, use all non-toxic and sample additional with replacement
            sampled_non_toxic = lang_non_toxic_df
            additional_needed = toxic_count - non_toxic_count
            if additional_needed > 0:
                additional_samples = lang_non_toxic_df.sample(n=additional_needed, replace=True, random_state=42)
                sampled_non_toxic = pd.concat([sampled_non_toxic, additional_samples], ignore_index=True)
                print(f"Using all {non_toxic_count:,} non-toxic comments and added {additional_needed:,} resampled to balance")
        
        # Combine toxic and non-toxic for this language
        lang_balanced = pd.concat([sampled_toxic, sampled_non_toxic], ignore_index=True)
        print(f"Final language size: {len(lang_balanced):,} ({len(sampled_toxic):,} toxic, {len(sampled_non_toxic):,} non-toxic)")
        balanced_dfs.append(lang_balanced)
    
    # Combine all balanced dataframes
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # If we have more than target size, sample down
    target_size = 51518  # Target size from the original requirement
    if len(balanced_df) > target_size:
        balanced_df = balanced_df.sample(n=target_size, random_state=42)
        print(f"\nSampled down to {target_size:,} comments")
    else:
        print(f"\nKept all {len(balanced_df):,} comments (less than target size {target_size:,})")
    
    # Get final statistics
    print("\nFinal dataset statistics:")
    print(f"Total comments: {len(balanced_df):,}")
    
    if 'lang' in balanced_df.columns:
        print("\nLanguage distribution in final dataset:")
        lang_dist = balanced_df['lang'].value_counts()
        for lang, count in lang_dist.items():
            toxic_in_lang = balanced_df[balanced_df['lang'] == lang][toxicity_cols].any(axis=1).sum()
            print(f"{lang}: {count:,} comments ({toxic_in_lang:,} toxic, {count-toxic_in_lang:,} non-toxic)")
    
    print("\nToxicity distribution in final dataset:")
    for col in toxicity_cols:
        toxic_count = (balanced_df[col] > 0).sum()
        print(f"{col.replace('_', ' ').title()}: {toxic_count:,} ({toxic_count/len(balanced_df)*100:.1f}%)")
    
    # Count comments with multiple toxicity types
    toxic_counts = balanced_df[toxicity_cols].astype(bool).sum(axis=1)
    print("\nComments by number of toxicity types:")
    for n_toxic, count in toxic_counts.value_counts().sort_index().items():
        print(f"{n_toxic} type{'s' if n_toxic != 1 else ''}: {count:,} ({count/len(balanced_df)*100:.1f}%)")
    
    # Save balanced dataset
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_balanced{ext}"
    
    print(f"\nSaving balanced dataset to: {output_file}")
    balanced_df.to_csv(output_file, index=False)
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    return balanced_df

if __name__ == "__main__":
    input_file = "dataset/processed/english_merged.csv"
    output_file = "dataset/processed/english_filtered.csv"
    
    filtered_df = filter_and_balance_comments(input_file, output_file) 