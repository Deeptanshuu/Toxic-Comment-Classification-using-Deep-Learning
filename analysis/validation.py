import json
import pandas as pd


def validate_dataset_distributions():
    """Validate dataset distributions and check for potential issues"""
    
    # Load datasets
    train_df = pd.read_csv('dataset/split/train.csv')
    val_df = pd.read_csv('dataset/split/val.csv')
    test_df = pd.read_csv('dataset/split/test.csv')
    
    # Load original weights for comparison
    with open('weights/class_weights.json', 'r') as f:
        weights_dict = json.load(f)
    
    toxicity_cols = [
        'toxic', 'severe_toxic', 'obscene', 
        'threat', 'insult', 'identity_hate'
    ]
    
    print("\n=== Dataset Distribution Validation ===\n")
    
    # 1. Per-language Label Distribution Analysis
    print("1. Per-Language Label Distribution Analysis:")
    print("-" * 50)
    
    for split_name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n{split_name} Set:")
        for lang in df['lang'].unique():
            lang_df = df[df['lang'] == lang]
            print(f"\n  {lang.upper()} ({len(lang_df)} samples):")
            
            # Compare with original distribution
            for col in toxicity_cols:
                curr_ratio = lang_df[col].mean()
                orig_ratio = weights_dict['language_weights'][lang][col]['positive_ratio']
                diff = abs(curr_ratio - orig_ratio)
                
                print(f"    {col:12s}: {curr_ratio:.4f} (Original: {orig_ratio:.4f}, Diff: {diff:.4f})")
                if diff > 0.05:  # Alert if difference is more than 5%
                    print(f"    ⚠️  Warning: Large distribution difference in {col} for {lang}")
    
    # 2. Label Leakage Check
    print("\n\n2. Label Leakage Check:")
    print("-" * 50)
    
    # Create comment fingerprints (simplified hash of cleaned text)
    def get_fingerprint(text):
        # Simple fingerprint: lowercase, remove spaces, take first 100 chars
        return str(text).lower().replace(' ', '')[:100]
    
    train_fps = set(train_df['comment_text'].apply(get_fingerprint))
    val_fps = set(val_df['comment_text'].apply(get_fingerprint))
    test_fps = set(test_df['comment_text'].apply(get_fingerprint))
    
    # Check overlaps
    train_val_overlap = len(train_fps & val_fps)
    train_test_overlap = len(train_fps & test_fps)
    val_test_overlap = len(val_fps & test_fps)
    
    print(f"\nOverlapping Comments:")
    print(f"  Train-Val : {train_val_overlap} ({train_val_overlap/len(train_df)*100:.2f}% of train)")
    print(f"  Train-Test: {train_test_overlap} ({train_test_overlap/len(train_df)*100:.2f}% of train)")
    print(f"  Val-Test  : {val_test_overlap} ({val_test_overlap/len(val_df)*100:.2f}% of val)")
    
    # 3. Rare Class Analysis
    print("\n\n3. Rare Class Analysis:")
    print("-" * 50)
    
    rare_classes = ['threat', 'identity_hate']
    for split_name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n{split_name} Set:")
        for col in rare_classes:
            positive_samples = df[df[col] > 0]
            print(f"\n  {col} (Total Positive: {len(positive_samples)}):")
            
            # Language distribution for rare class
            lang_dist = positive_samples['lang'].value_counts()
            for lang, count in lang_dist.items():
                print(f"    {lang:2s}: {count:5d} samples ({count/len(positive_samples)*100:.1f}%)")
            
            # Check for potential data quality issues
            if len(positive_samples) < 100:
                print(f"    ⚠️  Warning: Very few positive samples for {col}")
            
            # Calculate class imbalance ratio
            imbalance_ratio = len(df[df[col] == 0]) / max(1, len(df[df[col] > 0]))
            print(f"    Imbalance Ratio: {imbalance_ratio:.1f}:1")

if __name__ == "__main__":
    validate_dataset_distributions()
