from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import json

# Load data with language column
train_df = pd.read_csv('dataset/raw/MULTILINGUAL_TOXIC_DATASET_360K_7LANG.csv')
lang_weights = {}

for lang in train_df['lang'].unique():
    lang_df = train_df[train_df['lang'] == lang]
    lang_weights[lang] = {}
    
    for col in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        y = lang_df[col].values.astype(np.int32)  # Convert to int32
        try:
            weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y)
            lang_weights[lang][col] = {
                '0': float(weights[0]),  # Convert to native Python float
                '1': float(weights[1]),  # Convert to native Python float
                'support_0': int((y == 0).sum()),  # Convert to native Python int
                'support_1': int((y == 1).sum())   # Convert to native Python int
            }
            
            # Print statistics for verification
            print(f"\n{lang} - {col}:")
            print(f"  Class 0: weight={weights[0]:.2f}, samples={int((y == 0).sum()):,}")
            print(f"  Class 1: weight={weights[1]:.2f}, samples={int((y == 1).sum()):,}")
            
        except ValueError as e:
            # Handle single-class columns
            dominant_class = int(y[0])  # Convert to native Python int
            total_samples = len(y)
            lang_weights[lang][col] = {
                '0': 1.0 if dominant_class == 0 else 0.0,
                '1': 0.0 if dominant_class == 0 else 1.0,
                'support_0': int((y == 0).sum()),
                'support_1': int((y == 1).sum())
            }
            print(f"\n⚠️ Warning: {lang} - {col} has only class {dominant_class} ({total_samples:,} samples)")

# Save language-specific weights
with open('language_class_weights.json', 'w', encoding='utf-8') as f:
    json.dump(lang_weights, f, indent=2, ensure_ascii=False)

print("\n✓ Weights saved to language_class_weights.json")
