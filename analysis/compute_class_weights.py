from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import json
from typing import Dict, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def calculate_safe_weights(
    support_0: int,
    support_1: int,
    max_weight: float = 15.0,
    min_weight: float = 0.5,
    boost_factor: float = 1.0
) -> Dict[str, float]:
    """
    Calculate class weights with controlled scaling and safety limits.
    
    Args:
        support_0: Number of negative samples
        support_1: Number of positive samples
        max_weight: Maximum allowed weight
        min_weight: Minimum allowed weight
        boost_factor: Optional boost for specific classes
    """
    total = support_0 + support_1
    
    # Calculate balanced weights
    raw_weight_1 = (total / (2 * support_1)) * boost_factor
    raw_weight_0 = total / (2 * support_0)
    
    # Apply safety limits
    weight_1 = min(max_weight, max(min_weight, raw_weight_1))
    weight_0 = min(max_weight, max(min_weight, raw_weight_0))
    
    return {
        "0": round(weight_0, 2),
        "1": round(weight_1, 2),
        "support_0": support_0,
        "support_1": support_1,
        "raw_weight_1": round(raw_weight_1, 2)  # For debugging
    }

def get_language_specific_params(lang: str, toxicity_type: str) -> Dict:
    """
    Get language and class specific parameters for weight calculation.
    """
    # Default parameters
    default_params = {
        "max_weight": 15.0,
        "min_weight": 0.5,
        "boost_factor": 1.0
    }
    
    # Language-specific adjustments
    lang_adjustments = {
        "en": {
            "toxic": {"boost_factor": 2.0},  # Double emphasis on English toxic
            "threat": {"max_weight": 12.5},  # Cap English threat weights
            "identity_hate": {"max_weight": 8.4}  # Balance with Russian
        },
        "tr": {
            "threat": {"max_weight": 9.8},
            "identity_hate": {"max_weight": 7.9}
        },
        "ru": {
            "threat": {"max_weight": 10.0},
            "identity_hate": {"max_weight": 8.4}
        }
    }
    
    # Get language-specific params if they exist
    lang_params = lang_adjustments.get(lang, {})
    class_params = lang_params.get(toxicity_type, {})
    
    # Merge with defaults
    return {**default_params, **class_params}

def compute_language_weights(df: pd.DataFrame) -> Dict:
    """
    Compute controlled weights for each language and toxicity type.
    """
    lang_weights = {}
    toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Process each language
    for lang in df['lang'].unique():
        logging.info(f"\nProcessing language: {lang}")
        lang_df = df[df['lang'] == lang]
        lang_weights[lang] = {}
        
        # Process each toxicity type
        for col in toxicity_columns:
            y = lang_df[col].values.astype(np.int32)
            support_0 = int((y == 0).sum())
            support_1 = int((y == 1).sum())
            
            # Get language and class specific parameters
            params = get_language_specific_params(lang, col)
            
            # Calculate weights
            weights = calculate_safe_weights(
                support_0=support_0,
                support_1=support_1,
                max_weight=params['max_weight'],
                min_weight=params['min_weight'],
                boost_factor=params['boost_factor']
            )
            
            lang_weights[lang][col] = weights
            
            # Log the results
            logging.info(f"\n{lang} - {col}:")
            logging.info(f"  Class 0: weight={weights['0']:.2f}, samples={weights['support_0']:,}")
            logging.info(f"  Class 1: weight={weights['1']:.2f}, samples={weights['support_1']:,}")
            if weights['raw_weight_1'] > weights['1']:
                logging.info(f"  Note: Weight capped from {weights['raw_weight_1']:.2f} to {weights['1']:.2f}")
    
    return lang_weights

def main():
    # Load dataset
    input_file = 'dataset/raw/MULTILINGUAL_TOXIC_DATASET_360K_7LANG.csv'
    logging.info(f"Loading dataset from {input_file}")
    df = pd.read_csv(input_file)
    
    # Compute weights
    lang_weights = compute_language_weights(df)
    
    # Add metadata
    weights_data = {
        "metadata": {
            "total_samples": len(df),
            "language_distribution": df['lang'].value_counts().to_dict(),
            "weight_calculation": {
                "method": "controlled_inverse_frequency",
                "parameters": {
                    "default_max_weight": 15.0,
                    "default_min_weight": 0.5,
                    "language_specific_adjustments": True
                }
            }
        },
        "weights": lang_weights
    }
    
    # Save weights
    output_file = 'weights/language_class_weights.json'
    logging.info(f"\nSaving weights to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(weights_data, f, indent=2, ensure_ascii=False)
    
    logging.info("\nWeight calculation complete!")
    
    # Print summary statistics
    logging.info("\nSummary of adjustments made:")
    for lang in lang_weights:
        for col in ['threat', 'identity_hate']:
            if col in lang_weights[lang]:
                weight = lang_weights[lang][col]['1']
                raw = lang_weights[lang][col]['raw_weight_1']
                if raw != weight:
                    logging.info(f"{lang} {col}: Adjusted from {raw:.2f}× to {weight:.2f}×")

if __name__ == "__main__":
    main()
