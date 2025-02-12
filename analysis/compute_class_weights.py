from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import json
from typing import Dict, Tuple, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_parameters(params: Dict) -> Dict:
    """
    Validate weight calculation parameters to prevent dangerous combinations.
    """
    if params['boost_factor'] * params['max_weight'] > 30:
        raise ValueError(f"Dangerous weight scaling detected: boost_factor * max_weight = {params['boost_factor'] * params['max_weight']}")
    return params

def calculate_safe_weights(
    support_0: int,
    support_1: int,
    max_weight: float = 15.0,
    min_weight: float = 0.5,
    boost_factor: float = 1.0,
    num_classes: int = 6,
    lang: str = None,
    toxicity_type: str = None
) -> Dict[str, float]:
    """
    Calculate class weights with adaptive scaling and controlled limits.
    Uses logarithmic scaling for better numerical stability while preserving
    language-specific adjustments.
    
    Args:
        support_0: Number of negative samples
        support_1: Number of positive samples
        max_weight: Maximum allowed weight
        min_weight: Minimum allowed weight
        boost_factor: Optional boost for specific classes
        num_classes: Number of toxicity classes (default=6)
        lang: Language code for language-specific constraints
        toxicity_type: Type of toxicity for class-specific constraints
    """
    # Input validation with detailed error messages
    if support_0 < 0 or support_1 < 0:
        raise ValueError(f"Negative sample counts: support_0={support_0}, support_1={support_1}")
    if support_0 + support_1 == 0:
        logging.warning(f"Empty dataset for {toxicity_type} in {lang}")
        return {
            "0": 1.0,
            "1": 1.0,
            "support_0": support_0,
            "support_1": support_1,
            "raw_weight_1": 1.0,
            "calculation_metadata": {
                "formula": "default_weights_empty_dataset",
                "constraints_applied": ["empty_dataset_fallback"]
            }
        }
    
    # Handle zero support cases safely
    if support_1 == 0:
        logging.warning(f"No positive samples for {toxicity_type} in {lang}")
        return {
            "0": 1.0,
            "1": max_weight,  # Use max weight for extremely rare positives
            "support_0": support_0,
            "support_1": support_1,
            "raw_weight_1": max_weight,
            "calculation_metadata": {
                "formula": "max_weight_no_positives",
                "constraints_applied": ["no_positives_fallback"]
            }
        }
    
    # Determine effective maximum weight based on class and language
    if lang == 'en' and toxicity_type == 'threat':
        effective_max = min(max_weight, 15.0)  # Absolute cap for EN threat
    elif toxicity_type == 'identity_hate':
        effective_max = min(max_weight, 10.0)  # Cap for identity hate
    else:
        effective_max = max_weight
    
    try:
        # Calculate base weights using logarithmic scaling for better stability
        total = support_0 + support_1
        log_ratio = np.log1p(support_0) / np.log1p(max(1, support_1))
        
        # Scale the log ratio to match the expected weight range
        scale_factor = total / (num_classes * max(1, support_1))
        raw_weight_1 = (log_ratio * scale_factor * boost_factor)
        raw_weight_0 = 1.0 / max(raw_weight_1, 1e-6)  # Prevent division by zero
        
        # Apply adaptive scaling based on class distribution
        if toxicity_type in ['threat', 'identity_hate']:
            # More aggressive scaling for severe classes
            raw_weight_1 *= (1 + np.log1p(total) / np.log1p(support_1)) / 2
        
        # Detect potential numerical instability
        if not np.isfinite(raw_weight_1) or not np.isfinite(raw_weight_0):
            logging.error(f"Numerical instability detected for {toxicity_type} in {lang}")
            raw_weight_1 = effective_max
            raw_weight_0 = min_weight
    except Exception as e:
        logging.error(f"Weight calculation error: {str(e)}")
        raw_weight_1 = effective_max
        raw_weight_0 = min_weight
    
    # Apply safety limits with effective maximum
    weight_1 = min(effective_max, max(min_weight, raw_weight_1))
    weight_0 = min(effective_max, max(min_weight, raw_weight_0))
    
    # Round weights for consistency and to prevent floating point issues
    weight_1 = round(float(weight_1), 3)
    weight_0 = round(float(weight_0), 3)
    
    return {
        "0": weight_0,
        "1": weight_1,
        "support_0": support_0,
        "support_1": support_1,
        "raw_weight_1": round(float(raw_weight_1), 3),
        "calculation_metadata": {
            "formula": "log_scaled_inverse_freq",
            "raw_log_ratio": round(float(log_ratio), 3),
            "scale_factor": round(float(scale_factor), 3),
            "constraints_applied": [
                f"max_weight={effective_max}",
                f"boost={boost_factor}",
                f"numerical_stability=enforced",
                f"adaptive_scaling={'enabled' if toxicity_type in ['threat', 'identity_hate'] else 'disabled'}"
            ]
        }
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
    
    # Updated language-specific adjustments based on analysis
    lang_adjustments = {
        "en": {
            "toxic": {"boost_factor": 1.67},  # To achieve ~3.5x weight
            "threat": {"max_weight": 15.0},  # Absolute maximum cap
            "identity_hate": {"max_weight": 5.0},  # Reduced from 8.4
            "severe_toxic": {"max_weight": 3.9}  # Corrected weight
        },
        "tr": {
            "threat": {"max_weight": 12.8},  # Aligned with cross-lingual ratio
            "identity_hate": {"max_weight": 6.2}  # Adjusted for balance
        },
        "ru": {
            "threat": {"max_weight": 12.8},  # Aligned with cross-lingual ratio
            "identity_hate": {"max_weight": 7.0}  # Adjusted for balance
        },
        "fr": {
            "toxic": {"boost_factor": 1.2}  # To achieve ~2.2x weight
        }
    }
    
    # Get language-specific params and validate
    lang_params = lang_adjustments.get(lang, {})
    class_params = lang_params.get(toxicity_type, {})
    merged_params = {**default_params, **class_params}
    
    return validate_parameters(merged_params)

def check_cross_language_consistency(lang_weights: Dict) -> List[str]:
    """
    Check for consistency of weights across languages.
    Returns a list of warnings for significant disparities.
    """
    warnings = []
    baseline = lang_weights['en']
    
    for lang in lang_weights:
        if lang == 'en':
            continue
            
        for cls in ['threat', 'identity_hate']:
            if cls in lang_weights[lang] and cls in baseline:
                ratio = lang_weights[lang][cls]['1'] / baseline[cls]['1']
                if ratio > 1.5 or ratio < 0.67:
                    warning = f"Large {cls} weight disparity: {lang} vs en ({ratio:.2f}x)"
                    warnings.append(warning)
                    logging.warning(warning)
    
    return warnings

def validate_dataset_balance(df: pd.DataFrame) -> bool:
    """
    Validate dataset balance across languages.
    Returns False if imbalance exceeds threshold.
    """
    sample_counts = df.groupby('lang').size()
    cv = sample_counts.std() / sample_counts.mean()
    
    if cv > 0.15:  # 15% threshold for coefficient of variation
        logging.error(f"Dataset language imbalance exceeds 15% (CV={cv:.2%})")
        for lang, count in sample_counts.items():
            logging.warning(f"{lang}: {count:,} samples ({count/len(df):.1%})")
        return False
    return True

def compute_language_weights(df: pd.DataFrame) -> Dict:
    """
    Compute controlled weights for each language and toxicity type.
    """
    # Validate dataset balance first
    if not validate_dataset_balance(df):
        logging.warning("Proceeding with imbalanced dataset - weights may need manual adjustment")
    
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
            
            # Calculate weights with language and class context
            weights = calculate_safe_weights(
                support_0=support_0,
                support_1=support_1,
                max_weight=params['max_weight'],
                min_weight=params['min_weight'],
                boost_factor=params['boost_factor'],
                lang=lang,
                toxicity_type=col
            )
            
            lang_weights[lang][col] = weights
            
            # Log the results
            logging.info(f"\n{lang} - {col}:")
            logging.info(f"  Class 0: weight={weights['0']:.2f}, samples={weights['support_0']:,}")
            logging.info(f"  Class 1: weight={weights['1']:.2f}, samples={weights['support_1']:,}")
            if weights['raw_weight_1'] > weights['1']:
                logging.info(f"  Note: Weight capped from {weights['raw_weight_1']:.2f} to {weights['1']:.2f}")
    
    # Check cross-language consistency
    consistency_warnings = check_cross_language_consistency(lang_weights)
    
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
                "method": "log_inverse_frequency",
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
