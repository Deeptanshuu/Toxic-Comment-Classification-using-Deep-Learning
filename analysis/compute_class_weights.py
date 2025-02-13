import numpy as np
import pandas as pd
import json
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_parameters(params: Dict) -> Dict:
    """
    Validate weight calculation parameters to prevent dangerous combinations.
    Includes validation for focal loss parameters.
    """
    # Check for dangerous weight scaling
    if params['boost_factor'] * params['max_weight'] > 30:
        raise ValueError(f"Dangerous weight scaling detected: boost_factor * max_weight = {params['boost_factor'] * params['max_weight']}")
    
    # Validate focal loss parameters
    if not 0 < params['gamma'] <= 5.0:
        raise ValueError(f"Invalid gamma value: {params['gamma']}. Must be in (0, 5.0]")
    
    if not 0 < params['alpha'] < 1:
        raise ValueError(f"Invalid alpha value: {params['alpha']}. Must be in (0, 1)")
    
    # Check for potentially unstable combinations
    if params['gamma'] > 3.0 and params['boost_factor'] > 1.5:
        logging.warning(f"Potentially unstable combination: high gamma ({params['gamma']}) with high boost factor ({params['boost_factor']})")
    
    if params['alpha'] > 0.4 and params['boost_factor'] > 1.5:
        logging.warning(f"Potentially unstable combination: high alpha ({params['alpha']}) with high boost factor ({params['boost_factor']})")
    
    return params

def calculate_safe_weights(
    support_0: int,
    support_1: int,
    max_weight: float = 15.0,
    min_weight: float = 0.5,
    gamma: float = 2.0,
    alpha: float = 0.25,
    boost_factor: float = 1.0,
    num_classes: int = 6,
    lang: str = None,
    toxicity_type: str = None
) -> Dict[str, float]:
    """
    Calculate class weights with focal loss and adaptive scaling.
    Uses focal loss components for better handling of imbalanced classes
    while preserving language-specific adjustments.
    
    Args:
        support_0: Number of negative samples
        support_1: Number of positive samples
        max_weight: Maximum allowed weight
        min_weight: Minimum allowed weight
        gamma: Focal loss gamma parameter for down-weighting easy examples
        alpha: Focal loss alpha parameter for balancing positive/negative classes
        boost_factor: Optional boost for specific classes
        num_classes: Number of toxicity classes (default=6)
        lang: Language code for language-specific constraints
        toxicity_type: Type of toxicity for class-specific constraints
    """
    # Input validation with detailed error messages
    if support_0 < 0 or support_1 < 0:
        raise ValueError(f"Negative sample counts: support_0={support_0}, support_1={support_1}")
    
    eps = 1e-7  # Small epsilon for numerical stability
    total = support_0 + support_1 + eps
    
    # Handle empty dataset case
    if total <= eps:
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
            "1": max_weight,
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
        # Calculate class frequencies
        freq_1 = support_1 / total
        freq_0 = support_0 / total
        
        # Focal loss components
        pt = freq_1 + eps  # Probability of target class
        modulating_factor = (1 - pt) ** gamma
        balanced_alpha = alpha / (alpha + (1 - alpha) * (1 - pt))
        
        # Base weight calculation with focal loss
        raw_weight_1 = balanced_alpha * modulating_factor / (pt + eps)
        
        # Apply adaptive scaling for severe classes
        if toxicity_type in ['threat', 'identity_hate']:
            severity_factor = (1 + np.log1p(total) / np.log1p(support_1)) / 2
            raw_weight_1 *= severity_factor
        
        # Apply boost factor
        raw_weight_1 *= boost_factor
        
        # Detect potential numerical instability
        if not np.isfinite(raw_weight_1):
            logging.error(f"Numerical instability detected for {toxicity_type} in {lang}")
            raw_weight_1 = effective_max
            
    except Exception as e:
        logging.error(f"Weight calculation error: {str(e)}")
        raw_weight_1 = effective_max
    
    # Apply safety limits with effective maximum
    weight_1 = min(effective_max, max(min_weight, raw_weight_1))
    weight_0 = 1.0  # Reference weight for majority class
    
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
            "formula": "focal_loss_with_adaptive_scaling",
            "gamma": round(float(gamma), 3),
            "alpha": round(float(alpha), 3),
            "final_pt": round(float(pt), 4),
            "effective_max": round(float(effective_max), 3),
            "modulating_factor": round(float(modulating_factor), 4),
            "balanced_alpha": round(float(balanced_alpha), 4),
            "severity_adjusted": toxicity_type in ['threat', 'identity_hate'],
            "boost_factor": round(float(boost_factor), 3),
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
    Includes focal loss parameters and their adjustments per language/class.
    """
    # Default parameters
    default_params = {
        "max_weight": 15.0,
        "min_weight": 0.5,
        "boost_factor": 1.0,
        "gamma": 2.0,  # Default focal loss gamma
        "alpha": 0.25  # Default focal loss alpha
    }
    
    # Updated language-specific adjustments based on analysis
    lang_adjustments = {
        "en": {
            "toxic": {
                "boost_factor": 1.67,  # To achieve ~3.5x weight
                "gamma": 2.5  # More focus on hard examples for main class
            },
            "threat": {
                "max_weight": 15.0,  # Absolute maximum cap
                "gamma": 3.0,  # Higher gamma for severe class
                "alpha": 0.3  # Slightly higher alpha for better recall
            },
            "identity_hate": {
                "max_weight": 5.0,  # Reduced from 8.4
                "gamma": 3.0,  # Higher gamma for severe class
                "alpha": 0.3  # Slightly higher alpha for better recall
            },
            "severe_toxic": {
                "max_weight": 3.9,  # Corrected weight
                "gamma": 2.5  # Moderate gamma for balance
            }
        },
        "tr": {
            "threat": {
                "max_weight": 12.8,  # Aligned with cross-lingual ratio
                "gamma": 2.8  # Slightly lower than EN for stability
            },
            "identity_hate": {
                "max_weight": 6.2,  # Adjusted for balance
                "gamma": 2.8  # Slightly lower than EN for stability
            }
        },
        "ru": {
            "threat": {
                "max_weight": 12.8,  # Aligned with cross-lingual ratio
                "gamma": 2.8  # Slightly lower than EN for stability
            },
            "identity_hate": {
                "max_weight": 7.0,  # Adjusted for balance
                "gamma": 2.8  # Slightly lower than EN for stability
            }
        },
        "fr": {
            "toxic": {
                "boost_factor": 1.2,  # To achieve ~2.2x weight
                "gamma": 2.2  # Lower gamma for better stability
            }
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

def validate_weights(lang_weights: Dict) -> List[str]:
    """
    Ensure weights meet multilingual safety criteria.
    Validates weight ratios and focal loss parameters across languages.
    
    Args:
        lang_weights: Dictionary of weights per language and class
        
    Returns:
        List of validation warnings
        
    Raises:
        ValueError: If weights violate safety constraints
    """
    warnings = []
    
    for lang in lang_weights:
        for cls in lang_weights[lang]:
            w1 = lang_weights[lang][cls]['1']
            w0 = lang_weights[lang][cls]['0']
            
            # Check weight ratio sanity
            ratio = w1 / w0
            if ratio > 30:
                raise ValueError(
                    f"Dangerous weight ratio {ratio:.1f}x for {lang} {cls}. "
                    f"Weight_1={w1:.3f}, Weight_0={w0:.3f}"
                )
            elif ratio > 20:
                warnings.append(
                    f"High weight ratio {ratio:.1f}x for {lang} {cls}"
                )
            
            # Check focal parameter boundaries
            metadata = lang_weights[lang][cls]['calculation_metadata']
            gamma = metadata.get('gamma', 0.0)
            alpha = metadata.get('alpha', 0.0)
            
            if gamma > 5.0:
                raise ValueError(
                    f"Unsafe gamma={gamma:.1f} for {lang} {cls}. "
                    f"Must be <= 5.0"
                )
            elif gamma > 4.0:
                warnings.append(
                    f"High gamma={gamma:.1f} for {lang} {cls}"
                )
                
            if alpha > 0.9:
                raise ValueError(
                    f"Unsafe alpha={alpha:.2f} for {lang} {cls}. "
                    f"Must be < 0.9"
                )
            elif alpha > 0.7:
                warnings.append(
                    f"High alpha={alpha:.2f} for {lang} {cls}"
                )
            
            # Check for combined risk factors
            if gamma > 3.0 and ratio > 15:
                warnings.append(
                    f"Risky combination for {lang} {cls}: "
                    f"gamma={gamma:.1f}, ratio={ratio:.1f}x"
                )
    
    return warnings

def compute_language_weights(df: pd.DataFrame) -> Dict:
    """
    Compute weights with inter-language normalization to ensure consistent 
    weighting across languages while preserving relative class relationships.
    """
    # Validate dataset balance first
    if not validate_dataset_balance(df):
        logging.warning("Proceeding with imbalanced dataset - weights may need manual adjustment")
    
    lang_weights = {}
    toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # First pass: calculate raw weights for each language and class
    logging.info("\nFirst pass: Calculating raw weights")
    for lang in df['lang'].unique():
        logging.info(f"\nProcessing language: {lang}")
        lang_df = df[df['lang'] == lang]
        lang_weights[lang] = {}
        
        for col in toxicity_columns:
            y = lang_df[col].values.astype(np.int32)
            support_0 = int((y == 0).sum())
            support_1 = int((y == 1).sum())
            
            params = get_language_specific_params(lang, col)
            weights = calculate_safe_weights(
                support_0=support_0,
                support_1=support_1,
                max_weight=params['max_weight'],
                min_weight=params['min_weight'],
                gamma=params['gamma'],
                alpha=params['alpha'],
                boost_factor=params['boost_factor'],
                lang=lang,
                toxicity_type=col
            )
            lang_weights[lang][col] = weights
            
            # Log initial weights
            logging.info(f"  {col} - Initial weights:")
            logging.info(f"    Class 0: {weights['0']:.3f}, samples: {support_0:,}")
            logging.info(f"    Class 1: {weights['1']:.3f}, samples: {support_1:,}")
    
    # Second pass: normalize weights across languages
    logging.info("\nSecond pass: Normalizing weights across languages")
    for col in toxicity_columns:
        # Find maximum weight for this toxicity type across all languages
        max_weight = max(
            lang_weights[lang][col]['1'] 
            for lang in lang_weights
        )
        
        if max_weight > 0:  # Prevent division by zero
            logging.info(f"\nNormalizing {col}:")
            logging.info(f"  Maximum weight across languages: {max_weight:.3f}")
            
            # Normalize weights for each language
            for lang in lang_weights:
                original_weight = lang_weights[lang][col]['1']
                
                # Normalize and rescale
                normalized_weight = (original_weight / max_weight) * 15.0
                
                # Update weight while preserving metadata
                lang_weights[lang][col]['raw_weight_1'] = original_weight
                lang_weights[lang][col]['1'] = round(normalized_weight, 3)
                
                # Add normalization info to metadata
                lang_weights[lang][col]['calculation_metadata'].update({
                    'normalization': {
                        'original_weight': round(float(original_weight), 3),
                        'max_weight_across_langs': round(float(max_weight), 3),
                        'normalization_factor': round(float(15.0 / max_weight), 3)
                    }
                })
                
                # Log normalization results
                logging.info(f"  {lang}: {original_weight:.3f} → {normalized_weight:.3f}")
    
    # Validate final weights
    logging.info("\nValidating final weights:")
    for col in toxicity_columns:
        weights_range = [
            lang_weights[lang][col]['1']
            for lang in lang_weights
        ]
        logging.info(f"  {col}: range [{min(weights_range):.3f}, {max(weights_range):.3f}]")
    
    # Validate weights meet safety criteria
    validation_warnings = validate_weights(lang_weights)
    if validation_warnings:
        logging.warning("\nWeight validation warnings:")
        for warning in validation_warnings:
            logging.warning(f"  {warning}")
    
    # Check cross-language consistency
    consistency_warnings = check_cross_language_consistency(lang_weights)
    if consistency_warnings:
        logging.warning("\nCross-language consistency warnings:")
        for warning in consistency_warnings:
            logging.warning(f"  {warning}")
    
    return lang_weights

def main():
    # Load dataset
    input_file = 'dataset/processed/MULTILINGUAL_TOXIC_DATASET_AUGMENTED.csv'
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
                "method": "focal_loss_with_adaptive_scaling",
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
