import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys
from toxic_augment import ToxicAugmenter
import json
from sklearn.utils import resample

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_file = log_dir / f"balance_english_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)

logger = logging.getLogger(__name__)

def analyze_label_distribution(df, lang='en'):
    """Analyze label distribution for a specific language"""
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    lang_df = df[df['lang'] == lang]
    total = len(lang_df)
    
    if total == 0:
        logger.warning(f"No samples found for language {lang.upper()}.")
        return {}
    
    logger.info(f"\nLabel Distribution for {lang.upper()}:")
    logger.info("-" * 50)
    dist = {}
    for label in labels:
        count = lang_df[label].sum()
        percentage = (count / total) * 100
        dist[label] = {'count': int(count), 'percentage': percentage}
        logger.info(f"{label}: {count:,} ({percentage:.2f}%)")
    return dist

def analyze_language_distribution(df):
    """Analyze current language distribution"""
    lang_dist = df['lang'].value_counts()
    logger.info("\nCurrent Language Distribution:")
    logger.info("-" * 50)
    for lang, count in lang_dist.items():
        logger.info(f"{lang}: {count:,} comments ({count/len(df)*100:.2f}%)")
    return lang_dist

def calculate_required_samples(df):
    """Calculate how many English samples we need to generate"""
    lang_counts = df['lang'].value_counts()
    target_count = lang_counts.max()  # Use the largest language count as target
    en_count = lang_counts.get('en', 0)
    required_samples = target_count - en_count
    
    logger.info(f"\nTarget count per language: {target_count:,}")
    logger.info(f"Current English count: {en_count:,}")
    logger.info(f"Required additional English samples: {required_samples:,}")
    
    return required_samples

def generate_balanced_samples(df, required_samples):
    """Generate samples maintaining original class distribution ratios"""
    logger.info("\nGenerating balanced samples...")
    
    # Get English samples
    en_df = df[df['lang'] == 'en']
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Define class weights based on analysis
    CLASS_WEIGHTS = {
        'toxic': 1.03,
        'severe_toxic': 1.5,
        'obscene': 1.0,
        'threat': 2.01,
        'insult': 1.0,
        'identity_hate': 1.15
    }
    
    # Calculate label combination frequencies
    label_combinations = en_df.groupby(labels).size()
    total_en = len(en_df)
    
    # Calculate weighted target counts for each combination
    target_counts = pd.Series(0, index=label_combinations.index)
    for combo_labels in label_combinations.index:
        combo_dict = dict(zip(labels, combo_labels))
        
        # Calculate weight multiplier based on present labels
        weight_multiplier = 1.0
        present_labels = [label for label, value in combo_dict.items() if value == 1]
        if present_labels:
            weight_multiplier = max(CLASS_WEIGHTS[label] for label in present_labels)
            
            # Extra boost for combinations with multiple toxic types
            if len(present_labels) > 1:
                weight_multiplier *= (1 + 0.1 * len(present_labels))
        
        # Calculate weighted count
        base_count = (label_combinations[combo_labels] / total_en * required_samples)
        target_counts[combo_labels] = round(base_count * weight_multiplier)
    
    # Sort combinations by toxicity priority (but keep all combinations)
    def get_combo_priority(combo_labels):
        combo_dict = dict(zip(labels, combo_labels))
        num_toxic = sum(combo_dict.values())
        has_threat = combo_dict['threat']
        has_severe = combo_dict['severe_toxic']
        has_identity = combo_dict['identity_hate']
        return (num_toxic, has_threat, has_severe, has_identity)
    
    # Sort combinations by priority
    sorted_combinations = sorted(
        [(combo, count) for combo, count in target_counts.items()],
        key=lambda x: get_combo_priority(x[0]),
        reverse=True
    )
    
    logger.info("\nTarget sample counts after weighting (sorted by priority):")
    total_weighted = 0
    for combo_labels, count in sorted_combinations:
        combo_dict = dict(zip(labels, combo_labels))
        present_labels = [label for label, value in combo_dict.items() if value == 1]
        logger.info(f"Labels: {', '.join(present_labels) if present_labels else 'Clean'}")
        logger.info(f"Count: {count:,}")
        total_weighted += count
    logger.info(f"\nTotal weighted samples to generate: {total_weighted:,}")
    
    augmented_samples = []
    augmenter = ToxicAugmenter()
    total_generated = 0
    
    # Generate samples for each label combination in priority order
    for combo_labels, target_count in sorted_combinations:
        if target_count == 0:
            continue
            
        combo_dict = dict(zip(labels, combo_labels))
        present_labels = [label for label, value in combo_dict.items() if value == 1]
        
        logger.info(f"\nGenerating {target_count:,} samples for combination:")
        logger.info(f"Labels: {', '.join(present_labels) if present_labels else 'Clean'}")
        
        # Get seed texts with this label combination
        mask = pd.Series(True, index=en_df.index)
        for label, value in combo_dict.items():
            mask &= (en_df[label] == value)
        seed_texts = en_df[mask]['comment_text'].tolist()
        
        if not seed_texts:
            logger.warning("No seed texts found for this combination, skipping...")
            continue
        
        # Generate samples with timeout and strict count enforcement
        try:
            new_samples = augmenter.augment_dataset(
                target_samples=target_count,
                label_combo=combo_dict,
                seed_texts=seed_texts,
                timeout_minutes=30  # Add timeout to prevent getting stuck
            )
            
            if new_samples is not None and not new_samples.empty:
                # Ensure we don't exceed target count
                if len(new_samples) > target_count:
                    new_samples = new_samples.head(target_count)
                    
                augmented_samples.append(new_samples)
                total_generated += len(new_samples)
                logger.info(f"✓ Generated {len(new_samples):,} samples")
                
                # Log current progress
                logger.info(f"Progress: {total_generated:,}/{total_weighted:,} samples ({total_generated/total_weighted*100:.1f}%)")
                
                # Early stopping if we've generated enough samples
                if total_generated >= required_samples:
                    logger.info("Reached required sample count, stopping generation")
                    break
            else:
                logger.warning("Failed to generate samples for this combination")
                
        except TimeoutError:
            logger.warning(f"Timeout reached for combination {present_labels if present_labels else 'Clean'}, moving to next...")
            continue
    
    # Combine all generated samples
    if augmented_samples:
        augmented_df = pd.concat(augmented_samples, ignore_index=True)
        augmented_df['lang'] = 'en'
        
        # Ensure we don't exceed the required sample count
        if len(augmented_df) > required_samples:
            logger.info(f"Trimming excess samples from {len(augmented_df):,} to {required_samples:,}")
            augmented_df = augmented_df.head(required_samples)
        
        # Log final class distribution
        logger.info("\nFinal class distribution in generated samples:")
        for label in labels:
            count = augmented_df[label].sum()
            percentage = (count / len(augmented_df)) * 100
            logger.info(f"{label}: {count:,} ({percentage:.2f}%)")
        
        # Also log clean samples
        clean_count = len(augmented_df[augmented_df[labels].sum(axis=1) == 0])
        clean_percentage = (clean_count / len(augmented_df)) * 100
        logger.info(f"Clean samples: {clean_count:,} ({clean_percentage:.2f}%)")
        
        return augmented_df
    else:
        raise Exception("Failed to generate any valid samples")

def balance_english_data():
    """Main function to balance English data with other languages"""
    try:
        # Load dataset
        input_file = 'dataset/processed/MULTILINGUAL_TOXIC_DATASET_360K_7LANG_FINAL.csv'
        logger.info(f"Loading dataset from {input_file}")
        df = pd.read_csv(input_file)
        
        # Analyze current distribution
        logger.info("\nAnalyzing current distribution...")
        initial_dist = analyze_language_distribution(df)
        initial_label_dist = analyze_label_distribution(df, 'en')
        
        # Calculate required samples
        required_samples = calculate_required_samples(df)
        
        if required_samples <= 0:
            logger.info("English data is already balanced. No augmentation needed.")
            return
        
        # Generate balanced samples
        augmented_df = generate_balanced_samples(df, required_samples)
        
        # Merge with original dataset
        logger.info("\nMerging datasets...")
        output_file = 'dataset/processed/MULTILINGUAL_TOXIC_DATASET_BALANCED.csv'
        
        # Combine datasets
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        
        # Save balanced dataset
        combined_df.to_csv(output_file, index=False)
        logger.info(f"\nSaved balanced dataset to {output_file}")
        
        # Final distribution check
        logger.info("\nFinal distribution after balancing:")
        final_dist = analyze_language_distribution(combined_df)
        final_label_dist = analyze_label_distribution(combined_df, 'en')
        
        # Save distribution statistics
        stats = {
            'timestamp': timestamp,
            'initial_distribution': {
                'languages': initial_dist.to_dict(),
                'english_labels': initial_label_dist
            },
            'final_distribution': {
                'languages': final_dist.to_dict(),
                'english_labels': final_label_dist
            },
            'samples_generated': len(augmented_df),
            'total_samples': len(combined_df)
        }
        
        stats_file = f'logs/balance_stats_{timestamp}.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"\nSaved balancing statistics to {stats_file}")
        
    except Exception as e:
        logger.error(f"Error during balancing: {str(e)}")
        raise

def main():
    balance_english_data()

if __name__ == "__main__":
    logger.info("Starting English data balancing process...")
    main() 