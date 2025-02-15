import torch
from model.language_aware_transformer import LanguageAwareTransformer
from transformers import XLMRobertaTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, 
    confusion_matrix, hamming_loss, 
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime
import argparse
from torch.utils.data import Dataset, DataLoader
import gc
import multiprocessing
from pathlib import Path
import hashlib
import logging
from sklearn.metrics import make_scorer

# Set matplotlib to non-interactive backend
plt.switch_backend('agg')

# Set memory optimization environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

logger = logging.getLogger(__name__)

class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, config):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        
        # Convert labels to numpy array for efficiency
        self.labels = df[config.label_columns].values
        
        # Create language mapping
        self.lang_to_id = {
            'en': 0, 'ru': 1, 'tr': 2, 'es': 3,
            'fr': 4, 'it': 5, 'pt': 6
        }
        
        # Convert language codes to numeric indices
        self.langs = np.array([self.lang_to_id.get(lang, 0) for lang in df['lang']])
        
        print(f"Initialized dataset with {len(self)} samples")
        logger.info(f"Dataset initialized with {len(self)} samples")
        logger.info(f"Label columns: {config.label_columns}")
        logger.info(f"Unique languages: {np.unique(df['lang'])}")
        logger.info(f"Language mapping: {self.lang_to_id}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if idx % 1000 == 0:
            print(f"Loading sample {idx}")
            logger.debug(f"Loading sample {idx}")
        
        # Get text and labels
        text = self.df.iloc[idx]['comment_text']
        labels = torch.FloatTensor(self.labels[idx])
        lang = torch.tensor(self.langs[idx], dtype=torch.long)  # Ensure long dtype
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
            'lang': lang
        }

class ThresholdOptimizer(BaseEstimator, ClassifierMixin):
    """Custom estimator for threshold optimization"""
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.probabilities_ = None
        
    def fit(self, X, y):
        # Store probabilities for prediction
        self.probabilities_ = X
        return self
        
    def predict(self, X):
        # Apply threshold to probabilities
        return (X > self.threshold).astype(int)
        
    def score(self, X, y):
        # Return F1 score with proper handling of edge cases
        predictions = self.predict(X)
        
        # Handle edge case where all samples are negative
        if y.sum() == 0:
            return 1.0 if predictions.sum() == 0 else 0.0
            
        # Calculate metrics with zero_division=1
        try:
            precision = precision_score(y, predictions, zero_division=1)
            recall = recall_score(y, predictions, zero_division=1)
            
            # Calculate F1 manually to avoid warnings
            if precision + recall == 0:
                return 0.0
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
        except Exception:
            return 0.0

def load_model(model_path):
    """Load model and tokenizer from versioned checkpoint directory"""
    try:
        # Check if model_path points to a specific checkpoint or base directory
        model_dir = Path(model_path)
        if model_dir.is_dir():
            # Check for 'latest' symlink first
            latest_link = model_dir / 'latest'
            if latest_link.exists() and latest_link.is_symlink():
                model_dir = latest_link.resolve()
                logger.info(f"Using latest checkpoint: {model_dir}")
            else:
                # Find most recent checkpoint
                checkpoints = sorted([
                    d for d in model_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('checkpoint_epoch')
                ])
                if checkpoints:
                    model_dir = checkpoints[-1]
                    logger.info(f"Using most recent checkpoint: {model_dir}")
                else:
                    logger.info("No checkpoints found, using base directory")
        
        logger.info(f"Loading model from: {model_dir}")
        
        # Initialize the custom model architecture
        model = LanguageAwareTransformer(
            num_labels=6,
            hidden_size=1024,
            num_attention_heads=16,
            model_name='xlm-roberta-large'
        )
        
        # Load the trained weights
        weights_path = model_dir / 'pytorch_model.bin'
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
            
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully")
        
        # Load base XLM-RoBERTa tokenizer directly
        logger.info("Loading XLM-RoBERTa tokenizer...")
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        
        # Load training metadata if available
        metadata_path = model_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.info(f"Loaded checkpoint metadata: Epoch {metadata.get('epoch', 'unknown')}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None, None

def optimize_threshold(y_true, y_pred_proba, n_steps=50):
    """
    Optimize threshold using grid search to maximize F1 score
    """
    # Handle edge case where all samples are negative
    if y_true.sum() == 0:
        return {
            'threshold': 0.5,  # Use default threshold
            'f1_score': 1.0,   # Perfect score for all negative samples
            'support': 0,
            'total_samples': len(y_true)
        }
    
    # Create parameter grid
    param_grid = {
        'threshold': np.linspace(0.3, 0.7, n_steps)
    }
    
    # Initialize optimizer
    optimizer = ThresholdOptimizer()
    
    # Run grid search with custom scoring
    grid_search = GridSearchCV(
        optimizer,
        param_grid,
        scoring=make_scorer(f1_score, zero_division=1),
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    
    # Reshape probabilities to 2D array
    X = y_pred_proba.reshape(-1, 1)
    
    # Fit grid search
    grid_search.fit(X, y_true)
    
    # Get best results
    best_threshold = grid_search.best_params_['threshold']
    best_f1 = grid_search.best_score_
    
    return {
        'threshold': float(best_threshold),
        'f1_score': float(best_f1),
        'support': int(y_true.sum()),
        'total_samples': len(y_true)
    }

def calculate_optimal_thresholds(predictions, labels, langs):
    """Calculate optimal thresholds for each class and language combination using Bayesian optimization"""
    logger.info("Calculating optimal thresholds using Bayesian optimization...")
    
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    unique_langs = np.unique(langs)
    
    thresholds = {
        'global': {},
        'per_language': {}
    }
    
    # Calculate global thresholds
    logger.info("Computing global thresholds...")
    for i, class_name in enumerate(tqdm(toxicity_types, desc="Global thresholds")):
        thresholds['global'][class_name] = optimize_threshold(
            labels[:, i],
            predictions[:, i],
            n_steps=50
        )
    
    # Calculate language-specific thresholds
    logger.info("Computing language-specific thresholds...")
    for lang in tqdm(unique_langs, desc="Language thresholds"):
        lang_mask = langs == lang
        if not lang_mask.any():
            continue
            
        thresholds['per_language'][str(lang)] = {}
        lang_preds = predictions[lang_mask]
        lang_labels = labels[lang_mask]
        
        for i, class_name in enumerate(toxicity_types):
            # Only optimize if we have enough samples
            if lang_labels[:, i].sum() >= 100:  # Minimum samples threshold
                thresholds['per_language'][str(lang)][class_name] = optimize_threshold(
                    lang_labels[:, i],
                    lang_preds[:, i],
                    n_steps=30  # Fewer iterations for per-language optimization
                )
            else:
                # Use global threshold if not enough samples
                thresholds['per_language'][str(lang)][class_name] = thresholds['global'][class_name]
    
    return thresholds

def evaluate_model(model, val_loader, device, output_dir):
    """Evaluate model performance on validation set"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_langs = []
    
    total_samples = len(val_loader.dataset)
    total_batches = len(val_loader)
    
    logger.info(f"\nStarting evaluation on {total_samples:,} samples in {total_batches} batches")
    progress_bar = tqdm(
        val_loader,
        desc="Evaluating",
        total=total_batches,
        unit="batch",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    with torch.inference_mode():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            langs = batch['lang'].cpu().numpy()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lang_ids=batch['lang'].to(device)
            )
            
            predictions = outputs['probabilities'].cpu().numpy()
            
            all_predictions.append(predictions)
            all_labels.append(labels)
            all_langs.append(langs)
            
            # Update progress bar description with batch size
            progress_bar.set_description(f"Processed batch ({len(input_ids)} samples)")
    
    # Concatenate all batches with progress bar
    logger.info("\nProcessing results...")
    predictions = np.vstack(all_predictions)
    labels = np.vstack(all_labels)
    langs = np.concatenate(all_langs)
    
    logger.info(f"Computing metrics for {len(predictions):,} samples...")
    
    # Calculate metrics with progress indication
    results = calculate_metrics(predictions, labels, langs)
    
    # Save results with progress indication
    logger.info("Saving evaluation results...")
    save_results(
        results=results,
        predictions=predictions,
        labels=labels,
        langs=langs,
        output_dir=output_dir
    )
    
    # Plot metrics
    logger.info("Generating metric plots...")
    plot_metrics(results, output_dir, predictions=predictions, labels=labels)
    
    logger.info("Evaluation complete!")
    return results, predictions

def calculate_metrics(predictions, labels, langs):
    """Calculate detailed metrics"""
    results = {
        'default_thresholds': {
            'overall': {},
            'per_language': {},
            'per_class': {}
        },
        'optimized_thresholds': {
            'overall': {},
            'per_language': {},
            'per_class': {}
        }
    }
    
    # Default threshold of 0.5
    DEFAULT_THRESHOLD = 0.5
    
    # Calculate metrics with default threshold
    logger.info("Computing metrics with default threshold (0.5)...")
    binary_predictions_default = (predictions > DEFAULT_THRESHOLD).astype(int)
    results['default_thresholds']['overall'] = calculate_overall_metrics(predictions, labels, binary_predictions_default)
    
    # Calculate per-language metrics with default threshold
    unique_langs = np.unique(langs)
    logger.info(f"Computing per-language metrics with default threshold...")
    for lang in tqdm(unique_langs, desc="Language metrics (default)", ncols=100):
        lang_mask = langs == lang
        if not lang_mask.any():
            continue
            
        lang_preds = predictions[lang_mask]
        lang_labels = labels[lang_mask]
        lang_binary_preds = binary_predictions_default[lang_mask]
        
        results['default_thresholds']['per_language'][str(lang)] = calculate_overall_metrics(
            lang_preds, lang_labels, lang_binary_preds
        )
        results['default_thresholds']['per_language'][str(lang)]['sample_count'] = int(lang_mask.sum())
    
    # Calculate per-class metrics with default threshold
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    logger.info("Computing per-class metrics with default threshold...")
    for i, class_name in enumerate(tqdm(toxicity_types, desc="Class metrics (default)", ncols=100)):
        results['default_thresholds']['per_class'][class_name] = calculate_class_metrics(
            labels[:, i],
            predictions[:, i],
            binary_predictions_default[:, i],
            DEFAULT_THRESHOLD
        )
    
    # Calculate optimal thresholds and corresponding metrics
    logger.info("Computing optimal thresholds...")
    thresholds = calculate_optimal_thresholds(predictions, labels, langs)
    
    # Apply optimal thresholds
    logger.info("Computing metrics with optimized thresholds...")
    binary_predictions_opt = np.zeros_like(predictions, dtype=int)
    for i, class_name in enumerate(toxicity_types):
        opt_threshold = thresholds['global'][class_name]['threshold']
        binary_predictions_opt[:, i] = (predictions[:, i] > opt_threshold).astype(int)
    
    # Calculate overall metrics with optimized thresholds
    results['optimized_thresholds']['overall'] = calculate_overall_metrics(predictions, labels, binary_predictions_opt)
    
    # Calculate per-language metrics with optimized thresholds
    logger.info(f"Computing per-language metrics with optimized thresholds...")
    for lang in tqdm(unique_langs, desc="Language metrics (optimized)", ncols=100):
        lang_mask = langs == lang
        if not lang_mask.any():
            continue
            
        lang_preds = predictions[lang_mask]
        lang_labels = labels[lang_mask]
        lang_binary_preds = binary_predictions_opt[lang_mask]
        
        results['optimized_thresholds']['per_language'][str(lang)] = calculate_overall_metrics(
            lang_preds, lang_labels, lang_binary_preds
        )
        results['optimized_thresholds']['per_language'][str(lang)]['sample_count'] = int(lang_mask.sum())
    
    # Calculate per-class metrics with optimized thresholds
    logger.info("Computing per-class metrics with optimized thresholds...")
    for i, class_name in enumerate(tqdm(toxicity_types, desc="Class metrics (optimized)", ncols=100)):
        opt_threshold = thresholds['global'][class_name]['threshold']
        results['optimized_thresholds']['per_class'][class_name] = calculate_class_metrics(
            labels[:, i],
            predictions[:, i],
            binary_predictions_opt[:, i],
            opt_threshold
        )
    
    # Store the thresholds used
    results['thresholds'] = thresholds
    
    return results

def calculate_overall_metrics(predictions, labels, binary_predictions):
    """Calculate overall metrics for multi-label classification"""
    metrics = {}
    
    # AUC scores (threshold independent)
    try:
        metrics['auc_macro'] = roc_auc_score(labels, predictions, average='macro')
        metrics['auc_weighted'] = roc_auc_score(labels, predictions, average='weighted')
    except ValueError:
        # Handle case where a class has no positive samples
        metrics['auc_macro'] = 0.0
        metrics['auc_weighted'] = 0.0
    
    # Precision, recall, F1 (threshold dependent)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='macro', zero_division=1
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='weighted', zero_division=1
    )
    
    metrics.update({
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    })
    
    # Hamming loss
    metrics['hamming_loss'] = hamming_loss(labels, binary_predictions)
    
    # Exact match
    metrics['exact_match'] = accuracy_score(labels, binary_predictions)
    
    return metrics

def calculate_class_metrics(labels, predictions, binary_predictions, threshold):
    """Calculate metrics for a single class"""
    # Handle case where there are no positive samples
    if labels.sum() == 0:
        return {
            'auc': 0.0,
            'threshold': threshold,
            'precision': 1.0 if binary_predictions.sum() == 0 else 0.0,
            'recall': 1.0,  # All true negatives were correctly identified
            'f1': 1.0 if binary_predictions.sum() == 0 else 0.0,
            'support': 0,
            'brier': brier_score_loss(labels, predictions),
            'true_positives': 0,
            'false_positives': int(binary_predictions.sum()),
            'true_negatives': int((1 - binary_predictions).sum()),
            'false_negatives': 0
        }
    
    try:
        auc = roc_auc_score(labels, predictions)
    except ValueError:
        auc = 0.0
    
    # Calculate metrics with zero_division=1
    precision = precision_score(labels, binary_predictions, zero_division=1)
    recall = recall_score(labels, binary_predictions, zero_division=1)
    f1 = f1_score(labels, binary_predictions, zero_division=1)
    
    metrics = {
        'auc': auc,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': int(labels.sum()),
        'brier': brier_score_loss(labels, predictions)
    }
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(labels, binary_predictions).ravel()
    metrics.update({
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    })
    
    return metrics

def save_results(results, predictions, labels, langs, output_dir):
    """Save evaluation results and plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed metrics
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions for further analysis
    np.savez_compressed(
        os.path.join(output_dir, 'predictions.npz'),
        predictions=predictions,
        labels=labels,
        langs=langs
    )
    
    # Log summary of results
    logger.info("\nResults Summary:")
    logger.info("\nDefault Threshold (0.5):")
    logger.info(f"Macro F1: {results['default_thresholds']['overall']['f1_macro']:.3f}")
    logger.info(f"Weighted F1: {results['default_thresholds']['overall']['f1_weighted']:.3f}")
    
    logger.info("\nOptimized Thresholds:")
    logger.info(f"Macro F1: {results['optimized_thresholds']['overall']['f1_macro']:.3f}")
    logger.info(f"Weighted F1: {results['optimized_thresholds']['overall']['f1_weighted']:.3f}")
    
    # Log threshold comparison
    if 'thresholds' in results:
        logger.info("\nOptimal Thresholds:")
        for class_name, data in results['thresholds']['global'].items():
            logger.info(f"{class_name:>12}: {data['threshold']:.3f} (F1: {data['f1_score']:.3f})")

def plot_metrics(results, output_dir, predictions=None, labels=None):
    """Generate visualization plots comparing default vs optimized thresholds"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot comparison of metrics between default and optimized thresholds
    if results.get('default_thresholds') and results.get('optimized_thresholds'):
        plt.figure(figsize=(15, 8))
        
        # Get metrics to compare
        metrics = ['precision_macro', 'recall_macro', 'f1_macro']
        default_values = [results['default_thresholds']['overall'][m] for m in metrics]
        optimized_values = [results['optimized_thresholds']['overall'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, default_values, width, label='Default Threshold (0.5)')
        plt.bar(x + width/2, optimized_values, width, label='Optimized Thresholds')
        
        plt.ylabel('Score')
        plt.title('Comparison of Default vs Optimized Thresholds')
        plt.xticks(x, [m.replace('_', ' ').title() for m in metrics])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'threshold_comparison.png'))
        plt.close()
        
        # Plot per-class comparison
        plt.figure(figsize=(15, 8))
        toxicity_types = list(results['default_thresholds']['per_class'].keys())
        
        default_f1 = [results['default_thresholds']['per_class'][c]['f1'] for c in toxicity_types]
        optimized_f1 = [results['optimized_thresholds']['per_class'][c]['f1'] for c in toxicity_types]
        
        x = np.arange(len(toxicity_types))
        width = 0.35
        
        plt.bar(x - width/2, default_f1, width, label='Default Threshold (0.5)')
        plt.bar(x + width/2, optimized_f1, width, label='Optimized Thresholds')
        
        plt.ylabel('F1 Score')
        plt.title('Per-Class F1 Score Comparison')
        plt.xticks(x, toxicity_types, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'per_class_comparison.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate toxic comment classifier')
    parser.add_argument('--model_path', type=str, 
                      default='weights/toxic_classifier_xlm-roberta-large',
                      help='Path to model directory containing checkpoints')
    parser.add_argument('--checkpoint', type=str,
                      help='Specific checkpoint to evaluate (e.g., checkpoint_epoch05_20240213). If not specified, uses latest.')
    parser.add_argument('--test_file', type=str, default='dataset/split/val.csv',
                      help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Base directory to save results')
    parser.add_argument('--num_workers', type=int, default=16,
                      help='Number of workers for data loading')
    parser.add_argument('--cache_dir', type=str, default='cached_data',
                      help='Directory to store cached tokenized data')
    parser.add_argument('--force_retokenize', action='store_true',
                      help='Force retokenization even if cache exists')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                      help='Number of batches to prefetch per worker')
    parser.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length for tokenization')
    parser.add_argument('--gc_frequency', type=int, default=500,
                      help='Frequency of garbage collection')
    
    args = parser.parse_args()
    
    # Create timestamped directory for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save evaluation parameters
    eval_params = {
        'timestamp': timestamp,
        'model_path': args.model_path,
        'checkpoint': args.checkpoint,
        'test_file': args.test_file,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'cache_dir': args.cache_dir,
        'force_retokenize': args.force_retokenize,
        'prefetch_factor': args.prefetch_factor,
        'max_length': args.max_length,
        'gc_frequency': args.gc_frequency
    }
    with open(os.path.join(eval_dir, 'eval_params.json'), 'w') as f:
        json.dump(eval_params, f, indent=2)
    
    try:
        # Load model
        print("Loading multi-language toxic comment classifier model...")
        model, tokenizer, device = load_model(args.model_path)
        
        if model is None:
            return
            
        # Load test data
        print("\nLoading test dataset...")
        test_df = pd.read_csv(args.test_file)
        print(f"Loaded {len(test_df):,} test samples")
        
        # Create test dataset
        test_dataset = ToxicDataset(
            test_df, 
            tokenizer, 
            config=args
        )
        
        # Configure DataLoader with optimized settings
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True if args.num_workers > 0 else False,
            drop_last=False
        )
        
        # Evaluate model
        results = evaluate_model(model, test_loader, device, eval_dir)
        
        print(f"\nEvaluation complete! Results saved to {eval_dir}")
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise
    
    finally:
        # Cleanup
        plt.close('all')
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 