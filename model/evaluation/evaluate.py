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

# Set matplotlib to non-interactive backend
plt.switch_backend('agg')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logger = logging.getLogger(__name__)

class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, config):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache key based on data and config
        cache_key = self._create_cache_key(df, tokenizer, config)
        self.cache_file = self.cache_dir / f"cached_encodings_{cache_key}.pt"
        self.meta_file = self.cache_dir / f"meta_{cache_key}.json"
        
        # Initialize storage
        self.cached_encodings = []
        self.labels = None
        self.langs = None
        
        # Load or create cache
        if self._is_cache_valid():
            logger.info(f"Loading cached encodings from {self.cache_file}")
            self._load_cache()
        else:
            logger.info("Cache not found or invalid. Creating new cache...")
            self._create_cache()
            
        # Validate final state
        self._validate_dataset()
        
    def _create_cache_key(self, df, tokenizer, config):
        """Create unique cache key based on data and configuration"""
        key_components = [
            df.shape[0],
            df.columns.tolist(),
            tokenizer.__class__.__name__,
            config.max_length,
            config.model_name
        ]
        key_string = json.dumps(key_components, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()[:10]
    
    def _is_cache_valid(self):
        """Check if cache exists and is valid"""
        if not self.cache_file.exists() or not self.meta_file.exists():
            return False
            
        try:
            with open(self.meta_file, 'r') as f:
                meta = json.load(f)
            return (
                meta['num_samples'] == len(self.df) and
                meta['max_length'] == self.config.max_length and
                meta['model_name'] == self.config.model_name
            )
        except Exception as e:
            logger.warning(f"Cache validation failed: {str(e)}")
            return False
    
    def _create_cache(self):
        """Create and save cached encodings"""
        logger.info("Processing samples and creating cache...")
        failed_indices = []
        
        # Process all samples
        for idx in tqdm(range(len(self.df)), desc="Caching encodings"):
            try:
                encoding = self._process_sample(self.df.iloc[idx])
                self.cached_encodings.append(encoding)
            except Exception as e:
                logger.error(f"Failed to process sample {idx}: {str(e)}")
                failed_indices.append(idx)
        
        # Handle failed samples
        if failed_indices:
            logger.warning(f"Failed to process {len(failed_indices)} samples")
            # Fill failed samples with empty encodings
            for idx in failed_indices:
                self.cached_encodings.append(self._create_empty_encoding())
        
        # Save cache
        try:
            torch.save(self.cached_encodings, self.cache_file)
            meta = {
                'num_samples': len(self.df),
                'max_length': self.config.max_length,
                'model_name': self.config.model_name,
                'failed_indices': failed_indices
            }
            with open(self.meta_file, 'w') as f:
                json.dump(meta, f)
            logger.info(f"Cache saved to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {str(e)}")
    
    def _load_cache(self):
        """Load cached encodings"""
        try:
            self.cached_encodings = torch.load(self.cache_file)
            with open(self.meta_file, 'r') as f:
                meta = json.load(f)
            if meta.get('failed_indices'):
                logger.warning(f"Cache contains {len(meta['failed_indices'])} failed samples")
        except Exception as e:
            logger.error(f"Failed to load cache: {str(e)}")
            raise
    
    def _process_sample(self, row):
        """Process a single sample"""
        # Tokenize text
        encoding = self.tokenizer(
            row['comment_text'],
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert to expected format
        sample = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor([
                row['toxic'], row['severe_toxic'], row['obscene'],
                row['threat'], row['insult'], row['identity_hate']
            ], dtype=torch.float32),
            'lang': torch.tensor(row['lang_id'], dtype=torch.long)
        }
        return sample
    
    def _create_empty_encoding(self):
        """Create an empty encoding for failed samples"""
        return {
            'input_ids': torch.zeros(self.config.max_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.config.max_length, dtype=torch.long),
            'labels': torch.zeros(6, dtype=torch.float32),
            'lang': torch.tensor(0, dtype=torch.long)
        }
    
    def _validate_dataset(self):
        """Validate the final dataset state"""
        if len(self.cached_encodings) != len(self.df):
            raise ValueError(
                f"Cache size mismatch: {len(self.cached_encodings)} cached samples "
                f"vs {len(self.df)} input samples"
            )
        
        # Extract labels and langs for sampler
        self.labels = torch.stack([enc['labels'] for enc in self.cached_encodings])
        self.langs = torch.tensor([enc['lang'].item() for enc in self.cached_encodings])
        
        # Validate shapes
        expected_shapes = {
            'input_ids': (self.config.max_length,),
            'attention_mask': (self.config.max_length,),
            'labels': (6,),
            'lang': ()
        }
        
        for idx, encoding in enumerate(self.cached_encodings):
            for key, expected_shape in expected_shapes.items():
                if encoding[key].shape != expected_shape:
                    raise ValueError(
                        f"Shape mismatch in sample {idx}, key {key}: "
                        f"got {encoding[key].shape}, expected {expected_shape}"
                    )
    
    def __len__(self):
        return len(self.cached_encodings)
    
    def __getitem__(self, idx):
        if idx >= len(self.cached_encodings):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.cached_encodings)}")
        return self.cached_encodings[idx]

def load_model(model_path):
    """Load model and tokenizer"""
    try:
        # Initialize the custom model architecture
        model = LanguageAwareTransformer(
            num_labels=6,
            hidden_size=1024,
            num_attention_heads=16,
            model_name='xlm-roberta-large',
            dropout=0.1
        )
        
        # Load the trained weights
        state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'))
        model.load_state_dict(state_dict)
        
        try:
            tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        except:
            print("Loading base XLM-RoBERTa tokenizer...")
            tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None

def evaluate_model(model, val_loader, device, output_dir):
    """Evaluate model performance on validation set"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_langs = []
    
    print("\nGathering predictions...")
    with torch.inference_mode():
        for batch in val_loader:
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
    
    # Concatenate all batches
    predictions = np.vstack(all_predictions)
    labels = np.vstack(all_labels)
    langs = np.concatenate(all_langs)
    
    print(f"\nTotal samples: {len(predictions):,}")
    
    # Calculate metrics
    results = calculate_metrics(predictions, labels, langs)
    
    # Save evaluation results
    save_results(
        results=results,
        predictions=predictions,
        labels=labels,
        langs=langs,
        output_dir=output_dir
    )
    
    # Plot metrics
    plot_metrics(results, output_dir)
    
    return results, predictions

def calculate_metrics(predictions, labels, langs):
    """Calculate detailed metrics"""
    results = {
        'overall': {},
        'per_language': {},
        'per_class': {}
    }
    
    # Calculate overall metrics
    results['overall'] = calculate_overall_metrics(predictions, labels)
    
    # Calculate per-language metrics
    unique_langs = np.unique(langs)
    for lang in unique_langs:
        lang_mask = langs == lang
        if not lang_mask.any():
            continue
            
        lang_preds = predictions[lang_mask]
        lang_labels = labels[lang_mask]
        
        results['per_language'][str(lang)] = calculate_overall_metrics(
            lang_preds, lang_labels
        )
        results['per_language'][str(lang)]['sample_count'] = int(lang_mask.sum())
    
    # Calculate per-class metrics
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for i, class_name in enumerate(toxicity_types):
        results['per_class'][class_name] = calculate_class_metrics(
            labels[:, i],
            predictions[:, i],
            (predictions[:, i] > 0.5).astype(int),
            0.5
        )
    
    return results

def calculate_overall_metrics(predictions, labels):
    """Calculate overall metrics for multi-label classification"""
    binary_predictions = (predictions > 0.5).astype(int)
    
    metrics = {}
    
    # AUC scores
    metrics['auc_macro'] = roc_auc_score(labels, predictions, average='macro')
    metrics['auc_weighted'] = roc_auc_score(labels, predictions, average='weighted')
    
    # Precision, recall, F1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='weighted', zero_division=0
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
    metrics = {
        'auc': roc_auc_score(labels, predictions),
        'threshold': threshold,
        'precision': precision_score(labels, binary_predictions),
        'recall': recall_score(labels, binary_predictions),
        'f1': f1_score(labels, binary_predictions),
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

def plot_metrics(results, output_dir):
    """Generate visualization plots"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot per-class metrics
    if results.get('per_class'):
        plt.figure(figsize=(12, 6))
        toxicity_types = list(results['per_class'].keys())
        metrics = ['auc', 'precision', 'recall', 'f1']
        
        for metric in metrics:
            values = [results['per_class'][c][metric] for c in toxicity_types]
            plt.plot(toxicity_types, values, marker='o', label=metric.upper())
        
        plt.title('Per-Class Performance Metrics')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'per_class_metrics.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate toxic comment classifier')
    parser.add_argument('--model_path', type=str, default='weights/toxic_classifier_xlm-roberta-large',
                      help='Path to the trained model')
    parser.add_argument('--test_file', type=str, default='dataset/split/train.csv',
                      help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Base directory to save results')
    parser.add_argument('--num_workers', type=int, default=None,
                      help='Number of workers for data loading (default: CPU count - 1)')
    parser.add_argument('--cache_dir', type=str, default='cached_data',
                      help='Directory to store cached tokenized data')
    parser.add_argument('--force_retokenize', action='store_true',
                      help='Force retokenization even if cache exists')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                      help='Number of batches to prefetch per worker')
    parser.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length for tokenization')
    
    args = parser.parse_args()
    
    # Create timestamped directory for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save evaluation parameters
    eval_params = {
        'timestamp': timestamp,
        'model_path': args.model_path,
        'test_file': args.test_file,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'cache_dir': args.cache_dir,
        'force_retokenize': args.force_retokenize,
        'prefetch_factor': args.prefetch_factor,
        'max_length': args.max_length
    }
    with open(os.path.join(eval_dir, 'eval_params.json'), 'w') as f:
        json.dump(eval_params, f, indent=2)
    
    # Set number of workers
    if args.num_workers is None:
        args.num_workers = min(16, max(1, multiprocessing.cpu_count() - 1))
    
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
        
        # Configure DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
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