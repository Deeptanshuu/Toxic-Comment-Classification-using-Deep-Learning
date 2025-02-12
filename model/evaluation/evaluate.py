import torch
from model.language_aware_transformer import LanguageAwareTransformer
from transformers import XLMRobertaTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, 
    confusion_matrix, roc_curve, hamming_loss, 
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

# Set matplotlib to non-interactive backend
plt.switch_backend('agg')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128, cache_dir='cached_dataset'):
        self.texts = df['comment_text'].values
        self.labels = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
        
        # Define language mapping
        self.lang_to_id = {
            'en': 0, 'ru': 1, 'tr': 2, 'es': 3,
            'fr': 4, 'it': 5, 'pt': 6
        }
        
        # Add reverse mapping
        self.id_to_lang = {v: k for k, v in self.lang_to_id.items()}
        
        # Validate language column
        if 'lang' not in df.columns:
            raise ValueError("Language column 'lang' is required for evaluation")
        
        # Print language distribution before conversion
        print("\nLanguage distribution in dataset:")
        lang_counts = df['lang'].value_counts()
        total_samples = len(df)
        for lang, count in lang_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {lang}: {count:,} samples ({percentage:.2f}%)")
        
        # Convert language strings to IDs with validation
        self.langs = np.zeros(len(df), dtype=int)
        unknown_langs = set()
        valid_langs = set()
        
        for idx, lang in enumerate(df['lang'].values):
            lang_str = str(lang).lower().strip()
            if lang_str in self.lang_to_id:
                self.langs[idx] = self.lang_to_id[lang_str]
                valid_langs.add(lang_str)
            else:
                unknown_langs.add(lang_str)
                self.langs[idx] = 0  # Default to English
        
        # Print warnings for unknown languages
        if unknown_langs:
            print("\nWarning: Found unknown languages:")
            for lang in sorted(unknown_langs):
                mask = df['lang'].str.lower().str.strip() == lang
                count = mask.sum()
                percentage = (count / total_samples) * 100
                print(f"  {lang}: {count:,} samples ({percentage:.2f}%) - Will be treated as English (en)")
        
        # Print language ID distribution after conversion
        print("\nLanguage ID distribution after conversion:")
        unique_ids, counts = np.unique(self.langs, return_counts=True)
        for lang_id, count in zip(unique_ids, counts):
            lang_code = self.id_to_lang.get(lang_id, 'unknown')
            percentage = (count / total_samples) * 100
            print(f"  {lang_code} (ID: {lang_id}): {count:,} samples ({percentage:.2f}%)")
        
        # Validate that we have at least one valid language
        if not valid_langs:
            raise ValueError("No valid language codes found in dataset. "
                           f"Expected one of: {', '.join(sorted(self.lang_to_id.keys()))}")
        
        # Print label distribution by language
        print("\nLabel distribution by language:")
        for lang_id in unique_ids:
            lang_code = self.id_to_lang.get(lang_id, 'unknown')
            lang_mask = self.langs == lang_id
            lang_labels = self.labels[lang_mask]
            
            print(f"\n{lang_code.upper()} (ID: {lang_id}, {counts[unique_ids == lang_id][0]:,} samples):")
            label_sums = np.sum(lang_labels, axis=0)
            label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            for name, count in zip(label_names, label_sums):
                percentage = (count / len(lang_labels)) * 100 if len(lang_labels) > 0 else 0
                print(f"  {name}: {int(count):,} samples ({percentage:.2f}%)")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(
            cache_dir, 
            f'tokenized_data_{max_length}_{tokenizer.__class__.__name__}.pt'
        )
        
        # Try to load cached tokenized data
        self.cached_encodings = self._load_or_create_cache()

    def _load_or_create_cache(self):
        """Load cached tokenized data or create if not exists"""
        if os.path.exists(self.cache_file):
            return torch.load(self.cache_file)
        
        encodings = []
        for text in tqdm(self.texts, desc="Tokenizing", leave=False):
            encoding = self.tokenizer(
                str(text),
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            encodings.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            })
        
        torch.save(encodings, self.cache_file)
        return encodings

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Use cached encodings
        encoding = self.cached_encodings[idx]
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.FloatTensor(self.labels[idx]),
            'lang': torch.tensor(self.langs[idx], dtype=torch.long)
        }

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
    parser.add_argument('--test_file', type=str, default='dataset/split/test.csv',
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
        'prefetch_factor': args.prefetch_factor
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
            cache_dir=args.cache_dir
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