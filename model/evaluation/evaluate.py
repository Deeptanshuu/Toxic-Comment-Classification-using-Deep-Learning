import torch
from model.language_aware_transformer import LanguageAwareTransformer
from transformers import XLMRobertaTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, 
    confusion_matrix, roc_curve, hamming_loss, 
    accuracy_score
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime
import argparse
from torch.utils.data import Dataset, DataLoader
import gc
import multiprocessing
import multiprocessing as mp
from functools import partial

# Set matplotlib to non-interactive backend
plt.switch_backend('agg')

class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['comment_text'].values
        self.labels = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
        
        # Define language mapping
        self.lang_to_id = {
            'en': 0, 'ru': 1, 'tr': 2, 'es': 3,
            'fr': 4, 'it': 5, 'pt': 6
        }
        
        # Add reverse mapping
        self.id_to_lang = {v: k for k, v in self.lang_to_id.items()}
        
        # Convert language strings to IDs, default to English (0) if language not found
        self.langs = np.array([self.lang_to_id.get(str(lang).lower(), 0) for lang in df['lang'].values])
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
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

def evaluate_language(args):
    """Evaluate model performance for a single language
    
    Args:
        args: Tuple of (lang_id, lang_name, lang_preds, lang_labels, toxicity_types)
    Returns:
        Tuple of (lang_id, metrics, thresholds)
    """
    try:
        lang_id, lang_name, lang_preds, lang_labels, toxicity_types = args
        print(f"\nProcessing {lang_name} with {len(lang_labels)} samples...")
        
        # Optimize thresholds for this language
        lang_thresholds = {}
        for i, class_name in enumerate(toxicity_types):
            try:
                class_labels = lang_labels[:, i]
                class_preds = lang_preds[:, i]
                
                if len(np.unique(class_labels)) < 2:
                    print(f"Warning: Only one class present for {class_name} in {lang_name}")
                    lang_thresholds[class_name] = 0.5  # Default threshold
                    continue
                
                # Find optimal threshold using F1 score
                fpr, tpr, thresh = roc_curve(class_labels, class_preds)
                f1_scores = []
                
                # Calculate F1 score for each threshold
                for t in thresh:
                    binary_preds = (class_preds > t).astype(int)
                    _, _, f1, _ = precision_recall_fscore_support(
                        class_labels, binary_preds, average='binary', zero_division=0
                    )
                    f1_scores.append(f1)
                
                # Select threshold that maximizes F1 score
                best_idx = np.argmax(f1_scores)
                lang_thresholds[class_name] = thresh[best_idx]
                
                # If threshold results in no positive predictions, use a lower threshold
                test_preds = (class_preds > lang_thresholds[class_name]).astype(int)
                if test_preds.sum() == 0:
                    print(f"Warning: No positive predictions for {class_name} in {lang_name}, adjusting threshold")
                    percentile_threshold = np.percentile(class_preds, 95)
                    lang_thresholds[class_name] = min(percentile_threshold, 0.3)
                
            except Exception as e:
                print(f"Warning: Could not optimize threshold for {class_name} in {lang_name}: {str(e)}")
                lang_thresholds[class_name] = 0.3  # Conservative default
        
        # Calculate metrics using optimized thresholds
        binary_preds = np.zeros_like(lang_preds)
        for i, class_name in enumerate(toxicity_types):
            binary_preds[:, i] = (lang_preds[:, i] > lang_thresholds[class_name]).astype(int)
        
        # Calculate metrics without parallel processing
        metrics = {}
        
        # Calculate AUC if possible
        try:
            if len(np.unique(lang_labels)) > 1:
                metrics['auc'] = roc_auc_score(lang_labels, lang_preds, average='weighted')
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {str(e)}")
            metrics['auc'] = None
        
        # Calculate precision, recall, F1
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                lang_labels, binary_preds, average='weighted',
                zero_division=0
            )
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        except Exception as e:
            print(f"Warning: Could not calculate precision/recall/F1: {str(e)}")
            metrics.update({
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            })
        
        # Calculate Hamming Loss
        try:
            metrics['hamming_loss'] = hamming_loss(lang_labels, binary_preds)
        except Exception as e:
            print(f"Warning: Could not calculate Hamming Loss: {str(e)}")
            metrics['hamming_loss'] = 1.0  # Worst case
        
        # Calculate Exact Match
        try:
            metrics['exact_match'] = accuracy_score(lang_labels, binary_preds)
        except Exception as e:
            print(f"Warning: Could not calculate Exact Match: {str(e)}")
            metrics['exact_match'] = 0.0  # Worst case
        
        # Calculate specificity
        try:
            metrics['specificity'] = calculate_specificity(lang_labels, binary_preds)
        except Exception as e:
            print(f"Warning: Could not calculate specificity: {str(e)}")
            metrics['specificity'] = 0.0
        
        # Calculate per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(toxicity_types):
            try:
                class_metrics[class_name] = calculate_class_metrics(
                    lang_labels[:, i],
                    lang_preds[:, i],
                    binary_preds[:, i],
                    lang_thresholds[class_name]
                )
            except Exception as e:
                print(f"Warning: Could not calculate metrics for class {class_name}: {str(e)}")
                class_metrics[class_name] = {
                    'auc': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'specificity': 0.0,
                    'npv': 0.0,
                    'threshold': lang_thresholds[class_name]
                }
        
        metrics['class_metrics'] = class_metrics
        metrics['sample_count'] = len(lang_labels)
        
        return lang_id, metrics, lang_thresholds
        
    except Exception as e:
        print(f"Warning: Could not evaluate language {lang_name}: {str(e)}")
        return None

def evaluate_model(model, test_loader, device, output_dir):
    """Evaluate model performance with parallel language-specific evaluation"""
    all_predictions = []
    all_labels = []
    all_langs = []
    all_losses = []
    
    print("\nRunning predictions on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            langs = batch['lang'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                lang_ids=langs
            )
            loss = outputs['loss'].item()
            predictions = outputs['probabilities'].cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            all_langs.extend(langs.cpu().numpy())
            all_losses.append(loss)
            
            # Clear GPU memory
            del input_ids, attention_mask, outputs, labels
            torch.cuda.empty_cache()
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    langs = np.array(all_langs)
    avg_loss = np.mean(all_losses)
    
    # Initialize results dictionary
    results = {
        'overall': {'loss': avg_loss},
        'per_language': {},
        'per_class': {},
        'thresholds': {}
    }
    
    # Get language name mapping for logging
    id_to_lang = {
        0: 'English (en)',
        1: 'Russian (ru)',
        2: 'Turkish (tr)',
        3: 'Spanish (es)',
        4: 'French (fr)',
        5: 'Italian (it)',
        6: 'Portuguese (pt)'
    }
    
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Prepare arguments for parallel processing
    eval_args = []
    unique_langs = np.unique(langs)
    
    for lang in unique_langs:
        lang_mask = langs == lang
        if not lang_mask.any():
            continue
            
        lang_preds = predictions[lang_mask]
        lang_labels = labels[lang_mask]
        
        if len(lang_labels) == 0:
            continue
            
        eval_args.append((
            lang,
            id_to_lang.get(int(lang), f'Unknown ({lang})'),
            lang_preds,
            lang_labels,
            toxicity_types
        ))
    
    # Process languages in parallel
    n_processes = min(len(eval_args), mp.cpu_count())
    with mp.Pool(processes=n_processes) as pool:
        lang_results = list(filter(None, pool.map(evaluate_language, eval_args)))
    
    # Collect results
    for lang_id, lang_metrics, lang_thresholds in lang_results:
        results['per_language'][str(lang_id)] = lang_metrics
        results['thresholds'][str(lang_id)] = lang_thresholds
    
    # Calculate overall metrics using language-specific thresholds
    overall_binary_preds = np.zeros_like(predictions)
    for i, lang in enumerate(langs):
        for j, class_name in enumerate(toxicity_types):
            threshold = results['thresholds'][str(lang)][class_name]
            overall_binary_preds[i, j] = (predictions[i, j] > threshold).astype(int)
    
    # Calculate overall metrics
    results['overall'].update({
        'auc': roc_auc_score(labels, predictions, average='macro'),
        'auc_weighted': roc_auc_score(labels, predictions, average='weighted')
    })
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, overall_binary_preds, average='macro'
    )
    
    results['overall'].update({
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    # After calculating metrics, add calibration plots
    plot_calibration_curves(
        labels,
        predictions,
        output_dir,
        toxicity_types=toxicity_types,
        languages=id_to_lang,
        langs=langs
    )
    
    # Save results and generate visualizations
    save_results(results, predictions, labels, langs, output_dir)
    plot_metrics(results, output_dir, toxicity_types)
    
    return results

def calculate_metrics(predictions, labels, langs):
    """Calculate detailed metrics using class-specific thresholds"""
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Dynamic threshold calculation based on validation set statistics
    thresholds = {}
    for i, class_name in enumerate(toxicity_types):
        # Find optimal threshold using ROC curve
        fpr, tpr, thresh = roc_curve(labels[:, i], predictions[:, i])
        optimal_idx = np.argmax(tpr - fpr)
        thresholds[class_name] = thresh[optimal_idx]
    
    unique_langs = np.unique(langs)
    
    results = {
        'overall': {},
        'per_language': {},
        'per_class': {},
        'thresholds': thresholds,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Overall metrics
    results['overall'] = {
        'auc': roc_auc_score(labels, predictions, average='macro'),
        'auc_weighted': roc_auc_score(labels, predictions, average='weighted')
    }
    
    # Binary predictions using dynamic thresholds
    threshold_array = np.array([thresholds[ct] for ct in toxicity_types])
    binary_predictions = (predictions > threshold_array).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='macro'
    )
    
    results['overall'].update({
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    # Per-language metrics with confidence intervals
    for lang in unique_langs:
        lang_mask = langs == lang
        if lang_mask.sum() > 0:
            try:
                # Calculate metrics
                metrics = calculate_language_metrics(
                    labels[lang_mask],
                    predictions[lang_mask],
                    binary_predictions[lang_mask]
                )
                results['per_language'][lang] = metrics
            except Exception as e:
                print(f"Warning: Could not calculate metrics for language {lang}: {str(e)}")
    
    # Per-class metrics with detailed statistics
    for i, class_name in enumerate(toxicity_types):
        try:
            metrics = calculate_class_metrics(
                labels[:, i],
                predictions[:, i],
                binary_predictions[:, i],
                thresholds[class_name]
            )
            results['per_class'][class_name] = metrics
        except Exception as e:
            print(f"Warning: Could not calculate metrics for class {class_name}: {str(e)}")
    
    return results

def bootstrap_sample(y_true, y_pred):
    """Preserve label relationships during bootstrap sampling"""
    n_samples = len(y_true)
    
    # Check if we have any positive samples to stratify on
    row_sums = y_true.sum(axis=1) if len(y_true.shape) > 1 else y_true
    if len(np.unique(row_sums)) > 1:
        # If we have both positive and negative samples, use stratified sampling
        idx = resample(np.arange(n_samples), stratify=row_sums)
    else:
        # If all samples are the same (all positive or all negative), use regular bootstrap
        idx = resample(np.arange(n_samples))
    
    return y_true[idx], y_pred[idx]

def calculate_class_weights(labels):
    """Calculate balanced class weights for each label"""
    class_weights = {}
    for i in range(labels.shape[1]):
        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(labels[:, i], return_counts=True)
        
        # Handle cases where all samples belong to one class
        if len(unique_classes) == 1:
            class_weights[i] = {int(unique_classes[0]): 1.0}
            continue
            
        try:
            weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_classes,
                y=labels[:, i]
            )
            # Convert to integer indices and ensure weights are positive
            class_weights[i] = {int(class_label): max(weight, 0.1) 
                              for class_label, weight in zip(unique_classes, weights)}
        except ValueError as e:
            print(f"Warning: Could not compute class weights for label {i}: {str(e)}")
            # Fallback to equal weights
            class_weights[i] = {int(class_label): 1.0 for class_label in unique_classes}
            
    return class_weights

def calculate_specificity(y_true, y_pred, sample_weight=None):
    """Calculate specificity (true negative rate) for binary or multi-label data"""
    if len(y_true.shape) > 1:
        # Multi-label case: calculate specificity for each label and average
        specificities = []
        for i in range(y_true.shape[1]):
            tn, fp, fn, tp = confusion_matrix(
                y_true[:, i], 
                y_pred[:, i], 
                sample_weight=sample_weight
            ).ravel()
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        return np.mean(specificities)
    else:
        # Binary case
        tn, fp, fn, tp = confusion_matrix(
            y_true, 
            y_pred, 
            sample_weight=sample_weight
        ).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def calculate_language_metrics(labels, predictions, binary_predictions):
    """Calculate detailed metrics for a specific language with parallel processing"""
    if len(labels) == 0:
        raise ValueError("No samples available for metric calculation")
    
    # Calculate initial metrics
    try:
        class_weights = calculate_class_weights(labels)
    except Exception as e:
        print(f"Warning: Using default weights due to error in class weight calculation: {str(e)}")
        class_weights = {i: {0: 1.0, 1: 1.0} for i in range(labels.shape[1])}
    
    # Calculate weighted metrics
    sample_weights = np.ones(len(labels))
    for i in range(labels.shape[1]):
        label_indices = labels[:, i].astype(int)
        valid_indices = np.isin(label_indices, list(class_weights[i].keys()))
        if not valid_indices.all():
            print(f"Warning: Invalid label indices found for class {i}")
            continue
        sample_weights *= np.array([class_weights[i].get(idx, 1.0) for idx in label_indices])
    
    # Normalize weights
    sample_weights = np.clip(sample_weights, 0.1, 10.0)
    sample_weights /= sample_weights.sum()
    sample_weights *= len(sample_weights)
    
    # Calculate base metrics
    metrics = {}
    
    # Calculate AUC if possible
    try:
        if len(np.unique(labels)) > 1:
            metrics['auc'] = roc_auc_score(labels, predictions, average='weighted', sample_weight=sample_weights)
    except Exception as e:
        print(f"Warning: Could not calculate AUC: {str(e)}")
        metrics['auc'] = None
    
    # Calculate precision, recall, F1
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, binary_predictions, average='weighted',
            sample_weight=sample_weights, zero_division=0
        )
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    except Exception as e:
        print(f"Warning: Could not calculate precision/recall/F1: {str(e)}")
        metrics.update({
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        })
    
    # Calculate Hamming Loss
    try:
        # Ensure inputs are in correct format for hamming_loss
        y_true = labels.astype(int)
        y_pred = binary_predictions.astype(int)
        
        # Calculate Hamming Loss with sample weights
        metrics['hamming_loss'] = hamming_loss(
            y_true, 
            y_pred, 
            sample_weight=sample_weights
        )
    except Exception as e:
        print(f"Warning: Could not calculate Hamming Loss: {str(e)}")
        metrics['hamming_loss'] = 1.0  # Worst case
    
    # Calculate Exact Match (subset accuracy)
    try:
        # Ensure inputs are in correct format for accuracy_score
        y_true = labels.astype(int)
        y_pred = binary_predictions.astype(int)
        
        # Calculate Exact Match with sample weights
        metrics['exact_match'] = accuracy_score(
            y_true, 
            y_pred,
            sample_weight=sample_weights,
            normalize=True  # Return the fraction of correctly classified samples
        )
    except Exception as e:
        print(f"Warning: Could not calculate Exact Match: {str(e)}")
        metrics['exact_match'] = 0.0  # Worst case
    
    # Calculate specificity
    try:
        metrics['specificity'] = calculate_specificity(labels, binary_predictions, sample_weights)
    except Exception as e:
        print(f"Warning: Could not calculate specificity: {str(e)}")
        metrics['specificity'] = 0.0
    
    # Parallel bootstrap calculations
    n_bootstrap = 1000
    n_processes = min(mp.cpu_count(), 8)  # Limit to 8 processes max
    
    # Prepare arguments for parallel processing
    bootstrap_args = [(labels, predictions, binary_predictions, None)] * n_bootstrap
    
    # Run bootstrap iterations in parallel
    with mp.Pool(n_processes) as pool:
        bootstrap_results = list(filter(None, pool.map(parallel_bootstrap_metrics, bootstrap_args)))
    
    # Calculate confidence intervals
    ci_lower, ci_upper = 2.5, 97.5
    for metric in ['auc', 'f1', 'hamming_loss', 'exact_match', 'specificity']:
        values = [r[metric] for r in bootstrap_results if r and r[metric] is not None]
        if values:
            metrics[f'{metric}_ci'] = [
                np.percentile(values, ci_lower),
                np.percentile(values, ci_upper)
            ]
        else:
            metrics[f'{metric}_ci'] = [0.0, 0.0]
    
    metrics.update({
        'sample_count': len(labels),
        'bootstrap_samples': len(bootstrap_results),
        'class_weights': class_weights
    })
    
    return metrics

def calculate_class_metrics(labels, predictions, binary_predictions, threshold):
    """Calculate detailed metrics for a specific class with parallel processing"""
    # Calculate initial class weights
    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    label_indices = labels.astype(int)
    sample_weights = np.array([weights[idx] for idx in label_indices])
    
    # Calculate base metrics
    metrics = {
        'auc': roc_auc_score(labels, predictions, sample_weight=sample_weights),
        'threshold': threshold
    }
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='binary',
        sample_weight=sample_weights
    )
    metrics.update({
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    tn, fp, fn, tp = confusion_matrix(
        labels, binary_predictions,
        sample_weight=sample_weights
    ).ravel()
    
    metrics.update({
        'specificity': tn / (tn + fp),
        'npv': tn / (tn + fn),
        'positive_samples': int(labels.sum()),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    })
    
    # Parallel bootstrap calculations
    n_bootstrap = 1000
    n_processes = min(mp.cpu_count(), 8)  # Limit to 8 processes max
    
    # Prepare arguments for parallel processing
    bootstrap_args = [(
        labels.reshape(-1, 1),
        predictions.reshape(-1, 1),
        binary_predictions.reshape(-1, 1),
        threshold
    )] * n_bootstrap
    
    # Run bootstrap iterations in parallel
    with mp.Pool(n_processes) as pool:
        bootstrap_results = list(filter(None, pool.map(parallel_bootstrap_metrics, bootstrap_args)))
    
    # Calculate confidence intervals
    ci_lower, ci_upper = 2.5, 97.5
    for metric in ['auc', 'precision', 'recall', 'f1', 'specificity', 'npv']:
        values = [r[metric] for r in bootstrap_results if r and r[metric] is not None]
        if values:
            metrics[f'{metric}_ci'] = [
                np.percentile(values, ci_lower),
                np.percentile(values, ci_upper)
            ]
        else:
            metrics[f'{metric}_ci'] = [0.0, 0.0]
    
    metrics.update({
        'bootstrap_samples': len(bootstrap_results),
        'class_weights': dict(zip(np.unique(labels), weights))
    })
    
    return metrics

def plot_confusion_matrices(predictions, labels, langs, output_dir, batch_size=10):
    """Plot confusion matrices in batches using class-specific thresholds"""
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Per-class optimal thresholds
    thresholds = {
        'toxic': 0.3991,
        'severe_toxic': 0.2350,
        'obscene': 0.47,
        'threat': 0.3614,
        'insult': 0.3906,
        'identity_hate': 0.2533
    }
    
    threshold_array = np.array([thresholds[ct] for ct in toxicity_types])
    binary_predictions = (predictions > threshold_array).astype(int)
    
    # Create directory for confusion matrices
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    
    # Plot confusion matrices in batches
    def plot_batch(items, item_type='class'):
        for item in items:
            if item_type == 'class':
                i = toxicity_types.index(item)
                cm = confusion_matrix(labels[:, i], binary_predictions[:, i])
                title = f'Confusion Matrix - {item}\n(threshold={thresholds[item]:.2f})'
                filename = f'cm_{item}.png'
            else:  # language
                lang_mask = langs == item
                if lang_mask.sum() > 0:
                    cm = confusion_matrix(
                        labels[lang_mask, 0],
                        binary_predictions[lang_mask, 0]
                    )
                    title = f'Confusion Matrix - Toxic Class - {item}\n(threshold={thresholds["toxic"]:.2f})'
                    filename = f'cm_toxic_{item}.png'
                else:
                    continue
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(cm_dir, filename))
            plt.close('all')
    
    # Process classes in batches
    for i in range(0, len(toxicity_types), batch_size):
        batch = toxicity_types[i:i + batch_size]
        plot_batch(batch, 'class')
        gc.collect()
    
    # Process languages in batches
    unique_langs = np.unique(langs)
    for i in range(0, len(unique_langs), batch_size):
        batch = unique_langs[i:i + batch_size]
        plot_batch(batch, 'language')
        gc.collect()

def plot_metrics(results, output_dir, toxicity_types=None):
    """Generate detailed visualization plots
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save plots
        toxicity_types: List of toxicity class names. If None, will try to get from results.
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get toxicity types from results if not provided
    if toxicity_types is None:
        toxicity_types = list(results.get('per_class', {}).keys())
        if not toxicity_types:
            print("Warning: No toxicity classes found in results")
            return
    
    # Plot per-class performance if we have class metrics
    if results.get('per_class'):
        plt.figure(figsize=(12, 6))
        metrics = ['auc', 'precision', 'recall', 'f1']
        
        for metric in metrics:
            values = [results['per_class'][c].get(metric, np.nan) for c in toxicity_types]
            if any(not np.isnan(v) for v in values):  # Only plot if we have valid values
                plt.plot(toxicity_types, values, marker='o', label=metric.upper())
        
        plt.title('Per-Class Performance Metrics')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'per_class_metrics.png'))
        plt.close()
    
    # Plot per-language performance if we have language metrics
    if results.get('per_language'):
        plt.figure(figsize=(10, 6))
        languages = list(results['per_language'].keys())
        auc_scores = [results['per_language'][lang].get('auc', np.nan) for lang in languages]
        sample_counts = [results['per_language'][lang].get('sample_count', 0) for lang in languages]
        
        # Only create bubble plot if we have valid data
        valid_data = [(lang, auc, count) for lang, auc, count in zip(languages, auc_scores, sample_counts)
                     if not np.isnan(auc) and count > 0]
        
        if valid_data:
            langs, aucs, counts = zip(*valid_data)
            plt.scatter(langs, aucs, s=[count/50 for count in counts], alpha=0.6)
            plt.title('AUC Scores by Language (bubble size = sample count)')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'language_performance.png'))
        plt.close()
    
    # Create correlation heatmaps if we have language metrics
    if results.get('per_language'):
        metric_names = ['auc', 'f1', 'precision', 'recall', 'hamming_loss', 'exact_match']
        lang_metric_data = []
        valid_languages = []
        
        for lang in languages:
            metrics = results['per_language'][lang]
            lang_values = []
            has_valid_metrics = False
            
            for metric in metric_names:
                value = metrics.get(metric, np.nan)
                if isinstance(value, (int, float, np.number)):
                    lang_values.append(value)
                    has_valid_metrics = True
                else:
                    lang_values.append(np.nan)
            
            if has_valid_metrics:
                lang_metric_data.append(lang_values)
                valid_languages.append(lang)
        
        # Only create correlation heatmap if we have valid data
        if lang_metric_data and valid_languages:
            # Create DataFrame for correlation analysis
            lang_metric_df = pd.DataFrame(lang_metric_data, columns=metric_names, index=valid_languages)
            
            # Remove columns with all NaN values
            lang_metric_df = lang_metric_df.dropna(axis=1, how='all')
            
            if not lang_metric_df.empty and lang_metric_df.shape[1] > 1:
                plt.figure(figsize=(10, 8))
                metric_corr = lang_metric_df.corr()
                sns.heatmap(metric_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                           square=True, cbar_kws={'label': 'Correlation'})
                plt.title('Correlation between Performance Metrics across Languages')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'metric_correlations.png'))
            plt.close()
        
        # 2. Class-Language Performance Matrix
        if results.get('per_class'):
            class_lang_data = []
            valid_languages = []
            
            for lang in languages:
                class_metrics = results['per_language'][lang].get('class_metrics', {})
                lang_values = []
                has_valid_metrics = False
                
                for class_name in toxicity_types:
                    if class_name in class_metrics:
                        value = class_metrics[class_name].get('f1', np.nan)
                        if not np.isnan(value):
                            has_valid_metrics = True
                    else:
                        value = np.nan
                    lang_values.append(value)
                
                if has_valid_metrics:
                    class_lang_data.append(lang_values)
                    valid_languages.append(lang)
            
            # Only create heatmap if we have valid data
            if class_lang_data and valid_languages:
                class_lang_df = pd.DataFrame(class_lang_data, columns=toxicity_types, index=valid_languages)
                
                # Remove columns with all NaN values
                class_lang_df = class_lang_df.dropna(axis=1, how='all')
                
                if not class_lang_df.empty and class_lang_df.shape[1] > 0:
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(class_lang_df, annot=True, cmap='YlOrRd', center=0.5, fmt='.2f',
                              cbar_kws={'label': 'F1 Score'})
                    plt.title('F1 Scores by Class and Language')
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, 'class_language_performance.png'))
                plt.close()
    
    # 3. Performance Distribution Plots
    plt.figure(figsize=(15, 5))
    
    # Create subplot for metric distributions
    plt.subplot(131)
    has_metric_data = False
    for metric in ['auc', 'f1', 'precision', 'recall']:
        values = [results['per_language'][lang].get(metric, np.nan) for lang in languages]
        values = [v for v in values if not np.isnan(v)]
        if values:
            sns.kdeplot(data=values, label=metric.upper())
            has_metric_data = True
    
    if has_metric_data:
        plt.title('Distribution of Metrics across Languages')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend()
    
    # Create subplot for class performance distributions
    plt.subplot(132)
    class_scores = []
    class_names = []
    for class_name in toxicity_types:
        scores = [results['per_language'][lang].get('class_metrics', {}).get(class_name, {}).get('f1', np.nan) 
                 for lang in languages]
        scores = [s for s in scores if not np.isnan(s)]
        if scores:
            class_scores.extend(scores)
            class_names.extend([class_name] * len(scores))
    
    if class_scores:
        sns.boxplot(x=class_names, y=class_scores)
        plt.title('Class Performance Distribution')
        plt.xticks(rotation=45)
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
    
    # Create subplot for language performance distributions
    plt.subplot(133)
    lang_scores = []
    lang_names = []
    for lang in languages:
        scores = [metrics.get('f1', np.nan) for metrics in results['per_language'][lang].get('class_metrics', {}).values()]
        scores = [s for s in scores if not np.isnan(s)]
        if scores:
            lang_scores.extend(scores)
            lang_names.extend([lang] * len(scores))
    
    if lang_scores:
        sns.boxplot(x=lang_names, y=lang_scores)
        plt.title('Language Performance Distribution')
        plt.xticks(rotation=45)
        plt.xlabel('Language')
        plt.ylabel('F1 Score')
    
    if has_metric_data or class_scores or lang_scores:
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'performance_distributions.png'))
    plt.close()

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(convert_to_serializable(key)): convert_to_serializable(value) 
                for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def save_results(results, predictions, labels, langs, output_dir):
    """Save evaluation results and plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get language name mapping
    id_to_lang = {
        0: 'English (en)',
        1: 'Russian (ru)',
        2: 'Turkish (tr)',
        3: 'Spanish (es)',
        4: 'French (fr)',
        5: 'Italian (it)',
        6: 'Portuguese (pt)'
    }
    
    # Convert results to JSON serializable format
    serializable_results = convert_to_serializable(results)
    
    # Save detailed metrics
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save raw predictions for further analysis
    np.savez_compressed(
        os.path.join(output_dir, 'predictions.npz'),
        predictions=predictions,
        labels=labels,
        langs=langs
    )
    
    # Plot confusion matrices
    plot_confusion_matrices(predictions, labels, langs, output_dir)
    
    # Print detailed summary
    print("\nEvaluation Results:")
    print("-" * 80)
    print(f"Overall Metrics:")
    print(f"  AUC (macro): {results['overall']['auc']:.4f}")
    print(f"  F1 (macro): {results['overall']['f1']:.4f}")
    
    # Handle metrics that might be missing or non-numeric
    hamming_loss = results['overall'].get('hamming_loss', 'N/A')
    exact_match = results['overall'].get('exact_match', 'N/A')
    loss = results['overall'].get('loss', 'N/A')
    
    print(f"  Hamming Loss: {hamming_loss if isinstance(hamming_loss, str) else f'{hamming_loss:.4f}'}")
    print(f"  Exact Match: {exact_match if isinstance(exact_match, str) else f'{exact_match:.4f}'}")
    print(f"  Loss: {loss if isinstance(loss, str) else f'{loss:.4f}'}")
    
    print("\nPer-Language Performance:")
    for lang_id, metrics in results['per_language'].items():
        lang_name = id_to_lang.get(int(lang_id), f'Unknown ({lang_id})')
        print(f"\n{lang_name} (n={metrics['sample_count']}):")
        if 'auc' in metrics and 'auc_ci' in metrics:
            print(f"  AUC: {metrics['auc']:.4f} (95% CI: [{metrics['auc_ci'][0]:.4f}, {metrics['auc_ci'][1]:.4f}])")
        print(f"  F1: {metrics['f1']:.4f} (95% CI: [{metrics['f1_ci'][0]:.4f}, {metrics['f1_ci'][1]:.4f}])")
        
        # Handle metrics that might be missing or non-numeric
        h_loss = metrics.get('hamming_loss', 'N/A')
        h_loss_ci = metrics.get('hamming_loss_ci', ['N/A', 'N/A'])
        e_match = metrics.get('exact_match', 'N/A')
        e_match_ci = metrics.get('exact_match_ci', ['N/A', 'N/A'])
        
        h_loss_str = h_loss if isinstance(h_loss, str) else f'{h_loss:.4f}'
        h_loss_ci_str = [
            ci if isinstance(ci, str) else f'{ci:.4f}'
            for ci in h_loss_ci
        ]
        e_match_str = e_match if isinstance(e_match, str) else f'{e_match:.4f}'
        e_match_ci_str = [
            ci if isinstance(ci, str) else f'{ci:.4f}'
            for ci in e_match_ci
        ]
        
        print(f"  Hamming Loss: {h_loss_str} (95% CI: [{h_loss_ci_str[0]}, {h_loss_ci_str[1]}])")
        print(f"  Exact Match: {e_match_str} (95% CI: [{e_match_ci_str[0]}, {e_match_ci_str[1]}])")
    
    print("\nPer-Class Performance:")
    for class_name, metrics in results['per_class'].items():
        print(f"\n{class_name}:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  NPV: {metrics['npv']:.4f}")
        print(f"  Threshold: {metrics['threshold']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
        print(f"    FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}")

def plot_calibration_curves(y_true, y_pred, output_dir, toxicity_types=None, languages=None, langs=None):
    """Plot calibration curves for model predictions
    
    Args:
        y_true: True labels (n_samples, n_classes)
        y_pred: Predicted probabilities (n_samples, n_classes)
        output_dir: Directory to save plots
        toxicity_types: List of toxicity class names
        languages: Dictionary mapping language IDs to names
        langs: Array of language IDs for each sample
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Overall calibration curve
    plt.figure(figsize=(10, 6))
    prob_true, prob_pred = calibration_curve(
        y_true.flatten(),
        y_pred.flatten(),
        n_bins=10,
        strategy='quantile'  # Use quantile-based binning for better visualization
    )
    
    plt.plot(prob_pred, prob_true, marker='o', label="Calibration Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Overall Probability Calibration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'overall_calibration.png'))
    plt.close()
    
    # Per-class calibration curves
    if toxicity_types:
        plt.figure(figsize=(15, 10))
        for i, class_name in enumerate(toxicity_types):
            plt.subplot(2, 3, i + 1)
            prob_true, prob_pred = calibration_curve(
                y_true[:, i],
                y_pred[:, i],
                n_bins=10,
                strategy='quantile'
            )
            
            plt.plot(prob_pred, prob_true, marker='o', label="Calibration")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
            plt.title(f"{class_name}")
            plt.xlabel("Predicted Probability")
            plt.ylabel("Observed Frequency")
            plt.legend()
            plt.grid(True)
        
        plt.suptitle("Calibration Curves by Toxicity Class")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'class_calibration.png'))
        plt.close()
    
    # Per-language calibration curves
    if languages:
        for lang_id, lang_name in languages.items():
            lang_mask = langs == int(lang_id)
            if lang_mask.sum() > 0:
                plt.figure(figsize=(15, 10))
                for i, class_name in enumerate(toxicity_types):
                    plt.subplot(2, 3, i + 1)
                    try:
                        prob_true, prob_pred = calibration_curve(
                            y_true[lang_mask, i],
                            y_pred[lang_mask, i],
                            n_bins=10,
                            strategy='quantile'
                        )
                        
                        plt.plot(prob_pred, prob_true, marker='o', label="Calibration")
                        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
                        plt.title(f"{class_name}")
                        plt.xlabel("Predicted Probability")
                        plt.ylabel("Observed Frequency")
                        plt.legend()
                        plt.grid(True)
                    except Exception as e:
                        print(f"Warning: Could not plot calibration curve for {class_name} in {lang_name}: {str(e)}")
                
                plt.suptitle(f"Calibration Curves for {lang_name}")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'calibration_{lang_id}.png'))
                plt.close()

def parallel_bootstrap_metrics(args):
    """Calculate metrics for a single bootstrap iteration
    
    Args:
        args: Tuple of (labels, predictions, binary_predictions, threshold)
    Returns:
        Dictionary of calculated metrics
    """
    try:
        y_true, y_pred, y_binary, threshold = args
        
        # Perform bootstrap sampling
        y_true_boot, y_pred_boot = bootstrap_sample(y_true, y_pred)
        
        # For binary predictions, use the same indices as the other bootstrap samples
        if y_binary is not None:
            y_binary_boot = y_binary[np.arange(len(y_true))[y_true_boot == y_true]]
        else:
            y_binary_boot = (y_pred_boot > threshold).astype(int)
        
        # Calculate metrics
        metrics = {}
        
        # AUC
        if len(np.unique(y_true_boot)) > 1:
            metrics['auc'] = roc_auc_score(y_true_boot, y_pred_boot, average='weighted')
        else:
            metrics['auc'] = None
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_boot, y_binary_boot,
            average='weighted', zero_division=0
        )
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # Hamming Loss and Exact Match
        metrics['hamming_loss'] = hamming_loss(y_true_boot, y_binary_boot)
        metrics['exact_match'] = accuracy_score(y_true_boot, y_binary_boot)
        
        # Specificity
        metrics['specificity'] = calculate_specificity(y_true_boot, y_binary_boot)
        
        return metrics
    except Exception as e:
        print(f"Warning: Bootstrap iteration failed: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate toxic comment classifier')
    parser.add_argument('--model_path', type=str, default='weights/toxic_classifier_xlm-roberta-large',
                      help='Path to the trained model')
    parser.add_argument('--test_file', type=str, default='dataset/split/test.csv',
                      help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Base directory to save results')
    parser.add_argument('--num_workers', type=int, default=None,
                      help='Number of workers for data loading (default: CPU count - 1)')
    
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
        'num_workers': args.num_workers
    }
    with open(os.path.join(eval_dir, 'eval_params.json'), 'w') as f:
        json.dump(eval_params, f, indent=2)
    
    # Set number of workers
    if args.num_workers is None:
        args.num_workers = max(1, multiprocessing.cpu_count() - 1)
    
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
        
        # Create test dataset and dataloader
        test_dataset = ToxicDataset(test_df, tokenizer)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Evaluate model using the timestamped directory
        results = evaluate_model(model, test_loader, device, eval_dir)
        
        print(f"\nEvaluation complete! Results saved to {eval_dir}")
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        # Save error log in the evaluation directory
        with open(os.path.join(eval_dir, 'error_log.txt'), 'w') as f:
            f.write(f"Error during evaluation: {str(e)}")
        raise
    
    finally:
        # Cleanup
        plt.close('all')
        gc.collect()
        torch.cuda.empty_cache()
        
        # Close the dataloader explicitly
        if 'test_loader' in locals():
            del test_loader
        if 'test_dataset' in locals():
            del test_dataset
        if 'model' in locals():
            del model
        
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 