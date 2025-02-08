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

def evaluate_model(model, test_loader, device, output_dir):
    """Evaluate model performance with language-specific thresholds"""
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
    
    # Calculate language-specific thresholds and metrics
    unique_langs = np.unique(langs)
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    for lang in unique_langs:
        lang_mask = langs == lang
        if not lang_mask.any():
            continue
            
        lang_preds = predictions[lang_mask]
        lang_labels = labels[lang_mask]
        
        # Optimize thresholds for this language
        lang_thresholds = {}
        for i, class_name in enumerate(toxicity_types):
            try:
                # Find optimal threshold using ROC curve
                fpr, tpr, thresh = roc_curve(lang_labels[:, i], lang_preds[:, i])
                optimal_idx = np.argmax(tpr - fpr)
                lang_thresholds[class_name] = thresh[optimal_idx]
                
                # Calculate F1 scores for different thresholds
                f1_scores = []
                precisions = []
                recalls = []
                
                for t in thresh:
                    binary_preds = (lang_preds[:, i] > t).astype(int)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        lang_labels[:, i], 
                        binary_preds, 
                        average='binary',
                        zero_division=0  # Handle cases with no positive predictions
                    )
                    f1_scores.append(f1)
                    precisions.append(precision)
                    recalls.append(recall)
                
                # Use a combination of metrics to find the best threshold
                # Balance between F1 score and maintaining a reasonable positive prediction rate
                scores = np.array(f1_scores) * 0.7 + np.array(recalls) * 0.3
                best_idx = np.argmax(scores)
                
                # Only use the new threshold if it's better and produces some positive predictions
                if scores[best_idx] > scores[optimal_idx]:
                    test_preds = (lang_preds[:, i] > thresh[best_idx]).astype(int)
                    if test_preds.sum() > 0:  # Ensure we have some positive predictions
                        lang_thresholds[class_name] = thresh[best_idx]
                
                # If threshold results in no positive predictions, try a lower one
                test_preds = (lang_preds[:, i] > lang_thresholds[class_name]).astype(int)
                if test_preds.sum() == 0:
                    # Fall back to a lower threshold that gives some positive predictions
                    positive_pred_mask = test_preds.sum(axis=0) > 0
                    if positive_pred_mask.any():
                        percentile_threshold = np.percentile(lang_preds[:, i], 95)  # Use top 5% as positive
                        lang_thresholds[class_name] = min(percentile_threshold, 0.3)  # Cap at 0.3
                    else:
                        lang_thresholds[class_name] = 0.3  # Conservative default
            except Exception as e:
                print(f"Warning: Could not optimize threshold for {class_name} in language {lang}: {str(e)}")
                # Use a conservative default threshold
                lang_thresholds[class_name] = 0.3
        
        results['thresholds'][lang] = lang_thresholds
        
        # Calculate metrics using optimized thresholds
        binary_preds = np.zeros_like(lang_preds)
        for i, class_name in enumerate(toxicity_types):
            binary_preds[:, i] = (lang_preds[:, i] > lang_thresholds[class_name]).astype(int)
        
        # Calculate language-specific metrics
        try:
            lang_metrics = calculate_language_metrics(lang_labels, lang_preds, binary_preds)
            results['per_language'][lang] = lang_metrics
            
            # Calculate per-class metrics for this language
            class_metrics = {}
            for i, class_name in enumerate(toxicity_types):
                metrics = calculate_class_metrics(
                    lang_labels[:, i],
                    lang_preds[:, i],
                    binary_preds[:, i],
                    lang_thresholds[class_name]
                )
                class_metrics[class_name] = metrics
            
            results['per_language'][lang]['class_metrics'] = class_metrics
            
        except Exception as e:
            print(f"Warning: Could not calculate metrics for language {lang}: {str(e)}")
    
    # Calculate overall metrics using language-specific thresholds
    overall_binary_preds = np.zeros_like(predictions)
    for i, lang in enumerate(langs):
        for j, class_name in enumerate(toxicity_types):
            threshold = results['thresholds'][lang][class_name]
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
    
    # Save results and generate visualizations
    save_results(results, predictions, labels, langs, output_dir)
    plot_metrics(results, output_dir)
    
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
        unique_classes = np.unique(labels[:, i])
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=labels[:, i]
        )
        # Convert to integer indices
        class_weights[i] = {int(class_label): weight 
                          for class_label, weight in zip(unique_classes, weights)}
    return class_weights

def calculate_language_metrics(labels, predictions, binary_predictions):
    """Calculate detailed metrics for a specific language with class weights"""
    # Check if we have enough samples
    if len(labels) == 0:
        raise ValueError("No samples available for metric calculation")
    
    # Calculate class weights for this language
    class_weights = calculate_class_weights(labels)
    
    # Calculate weighted metrics
    sample_weights = np.ones(len(labels))
    for i in range(labels.shape[1]):
        # Convert labels to integers for indexing
        label_indices = labels[:, i].astype(int)
        sample_weights *= np.array([class_weights[i][idx] for idx in label_indices])
    
    # Calculate AUC only if we have both positive and negative samples
    try:
        auc = roc_auc_score(labels, predictions, average='weighted', sample_weight=sample_weights)
    except ValueError:
        # If we don't have both classes, set AUC to None
        auc = None
    
    # Calculate precision, recall, F1 with zero_division=0
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='weighted', 
        sample_weight=sample_weights, zero_division=0
    )
    
    # Add multi-label metrics
    h_loss = hamming_loss(labels, binary_predictions, sample_weight=sample_weights)
    exact_match = accuracy_score(labels, binary_predictions, sample_weight=sample_weights)
    
    # Calculate confidence intervals using stratified bootstrap
    n_bootstrap = 1000
    auc_scores = []
    f1_scores = []
    hamming_scores = []
    exact_match_scores = []
    
    for _ in range(n_bootstrap):
        try:
            # Use stratified bootstrap sampling
            boot_labels, boot_preds = bootstrap_sample(labels, predictions)
            boot_binary = (boot_preds > 0.5).astype(int)
            
            # Calculate class weights for bootstrap sample
            boot_weights = calculate_class_weights(boot_labels)
            boot_sample_weights = np.ones(len(boot_labels))
            for i in range(boot_labels.shape[1]):
                # Convert labels to integers for indexing
                boot_label_indices = boot_labels[:, i].astype(int)
                boot_sample_weights *= np.array([boot_weights[i][idx] for idx in boot_label_indices])
            
            # Calculate AUC only if possible
            try:
                if auc is not None:  # Only append AUC if main calculation succeeded
                    auc_scores.append(roc_auc_score(
                        boot_labels, boot_preds, average='weighted',
                        sample_weight=boot_sample_weights
                    ))
            except ValueError:
                pass
            
            # Calculate other metrics with zero_division=0
            _, _, f1, _ = precision_recall_fscore_support(
                boot_labels, boot_binary, average='weighted',
                sample_weight=boot_sample_weights, zero_division=0
            )
            f1_scores.append(f1)
            hamming_scores.append(hamming_loss(boot_labels, boot_binary, sample_weight=boot_sample_weights))
            exact_match_scores.append(accuracy_score(boot_labels, boot_binary, sample_weight=boot_sample_weights))
        except:
            continue
    
    # Calculate confidence intervals
    ci_lower = 2.5
    ci_upper = 97.5
    
    # Prepare results dictionary
    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_ci': [np.percentile(f1_scores, ci_lower), np.percentile(f1_scores, ci_upper)],
        'hamming_loss': h_loss,
        'hamming_loss_ci': [np.percentile(hamming_scores, ci_lower), np.percentile(hamming_scores, ci_upper)],
        'exact_match': exact_match,
        'exact_match_ci': [np.percentile(exact_match_scores, ci_lower), np.percentile(exact_match_scores, ci_upper)],
        'sample_count': len(labels),
        'bootstrap_samples': len(f1_scores),
        'class_weights': class_weights
    }
    
    # Add AUC metrics only if available
    if auc is not None:
        results['auc'] = auc
        if auc_scores:
            results['auc_ci'] = [np.percentile(auc_scores, ci_lower), np.percentile(auc_scores, ci_upper)]
    
    return results

def calculate_class_metrics(labels, predictions, binary_predictions, threshold):
    """Calculate detailed metrics for a specific class with class weights"""
    # Calculate class weights
    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    # Convert labels to integers for indexing
    label_indices = labels.astype(int)
    sample_weights = np.array([weights[idx] for idx in label_indices])
    
    # Calculate weighted metrics
    auc = roc_auc_score(labels, predictions, sample_weight=sample_weights)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='binary',
        sample_weight=sample_weights
    )
    
    # Calculate additional metrics
    tn, fp, fn, tp = confusion_matrix(
        labels, binary_predictions, 
        sample_weight=sample_weights
    ).ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    
    # Calculate confidence intervals using bootstrap
    n_bootstrap = 1000
    metrics_boot = {
        'auc': [], 'precision': [], 'recall': [], 
        'f1': [], 'specificity': [], 'npv': []
    }
    
    for _ in range(n_bootstrap):
        try:
            # Use stratified bootstrap sampling for single class
            boot_labels, boot_preds = bootstrap_sample(
                labels.reshape(-1, 1), 
                predictions.reshape(-1, 1)
            )
            boot_binary = (boot_preds > threshold).astype(int)
            
            # Calculate weights for bootstrap sample
            boot_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(boot_labels),
                y=boot_labels.ravel()
            )
            boot_sample_weights = np.array([boot_weights[l] for l in boot_labels.ravel()])
            
            # Calculate metrics for this bootstrap sample
            metrics_boot['auc'].append(
                roc_auc_score(boot_labels, boot_preds, sample_weight=boot_sample_weights)
            )
            p, r, f, _ = precision_recall_fscore_support(
                boot_labels, boot_binary, average='binary',
                sample_weight=boot_sample_weights
            )
            metrics_boot['precision'].append(p)
            metrics_boot['recall'].append(r)
            metrics_boot['f1'].append(f)
            
            tn, fp, fn, tp = confusion_matrix(
                boot_labels, boot_binary,
                sample_weight=boot_sample_weights
            ).ravel()
            metrics_boot['specificity'].append(tn / (tn + fp))
            metrics_boot['npv'].append(tn / (tn + fn))
        except:
            continue
    
    # Calculate confidence intervals
    ci_lower = 2.5
    ci_upper = 97.5
    
    return {
        'auc': auc,
        'auc_ci': [np.percentile(metrics_boot['auc'], ci_lower), 
                   np.percentile(metrics_boot['auc'], ci_upper)],
        'precision': precision,
        'precision_ci': [np.percentile(metrics_boot['precision'], ci_lower),
                        np.percentile(metrics_boot['precision'], ci_upper)],
        'recall': recall,
        'recall_ci': [np.percentile(metrics_boot['recall'], ci_lower),
                     np.percentile(metrics_boot['recall'], ci_upper)],
        'f1': f1,
        'f1_ci': [np.percentile(metrics_boot['f1'], ci_lower),
                  np.percentile(metrics_boot['f1'], ci_upper)],
        'specificity': specificity,
        'specificity_ci': [np.percentile(metrics_boot['specificity'], ci_lower),
                          np.percentile(metrics_boot['specificity'], ci_upper)],
        'npv': npv,
        'npv_ci': [np.percentile(metrics_boot['npv'], ci_lower),
                   np.percentile(metrics_boot['npv'], ci_upper)],
        'threshold': threshold,
        'positive_samples': int(labels.sum()),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'bootstrap_samples': len(metrics_boot['auc']),
        'class_weights': dict(zip(np.unique(labels), weights))  # Store class weights for reference
    }

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

def plot_metrics(results, output_dir):
    """Generate detailed visualization plots"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot per-class performance
    plt.figure(figsize=(12, 6))
    classes = list(results['per_class'].keys())
    metrics = ['auc', 'precision', 'recall', 'f1']
    
    for metric in metrics:
        values = [results['per_class'][c][metric] for c in classes]
        plt.plot(classes, values, marker='o', label=metric.upper())
    
    plt.title('Per-Class Performance Metrics')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'per_class_metrics.png'))
    plt.close()
    
    # Plot per-language performance
    plt.figure(figsize=(10, 6))
    languages = list(results['per_language'].keys())
    auc_scores = [results['per_language'][lang]['auc'] for lang in languages]
    sample_counts = [results['per_language'][lang]['sample_count'] for lang in languages]
    
    # Create bubble plot
    plt.scatter(languages, auc_scores, s=[count/50 for count in sample_counts], alpha=0.6)
    plt.title('AUC Scores by Language (bubble size = sample count)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'language_performance.png'))
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