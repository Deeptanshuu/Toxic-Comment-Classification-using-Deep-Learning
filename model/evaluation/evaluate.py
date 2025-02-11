import torch
from model.language_aware_transformer import LanguageAwareTransformer
from transformers import XLMRobertaTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, 
    confusion_matrix, roc_curve, hamming_loss, 
    accuracy_score, precision_score, recall_score
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

# Set matplotlib to non-interactive backend
plt.switch_backend('agg')

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
        
        # Convert language strings to IDs, default to English (0) if language not found
        self.langs = np.array([self.lang_to_id.get(str(lang).lower(), 0) for lang in df['lang'].values])
        
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

def evaluate_language(args):
    """Evaluate model performance for a single language"""
    try:
        lang_id, lang_name, lang_preds, lang_labels, toxicity_types = args
        print(f"\nEvaluating {lang_name} [{len(lang_labels)} samples]")
        
        # Class-specific configuration
        PRECISION_THRESHOLDS = {
            'toxic': 0.85,
            'severe_toxic': 0.90,
            'obscene': 0.85,
            'threat': 0.92,  # Increased for rare class
            'insult': 0.85,
            'identity_hate': 0.92  # Increased for rare class
        }
        
        # Class-specific minimum recall requirements
        MIN_RECALL = {
            'toxic': 0.20,
            'severe_toxic': 0.15,  # Lower for rare class
            'obscene': 0.20,
            'threat': 0.10,        # Lower for rare class
            'insult': 0.20,
            'identity_hate': 0.15  # Lower for rare class
        }
        
        # Optimize thresholds for this language
        lang_thresholds = {}
        for i, class_name in enumerate(toxicity_types):
            try:
                class_labels = lang_labels[:, i]
                class_preds = lang_preds[:, i]
                
                if len(np.unique(class_labels)) < 2:
                    lang_thresholds[class_name] = 0.8 if class_labels[0] == 0 else 0.6
                    continue
                
                pos_ratio = np.mean(class_labels)
                fpr, tpr, thresh = roc_curve(class_labels, class_preds)
                
                # Calculate class weights with stronger penalty for FP in rare classes
                pos_weight = 1.0 / (pos_ratio + 1e-7)  # Avoid division by zero
                neg_weight = 1.0
                
                # Increase positive weight for rare classes
                if pos_ratio < 0.01:  # Very rare class
                    pos_weight *= 2.0
                elif pos_ratio < 0.05:  # Rare class
                    pos_weight *= 1.5
                
                # Create sample weights
                sample_weights = np.where(class_labels == 1, pos_weight, neg_weight)
                
                # Normalize weights
                sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
                
                # Calculate scores for all thresholds with class-specific weighting
                metrics_by_threshold = []
                for t in thresh:
                    binary_preds = (class_preds > t).astype(int)
                    try:
                        precision = precision_score(
                            class_labels, binary_preds,
                            sample_weight=sample_weights,
                            zero_division=0
                        )
                        recall = recall_score(
                            class_labels, binary_preds,
                            sample_weight=sample_weights,
                            zero_division=0
                        )
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        hamming_acc = 1 - hamming_loss(class_labels, binary_preds)
                        
                        # Calculate false positive rate for this threshold
                        tn, fp, fn, tp = confusion_matrix(
                            class_labels, binary_preds,
                            sample_weight=sample_weights
                        ).ravel()
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 1.0
                        
                        metrics_by_threshold.append({
                            'threshold': t,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'hamming_accuracy': hamming_acc,
                            'fpr': fpr
                        })
                    except Exception:
                        continue
                
                # Filter thresholds meeting precision and minimum recall requirements
                required_precision = PRECISION_THRESHOLDS.get(class_name, 0.85)
                min_recall = MIN_RECALL.get(class_name, 0.20)
                
                # For rare classes, also consider FPR in threshold selection
                max_fpr = 0.1 if pos_ratio < 0.05 else 0.2  # Stricter FPR for rare classes
                
                valid_thresholds = [
                    m for m in metrics_by_threshold 
                    if m['precision'] >= required_precision and 
                       m['recall'] >= min_recall and
                       m['fpr'] <= max_fpr
                ]
                
                if valid_thresholds:
                    # Among valid thresholds, choose one with best balance of metrics
                    # Weight precision more heavily for rare classes
                    precision_weight = 0.7 if pos_ratio < 0.05 else 0.5
                    recall_weight = 0.3 if pos_ratio < 0.05 else 0.2
                    f1_weight = 0.0 if pos_ratio < 0.05 else 0.2  # Ignore F1 for rare classes
                    hamming_weight = 0.0 if pos_ratio < 0.05 else 0.1
                    
                    best_threshold = max(
                        valid_thresholds,
                        key=lambda x: (
                            x['precision'] * precision_weight +
                            x['recall'] * recall_weight +
                            x['f1'] * f1_weight +
                            x['hamming_accuracy'] * hamming_weight -
                            x['fpr'] * 0.1  # Penalize high FPR
                        )
                    )
                    initial_threshold = best_threshold['threshold']
                    achieved_metrics = best_threshold
                else:
                    # If no threshold meets all requirements, prioritize precision and low FPR
                    best_threshold = max(
                        metrics_by_threshold,
                        key=lambda x: (
                            x['precision'] * 0.7 +
                            (1 - x['fpr']) * 0.3  # Reward low FPR
                        )
                    )
                    initial_threshold = best_threshold['threshold']
                    achieved_metrics = best_threshold
                
                # Test and adjust the threshold if needed
                test_preds = (class_preds > initial_threshold).astype(int)
                if test_preds.sum() == 0 and pos_ratio > 0:
                    # Try multiple percentiles with precision focus
                    percentiles = [99, 95, 90, 85, 80] if pos_ratio < 0.05 else [95, 90, 85, 80, 75]
                    best_metrics = None
                    best_threshold = initial_threshold
                    
                    for p in percentiles:
                        threshold = np.percentile(class_preds, p)
                        test_preds = (class_preds > threshold).astype(int)
                        if test_preds.sum() > 0:
                            precision = precision_score(
                                class_labels, test_preds,
                                sample_weight=sample_weights,
                                zero_division=0
                            )
                            recall = recall_score(
                                class_labels, test_preds,
                                sample_weight=sample_weights,
                                zero_division=0
                            )
                            
                            # Calculate FPR
                            tn, fp, fn, tp = confusion_matrix(
                                class_labels, test_preds,
                                sample_weight=sample_weights
                            ).ravel()
                            fpr = fp / (fp + tn) if (fp + tn) > 0 else 1.0
                            
                            # For rare classes, be more strict about precision and FPR
                            if pos_ratio < 0.05:
                                if precision >= required_precision and fpr <= max_fpr:
                                    if best_metrics is None or precision > best_metrics['precision']:
                                        best_threshold = threshold
                                        best_metrics = {
                                            'precision': precision,
                                            'recall': recall,
                                            'fpr': fpr
                                        }
                            else:
                                if precision >= required_precision and recall >= min_recall:
                                    if best_metrics is None or precision > best_metrics['precision']:
                                        best_threshold = threshold
                                        best_metrics = {
                                            'precision': precision,
                                            'recall': recall,
                                            'fpr': fpr
                                        }
                    
                    lang_thresholds[class_name] = max(best_threshold, 0.5)
                    achieved_metrics = best_metrics or achieved_metrics
                else:
                    lang_thresholds[class_name] = initial_threshold
                
                # Log metrics with class ratio information
                print(f"  {class_name}: t={lang_thresholds[class_name]:.3f} "
                      f"[P={achieved_metrics['precision']:.3f}, R={achieved_metrics['recall']:.3f}, "
                      f"F1={achieved_metrics['f1']:.3f}, FPR={achieved_metrics['fpr']:.3f}] "
                      f"(pos_ratio={pos_ratio:.3%})")
                
            except Exception as e:
                lang_thresholds[class_name] = 0.6
        
        # Calculate metrics using optimized thresholds
        binary_preds = np.zeros_like(lang_preds)
        for i, class_name in enumerate(toxicity_types):
            binary_preds[:, i] = (lang_preds[:, i] > lang_thresholds[class_name]).astype(int)
        
        # Calculate metrics with class-specific weighting
        metrics = {}
        try:
            if len(np.unique(lang_labels)) > 1:
                # Calculate class weights based on positive ratios
                class_weights = []
                for i in range(lang_labels.shape[1]):
                    pos_ratio = np.mean(lang_labels[:, i])
                    weight = 1.0 / (pos_ratio + 1e-7)
                    if pos_ratio < 0.01:  # Very rare class
                        weight *= 2.0
                    elif pos_ratio < 0.05:  # Rare class
                        weight *= 1.5
                    class_weights.append(weight)
                
                # Normalize weights
                class_weights = np.array(class_weights)
                class_weights = class_weights / np.sum(class_weights) * len(class_weights)
                
                # Calculate AUC with class weights
                metrics['auc'] = roc_auc_score(
                    lang_labels, lang_preds,
                    average='weighted',
                    sample_weight=class_weights
                )
            
            # Calculate other metrics with class weights
            precision, recall, f1, _ = precision_recall_fscore_support(
                lang_labels, binary_preds,
                average='weighted',
                sample_weight=class_weights,
                zero_division=0
            )
            
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'hamming_accuracy': 1 - hamming_loss(lang_labels, binary_preds)
            })
            
        except Exception:
            metrics.update({
                'auc': None,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'hamming_accuracy': 0.0
            })
        
        return lang_id, metrics, lang_thresholds
        
    except Exception:
        return None

def evaluate_model(model, test_loader, device, output_dir):
    """Evaluate model performance with parallel language-specific evaluation"""
    all_predictions = []
    all_labels = []
    all_langs = []
    all_losses = []
    
    # Enable inference optimizations
    torch.backends.cudnn.benchmark = True
    
    # Enable mixed precision inference
    scaler = torch.cuda.amp.GradScaler()
    
    # Compile model if torch 2.0+ is available
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Using torch.compile() for faster inference")
        except Exception as e:
            print(f"Could not compile model: {e}")
    
    print("\nRunning predictions on test set...")
    model.eval()  # Ensure model is in eval mode
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device efficiently
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            langs = batch['lang'].to(device, non_blocking=True)
            
            # Run prediction
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                lang_ids=langs
            )
            
            # Gather predictions efficiently
            loss = outputs['loss'].item()
            predictions = outputs['probabilities'].cpu()
            
            # Store results
            all_predictions.append(predictions)
            all_labels.append(labels.cpu())
            all_langs.append(langs.cpu())
            all_losses.append(loss)
            
            # Clear GPU memory aggressively
            del input_ids, attention_mask, outputs, labels, langs, predictions
            torch.cuda.empty_cache()
    
    # Concatenate results efficiently
    predictions = torch.cat(all_predictions, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()  
    langs = torch.cat(all_langs, dim=0).numpy()
    avg_loss = np.mean(all_losses)
    
    # Clear temporary lists
    del all_predictions, all_labels, all_langs, all_losses
    torch.cuda.empty_cache()
    
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
    
    # Default thresholds in case evaluation fails
    default_thresholds = {
        'toxic': 0.3991,
        'severe_toxic': 0.2350,
        'obscene': 0.47,
        'threat': 0.3614,
        'insult': 0.3906,
        'identity_hate': 0.2533
    }
    
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
    
    # Collect results and ensure each language has thresholds
    for lang in unique_langs:
        # Find the result for this language
        lang_result = next((r for r in lang_results if r[0] == lang), None)
        
        if lang_result:
            lang_id, lang_metrics, lang_thresholds = lang_result
            results['per_language'][str(lang_id)] = lang_metrics
            results['thresholds'][str(lang_id)] = lang_thresholds
        else:
            # If evaluation failed, use default thresholds
            results['thresholds'][str(lang)] = default_thresholds
            print(f"Warning: Using default thresholds for language {id_to_lang.get(int(lang), f'Unknown ({lang})')}")
    
    # Calculate overall metrics using language-specific thresholds
    overall_binary_preds = np.zeros_like(predictions)
    for i, lang in enumerate(langs):
        lang_thresholds = results['thresholds'].get(str(lang), default_thresholds)
        for j, class_name in enumerate(toxicity_types):
            threshold = lang_thresholds.get(class_name, default_thresholds[class_name])
            overall_binary_preds[i, j] = (predictions[i, j] > threshold).astype(int)
    
    # Calculate AUC scores with both averaging methods
    try:
        results['overall'].update({
            'auc_macro': roc_auc_score(labels, predictions, average='macro'),
            'auc_weighted': roc_auc_score(labels, predictions, average='weighted')
        })
    except Exception as e:
        print(f"Warning: Could not calculate AUC scores: {str(e)}")
        results['overall'].update({
            'auc_macro': 0.0,
            'auc_weighted': 0.0
        })
    
    try:
        # Calculate both macro and weighted averages for precision, recall, and F1
        precision_macro, recall_macro, f1_macro, support_macro = precision_recall_fscore_support(
            labels, overall_binary_preds, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, support_weighted = precision_recall_fscore_support(
            labels, overall_binary_preds, average='weighted', zero_division=0
        )
        
        # Calculate per-class metrics and support
        per_class_precision, per_class_recall, per_class_f1, class_support = precision_recall_fscore_support(
            labels, overall_binary_preds, average=None, zero_division=0
        )
        
        # Update results with both macro and weighted metrics
        results['overall'].update({
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        })
        
        # Add per-class support information
        results['overall']['class_support'] = {
            class_name: int(support) for class_name, support in zip(toxicity_types, class_support)
        }
        
        # Add per-class metrics
        results['overall']['per_class_metrics'] = {
            class_name: {
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1),
                'support': int(support)
            } for class_name, prec, rec, f1, support in zip(
                toxicity_types, per_class_precision, per_class_recall, per_class_f1, class_support
            )
        }
        
        # Calculate class weights based on support
        total_samples = sum(class_support)
        class_weights = class_support / total_samples if total_samples > 0 else np.zeros_like(class_support)
        results['overall']['class_weights'] = {
            class_name: float(weight) for class_name, weight in zip(toxicity_types, class_weights)
        }
        
    except Exception as e:
        print(f"Warning: Could not calculate precision/recall/F1 scores: {str(e)}")
        # Set default values for all metrics
        default_metrics = {
            'precision_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_macro': 0.0,
            'recall_weighted': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'class_support': {class_name: 0 for class_name in toxicity_types},
            'per_class_metrics': {
                class_name: {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'support': 0
                } for class_name in toxicity_types
            },
            'class_weights': {class_name: 0.0 for class_name in toxicity_types}
        }
        results['overall'].update(default_metrics)
    
    # Calculate additional overall metrics with both averaging methods
    try:
        # Hamming Loss (already normalized by nature)
        results['overall']['hamming_loss'] = hamming_loss(labels, overall_binary_preds)
        
        # Exact Match (already normalized)
        results['overall']['exact_match'] = accuracy_score(labels, overall_binary_preds)
        
        # Calculate specificity with both averaging methods
        specificities = []
        weighted_specificities = []
        class_weights = []
        
        for i in range(labels.shape[1]):
            try:
                tn, fp, fn, tp = confusion_matrix(
                    labels[:, i],
                    overall_binary_preds[:, i]
                ).ravel()
                
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                specificities.append(specificity)
                
                # Weight by class size
                weight = (tn + fp + fn + tp)
                weighted_specificities.append(specificity * weight)
                class_weights.append(weight)
            except Exception as e:
                print(f"Warning: Could not calculate specificity for class {i}: {str(e)}")
                specificities.append(0.0)
                weighted_specificities.append(0.0)
                class_weights.append(0.0)
        
        # Calculate macro and weighted specificity
        results['overall'].update({
            'specificity_macro': np.mean(specificities),
            'specificity_weighted': (
                np.sum(weighted_specificities) / np.sum(class_weights) 
                if np.sum(class_weights) > 0 else 0.0
            )
        })
        
        # Add per-class specificity
        for i, class_name in enumerate(toxicity_types):
            if class_name in results['overall']['per_class_metrics']:
                results['overall']['per_class_metrics'][class_name]['specificity'] = specificities[i]
        
    except Exception as e:
        print(f"Warning: Could not calculate additional metrics: {str(e)}")
        results['overall'].update({
            'hamming_loss': 1.0,
            'exact_match': 0.0,
            'specificity_macro': 0.0,
            'specificity_weighted': 0.0
        })
        # Add default specificity to per-class metrics
        for class_name in toxicity_types:
            if class_name in results['overall']['per_class_metrics']:
                results['overall']['per_class_metrics'][class_name]['specificity'] = 0.0
    
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
    """Calculate specificity (true negative rate) for binary or multi-label data
    
    Args:
        y_true: Ground truth labels (n_samples,) or (n_samples, n_classes)
        y_pred: Predicted labels (n_samples,) or (n_samples, n_classes)
        sample_weight: Optional sample weights
    
    Returns:
        float: Specificity score (0.0 to 1.0) or None if calculation is invalid
    """
    try:
        # Input validation
        if y_true is None or y_pred is None:
            print("Warning: Null input to specificity calculation")
            return None
            
        if len(y_true) == 0 or len(y_pred) == 0:
            print("Warning: Empty input to specificity calculation")
            return None
            
        if y_true.shape != y_pred.shape:
            print(f"Warning: Shape mismatch in specificity calculation. y_true: {y_true.shape}, y_pred: {y_pred.shape}")
            return None
        
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Ensure binary values
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        
        if not all(x in [0, 1] for x in unique_true) or not all(x in [0, 1] for x in unique_pred):
            print("Warning: Non-binary values found in specificity calculation")
            return None
        
        if len(y_true.shape) > 1:
            # Multi-label case: calculate specificity for each label and average
            specificities = []
            weights = []
            
            for i in range(y_true.shape[1]):
                # Skip if no predictions made for this class
                if not np.any(y_pred[:, i] == y_true[:, i]):
                    continue
                    
                try:
                    cm = confusion_matrix(
                        y_true[:, i],
                        y_pred[:, i],
                        sample_weight=sample_weight
                    ).ravel()
                    
                    if len(cm) != 4:  # Ensure we have all confusion matrix elements
                        continue
                        
                    tn, fp, fn, tp = cm
                    
                    # Calculate specificity if we have negative samples
                    if (tn + fp) > 0:
                        spec = tn / (tn + fp)
                        specificities.append(spec)
                        weights.append(tn + fp)  # Weight by number of negative samples
                except Exception as e:
                    print(f"Warning: Error in specificity calculation for class {i}: {str(e)}")
                    continue
            
            # Return weighted average if we have any valid calculations
            if specificities:
                if sample_weight is not None:
                    return np.average(specificities, weights=weights)
                else:
                    return np.mean(specificities)
            else:
                print("Warning: No valid specificity calculations")
                return None
                
        else:
            # Binary case
            # Skip if no predictions match ground truth
            if not np.any(y_pred == y_true):
                print("Warning: No matching predictions in specificity calculation")
                return None
                
            try:
                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    sample_weight=sample_weight
                ).ravel()
                
                if len(cm) != 4:  # Ensure we have all confusion matrix elements
                    print("Warning: Invalid confusion matrix in specificity calculation")
                    return None
                    
                tn, fp, fn, tp = cm
                
                # Calculate specificity if we have negative samples
                if (tn + fp) > 0:
                    return tn / (tn + fp)
                else:
                    print("Warning: No negative samples for specificity calculation")
                    return None
                    
            except Exception as e:
                print(f"Warning: Error in binary specificity calculation: {str(e)}")
                return None
                
    except Exception as e:
        print(f"Warning: Unexpected error in specificity calculation: {str(e)}")
        return None

def calculate_weighted_hamming_loss(y_true, y_pred, sample_weight=None):
    """Calculate Hamming Loss with support for sample weights
    
    Args:
        y_true: Ground truth labels (n_samples,) or (n_samples, n_classes)
        y_pred: Predicted labels (n_samples,) or (n_samples, n_classes)
        sample_weight: Optional sample weights (n_samples,)
    
    Returns:
        float: Weighted Hamming Loss value
    """
    try:
        # Input validation
        if y_true is None or y_pred is None:
            print("Warning: Null input to Hamming Loss calculation")
            return None
            
        if len(y_true) == 0 or len(y_pred) == 0:
            print("Warning: Empty input to Hamming Loss calculation")
            return None
            
        if y_true.shape != y_pred.shape:
            print(f"Warning: Shape mismatch in Hamming Loss calculation. y_true: {y_true.shape}, y_pred: {y_pred.shape}")
            return None
        
        # Ensure inputs are numpy arrays and integer type
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        
        # Handle sample weights
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if len(sample_weight) != len(y_true):
                print(f"Warning: Sample weight length ({len(sample_weight)}) does not match input length ({len(y_true)})")
                return None
        
        if len(y_true.shape) > 1:
            # Multi-label case
            if sample_weight is not None:
                # Calculate weighted Hamming Loss for each sample
                sample_losses = np.sum(y_true != y_pred, axis=1) / y_true.shape[1]
                return np.average(sample_losses, weights=sample_weight)
            else:
                # Use sklearn's implementation for unweighted case
                return hamming_loss(y_true, y_pred)
        else:
            # Binary case
            if sample_weight is not None:
                return np.average(y_true != y_pred, weights=sample_weight)
            else:
                return np.mean(y_true != y_pred)
                
    except Exception as e:
        print(f"Warning: Error in Hamming Loss calculation: {str(e)}")
        return None

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
    
    # Calculate Hamming Loss with sample weights
    try:
        metrics['hamming_loss'] = calculate_weighted_hamming_loss(
            labels,
            binary_predictions,
            sample_weight=sample_weights
        )
        if metrics['hamming_loss'] is None:
            metrics['hamming_loss'] = 1.0  # Worst case
    except Exception as e:
        print(f"Warning: Could not calculate Hamming Loss: {str(e)}")
        metrics['hamming_loss'] = 1.0  # Worst case
    
    # Calculate Exact Match with sample weights
    try:
        metrics['exact_match'] = accuracy_score(
            labels,
            binary_predictions,
            sample_weight=sample_weights,
            normalize=True
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
    """Calculate detailed metrics for a specific class"""
    try:
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
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            'positive_samples': int(labels.sum()),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        })
        
        # Add confidence intervals as fixed values since we're not doing bootstrap
        for metric in ['auc', 'precision', 'recall', 'f1', 'specificity', 'npv']:
            metrics[f'{metric}_ci'] = [metrics[metric], metrics[metric]]
        
        metrics['class_weights'] = dict(zip(np.unique(labels), weights))
        
        return metrics
        
    except Exception as e:
        print(f"Warning: Error in calculate_class_metrics: {str(e)}")
        # Return default metrics
        return {
            'auc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'specificity': 0.0,
            'npv': 0.0,
            'threshold': threshold,
            'positive_samples': 0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'auc_ci': [0.0, 0.0],
            'precision_ci': [0.0, 0.0],
            'recall_ci': [0.0, 0.0],
            'f1_ci': [0.0, 0.0],
            'specificity_ci': [0.0, 0.0],
            'npv_ci': [0.0, 0.0],
            'class_weights': {0: 1.0, 1: 1.0}
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
    
    # Enhanced language performance visualization
    if results.get('per_language'):
        # Prepare data for plotting
        languages = list(results['per_language'].keys())
        metrics_to_plot = ['auc', 'f1', 'precision', 'recall']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
        
        # Plot 1: Main metrics by language
        x = np.arange(len(languages))
        width = 0.2  # Width of bars
        
        for i, metric in enumerate(metrics_to_plot):
            values = []
            errors = []
            for lang in languages:
                metric_value = results['per_language'][lang].get(metric, 0)
                metric_ci = results['per_language'][lang].get(f'{metric}_ci', [metric_value, metric_value])
                values.append(metric_value)
                errors.append((metric_value - metric_ci[0], metric_ci[1] - metric_value))
            
            errors = np.array(errors).T
            ax1.bar(x + i*width, values, width, label=metric.upper(),
                   yerr=errors, capsize=5)
        
        # Customize first subplot
        ax1.set_ylabel('Score')
        ax1.set_title('Language Performance Metrics')
        ax1.set_xticks(x + width * (len(metrics_to_plot)-1)/2)
        ax1.set_xticklabels([f'Lang {lang}' for lang in languages], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sample distribution
        sample_counts = [results['per_language'][lang].get('sample_count', 0) for lang in languages]
        total_samples = sum(sample_counts)
        
        # Handle zero division case
        if total_samples > 0:
            percentages = [count/total_samples * 100 for count in sample_counts]
        else:
            print("Warning: No samples found in results")
            percentages = [0] * len(languages)
        
        bars = ax2.bar(x, percentages, width*2)
        
        # Add value labels on the bars
        for i, (count, percentage) in enumerate(zip(sample_counts, percentages)):
            if total_samples > 0:
                ax2.text(i, percentage, f'{count:,}\n({percentage:.1f}%)', 
                        ha='center', va='bottom')
            else:
                ax2.text(i, 0, f'{count:,}\n(0.0%)', 
                        ha='center', va='bottom')
        
        # Customize second subplot
        ax2.set_ylabel('Sample Distribution (%)')
        ax2.set_title('Dataset Distribution Across Languages')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Lang {lang}' for lang in languages], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'language_performance.png'), dpi=300, bbox_inches='tight')
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
    
    # Performance Distribution Plots
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
    
    # Add global metrics summary to results
    results['overall']['summary'] = {
        'auc': {
            'macro': results['overall'].get('auc_macro', 0.0),
            'weighted': results['overall'].get('auc_weighted', 0.0)
        },
        'f1': {
            'macro': results['overall'].get('f1_macro', 0.0),
            'weighted': results['overall'].get('f1_weighted', 0.0)
        },
        'precision': {
            'macro': results['overall'].get('precision_macro', 0.0),
            'weighted': results['overall'].get('precision_weighted', 0.0)
        },
        'recall': {
            'macro': results['overall'].get('recall_macro', 0.0),
            'weighted': results['overall'].get('recall_weighted', 0.0)
        },
        'specificity': {
            'macro': results['overall'].get('specificity_macro', 0.0),
            'weighted': results['overall'].get('specificity_weighted', 0.0)
        },
        'other_metrics': {
            'hamming_loss': results['overall'].get('hamming_loss', 1.0),
            'exact_match': results['overall'].get('exact_match', 0.0)
        },
        'class_support': results['overall'].get('class_support', {})
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
    
    # Print detailed summary with both macro and weighted metrics
    print("\nEvaluation Results:")
    print("-" * 80)
    print(f"Overall Metrics:")
    
    print("\nAUC Scores:")
    print(f"  Macro-averaged: {results['overall'].get('auc_macro', 0.0):.4f}")
    print(f"  Weighted-averaged: {results['overall'].get('auc_weighted', 0.0):.4f}")
    
    print("\nF1 Scores:")
    print(f"  Macro-averaged: {results['overall'].get('f1_macro', 0.0):.4f}")
    print(f"  Weighted-averaged: {results['overall'].get('f1_weighted', 0.0):.4f}")
    
    print("\nPrecision:")
    print(f"  Macro-averaged: {results['overall'].get('precision_macro', 0.0):.4f}")
    print(f"  Weighted-averaged: {results['overall'].get('precision_weighted', 0.0):.4f}")
    
    print("\nRecall:")
    print(f"  Macro-averaged: {results['overall'].get('recall_macro', 0.0):.4f}")
    print(f"  Weighted-averaged: {results['overall'].get('recall_weighted', 0.0):.4f}")
    
    print("\nSpecificity:")
    print(f"  Macro-averaged: {results['overall'].get('specificity_macro', 0.0):.4f}")
    print(f"  Weighted-averaged: {results['overall'].get('specificity_weighted', 0.0):.4f}")
    
    print("\nOther Metrics:")
    print(f"  Exact Match: {results['overall'].get('exact_match', 0.0):.4f}")
    
    if 'class_support' in results['overall']:
        print("\nClass Support (number of samples):")
        for class_name, support in results['overall']['class_support'].items():
            print(f"  {class_name}: {support:,}")
    
    print("\nPer-Language Performance:")
    for lang_id, metrics in results['per_language'].items():
        lang_name = id_to_lang.get(int(lang_id), f'Unknown ({lang_id})')
        print(f"\n{lang_name} (n={metrics.get('sample_count', 0)}):")
        
        # Print AUC with CI if available
        if 'auc' in metrics:
            auc_str = f"{metrics['auc']:.4f}"
            if 'auc_ci' in metrics:
                auc_str += f" (95% CI: [{metrics['auc_ci'][0]:.4f}, {metrics['auc_ci'][1]:.4f}])"
            print(f"  AUC: {auc_str}")
        
        # Print F1 with CI if available
        if 'f1' in metrics:
            f1_str = f"{metrics['f1']:.4f}"
            if 'f1_ci' in metrics:
                f1_str += f" (95% CI: [{metrics['f1_ci'][0]:.4f}, {metrics['f1_ci'][1]:.4f}])"
            print(f"  F1: {f1_str}")
        
        # Handle Hamming Loss
        if 'hamming_loss' in metrics:
            h_loss_str = f"{metrics['hamming_loss']:.4f}"
            if 'hamming_loss_ci' in metrics:
                h_loss_str += f" (95% CI: [{metrics['hamming_loss_ci'][0]:.4f}, {metrics['hamming_loss_ci'][1]:.4f}])"
            print(f"  Hamming Loss: {h_loss_str}")
        
        # Handle Exact Match
        if 'exact_match' in metrics:
            e_match_str = f"{metrics['exact_match']:.4f}"
            if 'exact_match_ci' in metrics:
                e_match_str += f" (95% CI: [{metrics['exact_match_ci'][0]:.4f}, {metrics['exact_match_ci'][1]:.4f}])"
            print(f"  Exact Match: {e_match_str}")
    
    print("\nPer-Class Performance:")
    for class_name, metrics in results.get('per_class', {}).items():
        print(f"\n{class_name}:")
        print(f"  AUC: {metrics.get('auc', 0.0):.4f}")
        print(f"  F1: {metrics.get('f1', 0.0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0.0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0.0):.4f}")
        print(f"  Specificity: {metrics.get('specificity', 0.0):.4f}")
        print(f"  NPV: {metrics.get('npv', 0.0):.4f}")
        print(f"  Threshold: {metrics.get('threshold', 0.0):.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TP: {metrics.get('true_positives', 0)}, FP: {metrics.get('false_positives', 0)}")
        print(f"    FN: {metrics.get('false_negatives', 0)}, TN: {metrics.get('true_negatives', 0)}")
        if 'support' in metrics:
            print(f"  Support: {metrics['support']:,}")

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
    """Calculate metrics for a single bootstrap iteration"""
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
        
        # Hamming Loss with proper handling
        hamming_loss_value = calculate_weighted_hamming_loss(y_true_boot, y_binary_boot)
        metrics['hamming_loss'] = hamming_loss_value if hamming_loss_value is not None else 1.0
        
        # Exact Match
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
    parser.add_argument('--cache_dir', type=str, default='cached_data',
                      help='Directory to store cached tokenized data')
    parser.add_argument('--force_retokenize', action='store_true',
                      help='Force retokenization even if cache exists')
    
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
        'force_retokenize': args.force_retokenize
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
        
        # If force_retokenize, delete existing cache
        if args.force_retokenize and os.path.exists(args.cache_dir):
            print("Forcing retokenization - removing existing cache...")
            import shutil
            shutil.rmtree(args.cache_dir)
        
        # Load test data
        print("\nLoading test dataset...")
        test_df = pd.read_csv(args.test_file)
        print(f"Loaded {len(test_df):,} test samples")
        
        # Create test dataset with caching
        test_dataset = ToxicDataset(
            test_df, 
            tokenizer, 
            cache_dir=args.cache_dir
        )
        
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