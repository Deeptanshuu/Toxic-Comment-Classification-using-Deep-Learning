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
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
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
from sklearn.model_selection import train_test_split
import warnings

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

def calibrate_predictions(model, val_dataset, raw_predictions, labels, langs, device):
    """Calibrate model predictions using isotonic regression with proper data splitting"""
    n_classes = raw_predictions.shape[1]
    calibrated_predictions = np.zeros_like(raw_predictions)
    unique_langs = np.unique(langs)
    
    for class_idx in range(n_classes):
        print(f"\nCalibrating class {class_idx}...")
        for lang in unique_langs:
            lang_mask = langs == lang
            if not lang_mask.any():
                continue
            
            lang_preds = raw_predictions[lang_mask, class_idx]
            lang_labels = labels[lang_mask, class_idx]
            
            if not lang_labels.any():
                print(f"  Skipping language {lang} - no positive samples")
                calibrated_predictions[lang_mask, class_idx] = lang_preds
                continue
            
            try:
                # Split data for calibration
                calib_preds, val_preds, calib_labels, val_labels = train_test_split(
                    lang_preds, lang_labels, test_size=0.3, stratify=lang_labels
                )
                
                # Create a dummy classifier that just returns the input probabilities
                class DummyClassifier:
                    def predict_proba(self, X):
                        return np.vstack([1-X.ravel(), X.ravel()]).T
                    
                    def fit(self, X, y):
                        return self
                
                # Initialize and fit isotonic calibration
                calibrator = CalibratedClassifierCV(
                    estimator=DummyClassifier(),
                    method='isotonic',
                    cv='prefit'
                )
                
                # Reshape predictions for sklearn compatibility
                calib_preds_reshaped = calib_preds.reshape(-1, 1)
                calibrator.fit(calib_preds_reshaped, calib_labels)
                
                # Apply calibration to all predictions for this language/class
                lang_preds_reshaped = lang_preds.reshape(-1, 1)
                calibrated_lang_preds = calibrator.predict_proba(lang_preds_reshaped)[:, 1]
                calibrated_predictions[lang_mask, class_idx] = calibrated_lang_preds
                
                # Validate calibration quality
                val_preds_reshaped = val_preds.reshape(-1, 1)
                prob_true, prob_pred = calibration_curve(
                    val_labels,
                    calibrator.predict_proba(val_preds_reshaped)[:, 1],
                    n_bins=10
                )
                calib_error = np.mean(np.abs(prob_true - prob_pred))
                print(f"  Language {lang} calibration error: {calib_error:.4f}")
                
            except Exception as e:
                print(f"  Warning: Calibration failed for language {lang}, class {class_idx}: {str(e)}")
                calibrated_predictions[lang_mask, class_idx] = lang_preds
    
    return calibrated_predictions

def apply_calibration(raw_predictions, labels, langs, model, val_dataset, device):
    """Apply calibration transformations with proper error handling"""
    try:
        print("\nApplying probability calibration...")
        calibrated_predictions = calibrate_predictions(
            model, val_dataset, raw_predictions, labels, langs, device
        )
        
        # Validate calibration results
        if not np.all(np.isfinite(calibrated_predictions)):
            print("Warning: Invalid values in calibrated predictions")
            return raw_predictions
            
        if not (0 <= calibrated_predictions).all() or not (calibrated_predictions <= 1).all():
            print("Warning: Calibrated predictions outside [0,1] range")
            calibrated_predictions = np.clip(calibrated_predictions, 0, 1)
        
        # Compare calibration with raw predictions using Brier score
        raw_brier = np.mean([
            brier_score_loss(labels[:, i], raw_predictions[:, i])
            for i in range(labels.shape[1])
        ])
        cal_brier = np.mean([
            brier_score_loss(labels[:, i], calibrated_predictions[:, i])
            for i in range(labels.shape[1])
        ])
        
        print(f"\nBrier scores:")
        print(f"  Raw predictions: {raw_brier:.4f}")
        print(f"  Calibrated predictions: {cal_brier:.4f}")
        
        # Only use calibrated predictions if they improve or don't significantly degrade performance
        if cal_brier > raw_brier * 1.1:  # Allow 10% degradation
            print("Warning: Calibration significantly increased Brier score")
            return raw_predictions
        
        return calibrated_predictions
        
    except Exception as e:
        print(f"Warning: Calibration failed: {str(e)}")
        return raw_predictions

def evaluate_language(args):
    """Evaluate model performance for a single language"""
    try:
        lang_id, lang_name, lang_preds, lang_labels, toxicity_types = args
        
        def dynamic_constraints(pos_ratio):
            """Calculate dynamic constraints with better balance between precision and recall
            
            Args:
                pos_ratio: Positive class ratio (0 to 1)
                
            Returns:
                Tuple of (recall_req, precision_req, fpr_limit)
            """
            # Base requirements
            base_precision = 0.85
            base_recall = 0.10
            base_fpr = 0.01
            
            # Dynamic scaling based on class frequency
            precision_req = base_precision - 0.4 * pos_ratio  # Relax precision for rare classes
            recall_req = base_recall + 0.3 * np.sqrt(pos_ratio)  # Increase recall requirement gradually
            fpr_limit = base_fpr + 0.1 * pos_ratio  # Allow slightly higher FPR for common classes
            
            # Ensure reasonable bounds
            precision_req = np.clip(precision_req, 0.60, 0.95)  # Min 60%, max 95% precision
            recall_req = np.clip(recall_req, 0.10, 0.80)       # Min 10%, max 80% recall
            fpr_limit = np.clip(fpr_limit, 0.01, 0.15)         # Min 1%, max 15% FPR
            
            # Print constraint values for debugging
            print(f"    Dynamic constraints for pos_ratio={pos_ratio:.4f}:")
            print(f"      Precision requirement: {precision_req:.4f}")
            print(f"      Recall requirement: {recall_req:.4f}")
            print(f"      FPR limit: {fpr_limit:.4f}")
            
            return recall_req, precision_req, fpr_limit
        
        # Print evaluation header with sample count
        print(f"\nEvaluating {lang_name} [{len(lang_labels):,} samples]")
        
        # Optimize thresholds for this language
        lang_thresholds = {}
        for i, class_name in enumerate(toxicity_types):
            try:
                class_labels = lang_labels[:, i]
                class_preds = lang_preds[:, i]
                
                # Skip if no positive samples
                pos_count = np.sum(class_labels)
                if pos_count == 0:
                    print(f"  {class_name}: No positive samples found")
                    lang_thresholds[class_name] = 0.90  # Higher default threshold for safety
                    continue
                
                pos_ratio = pos_count / len(class_labels)
                print(f"  {class_name}: Found {pos_count:,} positive samples ({pos_ratio:.2%})")
                
                # Calculate dynamic constraints based on class frequency
                min_recall, required_precision, max_fpr = dynamic_constraints(pos_ratio)
                
                # Calculate ROC curve points
                fpr, tpr, thresh = roc_curve(class_labels, class_preds)
                
                # Enhanced class weighting with balanced emphasis
                pos_weight = 1.0 / (pos_ratio + 1e-7)  # Inverse frequency weighting
                pos_weight = np.clip(pos_weight, 1.0, 10.0)  # Limit maximum weight
                neg_weight = 1.0
                
                print(f"    Class weights: positive={pos_weight:.2f}, negative={neg_weight:.2f}")
                
                # Create sample weights
                sample_weights = np.where(class_labels == 1, pos_weight, neg_weight)
                sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
                
                # Calculate metrics for all thresholds
                metrics_by_threshold = []
                for t in thresh:
                    binary_preds = (class_preds > t).astype(int)
                    
                    # Skip if no predictions made
                    if not binary_preds.any():
                        continue
                    
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
                        
                        # Calculate confusion matrix for FPR
                        tn, fp, fn, tp = confusion_matrix(
                            class_labels, binary_preds,
                            sample_weight=sample_weights
                        ).ravel()
                        
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 1.0
                        
                        # Calculate calibration error at this threshold
                        pred_probs = class_preds[binary_preds == 1].mean() if binary_preds.any() else 0
                        actual_ratio = precision if precision > 0 else 0
                        calibration_error = abs(pred_probs - actual_ratio)
                        
                        metrics_by_threshold.append({
                            'threshold': t,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'fpr': fpr,
                            'support': pos_count,
                            'calibration_error': calibration_error
                        })
                    except Exception as e:
                        continue
                
                if not metrics_by_threshold:
                    print(f"  Warning: No valid thresholds found for {class_name}")
                    lang_thresholds[class_name] = 0.90  # Higher default threshold
                    continue
                
                # Find valid thresholds using dynamic constraints
                valid_thresholds = [
                    m for m in metrics_by_threshold 
                    if m['precision'] >= required_precision and 
                       m['recall'] >= min_recall and
                       m['fpr'] <= max_fpr
                ]
                
                if valid_thresholds:
                    # Enhanced metric weighting with balanced focus
                    weights = {
                        'precision': 0.4 + 0.2 * (1 - pos_ratio),  # Higher weight for rare classes
                        'recall': 0.3 + 0.2 * pos_ratio,          # Higher weight for common classes
                        'f1': 0.2,                                # Consistent F1 importance
                        'fpr': 0.1 * (1 - pos_ratio),            # FPR penalty for rare classes
                        'calibration': 0.1                        # Consistent calibration importance
                    }
                    
                    # Normalize weights
                    weight_sum = sum(weights.values())
                    weights = {k: v/weight_sum for k, v in weights.items()}
                    
                    print(f"    Metric weights: {weights}")
                    
                    # Select best threshold with balanced consideration
                    best_threshold = max(
                        valid_thresholds,
                        key=lambda x: (
                            x['precision'] * weights['precision'] +
                            x['recall'] * weights['recall'] +
                            x['f1'] * weights['f1'] -
                            x['fpr'] * weights['fpr'] -
                            x['calibration_error'] * weights['calibration']
                        )
                    )
                    
                    lang_thresholds[class_name] = best_threshold['threshold']
                    achieved_metrics = best_threshold
                    
                else:
                    print(f"  Warning: No thresholds meet criteria for {class_name}")
                    print("  Falling back to balanced threshold selection")
                    
                    # Find threshold with balanced precision-recall trade-off
                    best_threshold = max(
                        metrics_by_threshold,
                        key=lambda x: (
                            x['f1'] * 0.4 +                    # Emphasize F1 score
                            x['precision'] * 0.3 +             # Maintain reasonable precision
                            x['recall'] * 0.2 -                # Consider recall
                            x['fpr'] * 0.1 -                   # Limit false positives
                            x['calibration_error'] * 0.1       # Consider calibration
                        )
                    )
                    
                    lang_thresholds[class_name] = best_threshold['threshold']
                    achieved_metrics = best_threshold
                
                # Log detailed metrics
                print(f"  {class_name} results:")
                print(f"    Threshold: {lang_thresholds[class_name]:.3f}")
                print(f"    Precision: {achieved_metrics['precision']:.3f}")
                print(f"    Recall: {achieved_metrics['recall']:.3f}")
                print(f"    F1: {achieved_metrics['f1']:.3f}")
                print(f"    FPR: {achieved_metrics['fpr']:.3f}")
                print(f"    Calibration Error: {achieved_metrics['calibration_error']:.3f}")
                print(f"    Support: {achieved_metrics['support']:,} samples")
                
            except Exception as e:
                print(f"  Error evaluating {class_name}: {str(e)}")
                lang_thresholds[class_name] = 0.90  # Conservative default
        
        return lang_id, lang_thresholds
        
    except Exception as e:
        print(f"Error in evaluate_language: {str(e)}")
        return None

def evaluate_model(model, val_loader, device, output_dir):
    """Evaluate model performance on validation set with proper calibration"""
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
    
    # Apply calibration with proper validation split
    calibrated_predictions = apply_calibration(
        predictions, labels, langs, model, val_loader.dataset, device
    )
    
    # Calculate metrics and save results
    results = calculate_metrics(
        predictions, calibrated_predictions, labels, langs
    )
    
    # Save evaluation results
    save_results(
        results=results,
        raw_predictions=predictions,
        calibrated_predictions=calibrated_predictions,
        labels=labels,
        langs=langs,
        output_dir=output_dir
    )
    
    # Plot calibration curves
    plot_calibration_curves(
        y_true=labels,
        y_pred_raw=predictions,
        y_pred_calibrated=calibrated_predictions,
        output_dir=output_dir,
        toxicity_types=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
        languages={
            0: 'English',
            1: 'Russian',
            2: 'Turkish',
            3: 'Spanish',
            4: 'French',
            5: 'Italian',
            6: 'Portuguese'
        },
        langs=langs
    )
    
    return results, predictions, calibrated_predictions

def calculate_metrics(predictions, calibrated_predictions, labels, langs):
    """Calculate detailed metrics using class-specific thresholds with both macro and weighted averages
    
    Args:
        predictions: Raw model predictions
        calibrated_predictions: Calibrated model predictions
        labels: True labels
        langs: Language IDs for each sample
    """
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
    
    # Calculate both macro and weighted AUC for raw predictions
    results['overall'].update({
        'auc_macro': roc_auc_score(labels, predictions, average='macro'),
        'auc_weighted': roc_auc_score(labels, predictions, average='weighted')
    })
    
    # Binary predictions using dynamic thresholds for raw predictions
    threshold_array = np.array([thresholds[ct] for ct in toxicity_types])
    binary_predictions = (predictions > threshold_array).astype(int)
    calibrated_binary = (calibrated_predictions > threshold_array).astype(int)
    
    # Calculate metrics for raw predictions
    precision_macro, recall_macro, f1_macro, support_macro = precision_recall_fscore_support(
        labels, binary_predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, support_weighted = precision_recall_fscore_support(
        labels, binary_predictions, average='weighted', zero_division=0
    )
    
    # Store both macro and weighted metrics for raw predictions
    results['overall'].update({
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    })
    
    # Calculate metrics for calibrated predictions
    cal_precision_macro, cal_recall_macro, cal_f1_macro, _ = precision_recall_fscore_support(
        labels, calibrated_binary, average='macro', zero_division=0
    )
    cal_precision_weighted, cal_recall_weighted, cal_f1_weighted, _ = precision_recall_fscore_support(
        labels, calibrated_binary, average='weighted', zero_division=0
    )
    
    # Store calibrated metrics
    results['overall'].update({
        'calibrated_precision_macro': cal_precision_macro,
        'calibrated_precision_weighted': cal_precision_weighted,
        'calibrated_recall_macro': cal_recall_macro,
        'calibrated_recall_weighted': cal_recall_weighted,
        'calibrated_f1_macro': cal_f1_macro,
        'calibrated_f1_weighted': cal_f1_weighted
    })
    
    # Calculate per-class metrics for both raw and calibrated predictions
    for i, class_name in enumerate(toxicity_types):
        class_labels = labels[:, i]
        class_preds = predictions[:, i]
        class_binary = binary_predictions[:, i]
        cal_preds = calibrated_predictions[:, i]
        cal_binary = calibrated_binary[:, i]
        
        # Calculate metrics for this class
        try:
            class_metrics = {
                'raw': {
                    'auc': roc_auc_score(class_labels, class_preds),
                    'precision': precision_score(class_labels, class_binary, zero_division=0),
                    'recall': recall_score(class_labels, class_binary, zero_division=0),
                    'f1': f1_score(class_labels, class_binary, zero_division=0),
                    'support': np.sum(class_labels),
                    'threshold': thresholds[class_name]
                },
                'calibrated': {
                    'auc': roc_auc_score(class_labels, cal_preds),
                    'precision': precision_score(class_labels, cal_binary, zero_division=0),
                    'recall': recall_score(class_labels, cal_binary, zero_division=0),
                    'f1': f1_score(class_labels, cal_binary, zero_division=0)
                }
            }
            
            # Calculate confusion matrix for proper specificity
            tn, fp, fn, tp = confusion_matrix(class_labels, class_binary).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            class_metrics['raw']['specificity'] = specificity
            
            # Add confusion matrix metrics
            class_metrics['raw'].update({
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            })
            
            # Calculate calibrated confusion matrix
            cal_tn, cal_fp, cal_fn, cal_tp = confusion_matrix(class_labels, cal_binary).ravel()
            cal_specificity = cal_tn / (cal_tn + cal_fp) if (cal_tn + cal_fp) > 0 else 0.0
            class_metrics['calibrated']['specificity'] = cal_specificity
            
            class_metrics['calibrated'].update({
                'true_positives': int(cal_tp),
                'false_positives': int(cal_fp),
                'true_negatives': int(cal_tn),
                'false_negatives': int(cal_fn)
            })
            
            results['per_class'][class_name] = class_metrics
            
        except Exception as e:
            print(f"Warning: Could not calculate metrics for class {class_name}: {str(e)}")
            results['per_class'][class_name] = {
                'raw': {
                    'auc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                    'specificity': 0.0, 'support': 0, 'threshold': thresholds[class_name],
                    'true_positives': 0, 'false_positives': 0,
                    'true_negatives': 0, 'false_negatives': 0
                },
                'calibrated': {
                    'auc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                    'specificity': 0.0, 'true_positives': 0, 'false_positives': 0,
                    'true_negatives': 0, 'false_negatives': 0
                }
            }
    
    # Calculate overall specificity (both macro and weighted)
    specificities = []
    weighted_specificities = []
    class_weights = []
    
    for i, class_name in enumerate(toxicity_types):
        try:
            tn, fp, fn, tp = confusion_matrix(
                labels[:, i],
                binary_predictions[:, i]
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
    
    # Add macro and weighted specificity to results
    results['overall'].update({
        'specificity_macro': np.mean(specificities),
        'specificity_weighted': (
            np.sum(weighted_specificities) / np.sum(class_weights)
            if np.sum(class_weights) > 0 else 0.0
        )
    })
    
    # Per-language metrics with proper handling of rare classes
    for lang in unique_langs:
        lang_mask = langs == lang
        if lang_mask.sum() > 0:
            try:
                # Calculate metrics for this language
                lang_labels = labels[lang_mask]
                lang_preds = predictions[lang_mask]
                lang_binary = binary_predictions[lang_mask]
                cal_preds = calibrated_predictions[lang_mask]
                cal_binary = calibrated_binary[lang_mask]
                
                # Calculate both macro and weighted metrics for this language
                lang_metrics = {}
                
                # Raw predictions metrics
                lang_metrics['raw'] = {}
                
                # AUC scores
                lang_metrics['raw']['auc_macro'] = roc_auc_score(
                    lang_labels, lang_preds, average='macro'
                )
                lang_metrics['raw']['auc_weighted'] = roc_auc_score(
                    lang_labels, lang_preds, average='weighted'
                )
                
                # Precision, recall, F1
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    lang_labels, lang_binary, average='macro', zero_division=0
                )
                precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                    lang_labels, lang_binary, average='weighted', zero_division=0
                )
                
                lang_metrics['raw'].update({
                    'precision_macro': precision_macro,
                    'precision_weighted': precision_weighted,
                    'recall_macro': recall_macro,
                    'recall_weighted': recall_weighted,
                    'f1_macro': f1_macro,
                    'f1_weighted': f1_weighted
                })
                
                # Calibrated predictions metrics
                lang_metrics['calibrated'] = {}
                
                # AUC scores for calibrated predictions
                lang_metrics['calibrated']['auc_macro'] = roc_auc_score(
                    lang_labels, cal_preds, average='macro'
                )
                lang_metrics['calibrated']['auc_weighted'] = roc_auc_score(
                    lang_labels, cal_preds, average='weighted'
                )
                
                # Precision, recall, F1 for calibrated predictions
                cal_precision_macro, cal_recall_macro, cal_f1_macro, _ = precision_recall_fscore_support(
                    lang_labels, cal_binary, average='macro', zero_division=0
                )
                cal_precision_weighted, cal_recall_weighted, cal_f1_weighted, _ = precision_recall_fscore_support(
                    lang_labels, cal_binary, average='weighted', zero_division=0
                )
                
                lang_metrics['calibrated'].update({
                    'precision_macro': cal_precision_macro,
                    'precision_weighted': cal_precision_weighted,
                    'recall_macro': cal_recall_macro,
                    'recall_weighted': cal_recall_weighted,
                    'f1_macro': cal_f1_macro,
                    'f1_weighted': cal_f1_weighted
                })
                
                # Add sample count
                lang_metrics['sample_count'] = int(lang_mask.sum())
                
                # Calculate per-class metrics for this language
                lang_metrics['per_class'] = {}
                for i, class_name in enumerate(toxicity_types):
                    class_labels = lang_labels[:, i]
                    class_preds = lang_preds[:, i]
                    class_binary = lang_binary[:, i]
                    cal_preds = cal_preds[:, i]
                    cal_binary = cal_binary[:, i]
                    
                    try:
                        tn, fp, fn, tp = confusion_matrix(
                            class_labels, class_binary
                        ).ravel()
                        
                        cal_tn, cal_fp, cal_fn, cal_tp = confusion_matrix(
                            class_labels, cal_binary
                        ).ravel()
                        
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        cal_specificity = cal_tn / (cal_tn + cal_fp) if (cal_tn + cal_fp) > 0 else 0.0
                        support = np.sum(class_labels)
                        
                        lang_metrics['per_class'][class_name] = {
                            'raw': {
                                'precision': precision_score(class_labels, class_binary, zero_division=0),
                                'recall': recall_score(class_labels, class_binary, zero_division=0),
                                'f1': f1_score(class_labels, class_binary, zero_division=0),
                                'specificity': specificity,
                                'support': int(support),
                                'true_positives': int(tp),
                                'false_positives': int(fp),
                                'true_negatives': int(tn),
                                'false_negatives': int(fn)
                            },
                            'calibrated': {
                                'precision': precision_score(class_labels, cal_binary, zero_division=0),
                                'recall': recall_score(class_labels, cal_binary, zero_division=0),
                                'f1': f1_score(class_labels, cal_binary, zero_division=0),
                                'specificity': cal_specificity,
                                'true_positives': int(cal_tp),
                                'false_positives': int(cal_fp),
                                'true_negatives': int(cal_tn),
                                'false_negatives': int(cal_fn)
                            }
                        }
                    except Exception as e:
                        print(f"Warning: Could not calculate metrics for {class_name} in language {lang}: {str(e)}")
                
                results['per_language'][str(lang)] = lang_metrics
                
            except Exception as e:
                print(f"Warning: Could not calculate metrics for language {lang}: {str(e)}")
                
    return results

def bootstrap_sample(y_true, y_pred):
    """Preserve label relationships during bootstrap sampling for multi-label data
    
    Args:
        y_true: True labels (n_samples, n_classes) or (n_samples,)
        y_pred: Predicted probabilities (n_samples, n_classes) or (n_samples,)
        
    Returns:
        Tuple of bootstrapped true labels and predictions with preserved correlations
    """
    n_samples = len(y_true)
    
    # Handle single-label case
    if len(y_true.shape) == 1:
        # Check if we have any positive samples to stratify on
        if len(np.unique(y_true)) > 1:
            # If we have both positive and negative samples, use stratified sampling
            idx = resample(np.arange(n_samples), stratify=y_true)
        else:
            # If all samples are the same, use regular bootstrap
            idx = resample(np.arange(n_samples))
        return y_true[idx], y_pred[idx]
    
    # Multi-label case: preserve label correlations
    # Create label pattern groups to maintain relationships
    label_patterns = np.apply_along_axis(
        lambda x: ''.join(x.astype(str)), 
        axis=1, 
        arr=y_true
    )
    unique_patterns, pattern_counts = np.unique(label_patterns, return_counts=True)
    
    # Initialize arrays for stratified sampling
    sampled_indices = []
    target_counts = {}
    
    # Calculate target counts for each pattern to maintain class distribution
    for pattern, count in zip(unique_patterns, pattern_counts):
        # Use Poisson distribution to determine number of samples
        # This adds some randomness while maintaining approximate proportions
        target_count = np.random.poisson(count)
        target_counts[pattern] = max(1, target_count)  # Ensure at least one sample
    
    # Perform stratified sampling for each pattern
    for pattern in unique_patterns:
        pattern_mask = label_patterns == pattern
        pattern_indices = np.where(pattern_mask)[0]
        
        # Sample with replacement from this pattern group
        if len(pattern_indices) > 0:
            sampled_pattern_indices = resample(
                pattern_indices,
                replace=True,
                n_samples=target_counts[pattern]
            )
            sampled_indices.extend(sampled_pattern_indices)
    
    # Convert to numpy array and shuffle
    sampled_indices = np.array(sampled_indices)
    np.random.shuffle(sampled_indices)
    
    # Ensure we maintain original sample size
    if len(sampled_indices) > n_samples:
        sampled_indices = sampled_indices[:n_samples]
    elif len(sampled_indices) < n_samples:
        # Add random samples to reach original size if needed
        additional_samples = resample(
            np.arange(n_samples),
            n_samples=n_samples - len(sampled_indices)
        )
        sampled_indices = np.concatenate([sampled_indices, additional_samples])
    
    return y_true[sampled_indices], y_pred[sampled_indices]

def calculate_class_weights(labels, langs=None):
    """Calculate balanced class weights with focal weighting and per-language awareness
    
    Args:
        labels: Label matrix (n_samples, n_classes)
        langs: Optional language IDs for per-language weighting
    """
    # Define focal loss parameters
    gamma = 2.0  # Focusing parameter for hard examples
    
    # Class-specific alpha values for balancing positive/negative samples
    alpha = {
        'toxic': 0.1,        # Common class, lower alpha
        'severe_toxic': 0.7,  # Rare class, higher alpha
        'obscene': 0.3,      # Moderately common
        'threat': 0.8,       # Very rare class, highest alpha
        'insult': 0.4,       # Moderately common
        'identity_hate': 0.6  # Rare class
    }
    
    # Default alpha for unknown classes
    default_alpha = 0.25
    
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    class_weights = {}
    
    # Calculate global class weights
    for i, class_name in enumerate(toxicity_types):
        try:
            # Calculate positive ratio
            pos_count = np.sum(labels[:, i])
            total_samples = len(labels)
            pos_ratio = pos_count / total_samples if total_samples > 0 else 0
            
            # Get class-specific alpha value
            class_alpha = alpha.get(class_name, default_alpha)
            
            # Calculate focal weights
            pos_weight = class_alpha * (1 - np.clip(pos_ratio, 0.01, 0.99))**gamma
            neg_weight = (1 - class_alpha) * (np.clip(pos_ratio, 0.01, 0.99))**gamma
            
            # Ensure minimum weight values
            pos_weight = max(pos_weight, 0.1)
            neg_weight = max(neg_weight, 0.1)
            
            # Store weights
            class_weights[i] = {
                1: float(pos_weight),
                0: float(neg_weight)
            }
            
            # Per-language weighting if languages provided
            if langs is not None:
                lang_weights = {}
                unique_langs = np.unique(langs)
                
                for lang in unique_langs:
                    lang_mask = langs == lang
                    if not lang_mask.any():
                        continue
                    
                    # Calculate language-specific positive ratio
                    lang_pos_count = np.sum(labels[lang_mask, i])
                    lang_total = np.sum(lang_mask)
                    lang_pos_ratio = lang_pos_count / lang_total if lang_total > 0 else 0
                    
                    # Language-specific focal weights
                    lang_pos_weight = class_alpha * (1 - np.clip(lang_pos_ratio, 0.01, 0.99))**gamma
                    lang_neg_weight = (1 - class_alpha) * (np.clip(lang_pos_ratio, 0.01, 0.99))**gamma
                    
                    # Ensure minimum weights
                    lang_pos_weight = max(lang_pos_weight, 0.1)
                    lang_neg_weight = max(lang_neg_weight, 0.1)
                    
                    # Store language-specific weights
                    lang_weights[int(lang)] = {
                        1: float(lang_pos_weight),
                        0: float(lang_neg_weight)
                    }
                
                # Add language-specific weights
                class_weights[i] = {
                    'global': class_weights[i],
                    'per_language': lang_weights
                }
            
            # Print weight information for debugging
            print(f"\nClass weights for {class_name}:")
            print(f"  Positive ratio: {pos_ratio:.4f}")
            print(f"  Alpha: {class_alpha:.4f}")
            print(f"  Positive weight: {pos_weight:.4f}")
            print(f"  Negative weight: {neg_weight:.4f}")
            
        except Exception as e:
            print(f"Warning: Error in class weight calculation for {class_name}: {str(e)}")
            # Fallback to balanced weights
            class_weights[i] = {0: 1.0, 1: 1.0}
    
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

def calculate_language_metrics(labels, predictions, binary_predictions, langs=None):
    """Calculate detailed metrics for a specific language with parallel processing"""
    if len(labels) == 0:
        raise ValueError("No samples available for metric calculation")
    
    # Calculate class weights with language awareness
    try:
        class_weights = calculate_class_weights(labels, langs)
    except Exception as e:
        print(f"Warning: Using default weights due to error in class weight calculation: {str(e)}")
        class_weights = {i: {0: 1.0, 1: 1.0} for i in range(labels.shape[1])}
    
    # Calculate weighted metrics with language-specific weights
    sample_weights = np.ones(len(labels))
    for i in range(labels.shape[1]):
        label_indices = labels[:, i].astype(int)
        
        if isinstance(class_weights[i], dict) and 'per_language' in class_weights[i]:
            # Use language-specific weights
            for j, (label, lang) in enumerate(zip(label_indices, langs)):
                lang_weights = class_weights[i]['per_language'].get(int(lang), class_weights[i]['global'])
                sample_weights[j] *= lang_weights.get(label, 1.0)
        else:
            # Use global weights
            weights = class_weights[i]
            valid_indices = np.isin(label_indices, list(weights.keys()))
            if not valid_indices.all():
                print(f"Warning: Invalid label indices found for class {i}")
                continue
            sample_weights *= np.array([weights.get(idx, 1.0) for idx in label_indices])
    
    # Normalize and clip weights
    sample_weights = np.clip(sample_weights, 0.1, 10.0)
    sample_weights /= sample_weights.sum()
    sample_weights *= len(sample_weights)
    
    # Rest of the function remains unchanged
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
    
    # Add class weight information to metrics
    metrics['class_weights'] = class_weights
    
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
    
    # Helper function to check if value is numeric and not None
    def is_valid_number(v):
        if v is None:
            return False
        try:
            float_val = float(v)
            return not np.isnan(float_val)
        except (TypeError, ValueError):
            return False
    
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
            valid_values = [v for v in values if is_valid_number(v)]
            if valid_values:  # Only plot if we have valid values
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
                # Get metric value with default of 0.0 if None or invalid
                metric_value = results['per_language'][lang].get(metric, 0.0)
                if not is_valid_number(metric_value):
                    metric_value = 0.0
                
                # Get confidence interval with defaults if None or invalid
                metric_ci = results['per_language'][lang].get(f'{metric}_ci', [metric_value, metric_value])
                if metric_ci is None or not all(is_valid_number(v) for v in metric_ci):
                    metric_ci = [metric_value, metric_value]
                
                values.append(float(metric_value))
                # Calculate error bars, ensuring no None values
                lower_error = float(metric_value) - float(metric_ci[0])
                upper_error = float(metric_ci[1]) - float(metric_value)
                errors.append((lower_error, upper_error))
            
            # Convert errors to numpy array for proper plotting
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
                if is_valid_number(value):
                    lang_values.append(float(value))
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
    
    # Performance Distribution Plots
    plt.figure(figsize=(15, 5))
    
    # Create subplot for metric distributions
    plt.subplot(131)
    has_metric_data = False
    for metric in ['auc', 'f1', 'precision', 'recall']:
        values = [results['per_language'][lang].get(metric, np.nan) for lang in languages]
        valid_values = [float(v) for v in values if is_valid_number(v)]
        if valid_values:
            sns.kdeplot(data=valid_values, label=metric.upper())
            has_metric_data = True
    
    if has_metric_data:
        plt.title('Distribution of Metrics across Languages')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend()
    
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

def save_results(results, raw_predictions, calibrated_predictions, labels, langs, output_dir):
    """Save evaluation results and plots
    
    Args:
        results: Dictionary containing evaluation results
        raw_predictions: Raw model predictions
        calibrated_predictions: Calibrated model predictions
        labels: True labels
        langs: Language IDs
        output_dir: Directory to save results
    """
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
    
    # Create summary dictionaries for raw and calibrated predictions
    summary = {
        'raw': {
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
        },
        'calibrated': {
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
    }
    
    # Add summaries to results
    results['raw'] = summary['raw']
    results['calibrated'] = summary['calibrated']
    
    # Convert results to JSON serializable format
    serializable_results = convert_to_serializable(results)
    
    # Save detailed metrics
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save raw and calibrated predictions for further analysis
    np.savez_compressed(
        os.path.join(output_dir, 'predictions.npz'),
        raw_predictions=raw_predictions,
        calibrated_predictions=calibrated_predictions,
        labels=labels,
        langs=langs
    )
    
    # Plot confusion matrices for both raw and calibrated predictions
    plot_confusion_matrices(raw_predictions, labels, langs, 
                          os.path.join(output_dir, 'raw_predictions'))
    plot_confusion_matrices(calibrated_predictions, labels, langs, 
                          os.path.join(output_dir, 'calibrated_predictions'))
    
    # Print detailed summary
    print("\nEvaluation Results:")
    print("-" * 80)
    
    for pred_type in ['raw', 'calibrated']:
        print(f"\n{pred_type.upper()} PREDICTIONS:")
        print("-" * 40)
        
        metrics = summary[pred_type]
        
        print("\nAUC Scores:")
        print(f"  Macro-averaged: {metrics['auc']['macro']:.4f}")
        print(f"  Weighted-averaged: {metrics['auc']['weighted']:.4f}")
        
        print("\nF1 Scores:")
        print(f"  Macro-averaged: {metrics['f1']['macro']:.4f}")
        print(f"  Weighted-averaged: {metrics['f1']['weighted']:.4f}")
        
        print("\nPrecision:")
        print(f"  Macro-averaged: {metrics['precision']['macro']:.4f}")
        print(f"  Weighted-averaged: {metrics['precision']['weighted']:.4f}")
        
        print("\nRecall:")
        print(f"  Macro-averaged: {metrics['recall']['macro']:.4f}")
        print(f"  Weighted-averaged: {metrics['recall']['weighted']:.4f}")
        
        print("\nSpecificity:")
        print(f"  Macro-averaged: {metrics['specificity']['macro']:.4f}")
        print(f"  Weighted-averaged: {metrics['specificity']['weighted']:.4f}")
        
        print("\nOther Metrics:")
        print(f"  Exact Match: {metrics['other_metrics']['exact_match']:.4f}")
        
        if metrics['class_support']:
            print("\nClass Support (number of samples):")
            for class_name, support in metrics['class_support'].items():
                print(f"  {class_name}: {support:,}")
    
    print("\nPer-Language Performance:")
    for lang_id, metrics in results.get('per_language', {}).items():
        lang_name = id_to_lang.get(int(lang_id), f'Unknown ({lang_id})')
        print(f"\n{lang_name} (n={metrics.get('sample_count', 0)}):")
        
        # Print AUC with CI if available
        if 'auc' in metrics and metrics['auc'] is not None:
            auc_str = f"{metrics['auc']:.4f}"
            if 'auc_ci' in metrics:
                auc_str += f" (95% CI: [{metrics['auc_ci'][0]:.4f}, {metrics['auc_ci'][1]:.4f}])"
            print(f"  AUC: {auc_str}")
        else:
            print("  AUC: N/A")
        
        # Print F1 with CI if available
        if 'f1' in metrics and metrics['f1'] is not None:
            f1_str = f"{metrics['f1']:.4f}"
            if 'f1_ci' in metrics:
                f1_str += f" (95% CI: [{metrics['f1_ci'][0]:.4f}, {metrics['f1_ci'][1]:.4f}])"
            print(f"  F1: {f1_str}")
        else:
            print("  F1: N/A")
        
        # Handle Hamming Loss
        if 'hamming_loss' in metrics and metrics['hamming_loss'] is not None:
            h_loss_str = f"{metrics['hamming_loss']:.4f}"
            if 'hamming_loss_ci' in metrics:
                h_loss_str += f" (95% CI: [{metrics['hamming_loss_ci'][0]:.4f}, {metrics['hamming_loss_ci'][1]:.4f}])"
            print(f"  Hamming Loss: {h_loss_str}")
        else:
            print("  Hamming Loss: N/A")
        
        # Handle Exact Match
        if 'exact_match' in metrics and metrics['exact_match'] is not None:
            e_match_str = f"{metrics['exact_match']:.4f}"
            if 'exact_match_ci' in metrics:
                e_match_str += f" (95% CI: [{metrics['exact_match_ci'][0]:.4f}, {metrics['exact_match_ci'][1]:.4f}])"
            print(f"  Exact Match: {e_match_str}")
        else:
            print("  Exact Match: N/A")
    
    print("\nPer-Class Performance:")
    for class_name, metrics in results.get('per_class', {}).items():
        print(f"\n{class_name}:")
        
        # Print metrics with None checks
        def format_metric(name, format_str='.4f'):
            value = metrics.get(name)
            return f"{value:{format_str}}" if value is not None else "N/A"
        
        print(f"  AUC: {format_metric('auc')}")
        print(f"  F1: {format_metric('f1')}")
        print(f"  Precision: {format_metric('precision')}")
        print(f"  Recall: {format_metric('recall')}")
        print(f"  Specificity: {format_metric('specificity')}")
        print(f"  NPV: {format_metric('npv')}")
        print(f"  Threshold: {format_metric('threshold')}")
        print(f"  Confusion Matrix:")
        print(f"    TP: {metrics.get('true_positives', 0)}, FP: {metrics.get('false_positives', 0)}")
        print(f"    FN: {metrics.get('false_negatives', 0)}, TN: {metrics.get('true_negatives', 0)}")
        if 'support' in metrics:
            print(f"  Support: {metrics['support']:,}")

def plot_calibration_curves(y_true, y_pred_raw, y_pred_calibrated, output_dir, toxicity_types=None, languages=None, langs=None):
    """Plot calibration curves comparing raw and calibrated predictions
    
    Args:
        y_true: True labels (n_samples, n_classes)
        y_pred_raw: Raw predicted probabilities (n_samples, n_classes)
        y_pred_calibrated: Calibrated predicted probabilities (n_samples, n_classes)
        output_dir: Directory to save plots
        toxicity_types: List of toxicity class names
        languages: Dictionary mapping language IDs to names
        langs: Array of language IDs for each sample
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Overall calibration curve
    plt.figure(figsize=(10, 6))
    
    # Plot raw predictions
    prob_true_raw, prob_pred_raw = calibration_curve(
        y_true.flatten(),
        y_pred_raw.flatten(),
        n_bins=10,
        strategy='quantile'
    )
    plt.plot(prob_pred_raw, prob_true_raw, marker='o', label="Raw Predictions", color='red', alpha=0.7)
    
    # Plot calibrated predictions
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_true.flatten(),
        y_pred_calibrated.flatten(),
        n_bins=10,
        strategy='quantile'
    )
    plt.plot(prob_pred_cal, prob_true_cal, marker='s', label="Calibrated", color='blue', alpha=0.7)
    
    # Add perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration", color='gray')
    
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
            
            # Raw predictions
            prob_true_raw, prob_pred_raw = calibration_curve(
                y_true[:, i],
                y_pred_raw[:, i],
                n_bins=10,
                strategy='quantile'
            )
            plt.plot(prob_pred_raw, prob_true_raw, marker='o', label="Raw", color='red', alpha=0.7)
            
            # Calibrated predictions
            prob_true_cal, prob_pred_cal = calibration_curve(
                y_true[:, i],
                y_pred_calibrated[:, i],
                n_bins=10,
                strategy='quantile'
            )
            plt.plot(prob_pred_cal, prob_true_cal, marker='s', label="Calibrated", color='blue', alpha=0.7)
            
            # Perfect calibration line
            plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect", color='gray')
            
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
                        # Raw predictions
                        prob_true_raw, prob_pred_raw = calibration_curve(
                            y_true[lang_mask, i],
                            y_pred_raw[lang_mask, i],
                            n_bins=10,
                            strategy='quantile'
                        )
                        plt.plot(prob_pred_raw, prob_true_raw, marker='o', label="Raw", color='red', alpha=0.7)
                        
                        # Calibrated predictions
                        prob_true_cal, prob_pred_cal = calibration_curve(
                            y_true[lang_mask, i],
                            y_pred_calibrated[lang_mask, i],
                            n_bins=10,
                            strategy='quantile'
                        )
                        plt.plot(prob_pred_cal, prob_true_cal, marker='s', label="Calibrated", color='blue', alpha=0.7)
                        
                        # Perfect calibration line
                        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect", color='gray')
                        
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
        
        # If force_retokenize, delete existing cache
        if args.force_retokenize and os.path.exists(args.cache_dir):
            print("Forcing retokenization - removing existing cache...")
            import shutil
            shutil.rmtree(args.cache_dir)
        
        # Load test data
        print("\nLoading test dataset...")
        test_df = pd.read_csv(args.test_file)
        print(f"Loaded {len(test_df):,} test samples")
        
        # Print test dataset information
        print("\nTest dataset columns:", test_df.columns.tolist())
        print("\nTest dataset info:")
        print(test_df.info())
        
        if 'lang' in test_df.columns:
            print("\nLanguage distribution in test set:")
            print(test_df['lang'].value_counts())
        else:
            print("\nWarning: No language column in test dataset")
            print("Adding 'lang' column with default value 'en'")
            test_df['lang'] = 'en'
        
        # Create test dataset with caching
        test_dataset = ToxicDataset(
            test_df, 
            tokenizer, 
            cache_dir=args.cache_dir
        )
        
        # Configure DataLoader with optimized settings
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