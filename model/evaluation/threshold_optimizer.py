# threshold_optimizer.py
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve
import logging

logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    def __init__(self, min_samples=50, class_names=None):
        self.min_samples = min_samples
        self.thresholds = {}
        self.metrics = {}
        self.class_names = class_names or []
        
        # Language-specific threshold bounds
        self.threshold_bounds = {
            'en': {
                'threat': (0.18, 0.35),       # Higher precision needed for EN threats
                'identity_hate': (0.22, 0.4)  # EN has lower prevalence (4.61%)
            },
            'ru': {
                'identity_hate': (0.25, 0.45)  # RU: 5.34% prevalence
            },
            'default': {
                'threat': (0.15, 0.3),
                'identity_hate': (0.2, 0.35),
                'severe_toxic': (0.2, 0.4),    # Default bounds for critical classes
                'toxic': (0.25, 0.75),         # Wider bounds for general toxicity
                'obscene': (0.25, 0.75),
                'insult': (0.25, 0.75)
            }
        }
        
        # Map class names to indices
        self.class_indices = {name: idx for idx, name in enumerate(self.class_names)} if self.class_names else {}
        
    def validate_inputs(self, y_true, y_pred, languages, class_names):
        """Validate input arrays and parameters"""
        # Check for None inputs
        if y_true is None or y_pred is None or languages is None or class_names is None:
            raise ValueError("All inputs must not be None")
        
        # Check shapes
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")
        
        if len(y_true) != len(languages):
            raise ValueError(f"Length mismatch: y_true {len(y_true)} != languages {len(languages)}")
        
        if y_true.shape[1] != len(class_names):
            raise ValueError(f"Number of classes mismatch: y_true {y_true.shape[1]} != class_names {len(class_names)}")
        
        # Validate values
        if not np.all((y_true >= 0) & (y_true <= 1)):
            raise ValueError("y_true contains values outside [0, 1]")
        
        if not np.all((y_pred >= 0) & (y_pred <= 1)):
            raise ValueError("y_pred contains values outside [0, 1]")
        
        # Validate class names
        if len(set(class_names)) != len(class_names):
            raise ValueError("class_names contains duplicates")
        
        if not all(isinstance(name, str) for name in class_names):
            raise ValueError("All class_names must be strings")
        
    def get_threshold_bounds(self, lang: str, class_name: str) -> tuple:
        """Get language and class specific threshold bounds"""
        try:
            # First try language-specific bounds
            if lang in self.threshold_bounds and class_name in self.threshold_bounds[lang]:
                return self.threshold_bounds[lang][class_name]
            
            # Fallback to default bounds
            if class_name in self.threshold_bounds['default']:
                return self.threshold_bounds['default'][class_name]
            
            # Final fallback for unknown classes
            return (0.25, 0.75)
            
        except Exception as e:
            logger.warning(f"Error getting threshold bounds for {lang}/{class_name}: {str(e)}")
            return (0.25, 0.75)
    
    def optimize(self, y_true, y_pred, languages, class_names):
        """Optimize thresholds per language and class using F1 score with TPR boost."""
        try:
            # Validate inputs
            self.validate_inputs(y_true, y_pred, languages, class_names)
            
            self.thresholds = {}
            self.metrics = {}
            self.class_names = class_names
            
            # Group data by language
            lang_indices = defaultdict(list)
            for i, lang in enumerate(languages):
                lang_indices[lang].append(i)
                
            for lang in lang_indices:
                indices = lang_indices[lang]
                if len(indices) < self.min_samples:
                    logger.warning(f"Skipping threshold optimization for {lang} - insufficient samples ({len(indices)} < {self.min_samples})")
                    # Use default thresholds for languages with insufficient samples
                    self.thresholds[lang] = {name: 0.5 for name in class_names}
                    continue
                    
                lang_y_true = y_true[indices]
                lang_y_pred = y_pred[indices]
                
                self.thresholds[lang] = {}
                self.metrics[lang] = {}
                
                for class_idx, class_name in enumerate(class_names):
                    # Get language-specific bounds
                    min_thresh, max_thresh = self.get_threshold_bounds(lang, class_name)
                    
                    # Determine if this is a critical class
                    is_critical = class_name in ['threat', 'identity_hate', 'severe_toxic']
                    MIN_TPR = 0.65 if is_critical else 0.0  # Minimum TPR for critical classes
                    
                    # Get ROC curve points
                    fpr, tpr, thresholds = roc_curve(
                        lang_y_true[:, class_idx],
                        lang_y_pred[:, class_idx]
                    )
                    
                    # Filter thresholds within bounds
                    valid_indices = (thresholds >= min_thresh) & (thresholds <= max_thresh)
                    if not valid_indices.any():
                        logger.warning(f"No thresholds within bounds for {lang}/{class_name}, using closest valid threshold")
                        # Use closest valid threshold
                        if thresholds[0] > max_thresh:
                            valid_indices[0] = True
                        elif thresholds[-1] < min_thresh:
                            valid_indices[-1] = True
                    
                    fpr = fpr[valid_indices]
                    tpr = tpr[valid_indices]
                    thresholds = thresholds[valid_indices]
                    
                    # Calculate F1 scores for valid thresholds
                    best_threshold = None
                    best_score = -float('inf')
                    best_metrics = None
                    
                    for threshold in thresholds:
                        try:
                            y_pred_binary = (lang_y_pred[:, class_idx] >= threshold).astype(int)
                            
                            # Calculate metrics
                            tn, fp, fn, tp = confusion_matrix(
                                lang_y_true[:, class_idx],
                                y_pred_binary,
                                labels=[0, 1]
                            ).ravel()
                            
                            # Calculate F1 and TPR with error handling
                            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
                            
                            # Boost score for critical classes
                            score = f1
                            if is_critical:
                                score += 0.1 * tpr  # TPR boost for critical classes
                            
                            if score > best_score:
                                best_score = score
                                best_threshold = threshold
                                best_metrics = {
                                    'threshold': threshold,
                                    'f1': f1,
                                    'precision': precision,
                                    'recall': tpr,
                                    'tp': tp,
                                    'fp': fp,
                                    'tn': tn,
                                    'fn': fn
                                }
                        except Exception as e:
                            logger.warning(f"Error calculating metrics for threshold {threshold}: {str(e)}")
                            continue
                    
                    # Check if best metrics meet minimum TPR requirement
                    if best_metrics and best_metrics['recall'] < MIN_TPR:
                        logger.warning(f"Fallback to Youden's J for {lang} {class_name} - TPR too low: {best_metrics['recall']:.3f}")
                        
                        # Use Youden's J index as fallback
                        j_scores = tpr - fpr
                        best_idx = np.argmax(j_scores)
                        best_threshold = np.clip(thresholds[best_idx], min_thresh, max_thresh)
                        
                        # Calculate metrics for adjusted threshold
                        y_pred_binary = (lang_y_pred[:, class_idx] >= best_threshold).astype(int)
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            lang_y_true[:, class_idx],
                            y_pred_binary,
                            average='binary',
                            zero_division=0
                        )
                        
                        best_metrics = {
                            'threshold': best_threshold,
                            'f1': f1,
                            'precision': precision,
                            'recall': recall,
                            'tp': -1,  # Mark as fallback
                            'fp': -1,
                            'tn': -1,
                            'fn': -1
                        }
                    
                    self.thresholds[lang][class_name] = best_threshold
                    self.metrics[lang][class_name] = best_metrics
                    
                    # Log optimized thresholds for critical classes
                    if is_critical:
                        logger.info(
                            f"{lang} {class_name} threshold: {best_threshold:.3f} "
                            f"(F1: {best_metrics['f1']:.3f}, TPR: {best_metrics['recall']:.3f})"
                        )
            
            return self.thresholds
            
        except Exception as e:
            logger.error(f"Error in threshold optimization: {str(e)}")
            # Return default thresholds
            return {lang: {cls: 0.5 for cls in class_names} for lang in set(languages)}
        
    def get_threshold(self, language, class_name):
        """Get optimized threshold for a specific language and class."""
        try:
            return self.thresholds.get(language, {}).get(class_name, 0.5)
        except Exception as e:
            logger.warning(f"Error getting threshold for {language}/{class_name}: {str(e)}")
            return 0.5
        
    def get_metrics(self, language, class_name):
        """Get optimization metrics for a specific language and class."""
        try:
            return self.metrics.get(language, {}).get(class_name, None)
        except Exception as e:
            logger.warning(f"Error getting metrics for {language}/{class_name}: {str(e)}")
            return None 