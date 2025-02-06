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
                    # Determine search range based on class characteristics
                    is_critical = class_name in ['threat', 'identity_hate', 'severe_toxic']
                    
                    if is_critical and lang == 'en':
                        thresholds = np.linspace(0.25, 0.35, 40)  # Finer search for critical classes
                    else:
                        thresholds = np.linspace(0.1, 0.9, 80)
                        
                    best_threshold = 0.5  # Default
                    best_score = 0
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
                    
                    self.thresholds[lang][class_name] = best_threshold
                    self.metrics[lang][class_name] = best_metrics
                    
                    # Log optimized thresholds for critical classes
                    if is_critical:
                        logger.info(f"{lang} {class_name} threshold: {best_threshold:.3f} (F1: {best_metrics['f1']:.3f}, TPR: {best_metrics['recall']:.3f})")
            
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