import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    def __init__(self, min_samples=50):
        self.min_samples = min_samples
        self.thresholds = {}
        self.metrics = {}
        
    def optimize(self, y_true, y_pred, languages, class_names):
        """Optimize thresholds per language and class using F1 score with TPR boost."""
        self.thresholds = {}
        self.metrics = {}
        
        # Group data by language
        lang_indices = defaultdict(list)
        for i, lang in enumerate(languages):
            lang_indices[lang].append(i)
            
        for lang in lang_indices:
            indices = lang_indices[lang]
            if len(indices) < self.min_samples:
                logger.warning(f"Skipping threshold optimization for {lang} - insufficient samples ({len(indices)} < {self.min_samples})")
                continue
                
            lang_y_true = y_true[indices]
            lang_y_pred = y_pred[indices]
            
            self.thresholds[lang] = {}
            self.metrics[lang] = {}
            
            for class_idx, class_name in enumerate(class_names):
                # For threat class in English, use a tighter search range
                if lang == 'en' and class_name == 'threat':
                    thresholds = np.linspace(0.25, 0.28, 30)
                else:
                    thresholds = np.linspace(0.1, 0.9, 80)
                    
                best_threshold = 0.5  # Default
                best_score = 0
                best_metrics = None
                
                for threshold in thresholds:
                    y_pred_binary = (lang_y_pred[:, class_idx] >= threshold).astype(int)
                    
                    # Calculate metrics
                    tn, fp, fn, tp = confusion_matrix(
                        lang_y_true[:, class_idx],
                        y_pred_binary,
                        labels=[0, 1]
                    ).ravel()
                    
                    # Calculate F1 and TPR
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
                    
                    # Boost score with TPR to favor higher recall for critical classes
                    score = f1 + (0.1 * tpr if class_name == 'threat' else 0)
                    
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
                
                self.thresholds[lang][class_name] = best_threshold
                self.metrics[lang][class_name] = best_metrics
                
                # Log optimized thresholds for threat class
                if class_name == 'threat':
                    logger.info(f"{lang} threat class threshold: {best_threshold:.3f} (F1: {best_metrics['f1']:.3f}, TPR: {best_metrics['recall']:.3f})")
        
        return self.thresholds
        
    def get_threshold(self, language, class_name):
        """Get optimized threshold for a specific language and class."""
        return self.thresholds.get(language, {}).get(class_name, 0.5)  # Default to 0.5 if not optimized
        
    def get_metrics(self, language, class_name):
        """Get optimization metrics for a specific language and class."""
        return self.metrics.get(language, {}).get(class_name, None) 