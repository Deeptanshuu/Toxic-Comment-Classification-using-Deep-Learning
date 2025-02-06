from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import torch
import numpy as np
from pathlib import Path
import os
import warnings

@dataclass
class DynamicClassWeights:
    """Handles dynamic class weights per language with improved calculation"""
    weights_file: str = 'weights/language_class_weights.json'
    
    def __post_init__(self):
        try:
            self.load_weights()
            self.enforce_language_priority()
        except Exception as e:
            print(f"Warning: Could not initialize class weights: {str(e)}")
            print("Using default weights...")
            self._initialize_default_weights()
    
    def _initialize_default_weights(self):
        """Initialize default weights if loading fails"""
        self.toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.language_columns = ['en', 'es', 'fr', 'it', 'tr', 'pt', 'ru']
        
        # Create default weights
        self.weights = {}
        for lang in self.language_columns:
            self.weights[lang] = {}
            for col in self.toxicity_columns:
                self.weights[lang][col] = {'0': 0.5, '1': 1.0}
        
        # Set default weights (English)
        self.default_weights = torch.tensor([1.0] * len(self.toxicity_columns))
    
    def load_weights(self):
        """Load and adjust language-specific weights with error handling"""
        weights_path = Path(self.weights_file)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
        with open(weights_path, 'r') as f:
            self.weights_data = json.load(f)
            self.weights = self.weights_data['weights']
        
        # Get list of toxicity columns in order
        self.toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.language_columns = ['en', 'es', 'fr', 'it', 'tr', 'pt', 'ru']
        
        # Apply EN weight boosts for critical classes with validation
        try:
            self.weights['en']['toxic']['1'] = min(15.0, max(0.5, 3.5))
            self.weights['en']['threat']['1'] = min(15.0, max(0.5, 15.0))
            self.weights['en']['identity_hate']['1'] = min(15.0, max(0.5, 5.0))
        except KeyError as e:
            print(f"Warning: Could not apply EN weight boosts: {str(e)}")
        
        # Default weights (English) with validation
        try:
            self.default_weights = torch.tensor([
                float(self.weights['en'][col]['1']) 
                for col in self.toxicity_columns
            ]).clamp(0.1, 15.0)  # Ensure weights are in reasonable range
        except Exception as e:
            print(f"Warning: Could not set default weights: {str(e)}")
            self.default_weights = torch.tensor([1.0] * len(self.toxicity_columns))
    
    def enforce_language_priority(self):
        """Ensure EN weights >= other languages for critical classes with strict ratios"""
        critical_classes = {
            'toxic': {
                'ratio': 0.5,
                'min_weight': 3.5,
                'max_other_weight': 1.75
            },
            'threat': {
                'ratio': 0.6,
                'min_weight': 15.0,
                'max_other_weight': 9.0
            },
            'identity_hate': {
                'ratio': 0.7,
                'min_weight': 5.0,
                'max_other_weight': 3.5
            }
        }
        
        try:
            for cls, params in critical_classes.items():
                en_weight = float(self.weights['en'][cls]['1'])
                max_other_weight = min(params['max_other_weight'], en_weight * params['ratio'])
                
                for lang in self.language_columns:
                    if lang != 'en':
                        try:
                            current_weight = float(self.weights[lang][cls]['1'])
                            # Ensure weight is both below EN ratio and above minimum
                            if current_weight > max_other_weight or current_weight < params['min_weight']:
                                new_weight = max(
                                    params['min_weight'],
                                    min(max_other_weight, current_weight)
                                )
                                self.weights[lang][cls]['1'] = str(new_weight)
                                
                                # Update calculation metadata
                                if 'calculation_metadata' not in self.weights[lang][cls]:
                                    self.weights[lang][cls]['calculation_metadata'] = {
                                        'constraints_applied': []
                                    }
                                
                                self.weights[lang][cls]['calculation_metadata']['constraints_applied'].extend([
                                    f"language_priority_ratio={params['ratio']}",
                                    f"max_other_weight={max_other_weight}",
                                    f"adjusted_from={current_weight}_to={new_weight}"
                                ])
                        except (KeyError, ValueError) as e:
                            print(f"Warning: Could not adjust weights for {lang}/{cls}: {str(e)}")
                            continue
        except Exception as e:
            print(f"Warning: Could not enforce language priority: {str(e)}")
    
    def calculate_safe_weights(self, total_samples: int, support_1: int, toxicity_type: str) -> float:
        """Calculate weights with safety constraints and validation"""
        try:
            if total_samples <= 0 or support_1 <= 0:
                raise ValueError("Sample counts must be positive")
                
            if toxicity_type == 'toxic':
                num_classes = 2
                boost_factor = 1.67
            else:
                num_classes = 5
                boost_factor = 1.0
            
            raw_weight = (total_samples / (num_classes * support_1)) * boost_factor
            
            # Apply constraints with validation
            max_weight = 15.0 if toxicity_type == 'threat' else 10.0
            min_weight = 0.5
            
            return float(max(min_weight, min(max_weight, raw_weight)))
            
        except Exception as e:
            print(f"Warning: Weight calculation failed for {toxicity_type}: {str(e)}")
            return 1.0  # Return safe default
    
    def update_weights_based_on_performance(self, val_metrics: dict):
        """Adjust weights based on validation performance with error handling"""
        if not isinstance(val_metrics, dict):
            print("Warning: Invalid validation metrics format")
            return
            
        try:
            for lang in self.language_columns:
                if lang in val_metrics.get('per_language', {}):
                    lang_metrics = val_metrics['per_language'][lang]
                    for cls in self.toxicity_columns:
                        try:
                            if cls in lang_metrics.get('class_metrics', {}):
                                cls_metrics = lang_metrics['class_metrics'][cls]
                                # Adjust weight if F1 score is too low
                                if cls_metrics.get('f1', 1.0) < 0.3:
                                    current_weight = float(self.weights[lang][cls]['1'])
                                    self.weights[lang][cls]['1'] = str(min(current_weight * 1.2, 15.0))
                        except Exception as e:
                            print(f"Warning: Could not update weights for {lang}/{cls}: {str(e)}")
                            continue
        except Exception as e:
            print(f"Warning: Weight update based on performance failed: {str(e)}")
    
    def get_weights_for_batch(self, langs: List[str], device: torch.device) -> torch.Tensor:
        """Get language-specific weights for each sample in the batch with error handling"""
        try:
            batch_weights = []
            
            for lang in langs:
                try:
                    # Get weights for this language, fallback to English if language not found
                    lang_weights = []
                    for col in self.toxicity_columns:
                        try:
                            weight = float(self.weights[lang][col]['1'])
                        except (KeyError, ValueError):
                            # Fallback to English weights
                            weight = float(self.weights['en'][col]['1'])
                        lang_weights.append(weight)
                    
                    batch_weights.append(lang_weights)
                except Exception as e:
                    print(f"Warning: Using default weights for language {lang}: {str(e)}")
                    batch_weights.append(self.default_weights.tolist())
            
            # Convert to tensor and move to device with error handling
            weights = torch.tensor(batch_weights, dtype=torch.float32)
            try:
                weights = weights.to(device)
            except Exception as e:
                print(f"Warning: Could not move weights to device: {str(e)}")
                weights = weights.cpu()  # Fallback to CPU
            
            # Preserve per-sample weighting - no mean reduction
            return weights.clamp(0.1, 15.0)  # Ensure weights are in reasonable range
            
        except Exception as e:
            print(f"Warning: Could not get batch weights: {str(e)}")
            # Return safe default weights
            return torch.ones((len(langs), len(self.toxicity_columns)), 
                            dtype=torch.float32, device=device)

@dataclass
class MetricsTracker:
    """Tracks training and validation metrics with error handling"""
    best_auc: float = 0.0
    train_losses: List[float] = None
    val_losses: List[float] = None
    val_aucs: List[float] = None
    epoch_times: List[float] = None
    
    def __post_init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.epoch_times = []
    
    def update_train(self, loss: float):
        """Update training metrics with validation"""
        try:
            if not isinstance(loss, (int, float)) or np.isnan(loss) or np.isinf(loss):
                print(f"Warning: Invalid loss value: {loss}")
                return
            self.train_losses.append(float(loss))
        except Exception as e:
            print(f"Warning: Could not update training metrics: {str(e)}")
    
    def update_validation(self, metrics: Dict) -> bool:
        """Update validation metrics with error handling"""
        try:
            if not isinstance(metrics, dict):
                raise ValueError("Metrics must be a dictionary")
                
            loss = metrics.get('loss', float('inf'))
            auc = metrics.get('auc', 0.0)
            
            if np.isnan(loss) or np.isinf(loss):
                print(f"Warning: Invalid loss value: {loss}")
                loss = float('inf')
            
            if np.isnan(auc) or np.isinf(auc):
                print(f"Warning: Invalid AUC value: {auc}")
                auc = 0.0
            
            self.val_losses.append(float(loss))
            self.val_aucs.append(float(auc))
            
            # Update best AUC if needed
            if auc > self.best_auc:
                self.best_auc = auc
                return True
            return False
            
        except Exception as e:
            print(f"Warning: Could not update validation metrics: {str(e)}")
            return False
    
    def update_time(self, epoch_time: float):
        """Update timing metrics with validation"""
        try:
            if not isinstance(epoch_time, (int, float)) or epoch_time <= 0:
                print(f"Warning: Invalid epoch time: {epoch_time}")
                return
            self.epoch_times.append(float(epoch_time))
        except Exception as e:
            print(f"Warning: Could not update timing metrics: {str(e)}")
    
    def get_eta(self, current_epoch: int, total_epochs: int) -> str:
        """Calculate ETA based on average epoch time with error handling"""
        try:
            if not self.epoch_times:
                return "Calculating..."
                
            if current_epoch >= total_epochs:
                return "Complete"
                
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = total_epochs - current_epoch
            eta_seconds = avg_epoch_time * remaining_epochs
            
            hours = int(eta_seconds // 3600)
            minutes = int((eta_seconds % 3600) // 60)
            
            return f"{hours:02d}:{minutes:02d}:00"
            
        except Exception as e:
            print(f"Warning: Could not calculate ETA: {str(e)}")
            return "Unknown"

@dataclass
class EarlyStopping:
    """Early stopping to prevent overfitting with improved error handling"""
    patience: int = 2
    min_delta: float = 1e-4
    
    def __post_init__(self):
        if self.patience < 1:
            print("Warning: Invalid patience value, setting to 1")
            self.patience = 1
        if self.min_delta < 0:
            print("Warning: Invalid min_delta value, setting to 0")
            self.min_delta = 0
            
        self.best_value = None
        self.best_epoch = 0
        self.counter = 0
        self.stopped_epoch = 0
    
    def __call__(self, value: float, epoch: int) -> bool:
        """Check if training should stop with error handling"""
        try:
            if np.isnan(value) or np.isinf(value):
                print(f"Warning: Invalid metric value: {value}")
                return False
                
            if self.best_value is None:
                self.best_value = value
                self.best_epoch = epoch
                return False
                
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.best_epoch = epoch
                self.counter = 0
                return False
            
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                return True
                
            return False
            
        except Exception as e:
            print(f"Warning: Early stopping check failed: {str(e)}")
            return False
    
    def get_best_epoch(self) -> int:
        """Get the epoch with best performance"""
        return self.best_epoch
        
    def get_stop_reason(self) -> str:
        """Get the reason for early stopping"""
        try:
            if self.stopped_epoch > 0:
                return (f"No improvement after {self.patience} epochs. "
                       f"Best value: {self.best_value:.4f} at epoch {self.best_epoch}")
            return "Training completed normally"
        except Exception as e:
            print(f"Warning: Could not get stop reason: {str(e)}")
            return "Unknown stop reason"

@dataclass
class TrainingConfig:
    """Basic training configuration"""
    # Model parameters
    model_name: str = "xlm-roberta-large"
    max_length: int = 128
    num_labels: int = 6
    
    # Training parameters
    batch_size: int = 32
    grad_accum_steps: int = 4
    epochs: int = 10
    lr: float = 2e-5
    weight_decay: float = 0.01
    
    # System parameters
    num_workers: int = 4
    fp16: bool = True
    device: str = None
    
    def __post_init__(self):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set toxicity labels
        self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.num_labels = len(self.toxicity_labels)
        
        # Create output directories
        for directory in ['weights', 'logs']:
            os.makedirs(directory, exist_ok=True) 