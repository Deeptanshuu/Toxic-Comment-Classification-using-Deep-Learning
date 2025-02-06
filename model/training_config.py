from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import torch
import numpy as np
from pathlib import Path

@dataclass
class DynamicClassWeights:
    """Handles dynamic class weights per language with improved calculation"""
    weights_file: str = 'weights/language_class_weights.json'
    
    def __post_init__(self):
        self.load_weights()
        self.enforce_language_priority()
    
    def load_weights(self):
        """Load and adjust language-specific weights"""
        with open(self.weights_file, 'r') as f:
            self.weights_data = json.load(f)
            self.weights = self.weights_data['weights']
        
        # Get list of toxicity columns in order
        self.toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.language_columns = ['en', 'es', 'fr', 'it', 'tr', 'pt', 'ru']
        
        # Apply EN weight boosts for critical classes
        self.weights['en']['toxic']['1'] = 3.5       # Increased from 0.59
        self.weights['en']['threat']['1'] = 15.0     # Increased from 10.88
        self.weights['en']['identity_hate']['1'] = 5.0  # Increased from 3.58
        
        # Default weights (English)
        self.default_weights = torch.tensor([
            self.weights['en'][col]['1'] for col in self.toxicity_columns
        ])
    
    def enforce_language_priority(self):
        """Ensure EN weights >= other languages for critical classes with strict ratios"""
        critical_classes = {
            'toxic': {
                'ratio': 0.5,  # EN toxic must be 2x others
                'min_weight': 3.5,
                'max_other_weight': 1.75  # Half of EN weight
            },
            'threat': {
                'ratio': 0.6,  # EN threat must be 1.66x others
                'min_weight': 15.0,
                'max_other_weight': 9.0  # 60% of EN weight
            },
            'identity_hate': {
                'ratio': 0.7,
                'min_weight': 5.0,
                'max_other_weight': 3.5  # 70% of EN weight
            }
        }
        
        for cls, params in critical_classes.items():
            en_weight = self.weights['en'][cls]['1']
            max_other_weight = min(params['max_other_weight'], en_weight * params['ratio'])
            
            for lang in self.language_columns:
                if lang != 'en':
                    current_weight = self.weights[lang][cls]['1']
                    # Ensure weight is both below EN ratio and above minimum
                    if current_weight > max_other_weight or current_weight < params['min_weight']:
                        new_weight = max(
                            params['min_weight'],
                            min(max_other_weight, current_weight)
                        )
                        self.weights[lang][cls]['1'] = new_weight
                        
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
    
    def calculate_safe_weights(self, total_samples: int, support_1: int, toxicity_type: str) -> float:
        """Calculate weights with safety constraints"""
        if toxicity_type == 'toxic':
            num_classes = 2  # Treat as primary binary classification task
            boost_factor = 1.67  # Extra boost for main toxic class
        else:
            num_classes = 5  # Remaining secondary toxicity types
            boost_factor = 1.0
        
        raw_weight = (total_samples / (num_classes * support_1)) * boost_factor
        
        # Apply constraints
        max_weight = 15.0 if toxicity_type == 'threat' else 10.0
        min_weight = 0.5
        
        return max(min_weight, min(max_weight, raw_weight))
    
    def update_weights_based_on_performance(self, val_metrics: dict):
        """Adjust weights based on validation performance"""
        for lang in self.language_columns:
            if lang in val_metrics['per_language']:
                lang_metrics = val_metrics['per_language'][lang]
                for cls in self.toxicity_columns:
                    if cls in lang_metrics['class_metrics']:
                        cls_metrics = lang_metrics['class_metrics'][cls]
                        # Adjust weight if F1 score is too low
                        if cls_metrics['f1'] < 0.3:
                            current_weight = self.weights[lang][cls]['1']
                            self.weights[lang][cls]['1'] = min(current_weight * 1.2, 15.0)
    
    def get_weights_for_batch(self, langs: List[str], device: torch.device) -> torch.Tensor:
        """Get language-specific weights for each sample in the batch"""
        batch_weights = []
        
        for lang in langs:
            # Get weights for this language, fallback to English if language not found
            try:
                lang_weights = [
                    self.weights[lang][col]['1'] 
                    for col in self.toxicity_columns
                ]
            except KeyError:
                lang_weights = [
                    self.weights['en'][col]['1']  # Fallback to English weights
                    for col in self.toxicity_columns
                ]
            batch_weights.append(lang_weights)
        
        # Convert to tensor and move to device
        weights = torch.tensor(batch_weights, dtype=torch.float32).to(device)
        return weights  # Return per-sample weights [B, C]

@dataclass
class MetricsTracker:
    """Tracks training and validation metrics"""
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
        """Update training metrics"""
        self.train_losses.append(loss)
    
    def update_validation(self, metrics: Dict):
        """Update validation metrics"""
        self.val_losses.append(metrics['loss'])
        self.val_aucs.append(metrics['auc'])
        
        # Update best AUC if needed
        if metrics['auc'] > self.best_auc:
            self.best_auc = metrics['auc']
            return True
        return False
    
    def update_time(self, epoch_time: float):
        """Update timing metrics"""
        self.epoch_times.append(epoch_time)
    
    def get_eta(self, current_epoch: int, total_epochs: int) -> str:
        """Calculate ETA based on average epoch time"""
        if not self.epoch_times:
            return "Calculating..."
            
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = total_epochs - current_epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        
        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        
        return f"{hours:02d}:{minutes:02d}:00"

@dataclass
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    patience: int = 2  # Number of epochs to wait for improvement
    min_delta: float = 1e-4  # Minimum change to qualify as an improvement
    
    def __post_init__(self):
        self.best_value = None
        self.best_epoch = 0
        self.counter = 0
        self.stopped_epoch = 0
    
    def __call__(self, value: float, epoch: int) -> bool:
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
    
    def get_best_epoch(self) -> int:
        return self.best_epoch
        
    def get_stop_reason(self) -> str:
        if self.stopped_epoch > 0:
            return f"No improvement after {self.patience} epochs. Best value: {self.best_value:.4f} at epoch {self.best_epoch}"
        return "Training completed normally"

@dataclass
class TrainingConfig:
    """Training configuration with optimized defaults"""
    # Model parameters
    model_name: str = "xlm-roberta-large"
    max_length: int = 128
    num_labels: int = 6
    
    # Training parameters
    batch_size: int = 64  # Increased from 48 for better XLM-R utilization
    grad_accum_steps: int = 1  # Removed accumulation for stability
    epochs: int = 10
    lr: float = 2e-5  # Standard XLM-R fine-tuning rate
    warmup_ratio: float = 0.15
    weight_decay: float = 0.01
    
    # Gradient control parameters
    initial_max_norm: float = 1.0  # Start with moderate clipping (increased from 0.1)
    final_max_norm: float = 5.0    # Allow higher gradients as training progresses (increased from 1.0)
    min_max_norm: float = 0.5      # Increased minimum bound
    grad_norm_adjustment_steps: int = 100  # Steps to adjust norm
    
    # Mixed precision parameters
    fp16: bool = True
    mixed_precision: str = 'bf16'  # Better numerical stability than fp16
    
    # System parameters
    num_workers: int = 16
    pin_memory: bool = True
    prefetch_factor: int = 2
    gc_frequency: int = 100
    
    # Optimization flags
    activation_checkpointing: bool = True  # Enable for large batch size
    tensor_float_32: bool = True
    
    def __post_init__(self):
        # Initialize device
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enable TF32 if requested
        if torch.cuda.is_available() and self.tensor_float_32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Initialize metrics tracker
        self.metrics = MetricsTracker()
        
        # Initialize class weights
        self.class_weights = DynamicClassWeights()
        
        # Set toxicity labels
        self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.num_labels = len(self.toxicity_labels)
        
        # Create output directories
        Path('weights').mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        Path('tokenized').mkdir(exist_ok=True)  # For memory-mapped tokenization cache
        
        # Validate configuration
        self._validate_config()

    @property
    def device(self) -> torch.device:
        """Get the device to use"""
        return self._device
    
    def get_optimizer_groups(self, model: torch.nn.Module) -> list:
        """Create parameter groups for optimizer with layer-wise decay"""
        # Group parameters by layer depth for transformer
        layer_groups = []
        no_decay = ['bias', 'LayerNorm.weight']
        
        # Add embeddings
        layer_groups.append({
            'params': [p for n, p in model.roberta.embeddings.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': self.weight_decay,
            'lr': self.lr
        })
        
        # Add encoder layers with increasing learning rate
        num_layers = len(model.roberta.encoder.layer)
        for i, layer in enumerate(model.roberta.encoder.layer):
            lr_scale = 1.0 + (i / num_layers) * 0.1  # Gradually increase LR for higher layers
            layer_groups.append({
                'params': [p for n, p in layer.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay,
                'lr': self.lr * lr_scale
            })
        
        # Add classifier with higher learning rate
        layer_groups.append({
            'params': [p for n, p in model.classifier.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': self.weight_decay,
            'lr': self.lr * 1.5  # Higher LR for classifier
        })
        
        # Add all biases and LayerNorm parameters with no weight decay
        layer_groups.append({
            'params': [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': self.lr
        })
        
        return layer_groups

    def get_summary(self) -> Dict:
        """Get training configuration summary"""
        return {
            'grad_norm_max': self.final_max_norm,
            'throughput_avg': self.batch_size / self.metrics.epoch_times[-1] if self.metrics.epoch_times else "Calculating...",
            'peak_memory_gb': self.batch_size * self.num_workers * 4 / 1024**3 if torch.cuda.is_available() else 0,
            'hyperparameters': {
                'batch_size': self.batch_size,
                'grad_accum_steps': self.grad_accum_steps,
                'learning_rate': self.lr,
                'warmup_ratio': self.warmup_ratio
            }
        }

    def to_serializable_dict(self) -> dict:
        """Return a serializable dictionary of configuration parameters"""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "num_labels": self.num_labels,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "epochs": self.epochs,
            "lr": self.lr,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "initial_max_norm": self.initial_max_norm,
            "final_max_norm": self.final_max_norm,
            "min_max_norm": self.min_max_norm,
            "grad_norm_adjustment_steps": self.grad_norm_adjustment_steps,
            "fp16": self.fp16,
            "mixed_precision": self.mixed_precision,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "gc_frequency": self.gc_frequency,
            "activation_checkpointing": self.activation_checkpointing,
            "tensor_float_32": self.tensor_float_32,
            "toxicity_labels": self.toxicity_labels
        }

    def get_adaptive_max_norm(self, epoch: int, step: int) -> float:
        """Calculate adaptive gradient norm based on training progress"""
        # Calculate progress as combination of epochs and steps
        total_steps = self.grad_norm_adjustment_steps
        current_step = (epoch * total_steps / self.epochs) + (step / total_steps)
        progress = min(1.0, current_step / total_steps)
        
        # Start strict and gradually relax
        max_norm = self.initial_max_norm + (self.final_max_norm - self.initial_max_norm) * progress
        
        # Apply safety bounds
        max_norm = max(self.min_max_norm, min(self.final_max_norm, max_norm))
        
        return max_norm

    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate batch size
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.batch_size % 8 != 0:
            print(f"Warning: Batch size {self.batch_size} is not a multiple of 8")
            
        # Validate gradient accumulation steps
        if self.grad_accum_steps < 1:
            raise ValueError(f"Gradient accumulation steps must be positive, got {self.grad_accum_steps}")
            
        # Validate learning rate
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.lr}")
        if self.lr > 1e-3:
            print(f"Warning: Learning rate {self.lr} might be too high")
            
        # Validate warmup ratio
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError(f"Warmup ratio must be between 0 and 1, got {self.warmup_ratio}")
            
        # Validate weight decay
        if self.weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {self.weight_decay}")
            
        # Validate gradient norm parameters
        if self.initial_max_norm <= 0:
            raise ValueError(f"Initial max norm must be positive, got {self.initial_max_norm}")
        if self.final_max_norm < self.initial_max_norm:
            raise ValueError(f"Final max norm {self.final_max_norm} cannot be less than initial max norm {self.initial_max_norm}")
        if self.min_max_norm <= 0:
            raise ValueError(f"Minimum max norm must be positive, got {self.min_max_norm}")
        if self.min_max_norm > self.initial_max_norm:
            raise ValueError(f"Minimum max norm {self.min_max_norm} cannot be greater than initial max norm {self.initial_max_norm}")
            
        # Validate system parameters
        if self.num_workers < 0:
            raise ValueError(f"Number of workers must be non-negative, got {self.num_workers}")
        if self.gc_frequency < 1:
            raise ValueError(f"GC frequency must be positive, got {self.gc_frequency}")
            
        # Validate mixed precision settings
        if self.mixed_precision not in ['no', 'fp16', 'bf16']:
            raise ValueError(f"Mixed precision must be one of ['no', 'fp16', 'bf16'], got {self.mixed_precision}")
            
        # Set language columns
        self.language_columns = ['en', 'es', 'fr', 'it', 'tr', 'pt', 'ru']
        
        # Print configuration summary
        print("\nTraining Configuration:")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Batch size: {self.batch_size}")
        print(f"Gradient accumulation steps: {self.grad_accum_steps}")
        print(f"Effective batch size: {self.batch_size * self.grad_accum_steps}")
        print(f"Learning rate: {self.lr}")
        print(f"Warmup ratio: {self.warmup_ratio}")
        print(f"Weight decay: {self.weight_decay}")
        print(f"Gradient clipping: {self.initial_max_norm} → {self.final_max_norm}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Number of workers: {self.num_workers}")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
        
        return True 