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
    """Training configuration with optimized defaults and error handling"""
    # Model parameters
    model_name: str = "xlm-roberta-large"
    max_length: int = 128
    num_labels: int = 6
    
    # Training parameters with adjusted defaults for stability
    batch_size: int = 32  # Reduced batch size
    grad_accum_steps: int = 4  # Increased accumulation steps
    epochs: int = 10
    lr: float = 8e-6  # Reduced learning rate
    warmup_ratio: float = 0.15
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Added explicit grad norm
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999  # Increased beta2 for more stability
    adam_epsilon: float = 1e-8
    
    # Language parameters
    language_columns: list = None
    
    # Multi-GPU parameters
    distributed: bool = True
    world_size: int = 2
    dist_backend: str = 'nccl'
    dist_url: str = 'tcp://localhost:23456'
    find_unused_parameters: bool = False
    
    # Memory optimization parameters
    activation_checkpointing: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    max_split_size_mb: int = 512
    empty_cache_freq: int = 100
    
    # Gradient control parameters
    initial_max_norm: float = 0.8
    final_max_norm: float = 4.0
    min_max_norm: float = 0.5
    grad_norm_adjustment_steps: int = 200
    
    # Mixed precision parameters
    fp16: bool = True
    mixed_precision: str = 'bf16'
    
    # System parameters
    num_workers: int = 16
    gc_frequency: int = 100
    
    # Optimization flags
    tensor_float_32: bool = True
    
    def __post_init__(self):
        # Initialize language columns
        self.language_columns = ['en', 'es', 'fr', 'it', 'tr', 'pt', 'ru']
        
        try:
            # Load language-specific weights
            weights_path = Path('weights/language_class_weights.json')
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
            with open(weights_path, 'r') as f:
                self.weights_data = json.load(f)
                self.weights = self.weights_data['weights']
            
            # Get list of toxicity columns in order
            self.toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            
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
            
            # Validate parameters
            self._validate_parameters()
            
            # Initialize distributed setup
            if self.distributed and torch.cuda.is_available():
                if torch.cuda.device_count() < self.world_size:
                    print(f"Warning: Requested {self.world_size} GPUs but only {torch.cuda.device_count()} available")
                    self.world_size = torch.cuda.device_count()
                
                # Adjust batch size and learning rate for multi-GPU
                self.global_batch_size = self.batch_size * self.world_size
                self.lr = self.lr * (self.global_batch_size / 64)  # Linear scaling rule
                
                # Set memory optimization parameters
                torch.backends.cudnn.benchmark = True
                if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                # Set environment variables for distributed training
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '23456'
                os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{self.max_split_size_mb}'
            else:
                self.distributed = False
                self.world_size = 1
                self.global_batch_size = self.batch_size
            
            # Initialize device
            self._device = self._init_device()
            
            # Create output directories
            self._create_directories()
            
            # Initialize metrics tracker
            self.metrics = MetricsTracker()
            
            # Initialize class weights
            self.class_weights = DynamicClassWeights()
            
            # Set toxicity labels
            self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            self.num_labels = len(self.toxicity_labels)
            
        except Exception as e:
            print(f"Error initializing training config: {str(e)}")
            raise
    
    def _init_device(self) -> torch.device:
        """Initialize device with error handling"""
        try:
            if torch.cuda.is_available():
                # Try to initialize CUDA
                torch.cuda.init()
                return torch.device('cuda')
            return torch.device('cpu')
        except Exception as e:
            print(f"Warning: CUDA initialization failed: {str(e)}")
            return torch.device('cpu')
    
    def _create_directories(self):
        """Create necessary directories with error handling"""
        directories = ['weights', 'logs', 'tokenized']
        for directory in directories:
            try:
                Path(directory).mkdir(exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory {directory}: {str(e)}")
    
    def _validate_parameters(self):
        """Validate configuration parameters"""
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.batch_size % 8 != 0:
            warnings.warn(f"Batch size {self.batch_size} is not a multiple of 8")
            
        if self.grad_accum_steps < 1:
            raise ValueError(f"Gradient accumulation steps must be positive, got {self.grad_accum_steps}")
            
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.lr}")
        if self.lr > 1e-3:
            warnings.warn(f"Learning rate {self.lr} might be too high")
            
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError(f"Warmup ratio must be between 0 and 1, got {self.warmup_ratio}")
            
        if self.weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {self.weight_decay}")
            
        if self.initial_max_norm <= 0:
            raise ValueError(f"Initial max norm must be positive, got {self.initial_max_norm}")
        if self.final_max_norm < self.initial_max_norm:
            raise ValueError(f"Final max norm {self.final_max_norm} cannot be less than initial max norm {self.initial_max_norm}")
        if self.min_max_norm <= 0:
            raise ValueError(f"Minimum max norm must be positive, got {self.min_max_norm}")
        if self.min_max_norm > self.initial_max_norm:
            raise ValueError(f"Minimum max norm {self.min_max_norm} cannot be greater than initial max norm {self.initial_max_norm}")
            
        if self.num_workers < 0:
            raise ValueError(f"Number of workers must be non-negative, got {self.num_workers}")
        if self.gc_frequency < 1:
            raise ValueError(f"GC frequency must be positive, got {self.gc_frequency}")
            
        if self.mixed_precision not in ['no', 'fp16', 'bf16']:
            raise ValueError(f"Mixed precision must be one of ['no', 'fp16', 'bf16'], got {self.mixed_precision}")
    
    @property
    def device(self) -> torch.device:
        """Get the device to use"""
        return self._device
    
    def get_optimizer_groups(self, model: torch.nn.Module) -> list:
        """Create parameter groups for optimizer with layer-wise decay and error handling"""
        try:
            # Group parameters by layer depth for transformer
            layer_groups = []
            no_decay = ['bias', 'LayerNorm.weight']
            
            # Add embeddings
            try:
                layer_groups.append({
                    'params': [p for n, p in model.roberta.embeddings.named_parameters()
                              if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.weight_decay,
                    'lr': self.lr
                })
            except Exception as e:
                print(f"Warning: Could not add embedding parameters: {str(e)}")
            
            # Add encoder layers with increasing learning rate
            try:
                num_layers = len(model.roberta.encoder.layer)
                for i, layer in enumerate(model.roberta.encoder.layer):
                    lr_scale = 1.0 + (i / num_layers) * 0.1
                    layer_groups.append({
                        'params': [p for n, p in layer.named_parameters()
                                  if not any(nd in n for nd in no_decay)],
                        'weight_decay': self.weight_decay,
                        'lr': self.lr * lr_scale
                    })
            except Exception as e:
                print(f"Warning: Could not add encoder layer parameters: {str(e)}")
            
            # Add classifier with higher learning rate
            try:
                layer_groups.append({
                    'params': [p for n, p in model.classifier.named_parameters()
                              if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.weight_decay,
                    'lr': self.lr * 1.5
                })
            except Exception as e:
                print(f"Warning: Could not add classifier parameters: {str(e)}")
            
            # Add all biases and LayerNorm parameters with no weight decay
            try:
                layer_groups.append({
                    'params': [p for n, p in model.named_parameters()
                              if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': self.lr
                })
            except Exception as e:
                print(f"Warning: Could not add bias/LayerNorm parameters: {str(e)}")
            
            return layer_groups
            
        except Exception as e:
            print(f"Warning: Could not create optimizer groups: {str(e)}")
            # Return default parameter group
            return [{'params': model.parameters()}]
    
    def get_summary(self) -> Dict:
        """Get training configuration summary with error handling"""
        try:
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
        except Exception as e:
            print(f"Warning: Could not generate summary: {str(e)}")
            return {}
    
    def to_serializable_dict(self) -> dict:
        """Return a serializable dictionary of configuration parameters"""
        try:
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
        except Exception as e:
            print(f"Warning: Could not serialize config: {str(e)}")
            return {}
    
    def get_adaptive_max_norm(self, epoch: int, step: int) -> float:
        """Calculate adaptive gradient norm with error handling"""
        try:
            # Calculate progress as combination of epochs and steps
            total_steps = self.grad_norm_adjustment_steps
            current_step = (epoch * total_steps / self.epochs) + (step / total_steps)
            progress = min(1.0, current_step / total_steps)
            
            # Use a smoother transition curve (quadratic)
            smooth_progress = progress * progress
            
            # Start strict and gradually relax with smoother curve
            max_norm = self.initial_max_norm + (self.final_max_norm - self.initial_max_norm) * smooth_progress
            
            # Apply safety bounds
            max_norm = max(self.min_max_norm, min(self.final_max_norm, max_norm))
            
            return float(max_norm)
            
        except Exception as e:
            print(f"Warning: Could not calculate adaptive max norm: {str(e)}")
            return float(self.initial_max_norm)  # Return safe default 