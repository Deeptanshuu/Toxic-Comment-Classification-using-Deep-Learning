# training_config.py
from asyncio.log import logger
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import torch
import numpy as np
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict

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
        self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.language_columns = ['en', 'es', 'fr', 'it', 'tr', 'pt', 'ru']
        
        # Create default weights with proper clamping
        self.weights = {}
        for lang in self.language_columns:
            self.weights[lang] = {}
            for label in self.toxicity_labels:
                self.weights[lang][label] = {
                    '0': 0.5,  # Negative class weight
                    '1': self._clamp_weight(1.0)  # Positive class weight
                }
        
        # Set default weights (English)
        self.default_weights = torch.tensor([1.0] * len(self.toxicity_labels))
    
    def _clamp_weight(self, weight: float, min_val: float = 0.5, max_val: float = 15.0) -> float:
        """Safely clamp weight values within valid range"""
        try:
            return float(max(min_val, min(max_val, weight)))
        except (TypeError, ValueError) as e:
            print(f"Warning: Invalid weight value {weight}, using default")
            return 1.0
    
    def load_weights(self):
        """Load and adjust language-specific weights with error handling"""
        weights_path = Path(self.weights_file)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
        with open(weights_path, 'r') as f:
            self.weights_data = json.load(f)
            self.weights = self.weights_data['weights']
        
        # Get list of toxicity labels in order
        self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.language_columns = ['en', 'es', 'fr', 'it', 'tr', 'pt', 'ru']
        
        # Apply EN weight boosts for critical classes with validation
        try:
            critical_weights = {
                'toxic': 3.5,
                'threat': 15.0,
                'identity_hate': 5.0
            }
            
            for cls, weight in critical_weights.items():
                self.weights['en'][cls]['1'] = self._clamp_weight(weight)
                
        except KeyError as e:
            print(f"Warning: Could not apply EN weight boosts: {str(e)}")
        
        # Default weights (English) with validation
        try:
            self.default_weights = torch.tensor([
                self._clamp_weight(float(self.weights['en'][label]['1']))
                for label in self.toxicity_labels
            ])
        except Exception as e:
            print(f"Warning: Could not set default weights: {str(e)}")
            self.default_weights = torch.tensor([1.0] * len(self.toxicity_labels))
    
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
                # Get and validate English weight
                en_weight = self._clamp_weight(float(self.weights['en'][cls]['1']))
                self.weights['en'][cls]['1'] = str(en_weight)
                
                # Calculate maximum allowed weight for other languages
                max_other_weight = min(
                    params['max_other_weight'],
                    en_weight * params['ratio']
                )
                
                for lang in self.language_columns:
                    if lang != 'en':
                        try:
                            current_weight = float(self.weights[lang][cls]['1'])
                            
                            # Ensure weight is both below EN ratio and above minimum
                            new_weight = self._clamp_weight(
                                current_weight,
                                min_val=params['min_weight'],
                                max_val=max_other_weight
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
            
            return self._clamp_weight(raw_weight, min_weight, max_weight)
            
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
                    for label in self.toxicity_labels:
                        try:
                            if label in lang_metrics.get('class_metrics', {}):
                                cls_metrics = lang_metrics['class_metrics'][label]
                                # Adjust weight if F1 score is too low
                                if cls_metrics.get('f1', 1.0) < 0.3:
                                    current_weight = float(self.weights[lang][label]['1'])
                                    self.weights[lang][label]['1'] = str(min(current_weight * 1.2, 15.0))
                        except Exception as e:
                            print(f"Warning: Could not update weights for {lang}/{label}: {str(e)}")
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
                    for label in self.toxicity_labels:
                        try:
                            weight = float(self.weights[lang][label]['1'])
                        except (KeyError, ValueError):
                            # Fallback to English weights
                            weight = float(self.weights['en'][label]['1'])
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
            return torch.ones((len(langs), len(self.toxicity_labels)), 
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
    max_length: int = 512
    hidden_size: int = 1024
    num_attention_heads: int = 16
    model_dropout: float = 0.1
    freeze_layers: int = 8
    
    # Training parameters
    batch_size: int = 32
    grad_accum_steps: int = 1
    epochs: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.05
    
    # Language-specific learning rate multipliers
    lang_lr_multipliers: Dict[str, float] = None
    
    # System parameters
    num_workers: int = 12
    fp16: bool = False
    mixed_precision: str = "bf16"  # Options: "no", "fp16", "bf16"
    device: str = None
    activation_checkpointing: bool = False
    tensor_float_32: bool = True
    gc_frequency: int = 100
    
    # Distributed training parameters
    distributed: bool = False
    world_size: int = 1
    dist_backend: str = "nccl"
    dist_url: str = "env://"
    local_rank: int = -1
    find_unused_parameters: bool = False
    
    def __post_init__(self):
        """Initialize and validate configuration"""
        # Initialize language-specific learning rate multipliers with defaults
        self.validate_lr_multipliers()
        
        # Validate learning rate multipliers
        for lang, multiplier in self.lang_lr_multipliers.items():
            if multiplier <= 0:
                raise ValueError(f"Invalid learning rate multiplier for {lang}: {multiplier}")
            if multiplier > 2.0:  # Reasonable upper bound
                logger.warning(f"High learning rate multiplier for {lang}: {multiplier}")
        
        # Validate mixed precision settings
        valid_precisions = ["no", "fp16", "bf16"]
        if self.mixed_precision not in valid_precisions:
            raise ValueError(f"Invalid mixed precision mode: {self.mixed_precision}. Must be one of {valid_precisions}")
        
        # Set use_mixed_precision flag
        self.use_mixed_precision = self.mixed_precision != "no"
        
        # Validate parameters
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {self.batch_size}")
        if self.grad_accum_steps <= 0:
            raise ValueError(f"Invalid grad_accum_steps: {self.grad_accum_steps}")
        if self.epochs <= 0:
            raise ValueError(f"Invalid epochs: {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"Invalid learning rate: {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {self.weight_decay}")
        if self.num_workers < 0:
            raise ValueError(f"Invalid num_workers: {self.num_workers}")
        if self.gc_frequency <= 0:
            raise ValueError(f"Invalid gc_frequency: {self.gc_frequency}")
        if not 0 <= self.model_dropout < 1:
            raise ValueError(f"Invalid model_dropout: {self.model_dropout}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"Invalid max_grad_norm: {self.max_grad_norm}")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError(f"Invalid warmup_ratio: {self.warmup_ratio}")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError(f"Invalid label_smoothing: {self.label_smoothing}")
        if self.freeze_layers < 0:
            raise ValueError(f"Invalid freeze_layers: {self.freeze_layers}")
        
        # Validate distributed training parameters
        if self.distributed:
            if self.world_size <= 0:
                raise ValueError(f"Invalid world_size for distributed training: {self.world_size}")
            if self.local_rank < -1:
                raise ValueError(f"Invalid local_rank: {self.local_rank}")
            if self.dist_backend not in ["nccl", "gloo", "mpi"]:
                raise ValueError(f"Invalid dist_backend: {self.dist_backend}")
        
        # Set device with error handling
        if torch.cuda.is_available():
            try:
                    torch.cuda.init()
                    self.device = torch.device('cuda')
                    
                    # Check if GPU supports BF16
                    if self.mixed_precision == "bf16":
                        if not torch.cuda.is_bf16_supported():
                            print("Warning: BF16 not supported on this GPU. Falling back to FP16")
                            self.mixed_precision = "fp16"
                            self.fp16 = True
                    
                    # Enable TF32 if requested and available
                    if self.tensor_float_32:
                        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                            torch.backends.cuda.matmul.allow_tf32 = True
                            torch.backends.cudnn.allow_tf32 = True
                    else:
                        print("Warning: TF32 not supported on this GPU. Disabling.")
                        self.tensor_float_32 = False
                    
            except Exception as e:
                print(f"Warning: CUDA initialization failed: {str(e)}")
                self.device = torch.device('cpu')
                self.mixed_precision = "no"
                self.fp16 = False
        else:
            self.device = torch.device('cpu')
            if self.mixed_precision != "no" or self.fp16:
                print("Warning: Mixed precision not supported on CPU. Disabling.")
                self.mixed_precision = "no"
            self.fp16 = False
        
        # Create directories with error handling
        try:
            for directory in ["weights", "logs"]:
                dir_path = Path(directory)
                if not dir_path.exists():
                    dir_path.mkdir(parents=True)
                elif not dir_path.is_dir():
                    raise NotADirectoryError(f"{directory} exists but is not a directory")
        except Exception as e:
            print(f"Error creating directories: {str(e)}")
            raise
        
        # Initialize toxicity labels
        self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.num_labels = len(self.toxicity_labels)
        
    @property
    def dtype(self) -> torch.dtype:
        """Get the appropriate dtype based on mixed precision settings"""
        if self.mixed_precision == "bf16":
            return torch.bfloat16
        elif self.mixed_precision == "fp16":
            return torch.float16
        return torch.float32
    
    def get_autocast_context(self):
        """Get the appropriate autocast context based on configuration."""
        if not self.use_mixed_precision:
            return nullcontext()
            
        dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)
    
    def to_serializable_dict(self):
        """Convert config to a dictionary for saving."""
        config_dict = asdict(self)
        # Add any non-serializable field handling here
        return config_dict
    
    def validate_lr_multipliers(self):
        """Validate language-specific learning rate multipliers"""
        try:
            # Initialize with defaults if not set
            if self.lang_lr_multipliers is None:
                self.lang_lr_multipliers = {
                    'en': 1.2,  # Higher LR for English (most data)
                    'ru': 0.9,  # Lower for Russian (different script)
                    'tr': 0.95, # Slightly lower for Turkish
                    'es': 1.0,  # Default for Spanish
                    'fr': 1.0,  # Default for French
                    'it': 0.95, # Slightly lower for Italian (less data)
                    'pt': 0.95, # Slightly lower for Portuguese (less data)
                    'default': 1.0  # Default multiplier for other languages
                }
            
            # Ensure all required languages are present
            required_langs = {'en', 'ru', 'tr', 'es', 'fr', 'it', 'pt', 'default'}
            missing_langs = required_langs - set(self.lang_lr_multipliers.keys())
            if missing_langs:
                print(f"Warning: Missing language multipliers for {missing_langs}")
                # Add missing languages with default multiplier
                for lang in missing_langs:
                    self.lang_lr_multipliers[lang] = 1.0
            
            # Validate multiplier values
            for lang, multiplier in self.lang_lr_multipliers.items():
                try:
                    multiplier_float = float(multiplier)
                    if multiplier_float <= 0:
                        print(f"Warning: Invalid multiplier {multiplier} for {lang}, using 1.0")
                        self.lang_lr_multipliers[lang] = 1.0
                    elif multiplier_float > 2.0:
                        print(f"Warning: High multiplier {multiplier} for {lang}")
                except (TypeError, ValueError):
                    print(f"Warning: Invalid multiplier format for {lang}, using 1.0")
                    self.lang_lr_multipliers[lang] = 1.0
            
            # Ensure English has slightly higher learning rate
            en_multiplier = self.lang_lr_multipliers['en']
            if en_multiplier <= 1.0:
                print("Warning: English learning rate should be higher, adjusting to 1.2")
                self.lang_lr_multipliers['en'] = 1.2
            
            # Log final multipliers
            print("Language-specific learning rate multipliers:")
            for lang, multiplier in sorted(self.lang_lr_multipliers.items()):
                print(f"  {lang}: {multiplier}")
                
        except Exception as e:
            print(f"Error validating learning rate multipliers: {str(e)}")
            # Reset to safe defaults
            self.lang_lr_multipliers = {lang: 1.0 for lang in required_langs}
            self.lang_lr_multipliers['en'] = 1.2  # Keep English higher
    
    def get_param_groups(self, model):
        """Get parameter groups with language-specific learning rates"""
        try:
            param_groups = []
            
            # Track parameters to ensure no duplicates
            seen_params = set()
            
            for name, param in model.named_parameters():
                if not param.requires_grad or id(param) in seen_params:
                    continue
                
                seen_params.add(id(param))
                
                # Base learning rate
                lr = self.lr
                
                # Apply language-specific multiplier if applicable
                for lang, multiplier in self.lang_lr_multipliers.items():
                    if lang != 'default' and lang in name.lower():
                        lr *= multiplier
                        break
                else:
                    # Apply default multiplier if no specific language found
                    lr *= self.lang_lr_multipliers['default']
                
                param_groups.append({
                    'params': [param],
                    'lr': lr,
                    'weight_decay': self.weight_decay
                })
            
            return param_groups
            
        except Exception as e:
            print(f"Error creating parameter groups: {str(e)}")
            # Fallback to simple parameter group
            return [{'params': model.parameters(), 'lr': self.lr}] 