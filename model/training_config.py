# training_config.py
from asyncio.log import logger
from dataclasses import dataclass
from typing import Dict, List
import json
import torch
import numpy as np
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict

@dataclass
class DynamicClassWeights:
    """Handles class weights per language using pre-calculated values with focal loss"""
    weights_file: str = 'weights/language_class_weights.json'
    
    def __post_init__(self):
        self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.language_columns = ['en', 'es', 'fr', 'it', 'tr', 'pt', 'ru']
        self.weights = {}
        self.default_weights = None
        self.focal_params = {}
        
        try:
            with open(self.weights_file, 'r') as f:
                data = json.load(f)
                self.weights = data['weights']
                self.metadata = data.get('metadata', {})
                self.default_weights = torch.tensor([
                    float(self.weights['en'][label]['1']) 
                    for label in self.toxicity_labels
                ])
                
                # Extract focal loss parameters from metadata
                for lang in self.weights:
                    self.focal_params[lang] = {}
                    for label in self.weights[lang]:
                        metadata = self.weights[lang][label].get('calculation_metadata', {})
                        self.focal_params[lang][label] = {
                            'gamma': metadata.get('gamma', 2.0),
                            'alpha': metadata.get('alpha', 0.25)
                        }
        except Exception as e:
            print(f"Warning: Could not load weights from {self.weights_file}: {str(e)}")
            print("Using default weights...")
            self._initialize_default_weights()
    
    def _initialize_default_weights(self):
        """Initialize safe default weights if loading fails"""
        self.weights = {}
        self.focal_params = {}
        for lang in self.language_columns:
            self.weights[lang] = {}
            self.focal_params[lang] = {}
            for label in self.toxicity_labels:
                self.weights[lang][label] = {'0': 0.5, '1': 1.0}
                self.focal_params[lang][label] = {'gamma': 2.0, 'alpha': 0.25}
        self.default_weights = torch.tensor([1.0] * len(self.toxicity_labels))
    
    def get_weights_for_batch(self, langs: List[str], label_counts: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Get weights and focal loss parameters for batch
        Returns dict with 'weights' tensor and 'focal_params' dict
        """
        try:
            batch_weights = []
            batch_gamma = []
            batch_alpha = []
            
            for lang, counts in zip(langs, label_counts):
                lang_weights = []
                label_gamma = []
                label_alpha = []
                
                for idx, label in enumerate(self.toxicity_labels):
                    try:
                        # Get base weight with English fallback
                        base_weight = float(self.weights.get(lang, self.weights['en'])[label]['1'])
                        
                        # Get focal parameters
                        focal_params = self.focal_params.get(lang, {}).get(label, {})
                        gamma = focal_params.get('gamma', 2.0)
                        alpha = focal_params.get('alpha', 0.25)
                        
                        # Calculate class frequency in current batch
                        freq = counts[idx].float() / counts.sum().float()
                        
                        # Adjust weight based on batch distribution
                        adjusted = base_weight * (1 + 2 * (1 - freq))
                        
                        lang_weights.append(adjusted)
                        label_gamma.append(gamma)
                        label_alpha.append(alpha)
                        
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Weight calculation failed for {lang}-{label}: {str(e)}")
                        lang_weights.append(1.0)
                        label_gamma.append(2.0)
                        label_alpha.append(0.25)
                
                batch_weights.append(lang_weights)
                batch_gamma.append(label_gamma)
                batch_alpha.append(label_alpha)
            
            # Convert to tensors and apply safety bounds
            weights = torch.tensor(batch_weights, dtype=torch.float32, device=device)
            gamma = torch.tensor(batch_gamma, dtype=torch.float32, device=device)
            alpha = torch.tensor(batch_alpha, dtype=torch.float32, device=device)
            
            return {
                'weights': weights.clamp(0.1, 15.0),  # Prevent extreme values
                'gamma': gamma.clamp(1.0, 5.0),
                'alpha': alpha.clamp(0.1, 0.9)
            }
            
        except Exception as e:
            logger.error(f"Error computing batch weights: {str(e)}")
            # Fallback to safe default values
            batch_size = len(langs)
            num_labels = len(self.toxicity_labels)
            return {
                'weights': torch.ones((batch_size, num_labels), dtype=torch.float32, device=device),
                'gamma': torch.full((batch_size, num_labels), 2.0, dtype=torch.float32, device=device),
                'alpha': torch.full((batch_size, num_labels), 0.25, dtype=torch.float32, device=device)
            }

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
class TrainingConfig:
    """Basic training configuration with consolidated default values"""
    # Model parameters
    model_name: str = "xlm-roberta-large"
    max_length: int = 128
    hidden_size: int = 1024
    num_attention_heads: int = 16
    model_dropout: float = 0.0
    freeze_layers: int = 8
    
    # Dataset parameters
    cache_dir: str = 'cached_dataset'
    
    # Training parameters
    batch_size: int = 64
    grad_accum_steps: int = 1
    epochs: int = 10
    lr: float = 2e-5  # Base learning rate
    weight_decay: float = 0.005
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.01
    min_lr_ratio: float = 0.01  # Minimum learning rate will be 1% of base lr
    
    # Cosine scheduler parameters
    num_cycles: int = 3  # Number of cosine cycles
    
    # System parameters
    num_workers: int = 16
    mixed_precision: str = "fp16"
    device: str = None
    activation_checkpointing: bool = False
    tensor_float_32: bool = True
    gc_frequency: int = 500
    
    def __post_init__(self):
        """Initialize and validate configuration"""
        # Validate learning rate first
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.lr}")
        if self.lr < 1e-7:
            raise ValueError(f"Learning rate too small: {self.lr}")
        if self.lr > 1.0:
            raise ValueError(f"Learning rate too large: {self.lr}")
            
        # Validate weight decay and learning rate combination
        if self.weight_decay > 0:
            if self.lr < 1e-4:
                logger.warning(
                    "Weight decay (%.4f) may be too high for learning rate %.2e", 
                    self.weight_decay, self.lr
                )
            # Calculate effective learning rate after weight decay
            effective_lr = self.lr * (1 - self.weight_decay)
            if effective_lr < 1e-7:
                logger.warning(
                    "Effective learning rate %.2e after weight decay may be too small",
                    effective_lr
                )
        
        # Rest of the validation checks
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {self.batch_size}")
        if self.grad_accum_steps <= 0:
            raise ValueError(f"Invalid grad_accum_steps: {self.grad_accum_steps}")
        if self.epochs <= 0:
            raise ValueError(f"Invalid epochs: {self.epochs}")
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
        if not 0 < self.min_lr_ratio < 1:
            raise ValueError(f"Invalid min_lr_ratio: {self.min_lr_ratio}")
            
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
        else:
            self.device = torch.device('cpu')
            if self.mixed_precision != "no":
                print("Warning: Mixed precision not supported on CPU. Disabling.")
                self.mixed_precision = "no"
        
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
        
        # Set use_mixed_precision flag
        self.use_mixed_precision = self.mixed_precision != "no"

    def validate_model_config(self, model):
        """Validate configuration against model architecture"""
        try:
            # Validate layer freezing
            if self.freeze_layers > 0:
                total_layers = len(list(model.base_model.encoder.layer))
                if self.freeze_layers > total_layers:
                    raise ValueError(f"Can't freeze {self.freeze_layers} layers in {total_layers}-layer model")
                logger.info(f"Freezing {self.freeze_layers} out of {total_layers} layers")
            
            # Validate parameter groups and weight decay
            param_groups = self.get_param_groups(model)
            if self.weight_decay > 0:
                low_lr_groups = [g for g in param_groups if g['lr'] < 0.01]
                if low_lr_groups:
                    logger.warning("Found parameter groups with low learning rates (< 0.01) and non-zero weight decay:")
                    for group in low_lr_groups:
                        logger.warning(f"Group with lr={group['lr']:.4f}")
            
            return True
        except Exception as e:
            logger.error(f"Model configuration validation failed: {str(e)}")
            raise

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
        return config_dict
    
    def get_param_groups(self, model):
        """Get parameter groups with base learning rate"""
        return [{'params': model.parameters(), 'lr': self.lr}]
        
    @property
    def use_amp(self):
        """Check if AMP should be used based on device and mixed precision setting"""
        return self.device.type == 'cuda' and self.mixed_precision != "no"
    
    @property
    def grad_norm_clip(self):
        """Adaptive gradient clipping based on precision"""
        if self.mixed_precision == "bf16":
            return 1.5  # BF16 can handle slightly higher gradients than FP16
        if self.mixed_precision == "fp16":
            return 1.0  # Most conservative for FP16 due to lower precision
        return 5.0  # Full precision can handle larger gradients 