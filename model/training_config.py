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
    """Handles class weights per language using pre-calculated values"""
    weights_file: str = 'weights/language_class_weights.json'
    
    def __post_init__(self):
        self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.language_columns = ['en', 'es', 'fr', 'it', 'tr', 'pt', 'ru']
        self.weights = {}
        self.default_weights = None
        
        try:
            with open(self.weights_file, 'r') as f:
                data = json.load(f)
                self.weights = data['weights']
                self.default_weights = torch.tensor([
                    float(self.weights['en'][label]['1']) 
                    for label in self.toxicity_labels
                ])
        except Exception as e:
            print(f"Warning: Could not load weights from {self.weights_file}: {str(e)}")
            print("Using default weights...")
            self._initialize_default_weights()
    
    def _initialize_default_weights(self):
        """Initialize safe default weights if loading fails"""
        self.weights = {}
        for lang in self.language_columns:
            self.weights[lang] = {}
            for label in self.toxicity_labels:
                self.weights[lang][label] = {'0': 0.5, '1': 1.0}
        self.default_weights = torch.tensor([1.0] * len(self.toxicity_labels))
    
    def get_weights_for_batch(self, langs: List[str], device: torch.device) -> torch.Tensor:
        """Get language-specific weights for each sample in the batch"""
        try:
            batch_weights = []
            for lang in langs:
                # Get weights for this language, fallback to English if not found
                lang_weights = []
                for label in self.toxicity_labels:
                    try:
                        weight = float(self.weights.get(lang, self.weights['en'])[label]['1'])
                    except (KeyError, ValueError):
                        weight = float(self.weights['en'][label]['1'])
                    lang_weights.append(weight)
                batch_weights.append(lang_weights)
            
            weights = torch.tensor(batch_weights, dtype=torch.float32, device=device)
            return weights.clamp(0.1, 15.0)  # Safety clamp
            
        except Exception as e:
            print(f"Warning: Using default weights due to error: {str(e)}")
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
class TrainingConfig:
    """Basic training configuration with consolidated default values"""
    # Model parameters
    model_name: str = "xlm-roberta-large"
    max_length: int = 128
    hidden_size: int = 1024
    num_attention_heads: int = 16
    model_dropout: float = 0.0
    freeze_layers: int = 8
    
    # Training parameters
    batch_size: int = 64
    grad_accum_steps: int = 1
    epochs: int = 10
    lr: float = 2e-5
    weight_decay: float = 0.005
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.01
    
    # Cosine scheduler parameters
    num_cycles: int = 3  # Number of cosine cycles
    min_lr_ratio: float = 0.1  # Minimum LR as a fraction of max LR
    
    # System parameters
    num_workers: int = 16
    mixed_precision: str = "fp16"
    device: str = None
    activation_checkpointing: bool = False
    tensor_float_32: bool = True
    gc_frequency: int = 500
    
    def __post_init__(self):
        """Initialize and validate configuration"""
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