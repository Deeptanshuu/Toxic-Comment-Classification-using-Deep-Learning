from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import torch
import numpy as np
from pathlib import Path

@dataclass
class DynamicClassWeights:
    """Handles dynamic class weights per language"""
    weights_file: str = 'weights/language_class_weights.json'
    
    def __post_init__(self):
        self.load_weights()
        
    def load_weights(self):
        """Load language-specific weights from JSON file"""
        with open(self.weights_file, 'r') as f:
            self.weights_data = json.load(f)
            self.weights = self.weights_data['weights']
            
        # Get list of toxicity columns in order
        self.toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Default weights (English)
        self.default_weights = torch.tensor([
            self.weights['en'][col]['1'] for col in self.toxicity_columns
        ])
    
    def get_weights_for_batch(self, langs: List[str], device: torch.device) -> torch.Tensor:
        """Get language-specific weights for each sample in the batch"""
        batch_weights = []
        
        for lang in langs:
            # Get weights for this language
            lang_weights = [
                self.weights[lang][col]['1'] 
                for col in self.toxicity_columns
            ]
            batch_weights.append(lang_weights)
        
        # Convert to tensor and move to device
        weights = torch.tensor(batch_weights, dtype=torch.float32).to(device)
        
        # Take mean across batch if shape mismatch
        if len(weights.shape) > 1:
            weights = weights.mean(dim=0)
            
        return weights

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
class TrainingConfig:
    """Training configuration with optimized defaults"""
    # Model parameters
    model_name: str = "xlm-roberta-large"
    max_length: int = 128
    num_labels: int = 6
    
    # Training parameters
    batch_size: int = 32
    grad_accum_steps: int = 4
    epochs: int = 10
    lr: float = 1e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Mixed precision parameters
    fp16: bool = True
    mixed_precision: str = 'bf16'
    
    # System parameters
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    gc_frequency: int = 100
    
    # Optimization flags
    activation_checkpointing: bool = False
    tensor_float_32: bool = True
    
    def __post_init__(self):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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

    @property
    def device(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_optimizer_groups(self, model: torch.nn.Module) -> list:
        """
        Create parameter groups for discriminative learning rates.
        """
        return [
            {'params': model.roberta.parameters(), 'lr': self.lr},
            {'params': model.classifier.parameters(), 'lr': self.lr}
        ]

    def get_summary(self) -> Dict:
        return {
            'grad_norm_max': self.max_grad_norm,
            'throughput_avg': self.batch_size / self.epoch_times[-1] if self.epoch_times else "Calculating...",
            'peak_memory_gb': self.batch_size * self.num_workers * 4 / 1024**3 if torch.cuda.is_available() else 0
        } 