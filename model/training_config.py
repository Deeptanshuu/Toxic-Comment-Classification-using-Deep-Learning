from dataclasses import dataclass
from typing import Dict, Optional
import torch
from pathlib import Path

@dataclass
class TrainingConfig:
    # Model Configuration
    model_name: str = "xlm-roberta-large"
    max_length: int = 128
    dropout: float = 0.1
    
    # Hardware Optimization
    batch_size: int = 64  # Optimized for 24GB VRAM
    num_workers: int = 12  # Xeon Gold optimal
    grad_accum_steps: int = 2  # Effective batch size of 128
    mixed_precision: str = 'bf16'  # Native RTX 6000 support
    tensor_float_32: bool = True  # Enable TF32 math
    activation_checkpointing: bool = True  # Memory optimization
    
    # Training Parameters
    epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gc_frequency: int = 500  # Garbage collection interval
    
    # Optimizer Settings
    learning_rates: Dict[str, float] = None
    weight_decay: float = 0.01
    
    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = {
                'roberta': 1e-5,
                'classifier': 2e-5
            }
        
        # Set device-specific optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self.tensor_float_32
            torch.backends.cudnn.allow_tf32 = self.tensor_float_32
        
    @property
    def device(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_optimizer_groups(self, model: torch.nn.Module) -> list:
        """
        Create parameter groups for discriminative learning rates.
        """
        return [
            {'params': model.roberta.parameters(), 'lr': self.learning_rates['roberta']},
            {'params': model.classifier.parameters(), 'lr': self.learning_rates['classifier']}
        ]

class DynamicClassWeights:
    """
    Manages dynamic class weights that evolve during training.
    """
    def __init__(
        self,
        initial_weights: Dict[str, float],
        min_weights: Dict[str, float],
        max_weights: Dict[str, float],
        decay_rates: Dict[str, float]
    ):
        self.weights = initial_weights.copy()
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.decay_rates = decay_rates
    
    def update(self, epoch: int) -> Dict[str, float]:
        """
        Update weights based on epoch and performance metrics.
        """
        for class_name in self.weights:
            if class_name == 'threat':
                # Decay from max to min
                self.weights[class_name] = max(
                    self.min_weights[class_name],
                    self.max_weights[class_name] - epoch * self.decay_rates[class_name]
                )
            elif class_name == 'identity_hate':
                # Grow from min to max
                self.weights[class_name] = min(
                    self.max_weights[class_name],
                    self.min_weights[class_name] + epoch * self.decay_rates[class_name]
                )
        return self.weights

class MetricsTracker:
    """
    Tracks and logs advanced training metrics.
    """
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics = {
            'grad_norm': [],
            'threat_recall': {'en': [], 'other': []},
            'threat_false_negatives': [],
            'memory_usage': [],
            'throughput': []
        }
    
    def update_grad_norm(self, grad_norm: float):
        self.metrics['grad_norm'].append(grad_norm)
    
    def update_threat_metrics(self, recall_en: float, recall_other: float, false_neg: int):
        self.metrics['threat_recall']['en'].append(recall_en)
        self.metrics['threat_recall']['other'].append(recall_other)
        self.metrics['threat_false_negatives'].append(false_neg)
    
    def update_performance_metrics(self, batch_time: float, memory_used: float):
        self.metrics['throughput'].append(self.config.batch_size / batch_time)
        self.metrics['memory_usage'].append(memory_used)
    
    def get_summary(self) -> Dict:
        return {
            'grad_norm_max': max(self.metrics['grad_norm']),
            'threat_recall_en_avg': sum(self.metrics['threat_recall']['en']) / len(self.metrics['threat_recall']['en']),
            'threat_false_negatives_total': sum(self.metrics['threat_false_negatives']),
            'throughput_avg': sum(self.metrics['throughput']) / len(self.metrics['throughput']),
            'peak_memory_gb': max(self.metrics['memory_usage']) / 1024**3
        } 