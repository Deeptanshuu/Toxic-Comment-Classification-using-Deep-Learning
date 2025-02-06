import torch
import torch.nn as nn
import logging
import os
import gc
import wandb
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

logger = logging.getLogger(__name__)

from transformers import (
    XLMRobertaTokenizer,
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
import wandb
from dataclasses import dataclass, asdict
import os
import warnings
from torch.amp import autocast, GradScaler
import time
from datetime import datetime
import GPUtil
import json
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import sys
import signal
import atexit
from pathlib import Path
from model.training_config import TrainingConfig, DynamicClassWeights, EarlyStopping
from model.data.sampler import MultilabelStratifiedSampler
from model.evaluation.threshold_optimizer import ThresholdOptimizer
from model.language_aware_transformer import LanguageAwareTransformer

# Set environment variables if not already set
os.environ['TF_CPP_MIN_LOG_LEVEL'] = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '2')
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

# Initialize global variables with None
_model = None
_optimizer = None
_scheduler = None
_cleanup_handlers = []

def register_cleanup(handler):
    """Register cleanup handlers that will be called on exit"""
    _cleanup_handlers.append(handler)

def cleanup():
    """Cleanup function to be called on exit"""
    global _model, _optimizer, _scheduler
    
    print("\nPerforming cleanup...")
    
    # Call all registered cleanup handlers
    for handler in _cleanup_handlers:
        try:
            handler()
        except Exception as e:
            print(f"Warning: Cleanup handler failed: {str(e)}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Could not clear CUDA cache: {str(e)}")
    
    # Delete model and optimizer
    try:
        if _model is not None:
            del _model
        if _optimizer is not None:
            del _optimizer
        if _scheduler is not None:
            del _scheduler
    except Exception as e:
        print(f"Warning: Error during cleanup: {str(e)}")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Register cleanup function
atexit.register(cleanup)

# Handle termination signals
def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}. Cleaning up...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@dataclass
class Config:
    model_name: str = "xlm-roberta-large"
    max_length: int = 128
    batch_size: int = 32
    grad_accum_steps: int = 4
    epochs: int = 10
    lr: float = 1e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.01
    device: str = None
    fp16: bool = True
    mixed_precision: str = 'bf16'
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    activation_checkpointing: bool = False
    tensor_float_32: bool = True
    gc_frequency: int = 100
    
    def __post_init__(self):
        try:
            # Load language-specific weights
            weights_path = Path('weights/language_class_weights.json')
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
                
            with open(weights_path, 'r') as f:
                weights_data = json.load(f)
                self.lang_weights = weights_data['weights']
        except Exception as e:
            print(f"Error loading language weights: {str(e)}")
            print("Using default weights...")
            self.lang_weights = self._get_default_weights()
            
        # Set device with error handling
        if torch.cuda.is_available():
            try:
                torch.cuda.init()
                self.device = torch.device('cuda')
            except Exception as e:
                print(f"Warning: CUDA initialization failed: {str(e)}")
                self.device = torch.device('cpu')
        
        # Set TF32 if requested and available
        if torch.cuda.is_available() and self.tensor_float_32:
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            else:
                print("Warning: TF32 not supported on this GPU. Disabling.")
                self.tensor_float_32 = False
            
        # Define toxicity labels
        self.toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.num_labels = len(self.toxicity_labels)
    
    def _get_default_weights(self):
        """Provide default weights if loading fails"""
        return {
            'en': {label: {'0': 0.5, '1': 1.0} for label in self.toxicity_labels},
            'default': {label: {'0': 0.5, '1': 1.0} for label in self.toxicity_labels}
        }

def init_model(config):
    """Initialize model with error handling"""
    global _model
    
    try:
        # Initialize custom model
        _model = LanguageAwareTransformer(
            num_labels=config.num_labels,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            model_name=config.model_name,
            dropout=config.model_dropout
        )
        
        # Enable gradient checkpointing if requested
        if config.activation_checkpointing:
            _model.gradient_checkpointing_enable()
        
        # Move model to device
        _model = _model.to(config.device)
        
        return _model
        
    except Exception as e:
        print(f"Fatal error initializing model: {str(e)}")
        raise

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, config, is_best=False):
    """Save training checkpoint with error handling"""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'metrics': metrics,
            'config': asdict(config)
        }
        
        # Create weights directory if it doesn't exist
        Path('weights').mkdir(exist_ok=True)
        
        # Save latest checkpoint
        checkpoint_path = Path('weights') / 'latest_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = Path('weights') / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
            # Log best model to wandb if initialized
            if wandb.run is not None:
                try:
                    artifact = wandb.Artifact(
                        name=f"best-model-auc{metrics['auc']:.4f}",
                        type="model",
                        description=f"Best model checkpoint with AUC {metrics['auc']:.4f}"
                    )
                    artifact.add_file(str(best_path))
                    wandb.log_artifact(artifact)
                except Exception as e:
                    print(f"Warning: Could not log best model to wandb: {str(e)}")
        
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {str(e)}")
        # Try to save with pickle protocol 4 as fallback
        try:
            torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False, pickle_protocol=4)
        except Exception as e2:
            print(f"Error: Could not save checkpoint even with fallback method: {str(e2)}")

def load_checkpoint(model, optimizer, scheduler, scaler, config):
    """Load training checkpoint with error handling"""
    checkpoint_path = Path('weights') / 'latest_checkpoint.pt'
    if not checkpoint_path.exists():
        return 0, 0  # Start from epoch 0
        
    try:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        
        # Load model state with error handling
        try:
            if config.ddp:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load model state: {str(e)}")
            return 0, 0
        
        # Load optimizer state
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {str(e)}")
        
        # Load scheduler state if it exists
        if scheduler and checkpoint['scheduler_state_dict']:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {str(e)}")
        
        # Load scaler state if it exists
        if scaler and checkpoint['scaler_state_dict']:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load scaler state: {str(e)}")
        
        return checkpoint['epoch'] + 1, checkpoint.get('metrics', {}).get('auc', 0)
        
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return 0, 0

def calculate_lang_specific_loss(batch, weighted_loss, device):
    """Helper function to calculate language-specific losses with proper error handling"""
    lang_losses = {}
    try:
        # Get batch size and number of classes
        batch_size = weighted_loss.size(0)
        num_classes = weighted_loss.size(1) if weighted_loss.dim() > 1 else 1
        
        # Convert batch['lang'] to list if it's a tensor
        langs = batch['lang']
        if isinstance(langs, torch.Tensor):
            langs = langs.tolist()
        
        for lang in set(langs):  # Use set for unique languages
            if lang not in lang_losses:
                lang_losses[lang] = []
            
            try:
                # Create mask with correct shape
                lang_mask = torch.tensor([l == lang for l in langs], 
                                       device=device, dtype=torch.bool)
                
                # Skip if no samples for this language
                if not lang_mask.any():
                    continue
                
                # Ensure mask has correct shape for broadcasting
                if weighted_loss.dim() > 1:
                    # For 2D tensor [batch_size, num_classes]
                    lang_mask = lang_mask.view(-1, 1).expand(-1, num_classes)
                
                # Apply mask and calculate mean loss
                masked_loss = weighted_loss[lang_mask].view(-1, num_classes)
                if len(masked_loss) > 0:
                    lang_loss = masked_loss.mean().item()
                    lang_losses[lang].append(lang_loss)
            
            except Exception as e:
                print(f"Warning: Error processing language {lang}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Warning: Error in language-specific loss calculation: {str(e)}")
        return {}
    
    return lang_losses

def train(model, train_loader, val_loader, config):
    """Training loop with enhanced optimization"""
    model.train()
    
    # Calculate training steps
    num_training_steps = len(train_loader) * config.epochs
    steps_per_epoch = len(train_loader)
    
    # Initialize optimizer with automatic gradient accumulation
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    # Initialize OneCycle scheduler for the first half of training
    onecycle_scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        epochs=config.epochs // 2,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        cycle_momentum=True,
        div_factor=25.0,
        final_div_factor=1e4
    )
    
    # Initialize Cosine Annealing with Warm Restarts for the second half
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=steps_per_epoch * 2,  # Restart every 2 epochs
        T_mult=2,  # Double the restart interval after each restart
        eta_min=config.lr / 100  # Minimum learning rate
    )
    
    scaler = GradScaler(enabled=config.fp16)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=3, min_delta=1e-4)
    best_auc = 0
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        # Use appropriate scheduler based on training phase
        scheduler = onecycle_scheduler if epoch < config.epochs // 2 else cosine_scheduler
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move batch to device
                inputs = {
                    k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass with automatic mixed precision
                with autocast(enabled=config.fp16):
                    outputs = model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=inputs['labels']
                    )
                    loss = outputs['loss']
                
                # Scale loss and backward pass
                scaled_loss = loss / config.grad_accum_steps
                scaler.scale(scaled_loss).backward()
                
                # Step if we've accumulated enough gradients
                if (batch_idx + 1) % config.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item()
                
                # Log progress
                if batch_idx % 10 == 0:
                    lr = scheduler.get_last_lr()[0]
                    avg_loss = total_loss / (batch_idx + 1)
                    print(
                        f"\rEpoch {epoch+1}/{config.epochs} "
                        f"[{batch_idx}/{len(train_loader)}] "
                        f"Loss: {avg_loss:.4f} LR: {lr:.2e}",
                        end=""
                    )
                    
                    # Log to wandb
                    if wandb.run is not None:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/epoch': epoch,
                            'train/gate_values': outputs['gate_values'].mean().item(),
                            'train/thresholds': outputs['thresholds'].detach().cpu().numpy()
                        })
                
            except Exception as e:
                print(f"\nError in training batch: {str(e)}")
                continue
        
        # Validation
        val_metrics = evaluate(model, val_loader, config)
        print(f"\nValidation - Loss: {val_metrics['loss']:.4f} AUC: {val_metrics['auc']:.4f}")
        
        # Check early stopping
        if early_stopping(val_metrics['auc'], epoch):
            print(f"Early stopping triggered. Best AUC: {best_auc:.4f}")
            break
            
        # Update best AUC and save model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            save_model(model, config, epoch, best_auc)
        
        # Log validation metrics
        if wandb.run is not None:
            wandb.log({
                'val/loss': val_metrics['loss'],
                'val/auc': val_metrics['auc'],
                'val/epoch': epoch,
                'val/best_auc': best_auc
            })
            
            # Log per-class metrics
            for label, metrics in val_metrics['class_metrics'].items():
                wandb.log({
                    f'val/class/{label}/auc': metrics['auc'],
                    f'val/class/{label}/f1': metrics['f1'],
                    f'val/class/{label}/threshold': metrics['threshold']
                })

def get_grad_norm(model):
    """Calculate gradient norm for monitoring"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_gpu_stats():
    """Get GPU statistics"""
    gpu = GPUtil.getGPUs()[0]
    return {
        'utilization': gpu.load * 100,
        'memory': gpu.memoryUtil * 100,
        'temperature': gpu.temperature
    }

def save_model(model, config, epoch, auc):
    """Save model checkpoint"""
    model_save_path = f"weights/toxic_classifier_{config.model_name}"
    model.save_pretrained(model_save_path)
    
    # Log model as wandb artifact
    artifact = wandb.Artifact(
        name=f"model-epoch{epoch+1}",
        type="model",
        description=f"Model checkpoint from epoch {epoch+1} with AUC {auc:.4f}"
    )
    artifact.add_dir(model_save_path)
    wandb.log_artifact(artifact)

def log_epoch_metrics(epoch, metrics, epoch_time, eta):
    """Log epoch-level metrics to wandb"""
    wandb.log({
        'val/auc': metrics['auc'],
        'val/loss': metrics['loss'],
        'val/precision': metrics['precision'],
        'val/recall': metrics['recall'],
        'val/f1': metrics['f1'],
        'time/epoch_minutes': epoch_time / 60,
        'time/epoch_eta': eta,
        'epoch': epoch + 1
    })
    
    # Log per-language metrics
    for lang, lang_metrics in metrics['lang_metrics'].items():
        wandb.log({
            f'val/{lang}/auc': lang_metrics['auc'],
            f'val/{lang}/precision': lang_metrics['precision'],
            f'val/{lang}/recall': lang_metrics['recall'],
            f'val/{lang}/f1': lang_metrics['f1']
        })

# Custom Dataset
class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, config):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        
        required_columns = ['comment_text'] + config.toxicity_labels + ['lang']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create cache directory
        Path('cache').mkdir(exist_ok=True)
        
        # Generate cache key based on data and config
        cache_key = f"tokenized_{len(df)}_{config.max_length}_{tokenizer.name_or_path.replace('/', '_')}"
        self.cache_file = Path('cache') / f"{cache_key}.pt"
        
        if self.cache_file.exists():
            print(f"Loading cached tokenized data from {self.cache_file}")
            cached_data = torch.load(self.cache_file, map_location='cpu')
            self.encodings = {
                k: cached_data[k].pin_memory() if torch.cuda.is_available() else cached_data[k]
                for k in ['input_ids', 'attention_mask']
            }
        else:
            print("Pre-tokenizing texts...")
            # Process in batches to manage memory
            batch_size = 1000
            self.encodings = {'input_ids': [], 'attention_mask': []}
            
            for i in range(0, len(df), batch_size):
                batch_texts = df['comment_text'].fillna('').iloc[i:i+batch_size].tolist()
                batch_encodings = self.tokenizer(
                    batch_texts,
                    max_length=config.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                self.encodings['input_ids'].append(batch_encodings['input_ids'])
                self.encodings['attention_mask'].append(batch_encodings['attention_mask'])
                
                if i % 10000 == 0:
                    print(f"Processed {i}/{len(df)} texts")
            
            # Concatenate all batches
            self.encodings = {
                k: torch.cat(v).pin_memory() if torch.cuda.is_available() else torch.cat(v)
                for k, v in self.encodings.items()
            }
            
            # Save to cache
            torch.save(self.encodings, self.cache_file)
        
        # Convert labels to tensor
        self.labels = torch.FloatTensor(df[config.toxicity_labels].fillna(0).values)
        if torch.cuda.is_available():
            self.labels = self.labels.pin_memory()
        
        self.langs = df['lang'].fillna('en').values
        
        # Calculate and store feature statistics for distribution shift monitoring
        self._calculate_feature_stats()
    
    def _calculate_feature_stats(self):
        """Calculate feature statistics for monitoring distribution shift"""
        # Calculate mean token usage
        self.token_usage = (self.encodings['attention_mask'] == 1).float().mean(dim=1)
        
        # Calculate class distribution per language
        self.lang_class_dist = {}
        for lang in np.unique(self.langs):
            lang_mask = self.langs == lang
            self.lang_class_dist[lang] = {
                'mean': self.labels[lang_mask].mean(dim=0).numpy(),
                'std': self.labels[lang_mask].std(dim=0).numpy()
            }
    
    def check_distribution_shift(self, other_dataset):
        """Check for distribution shift between this dataset and another"""
        shifts = {}
        
        # Check token usage distribution
        shifts['token_usage'] = {
            'mean_diff': (self.token_usage.mean() - other_dataset.token_usage.mean()).item(),
            'std_diff': (self.token_usage.std() - other_dataset.token_usage.std()).item()
        }
        
        # Check class distribution shifts per language
        for lang in self.lang_class_dist:
            if lang in other_dataset.lang_class_dist:
                mean_diff = np.abs(
                    self.lang_class_dist[lang]['mean'] - 
                    other_dataset.lang_class_dist[lang]['mean']
                )
                std_diff = np.abs(
                    self.lang_class_dist[lang]['std'] - 
                    other_dataset.lang_class_dist[lang]['std']
                )
                shifts[f'lang_{lang}'] = {
                    'mean_diff': mean_diff.tolist(),
                    'std_diff': std_diff.tolist()
                }
        
        return shifts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx],
            'lang': self.langs[idx]
        }
        return item

# Weighted Focal Loss
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, label_smoothing=0.01):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.class_weights = DynamicClassWeights()
    
    def forward(self, preds, targets, langs=None):
        # Apply label smoothing
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Ensure all inputs have correct dimensions [B, C]
        if preds.dim() == 1:
            preds = preds.unsqueeze(-1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        
        # Get weights with proper dimensions
        if langs is None:
            # Use default weights if no language information
            weights = self.class_weights.default_weights.to(preds.device)
            if weights.dim() == 1:
                weights = weights.unsqueeze(0)  # [C] → [1, C]
        else:
            # Get language-specific weights
            weights = self.class_weights.get_weights_for_batch(langs, preds.device)  # [B, C]
        
        # Ensure weights have correct shape through proper broadcasting
        if weights.shape != preds.shape:
            weights = weights.expand_as(preds)  # Match dimensions exactly [B, C]
        
        # Calculate BCE loss with proper reduction
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)  # [B, C]
        
        # Calculate focal weights
        pt = torch.exp(-bce_loss)
        pt = torch.clamp(pt, min=1e-7, max=1.0)  # Numerical stability
        focal_weights = (1 - pt) ** self.gamma
        
        # Apply focal weighting and class weights
        loss = self.alpha * focal_weights * bce_loss  # [B, C]
        weighted_loss = loss * weights  # [B, C]
        
        # First sum over classes to preserve per-sample weighting, then average over batch
        return weighted_loss.sum(dim=-1).mean()

def calculate_metrics(labels, probs, thresholds, class_names):
    """Calculate classification metrics using optimized thresholds"""
    binary_preds = (probs > thresholds).astype(int)
    
    # Calculate overall metrics
    auc = roc_auc_score(labels, probs, average='macro')
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_preds, average='macro'
    )
    
    # Calculate per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            'auc': roc_auc_score(labels[:, i], probs[:, i]),
            'precision': precision_recall_fscore_support(labels[:, i], binary_preds[:, i], average='binary')[0],
            'recall': precision_recall_fscore_support(labels[:, i], binary_preds[:, i], average='binary')[1],
            'f1': precision_recall_fscore_support(labels[:, i], binary_preds[:, i], average='binary')[2],
            'threshold': thresholds[i]
        }
    
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': class_metrics
    }

def evaluate(model, loader, config):
    """Evaluation function for language-aware model"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    all_langs = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = {
                k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['labels']
            )
            
            total_loss += outputs['loss'].item()
            all_labels.append(inputs['labels'].cpu().numpy())
            all_probs.append(outputs['probabilities'].cpu().numpy())
            all_langs.extend(inputs['lang'])
    
    # Calculate metrics
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    
    # Optimize thresholds per language
    threshold_optimizer = ThresholdOptimizer(min_samples=50, class_names=config.toxicity_labels)
    threshold_optimizer.optimize(probs, labels, all_langs)
    
    # Calculate metrics using optimized thresholds
    metrics = calculate_metrics(labels, probs, threshold_optimizer.thresholds, config.toxicity_labels)
    metrics['loss'] = total_loss / len(loader)
    
    return metrics

def predict_toxicity(text, model, tokenizer, config):
    """
    Predict toxicity labels for a given text.
    Returns probabilities for each toxicity type.
    """
    # Prepare model
    model.eval()
    
    # Tokenize text
    encoding = tokenizer(
        text,
        max_length=config.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(config.device)
    attention_mask = encoding['attention_mask'].to(config.device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.sigmoid(outputs.logits)
    
    # Convert to probabilities
    probabilities = predictions[0].cpu().numpy()
    
    # Create results dictionary
    results = {}
    for label, prob in zip(config.toxicity_labels, probabilities):
        results[label] = float(prob)
    return results

class MultilabelStratifiedSampler(torch.utils.data.Sampler):
    """Samples elements maintaining balanced representation of classes per language"""
    def __init__(self, labels, groups, batch_size):
        self.labels = torch.tensor(labels)
        self.groups = np.array(groups)
        self.batch_size = batch_size
        self.unique_groups = np.unique(groups)
        self.num_classes = labels.shape[1]
        
        # Calculate weights for balanced sampling
        self.weights = self._calculate_weights()
        
        # Calculate samples per group ensuring minimum representation
        self.samples_per_group = max(2, self.batch_size // len(self.unique_groups))
        self.remainder = self.batch_size - (self.samples_per_group * len(self.unique_groups))
        
        # Validate minimum samples per group
        if self.samples_per_group < 2:
            print(f"Warning: Batch size {batch_size} too small for {len(self.unique_groups)} groups.")
            print(f"Consider increasing batch size to at least {2 * len(self.unique_groups)}")
        
    def _calculate_weights(self):
        weights = torch.ones(len(self.labels))
        for group in self.unique_groups:
            group_mask = self.groups == group
            group_labels = self.labels[group_mask]
            
            if len(group_labels) == 0:
                continue
            
            # Calculate inverse frequency for each class in this group
            class_weights = []
            for c in range(self.num_classes):
                pos_count = group_labels[:, c].sum()
                if pos_count > 0:
                    w = len(group_labels) / (2 * pos_count)
                    class_weights.append(w)
                else:
                    class_weights.append(1.0)
            
            # Apply weights to samples in this group
            group_weights = torch.tensor(class_weights).mean(dim=0)
            weights[group_mask] = group_weights
        
        return weights
    
    def __iter__(self):
        # Convert weights to probabilities
        probs = self.weights / self.weights.sum()
        
        # Generate indices ensuring each batch has samples from each group
        indices = []
        while len(indices) < len(self.labels):
            batch_indices = []
            
            # First, ensure minimum samples from each group
            for group in self.unique_groups:
                group_mask = self.groups == group
                group_probs = probs[group_mask].clone()
                
                if group_probs.sum() > 0:  # Only sample if group has samples
                    group_probs /= group_probs.sum()
                    
                    # Sample indices for this group
                    try:
                        group_indices = np.random.choice(
                            np.where(group_mask)[0],
                            size=self.samples_per_group,
                            p=group_probs.numpy(),
                            replace=False
                        )
                        batch_indices.extend(group_indices)
                    except ValueError as e:
                        print(f"Warning: Could not sample {self.samples_per_group} samples from group {group}")
                        # Fall back to sampling with replacement if necessary
                        group_indices = np.random.choice(
                            np.where(group_mask)[0],
                            size=self.samples_per_group,
                            p=group_probs.numpy(),
                            replace=True
                        )
                        batch_indices.extend(group_indices)
            
            # Add remaining samples randomly but weighted by class distribution
            if self.remainder > 0:
                remaining_probs = probs.clone()
                remaining_probs /= remaining_probs.sum()
                
                remaining_indices = np.random.choice(
                    len(self.labels),
                    size=self.remainder,
                    p=remaining_probs.numpy(),
                    replace=False
                )
                batch_indices.extend(remaining_indices)
            
            indices.extend(batch_indices)
        
        return iter(indices[:len(self.labels)])
    
    def __len__(self):
        return len(self.labels)

def setup_distributed(config, rank):
    """Initialize distributed training"""
    if config.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '23456'
        
        # Initialize process group
        dist.init_process_group(
            backend=config.dist_backend,
            init_method=config.dist_url,
            world_size=config.world_size,
            rank=rank
        )
        
        # Set device
        torch.cuda.set_device(rank)
        config._device = torch.device(f'cuda:{rank}')
        
        print(f"Initialized process group for rank {rank}")
        return True
    return False

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def create_dataloaders(train_dataset, val_dataset, config, rank=None):
    """Create optimized DataLoaders with distributed support"""
    # Create samplers
    if config.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=config.world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = MultilabelStratifiedSampler(
            labels=train_dataset.labels,
            groups=train_dataset.langs,
            batch_size=config.batch_size
        )
        val_sampler = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return train_loader, val_loader

def check_class_wise_gradient_flow(model):
    """Monitor gradient flow for each class head"""
    grad_stats = {}
    
    try:
        # Get the base model from DDP if needed
        base_model = model.module if hasattr(model, 'module') else model
        
        # Get gradients for class-specific parameters
        classifier_params = list(base_model.classifier.parameters())
        if not classifier_params[-1].grad is None:  # Check if gradients exist
            class_grads = classifier_params[-1].grad  # Shape might be [num_classes, hidden_dim]
            
            # Ensure class_grads has the right shape
            if class_grads.dim() == 1:
                # If 1D, reshape to [1, hidden_dim]
                class_grads = class_grads.unsqueeze(0)
            
            # Calculate statistics along the correct dimension
            grad_stats['mean'] = class_grads.mean(dim=-1).cpu().numpy()  # Mean across hidden dimensions
            grad_stats['std'] = class_grads.std(dim=-1).cpu().numpy()    # Std across hidden dimensions
            grad_stats['norm'] = torch.norm(class_grads, dim=-1).cpu().numpy()  # Norm across hidden dimensions
            
            # Calculate gradient diversity only if we have multiple classes
            if class_grads.size(0) > 1:
                # Reshape for correlation calculation if needed
                flat_grads = class_grads.view(class_grads.size(0), -1)
                # Calculate correlation matrix
                grad_corr = torch.corrcoef(flat_grads)
                grad_stats['correlation'] = grad_corr.cpu().numpy()
    except Exception as e:
        print(f"Warning: Could not calculate gradient stats: {str(e)}")
        return None
    
    return grad_stats

def validate_distribution_shift(train_dataset, val_dataset):
    """Check for distribution shift between train and validation sets"""
    shifts = train_dataset.check_distribution_shift(val_dataset)
    
    # Log shifts to wandb
    wandb.log({
        'distribution_shift/token_usage_mean': shifts['token_usage']['mean_diff'],
        'distribution_shift/token_usage_std': shifts['token_usage']['std_diff']
    })
    
    # Log language-specific shifts
    for lang, lang_shifts in shifts.items():
        if lang.startswith('lang_'):
            for cls_idx, (mean_diff, std_diff) in enumerate(zip(
                lang_shifts['mean_diff'], lang_shifts['std_diff']
            )):
                wandb.log({
                    f'distribution_shift/{lang}_class{cls_idx}_mean': mean_diff,
                    f'distribution_shift/{lang}_class{cls_idx}_std': std_diff
                })
    
    return shifts

def train_worker(rank, config, train_dataset, val_dataset):
    """Worker function for distributed training"""
    # Setup distributed training
    is_distributed = setup_distributed(config, rank)
    
    try:
        # Create model
        model = init_model(config)
        if is_distributed:
            model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        
        # Create dataloaders with proper error handling
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, config, rank
        )
        
        # Initialize wandb only on master process and ensure it's done before training
        if not is_distributed or rank == 0:
            try:
                wandb.init(
                    project="toxic-comment-classification",
                    config=config.to_serializable_dict(),
                    reinit=True
                )
            except Exception as e:
                print(f"Warning: Could not initialize wandb: {str(e)}")
        
        # Train model
        train(model, train_loader, val_loader, config)
        
    except Exception as e:
        print(f"Error in worker {rank}: {str(e)}")
        raise
    finally:
        # Cleanup
        cleanup_distributed()
        if model is not None:
            del model
        torch.cuda.empty_cache()
        gc.collect()

# Training Configuration
TRAINING_CONFIG = {
    "model_name": "xlm-roberta-large",
    "batch_size": 32,
    "grad_accum_steps": 2,
    "epochs": 4,
    "lr": 2e-5,
    "mixed_precision": "bf16",
    "num_workers": 12,
    "activation_checkpointing": True,
    "tensor_float_32": True,
    "gc_frequency": 500,
    "weight_decay": 0.01,
    "max_length": 128,
    "fp16": False,
    "distributed": False,
    "world_size": 1,
    # New architecture parameters
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "model_dropout": 0.1
}

def main():
    """Main training function with enhanced error handling"""
    try:
        # Initialize wandb
        try:
            wandb.init(
                project="toxic-comment-classification",
                name=f"toxic-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=TRAINING_CONFIG
            )
            print("Initialized wandb logging")
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {str(e)}")
        
        # Initialize config with parameters from TRAINING_CONFIG
        config = TrainingConfig(**TRAINING_CONFIG)
        
        # Load datasets with error handling
        print("Loading datasets...")
        try:
            train_df = pd.read_csv("dataset/split/train.csv")
            val_df = pd.read_csv("dataset/split/val.csv")
        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
            raise
        
        # Create datasets with error handling
        try:
            tokenizer = XLMRobertaTokenizer.from_pretrained(config.model_name)
            train_dataset = ToxicDataset(train_df, tokenizer, config)
            val_dataset = ToxicDataset(val_df, tokenizer, config)
        except Exception as e:
            print(f"Error creating datasets: {str(e)}")
            raise
        
        # Create dataloaders and train
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Initialize model
        model = init_model(config)
        
        # Train model
        train(model, train_loader, val_loader, config)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        cleanup()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if wandb.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: Could not finish wandb run: {str(e)}")
        cleanup()

if __name__ == "__main__":
    # Set numerical precision options
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    
    # Run main with error handling
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        cleanup()
        sys.exit(1)
