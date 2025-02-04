import torch
import torch.nn as nn
from transformers import (
    XLMRobertaForSequenceClassification, 
    XLMRobertaTokenizer,
    get_cosine_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
import wandb
import argparse
from dataclasses import dataclass, asdict
import os
import warnings
from torch.cuda.amp import autocast, GradScaler
import time
from datetime import datetime, timedelta
import psutil
import GPUtil
import json
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from itertools import chain
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

from .training_config import TrainingConfig, DynamicClassWeights, MetricsTracker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

@dataclass
class Config:
    model_name: str = "xlm-roberta-large"
    max_length: int = 128
    batch_size: int = 48  # Optimized for RTX 6000
    grad_accum_steps: int = 1  # Reduced since we increased batch size
    epochs: int = 5
    lr: float = 2e-5
    warmup_steps: int = 500
    languages: list = None
    device: str = None
    fp16: bool = True
    num_workers: int = 8  # Optimized for Xeon Gold
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    def __post_init__(self):
        # Load language-specific weights
        with open('weights/language_class_weights.json', 'r') as f:
            weights_data = json.load(f)
            self.lang_weights = weights_data['weights']
            
        # Set available languages from the weights file
        self.languages = list(self.lang_weights.keys())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_model(config):
    """Initialize model"""
    model = XLMRobertaForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(config.lang_weights),
        problem_type="multi_label_classification"
    )
    
    return model.to(config.device)

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, config, is_best=False):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if config.ddp else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'config': asdict(config)
    }
    
    # Save latest checkpoint
    checkpoint_path = Path('weights') / 'latest_checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately
    if is_best:
        best_path = Path('weights') / 'best_model.pt'
        torch.save(checkpoint, best_path)
        
        # Log best model to wandb
        if not config.ddp or (config.ddp and dist.get_rank() == 0):
            artifact = wandb.Artifact(
                name=f"best-model-auc{metrics['auc']:.4f}",
                type="model",
                description=f"Best model checkpoint with AUC {metrics['auc']:.4f}"
            )
            artifact.add_file(str(best_path))
            wandb.log_artifact(artifact)

def load_checkpoint(model, optimizer, scheduler, scaler, config):
    """Load training checkpoint"""
    checkpoint_path = Path('weights') / 'latest_checkpoint.pt'
    if not checkpoint_path.exists():
        return 0, 0  # Start from epoch 0
        
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    # Load model state
    if config.ddp:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if it exists
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load scaler state if it exists
    if scaler and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'] + 1, checkpoint['metrics'].get('auc', 0)

def train(model, train_loader, val_loader, config):
    """Optimized training loop with language-specific weights"""
    
    # Initialize optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=config.fp16)
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = config.warmup_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Try to load checkpoint
    start_epoch, best_auc = load_checkpoint(model, optimizer, scheduler, scaler, config)
    
    # Initialize loss function with language-specific weights
    loss_fn = WeightedFocalLoss()
    start_time = time.time()
    
    # Training loop
    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_loss = 0
        epoch_start = time.time()
        
        # Enable automatic mixed precision
        with autocast(enabled=config.fp16):
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to GPU and clear cache if needed
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                
                inputs = {
                    'input_ids': batch['input_ids'].to(config.device, non_blocking=True),
                    'attention_mask': batch['attention_mask'].to(config.device, non_blocking=True),
                    'labels': batch['labels'].to(config.device, non_blocking=True)
                }
                
                # Forward pass with gradient scaling
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, inputs['labels'], batch['lang'])
                scaled_loss = scaler.scale(loss)
                
                # Backward pass with gradient scaling
                scaled_loss.backward()
                
                # Gradient clipping and optimizer step
                if (batch_idx + 1) % config.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Log metrics
                if batch_idx % 50 == 0:
                    gpu_stats = get_gpu_stats()
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + (batch_idx / len(train_loader)),
                        'system/gpu_utilization': gpu_stats['utilization'],
                        'system/gpu_memory': gpu_stats['memory'],
                        'system/gpu_temp': gpu_stats['temperature']
                    })
        
        # Validation
        val_metrics = evaluate(model, val_loader, config)
        
        # Save checkpoint
        is_best = val_metrics['auc'] > best_auc
        if is_best:
            best_auc = val_metrics['auc']
        
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch, val_metrics, config, is_best
        )
        
        # Log epoch metrics
        log_epoch_metrics(epoch, val_metrics, time.time() - epoch_start)
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

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

def log_epoch_metrics(epoch, metrics, epoch_time):
    """Log epoch-level metrics"""
    wandb.log({
        'val/auc': metrics['auc'],
        'val/loss': metrics['loss'],
        'val/precision': metrics['precision'],
        'val/recall': metrics['recall'],
        'val/f1': metrics['f1'],
        'time/epoch_seconds': epoch_time,
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train toxic comment classifier')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=2,
                      help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                      help='Learning rate')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-large',
                      help='Model name')
    parser.add_argument('--fp16', action='store_true',
                      help='Use mixed precision training')
    return parser.parse_args()

# Custom Dataset
class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, config):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        
        required_columns = ['comment_text'] + list(config.lang_weights.keys()) + ['lang']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Pre-tokenize all texts for faster training
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
        self.encodings['input_ids'] = torch.cat(self.encodings['input_ids'])
        self.encodings['attention_mask'] = torch.cat(self.encodings['attention_mask'])
        
        # Convert labels to tensor
        self.labels = torch.FloatTensor(df[list(config.lang_weights.keys())].fillna(0).values)
        self.langs = df['lang'].fillna('en').values
        
        # Pin memory for faster transfer to GPU
        if torch.cuda.is_available():
            self.encodings['input_ids'] = self.encodings['input_ids'].pin_memory()
            self.encodings['attention_mask'] = self.encodings['attention_mask'].pin_memory()
            self.labels = self.labels.pin_memory()

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
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        # Load language-specific weights
        with open('weights/language_class_weights.json', 'r') as f:
            weights_data = json.load(f)
            self.lang_weights = weights_data['weights']
        
        # Get list of toxicity columns in order
        self.toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Default weights as fallback (use English weights)
        self.default_weights = torch.tensor([
            self.lang_weights['en'][col]['1'] for col in self.toxicity_columns
        ])
    
    def get_weights_for_batch(self, langs, device):
        """Get language-specific weights for each sample in the batch"""
        batch_weights = []
        
        for lang in langs:
            # Get weights for this language
            lang_weights = [
                self.lang_weights[lang][col]['1'] 
                for col in self.toxicity_columns
            ]
            batch_weights.append(lang_weights)
        
        # Convert to tensor and move to device
        weights = torch.tensor(batch_weights, dtype=torch.float32).to(device)
        
        # Take mean across batch if shape mismatch (shouldn't happen but just in case)
        if len(weights.shape) > 1:
            weights = weights.mean(dim=0)
            
        return weights

    def forward(self, preds, targets, langs=None):
        if langs is None:
            # Use default weights if no language information
            weights = self.default_weights.to(preds.device)
        else:
            # Get language-specific weights
            weights = self.get_weights_for_batch(langs, preds.device)
        
        # Calculate focal loss with dynamic weights
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * bce_loss) * weights
        
        return focal_loss.mean()

# Evaluation
def evaluate(model, loader, config):
    """Enhanced evaluation with more metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_langs = []  # Track languages for per-language metrics
    loss_fn = WeightedFocalLoss()
    
    with torch.no_grad():
        for batch in loader:
            inputs = {
                'input_ids': batch['input_ids'].to(config.device),
                'attention_mask': batch['attention_mask'].to(config.device),
                'labels': batch['labels'].to(config.device)
            }
            
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, inputs['labels'], batch['lang'])
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs.logits).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(batch['labels'].cpu().numpy())
            all_langs.extend(batch['lang'])  # Collect languages
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate overall metrics
    auc = roc_auc_score(all_targets, all_preds, average='macro')
    binary_preds = (all_preds > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, binary_preds, average='macro'
    )
    
    # Calculate per-language metrics
    lang_metrics = {}
    unique_langs = set(all_langs)
    
    for lang in unique_langs:
        lang_mask = np.array(all_langs) == lang
        if np.sum(lang_mask) > 0:  # Only calculate if we have samples for this language
            lang_preds = all_preds[lang_mask]
            lang_targets = all_targets[lang_mask]
            
            try:
                lang_auc = roc_auc_score(lang_targets, lang_preds, average='macro')
                lang_binary_preds = (lang_preds > 0.5).astype(int)
                lang_precision, lang_recall, lang_f1, _ = precision_recall_fscore_support(
                    lang_targets, lang_binary_preds, average='macro'
                )
                
                lang_metrics[lang] = {
                    'auc': lang_auc,
                    'precision': lang_precision,
                    'recall': lang_recall,
                    'f1': lang_f1
                }
            except Exception as e:
                print(f"Warning: Could not calculate metrics for language {lang}: {str(e)}")
    
    # Log per-language metrics to wandb
    for lang, metrics in lang_metrics.items():
        wandb.log({
            f'val/{lang}/auc': metrics['auc'],
            f'val/{lang}/precision': metrics['precision'],
            f'val/{lang}/recall': metrics['recall'],
            f'val/{lang}/f1': metrics['f1']
        })
    
    return {
        'auc': auc,
        'loss': total_loss / len(loader),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'lang_metrics': lang_metrics
    }

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
    for label, prob in zip(config.lang_weights.keys(), probabilities):
        results[label] = float(prob)
    
    return results

def create_dataloaders(train_dataset, val_dataset, config):
    """Create optimized DataLoaders"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return train_loader, val_loader

def main():
    """Main training function"""
    try:
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use first GPU
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Parse arguments and initialize config
        args = parse_args()
        config = Config(
            model_name=args.model_name,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            epochs=args.epochs,
            lr=args.lr,
            fp16=args.fp16
        )
        
        # Initialize wandb
        wandb.init(project="toxic-comments", config=asdict(config))
        print("Initialized wandb")
        
        # Load data
        print("Loading datasets...")
        train_df = pd.read_csv("dataset/split/train.csv")
        val_df = pd.read_csv("dataset/split/val.csv")
        print(f"Loaded {len(train_df)} training samples")
        print(f"Loaded {len(val_df)} validation samples")
        
        # Initialize model and tokenizer
        print("Initializing model and tokenizer...")
        tokenizer = XLMRobertaTokenizer.from_pretrained(config.model_name)
        model = init_model(config)
        
        # Create datasets and dataloaders
        train_dataset = ToxicDataset(train_df, tokenizer, config)
        val_dataset = ToxicDataset(val_df, tokenizer, config)
        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)
        
        print("Starting training...")
        
        # Train
        train(model, train_loader, val_loader, config)
        
        print("Training completed successfully")
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        wandb.finish()
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set better error formatting
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    
    main()
