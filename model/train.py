import torch
import torch.nn as nn
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import wandb
import argparse
from dataclasses import dataclass, asdict
import os
import warnings
from torch.cuda.amp import autocast, GradScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

@dataclass
class Config:
    model_name: str = "xlm-roberta-large"
    max_length: int = 128
    batch_size: int = 32
    grad_accum_steps: int = 2
    epochs: int = 5
    lr: float = 2e-5
    warmup_steps: int = 500
    class_weights: dict = None
    languages: list = None
    device: str = None
    fp16: bool = True  # Enable mixed precision training

    def __post_init__(self):
        self.class_weights = {
            'toxic': 0.54,
            'severe_toxic': 5.88,
            'obscene': 1.0,
            'threat': 33.33,
            'insult': 0.91,
            'identity_hate': 5.45
        }
        self.languages = ['en', 'ru', 'fr', 'it', 'es', 'pt', 'tr']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_model(config):
    """Initialize model"""
    model = XLMRobertaForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(config.class_weights),
        problem_type="multi_label_classification"
    )
    
    return model.to(config.device)

def train(model, train_loader, val_loader, config):
    """Training loop with mixed precision support"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = WeightedFocalLoss()
    scaler = GradScaler() if config.fp16 else None
    
    best_auc = 0
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            with autocast(enabled=config.fp16):
                inputs = {
                    'input_ids': batch['input_ids'].to(config.device),
                    'attention_mask': batch['attention_mask'].to(config.device),
                    'labels': batch['labels'].to(config.device)
                }
                
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, inputs['labels'])
                loss = loss / config.grad_accum_steps
            
            if config.fp16:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % config.grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % config.grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * config.grad_accum_steps
            
            if batch_idx % 50 == 0:
                wandb.log({
                    'train_loss': total_loss/(batch_idx+1),
                    'lr': scheduler.get_last_lr()[0]
                })
        
        scheduler.step()
        
        val_auc = evaluate(model, val_loader, config)
        print(f"Epoch {epoch+1}: Val AUC = {val_auc:.4f}")
        wandb.log({'val_auc': val_auc, 'epoch': epoch+1})
        
        if val_auc > best_auc:
            best_auc = val_auc
            model_save_path = f"weights/toxic_classifier_{config.model_name}"
            model.save_pretrained(model_save_path)

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
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        
        # Verify required columns
        required_columns = ['comment_text'] + list(Config().class_weights.keys()) + ['lang']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {list(df.columns)}")
        
        # Get text and labels
        self.texts = df['comment_text'].fillna('').values  # Handle any potential NaN values
        self.labels = df[list(Config().class_weights.keys())].fillna(0).values  # Handle any potential NaN values
        self.langs = df['lang'].fillna('en').values  # Default to 'en' for any NaN values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        lang = self.langs[idx]
        label = self.labels[idx]
        
        # Ensure text is not empty
        if not text.strip():
            text = '[UNK]'  # Use unknown token for empty text
        
        encoding = self.tokenizer(
            text,
            max_length=Config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label),
            'lang': lang
        }

# Weighted Focal Loss
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = torch.tensor(list(Config().class_weights.values())).to(Config().device)

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * bce_loss) * self.weights
        return focal_loss.mean()

# Evaluation
def evaluate(model, loader, config):
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = {
                'input_ids': batch['input_ids'].to(config.device),
                'attention_mask': batch['attention_mask'].to(config.device)
            }
            
            outputs = model(**inputs)
            preds.append(torch.sigmoid(outputs.logits).cpu().numpy())
            targets.append(batch['labels'].cpu().numpy())
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return roc_auc_score(targets, preds, average='macro')

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
    for label, prob in zip(config.class_weights.keys(), probabilities):
        results[label] = float(prob)
    
    return results

# Main
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Initialize config
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
    
    try:
        # Load data
        print("Loading datasets...")
        train_df = pd.read_csv("dataset/split/train.csv", encoding='utf-8')
        val_df = pd.read_csv("dataset/split/val.csv", encoding='utf-8')
        print(f"Loaded training set with {len(train_df)} samples")
        print(f"Columns available: {list(train_df.columns)}")
        
        # Initialize model and tokenizer
        print("Initializing model and tokenizer...")
        tokenizer = XLMRobertaTokenizer.from_pretrained(config.model_name)
        model = init_model(config)
        
        # Create datasets
        train_dataset = ToxicDataset(train_df, tokenizer)
        val_dataset = ToxicDataset(val_df, tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size*2,
            num_workers=4,
            pin_memory=True
        )
        
        # Train
        train(model, train_loader, val_loader, config)
        
        # Example predictions
        print("\nTesting model with example texts...")
        examples = [
            "You are a wonderful person!",  # Non-toxic English
            "Va te faire foutre, idiot!",   # Toxic French
            "Eres un estúpido imbécil",     # Toxic Spanish
            "Sei una persona molto gentile", # Non-toxic Italian
        ]
        
        for text in examples:
            print(f"\nText: {text}")
            predictions = predict_toxicity(text, model, tokenizer, config)
            print("Predictions:")
            for label, prob in predictions.items():
                if prob > 0.5:
                    print(f"- {label}: {prob:.2%}")
    
    finally:
        wandb.finish()
