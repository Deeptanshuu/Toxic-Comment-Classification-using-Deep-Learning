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

@dataclass
class Config:
    model_name: str = "xlm-roberta-large"
    max_length: int = 128
    batch_size: int = 16
    grad_accum_steps: int = 4
    epochs: int = 5
    lr: float = 2e-5
    class_weights: dict = None
    languages: list = None
    device: str = None
    num_gpus: int = None

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
        self.num_gpus = torch.cuda.device_count()

# Custom Dataset
class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        
        # Handle both possible text column names
        if 'comment_text' in df.columns:
            self.texts = df['comment_text'].values
        elif 'text' in df.columns:
            self.texts = df['text'].values
        else:
            raise ValueError("Dataset must contain either 'comment_text' or 'text' column")
            
        self.labels = df[list(Config().class_weights.keys())].values
        self.langs = df['lang'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        lang = self.langs[idx]
        label = self.labels[idx]
        
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train toxic comment classifier')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-large', help='Model name')
    return parser.parse_args()

# Initialize Model
def init_model(config):
    model = XLMRobertaForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(config.class_weights),
        problem_type="multi_label_classification"
    )
    
    if config.num_gpus > 1:
        model = nn.DataParallel(model)
    
    return model.to(config.device)

# Training Loop
def train(model, train_loader, val_loader, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = WeightedFocalLoss()
    
    best_auc = 0
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = {
                'input_ids': batch['input_ids'].to(config.device),
                'attention_mask': batch['attention_mask'].to(config.device),
                'labels': batch['labels'].to(config.device)
            }
            
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, inputs['labels'])
            loss = loss / config.grad_accum_steps  # Normalize loss for gradient accumulation
            
            loss.backward()
            
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item() * config.grad_accum_steps  # Denormalize for logging
            
            # Log metrics
            if batch_idx % 50 == 0:
                wandb.log({
                    'train_loss': total_loss/(batch_idx+1),
                    'lr': scheduler.get_last_lr()[0]
                })
        
        # Validation
        val_auc = evaluate(model, val_loader, config)
        print(f"Epoch {epoch+1}: Val AUC = {val_auc:.4f}")
        wandb.log({'val_auc': val_auc, 'epoch': epoch+1})
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"best_model_{config.model_name}.pt")
            
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

# Main
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Initialize config with command line arguments
    config = Config(
        model_name=args.model_name,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        lr=args.lr
    )
    
    # Initialize Weights & Biases
    wandb.init(project="toxic-comments", config=asdict(config))
    
    # Load data
    train_df = pd.read_csv("dataset/split/train.csv")
    val_df = pd.read_csv("dataset/split/val.csv")
    
    # Initialize tokenizer and model
    tokenizer = XLMRobertaTokenizer.from_pretrained(config.model_name)
    model = init_model(config)
    
    # Create datasets and loaders
    train_dataset = ToxicDataset(train_df, tokenizer)
    val_dataset = ToxicDataset(val_df, tokenizer)
    
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
    
    # Save final model
    torch.save(model.state_dict(), f"final_model_{config.model_name}.pt")
    tokenizer.save_pretrained(config.model_name)
