import torch
import torch.nn as nn
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import wandb  # For experiment tracking

# Configuration
class Config:
    model_name = "xlm-roberta-large"
    max_length = 128
    batch_size = 16
    grad_accum_steps = 4
    epochs = 5
    lr = 2e-5
    class_weights = {
        'toxic': 0.54,
        'severe_toxic': 5.88,
        'obscene': 1.0,
        'threat': 33.33,
        'insult': 0.91,
        'identity_hate': 5.45
    }
    languages = ['en', 'ru', 'fr', 'it', 'es', 'pt', 'tr']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()

# Custom Dataset
class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        self.texts = df['comment_text'].values
        self.labels = df[Config.class_weights.keys()].values
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
        self.weights = torch.tensor(list(Config.class_weights.values())).to(Config.device)

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * bce_loss) * self.weights
        return focal_loss.mean()

# Initialize Model
def init_model():
    model = XLMRobertaForSequenceClassification.from_pretrained(
        Config.model_name,
        num_labels=len(Config.class_weights),
        problem_type="multi_label_classification"
    )
    
    if Config.num_gpus > 1:
        model = nn.DataParallel(model)
    
    return model.to(Config.device)

# Training Loop
def train(model, train_loader, val_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)
    loss_fn = WeightedFocalLoss()
    
    best_auc = 0
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = {
                'input_ids': batch['input_ids'].to(Config.device),
                'attention_mask': batch['attention_mask'].to(Config.device),
                'labels': batch['labels'].to(Config.device)
            }
            
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, inputs['labels'])
            
            loss.backward()
            
            if (batch_idx + 1) % Config.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item()
            
            # Log metrics
            if batch_idx % 50 == 0:
                wandb.log({
                    'train_loss': total_loss/(batch_idx+1),
                    'lr': scheduler.get_last_lr()[0]
                })
        
        # Validation
        val_auc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: Val AUC = {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"best_model_{Config.model_name}.pt")
            
# Evaluation
def evaluate(model, loader):
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = {
                'input_ids': batch['input_ids'].to(Config.device),
                'attention_mask': batch['attention_mask'].to(Config.device)
            }
            
            outputs = model(**inputs)
            preds.append(torch.sigmoid(outputs.logits).cpu().numpy())
            targets.append(batch['labels'].cpu().numpy())
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return roc_auc_score(targets, preds, average='macro')

# Main
if __name__ == "__main__":
    # Initialize Weights & Biases
    wandb.init(project="toxic-comments", config=vars(Config))
    
    # Load data
    train_df = pd.read_csv("dataset/split/train.csv")
    val_df = pd.read_csv("dataset/split/val.csv")
    
    # Initialize tokenizer and model
    tokenizer = XLMRobertaTokenizer.from_pretrained(Config.model_name)
    model = init_model()
    
    # Create datasets and loaders
    train_dataset = ToxicDataset(train_df, tokenizer)
    val_dataset = ToxicDataset(val_df, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size*2,
        num_workers=4,
        pin_memory=True
    )
    
    # Train
    train(model, train_loader, val_loader)
    
    # Save final model
    torch.save(model.state_dict(), f"final_model_{Config.model_name}.pt")
    tokenizer.save_pretrained(Config.model_name)
