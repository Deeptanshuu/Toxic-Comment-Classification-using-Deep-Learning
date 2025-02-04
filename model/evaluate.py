import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import torch.backends.cudnn as cudnn

# Enable cuDNN auto-tuner
cudnn.benchmark = True

class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['comment_text'].values
        self.labels = torch.FloatTensor(df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values)
        self.langs = df['lang'].values
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-encode all texts
        print("Pre-encoding texts...")
        self.encodings = self.tokenizer(
            list(self.texts),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx],
            'lang': self.langs[idx]
        }

def load_model(model_path):
    """Load model and tokenizer with optimizations"""
    try:
        # Load model with FP16 support
        model = XLMRobertaForSequenceClassification.from_pretrained(
            model_path,
            torchscript=True  # Enable TorchScript optimization
        )
        try:
            tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        except:
            print("Loading base XLM-RoBERTa tokenizer...")
            tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Convert model to TorchScript
        if device.type == 'cuda':
            model = torch.jit.script(model)
        
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None

@torch.no_grad()
def evaluate_model(model, test_loader, device, output_dir):
    """Evaluate model performance with optimizations"""
    all_predictions = []
    all_labels = []
    all_langs = []
    
    print("\nRunning predictions on test set...")
    # Enable automatic mixed precision
    with autocast(enabled=True):
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels']
            langs = batch['lang']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs.logits).cpu()
            
            all_predictions.append(predictions)
            all_labels.append(labels)
            all_langs.extend(langs)
    
    # Concatenate results efficiently
    predictions = torch.cat(all_predictions, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    langs = np.array(all_langs)
    
    # Calculate metrics
    results = calculate_metrics(predictions, labels, langs)
    save_results(results, predictions, labels, langs, output_dir)
    
    return results

def calculate_metrics(predictions, labels, langs):
    """Calculate metrics efficiently"""
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    unique_langs = np.unique(langs)
    
    results = {
        'overall': {},
        'per_language': {},
        'per_class': {},
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Calculate all metrics at once
    binary_predictions = (predictions > 0.5).astype(np.int32)
    
    # Overall metrics
    results['overall'] = {
        'auc': float(roc_auc_score(labels, predictions, average='macro')),
        'auc_weighted': float(roc_auc_score(labels, predictions, average='weighted'))
    }
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='macro'
    )
    
    results['overall'].update({
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    })
    
    # Per-language metrics (parallel processing could be added here if needed)
    for lang in unique_langs:
        lang_mask = langs == lang
        if lang_mask.sum() > 0:
            try:
                lang_metrics = calculate_lang_metrics(
                    labels[lang_mask],
                    predictions[lang_mask],
                    binary_predictions[lang_mask]
                )
                lang_metrics['sample_count'] = int(lang_mask.sum())
                results['per_language'][lang] = lang_metrics
            except:
                print(f"Warning: Could not calculate metrics for language {lang}")
    
    # Per-class metrics
    for i, class_name in enumerate(toxicity_types):
        try:
            class_metrics = calculate_class_metrics(
                labels[:, i],
                predictions[:, i],
                binary_predictions[:, i]
            )
            results['per_class'][class_name] = class_metrics
        except:
            print(f"Warning: Could not calculate metrics for class {class_name}")
    
    return results

def calculate_lang_metrics(labels, predictions, binary_predictions):
    """Helper function to calculate language-specific metrics"""
    return {
        'auc': float(roc_auc_score(labels, predictions, average='macro')),
        'precision': float(precision_recall_fscore_support(
            labels, binary_predictions, average='macro'
        )[0]),
        'recall': float(precision_recall_fscore_support(
            labels, binary_predictions, average='macro'
        )[1]),
        'f1': float(precision_recall_fscore_support(
            labels, binary_predictions, average='macro'
        )[2])
    }

def calculate_class_metrics(labels, predictions, binary_predictions):
    """Helper function to calculate class-specific metrics"""
    return {
        'auc': float(roc_auc_score(labels, predictions)),
        'precision': float(precision_recall_fscore_support(
            labels, binary_predictions, average='binary'
        )[0]),
        'recall': float(precision_recall_fscore_support(
            labels, binary_predictions, average='binary'
        )[1]),
        'f1': float(precision_recall_fscore_support(
            labels, binary_predictions, average='binary'
        )[2]),
        'positive_samples': int(labels.sum())
    }

def plot_confusion_matrices(predictions, labels, langs, output_dir):
    """Plot confusion matrices efficiently"""
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    binary_predictions = (predictions > 0.5).astype(np.int32)
    
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    
    plt.style.use('dark_background')  # Better looking plots
    
    # Plot matrices in parallel using ThreadPoolExecutor if needed
    for i, class_name in enumerate(toxicity_types):
        cm = confusion_matrix(labels[:, i], binary_predictions[:, i])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {class_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(cm_dir, f'cm_{class_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    for lang in np.unique(langs):
        lang_mask = langs == lang
        if lang_mask.sum() > 0:
            cm = confusion_matrix(
                labels[lang_mask, 0],
                binary_predictions[lang_mask, 0]
            )
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Toxic Class - {lang}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(cm_dir, f'cm_toxic_{lang}.png'), dpi=300, bbox_inches='tight')
            plt.close()

def save_results(results, predictions, labels, langs, output_dir):
    """Save evaluation results and plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrices
    plot_confusion_matrices(predictions, labels, langs, output_dir)
    
    # Print summary
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"Overall AUC (macro): {results['overall']['auc']:.4f}")
    print(f"Overall F1 (macro): {results['overall']['f1']:.4f}")
    
    print("\nPer-Language Performance (AUC):")
    for lang, metrics in results['per_language'].items():
        print(f"{lang}: {metrics['auc']:.4f} (n={metrics['sample_count']})")
    
    print("\nPer-Class Performance (AUC):")
    for class_name, metrics in results['per_class'].items():
        print(f"{class_name}: {metrics['auc']:.4f} (pos={metrics['positive_samples']})")

def main():
    parser = argparse.ArgumentParser(description='Evaluate toxic comment classifier')
    parser.add_argument('--model_path', type=str, default='weights/toxic_classifier_xlm-roberta-large',
                      help='Path to the trained model')
    parser.add_argument('--test_file', type=str, default='dataset/split/test.csv',
                      help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=64,  # Increased batch size
                      help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save results')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of worker processes for data loading')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model(args.model_path)
    
    if model is None:
        return
    
    # Load test data
    print("\nLoading test dataset...")
    test_df = pd.read_csv(args.test_file)
    print(f"Loaded {len(test_df):,} test samples")
    
    # Create test dataset and dataloader
    test_dataset = ToxicDataset(test_df, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, args.output_dir)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 