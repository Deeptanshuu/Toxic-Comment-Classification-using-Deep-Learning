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

class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['comment_text'].values
        self.labels = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
        self.langs = df['lang'].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx]),
            'lang': self.langs[idx]
        }

def load_model(model_path):
    """Load model and tokenizer"""
    try:
        model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
        try:
            tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        except:
            print("Loading base XLM-RoBERTa tokenizer...")
            tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None

def evaluate_model(model, test_loader, device, output_dir):
    """Evaluate model performance"""
    all_predictions = []
    all_labels = []
    all_langs = []
    
    print("\nRunning predictions on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()
            langs = batch['lang']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs.logits).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_langs.extend(langs)
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    langs = np.array(all_langs)
    
    # Calculate overall metrics
    results = calculate_metrics(predictions, labels, langs)
    
    # Save results
    save_results(results, predictions, labels, langs, output_dir)
    
    return results

def calculate_metrics(predictions, labels, langs):
    """Calculate detailed metrics"""
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    unique_langs = np.unique(langs)
    
    results = {
        'overall': {},
        'per_language': {},
        'per_class': {},
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Overall metrics
    results['overall'] = {
        'auc': roc_auc_score(labels, predictions, average='macro'),
        'auc_weighted': roc_auc_score(labels, predictions, average='weighted')
    }
    
    # Binary predictions using 0.5 threshold
    binary_predictions = (predictions > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='macro'
    )
    
    results['overall'].update({
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    # Per-language metrics
    for lang in unique_langs:
        lang_mask = langs == lang
        if lang_mask.sum() > 0:
            try:
                lang_auc = roc_auc_score(
                    labels[lang_mask], 
                    predictions[lang_mask], 
                    average='macro'
                )
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels[lang_mask],
                    binary_predictions[lang_mask],
                    average='macro'
                )
                results['per_language'][lang] = {
                    'auc': lang_auc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'sample_count': int(lang_mask.sum())
                }
            except:
                print(f"Warning: Could not calculate metrics for language {lang}")
    
    # Per-class metrics
    for i, class_name in enumerate(toxicity_types):
        try:
            class_auc = roc_auc_score(labels[:, i], predictions[:, i])
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels[:, i],
                binary_predictions[:, i],
                average='binary'
            )
            results['per_class'][class_name] = {
                'auc': class_auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'positive_samples': int(labels[:, i].sum())
            }
        except:
            print(f"Warning: Could not calculate metrics for class {class_name}")
    
    return results

def plot_confusion_matrices(predictions, labels, langs, output_dir):
    """Plot confusion matrices for each class and language"""
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Create directory for confusion matrices
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    
    # Overall confusion matrices per class
    for i, class_name in enumerate(toxicity_types):
        cm = confusion_matrix(labels[:, i], binary_predictions[:, i])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {class_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(cm_dir, f'cm_{class_name}.png'))
        plt.close()
    
    # Per-language confusion matrices for toxic class
    for lang in np.unique(langs):
        lang_mask = langs == lang
        if lang_mask.sum() > 0:
            cm = confusion_matrix(
                labels[lang_mask, 0],  # toxic class
                binary_predictions[lang_mask, 0]
            )
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Toxic Class - {lang}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(cm_dir, f'cm_toxic_{lang}.png'))
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
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save results')
    
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
        num_workers=4
    )
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, args.output_dir)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 