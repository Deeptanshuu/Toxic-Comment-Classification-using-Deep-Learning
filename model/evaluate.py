import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime
import argparse
from torch.utils.data import Dataset, DataLoader
import gc
import multiprocessing

# Set matplotlib to non-interactive backend
plt.switch_backend('agg')

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
    all_losses = []
    
    print("\nRunning predictions on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            langs = batch['lang']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.item()
            predictions = torch.sigmoid(outputs.logits).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            all_langs.extend(langs)
            all_losses.append(loss)
            
            # Clear GPU memory
            del input_ids, attention_mask, outputs, labels
            torch.cuda.empty_cache()
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    langs = np.array(all_langs)
    avg_loss = np.mean(all_losses)
    
    # Calculate overall metrics
    results = calculate_metrics(predictions, labels, langs)
    results['overall']['loss'] = avg_loss
    
    # Save results
    save_results(results, predictions, labels, langs, output_dir)
    
    # Generate and save visualizations
    plot_metrics(results, output_dir)
    
    return results

def calculate_metrics(predictions, labels, langs):
    """Calculate detailed metrics using class-specific thresholds"""
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Dynamic threshold calculation based on validation set statistics
    thresholds = {}
    for i, class_name in enumerate(toxicity_types):
        # Find optimal threshold using ROC curve
        fpr, tpr, thresh = roc_curve(labels[:, i], predictions[:, i])
        optimal_idx = np.argmax(tpr - fpr)
        thresholds[class_name] = thresh[optimal_idx]
    
    unique_langs = np.unique(langs)
    
    results = {
        'overall': {},
        'per_language': {},
        'per_class': {},
        'thresholds': thresholds,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Overall metrics
    results['overall'] = {
        'auc': roc_auc_score(labels, predictions, average='macro'),
        'auc_weighted': roc_auc_score(labels, predictions, average='weighted')
    }
    
    # Binary predictions using dynamic thresholds
    threshold_array = np.array([thresholds[ct] for ct in toxicity_types])
    binary_predictions = (predictions > threshold_array).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='macro'
    )
    
    results['overall'].update({
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    # Per-language metrics with confidence intervals
    for lang in unique_langs:
        lang_mask = langs == lang
        if lang_mask.sum() > 0:
            try:
                # Calculate metrics
                metrics = calculate_language_metrics(
                    labels[lang_mask],
                    predictions[lang_mask],
                    binary_predictions[lang_mask]
                )
                results['per_language'][lang] = metrics
            except Exception as e:
                print(f"Warning: Could not calculate metrics for language {lang}: {str(e)}")
    
    # Per-class metrics with detailed statistics
    for i, class_name in enumerate(toxicity_types):
        try:
            metrics = calculate_class_metrics(
                labels[:, i],
                predictions[:, i],
                binary_predictions[:, i],
                thresholds[class_name]
            )
            results['per_class'][class_name] = metrics
        except Exception as e:
            print(f"Warning: Could not calculate metrics for class {class_name}: {str(e)}")
    
    return results

def calculate_language_metrics(labels, predictions, binary_predictions):
    """Calculate detailed metrics for a specific language"""
    auc = roc_auc_score(labels, predictions, average='macro')
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='macro'
    )
    
    # Calculate confidence intervals using bootstrap
    n_bootstrap = 1000
    auc_scores = []
    f1_scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, len(labels), len(labels))
        try:
            auc_scores.append(roc_auc_score(labels[indices], predictions[indices], average='macro'))
            _, _, f1, _ = precision_recall_fscore_support(
                labels[indices], binary_predictions[indices], average='macro'
            )
            f1_scores.append(f1)
        except:
            continue
    
    return {
        'auc': auc,
        'auc_ci': [np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5)],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_ci': [np.percentile(f1_scores, 2.5), np.percentile(f1_scores, 97.5)],
        'sample_count': len(labels)
    }

def calculate_class_metrics(labels, predictions, binary_predictions, threshold):
    """Calculate detailed metrics for a specific class"""
    auc = roc_auc_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='binary'
    )
    
    # Calculate additional metrics
    tn, fp, fn, tp = confusion_matrix(labels, binary_predictions).ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'npv': npv,
        'threshold': threshold,
        'positive_samples': int(labels.sum()),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }

def plot_confusion_matrices(predictions, labels, langs, output_dir, batch_size=10):
    """Plot confusion matrices in batches using class-specific thresholds"""
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Per-class optimal thresholds
    thresholds = {
        'toxic': 0.3991,
        'severe_toxic': 0.2350,
        'obscene': 0.47,
        'threat': 0.3614,
        'insult': 0.3906,
        'identity_hate': 0.2533
    }
    
    threshold_array = np.array([thresholds[ct] for ct in toxicity_types])
    binary_predictions = (predictions > threshold_array).astype(int)
    
    # Create directory for confusion matrices
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    
    # Plot confusion matrices in batches
    def plot_batch(items, item_type='class'):
        for item in items:
            if item_type == 'class':
                i = toxicity_types.index(item)
                cm = confusion_matrix(labels[:, i], binary_predictions[:, i])
                title = f'Confusion Matrix - {item}\n(threshold={thresholds[item]:.2f})'
                filename = f'cm_{item}.png'
            else:  # language
                lang_mask = langs == item
                if lang_mask.sum() > 0:
                    cm = confusion_matrix(
                        labels[lang_mask, 0],
                        binary_predictions[lang_mask, 0]
                    )
                    title = f'Confusion Matrix - Toxic Class - {item}\n(threshold={thresholds["toxic"]:.2f})'
                    filename = f'cm_toxic_{item}.png'
                else:
                    continue
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(cm_dir, filename))
            plt.close('all')
    
    # Process classes in batches
    for i in range(0, len(toxicity_types), batch_size):
        batch = toxicity_types[i:i + batch_size]
        plot_batch(batch, 'class')
        gc.collect()
    
    # Process languages in batches
    unique_langs = np.unique(langs)
    for i in range(0, len(unique_langs), batch_size):
        batch = unique_langs[i:i + batch_size]
        plot_batch(batch, 'language')
        gc.collect()

def plot_metrics(results, output_dir):
    """Generate detailed visualization plots"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot per-class performance
    plt.figure(figsize=(12, 6))
    classes = list(results['per_class'].keys())
    metrics = ['auc', 'precision', 'recall', 'f1']
    
    for metric in metrics:
        values = [results['per_class'][c][metric] for c in classes]
        plt.plot(classes, values, marker='o', label=metric.upper())
    
    plt.title('Per-Class Performance Metrics')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'per_class_metrics.png'))
    plt.close()
    
    # Plot per-language performance
    plt.figure(figsize=(10, 6))
    languages = list(results['per_language'].keys())
    auc_scores = [results['per_language'][lang]['auc'] for lang in languages]
    sample_counts = [results['per_language'][lang]['sample_count'] for lang in languages]
    
    # Create bubble plot
    plt.scatter(languages, auc_scores, s=[count/50 for count in sample_counts], alpha=0.6)
    plt.title('AUC Scores by Language (bubble size = sample count)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'language_performance.png'))
    plt.close()

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def save_results(results, predictions, labels, langs, output_dir):
    """Save evaluation results and plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to JSON serializable format
    serializable_results = convert_to_serializable(results)
    
    # Save detailed metrics
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save raw predictions for further analysis
    np.savez_compressed(
        os.path.join(output_dir, 'predictions.npz'),
        predictions=predictions,
        labels=labels,
        langs=langs
    )
    
    # Plot confusion matrices
    plot_confusion_matrices(predictions, labels, langs, output_dir)
    
    # Print detailed summary
    print("\nEvaluation Results:")
    print("-" * 80)
    print(f"Overall Metrics:")
    print(f"  AUC (macro): {results['overall']['auc']:.4f}")
    print(f"  F1 (macro): {results['overall']['f1']:.4f}")
    print(f"  Loss: {results['overall'].get('loss', 'N/A')}")
    
    print("\nPer-Language Performance:")
    for lang, metrics in results['per_language'].items():
        print(f"\n{lang} (n={metrics['sample_count']}):")
        print(f"  AUC: {metrics['auc']:.4f} (95% CI: [{metrics['auc_ci'][0]:.4f}, {metrics['auc_ci'][1]:.4f}])")
        print(f"  F1: {metrics['f1']:.4f} (95% CI: [{metrics['f1_ci'][0]:.4f}, {metrics['f1_ci'][1]:.4f}])")
    
    print("\nPer-Class Performance:")
    for class_name, metrics in results['per_class'].items():
        print(f"\n{class_name}:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  NPV: {metrics['npv']:.4f}")
        print(f"  Threshold: {metrics['threshold']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
        print(f"    FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}")

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
    parser.add_argument('--num_workers', type=int, default=None,
                      help='Number of workers for data loading (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    # Set number of workers
    if args.num_workers is None:
        args.num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    try:
        # Load model
        print("Loading multi-language toxic comment classifier model...")
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
            pin_memory=True
        )
        
        # Evaluate model
        results = evaluate_model(model, test_loader, device, args.output_dir)
        
        print(f"\nEvaluation complete! Results saved to {args.output_dir}")
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise
    
    finally:
        # Cleanup
        plt.close('all')
        gc.collect()
        torch.cuda.empty_cache()
        
        # Close the dataloader explicitly
        if 'test_loader' in locals():
            del test_loader
        if 'test_dataset' in locals():
            del test_dataset
        if 'model' in locals():
            del model
        
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 