import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import json
from pathlib import Path

def plot_roc_curves(predictions_path, output_dir=None):
    """
    Plot ROC curves from model predictions
    
    Args:
        predictions_path (str): Path to the .npz file containing predictions
        output_dir (str, optional): Directory to save plots. If None, will use same directory as predictions
    """
    # Load predictions
    data = np.load(predictions_path)
    predictions = data['predictions']
    labels = data['labels']
    langs = data['langs']
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.dirname(predictions_path)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Define toxicity types
    toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Define language mapping
    id_to_lang = {
        0: 'English (en)',
        1: 'Russian (ru)',
        2: 'Turkish (tr)',
        3: 'Spanish (es)',
        4: 'French (fr)',
        5: 'Italian (it)',
        6: 'Portuguese (pt)'
    }
    
    # Plot overall ROC curves (one per class)
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(toxicity_types):
        fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Classes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_all_classes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot per-class ROC curves with confidence intervals
    n_bootstrap = 1000
    n_classes = len(toxicity_types)
    
    for i, class_name in enumerate(toxicity_types):
        plt.figure(figsize=(8, 6))
        
        # Calculate main ROC curve
        fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Plot main curve
        plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.3f})')
        
        # Bootstrap for confidence intervals
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            indices = np.random.randint(0, len(labels), len(labels))
            if len(np.unique(labels[indices, i])) < 2:
                continue
                
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(labels[indices, i], predictions[indices, i])
            
            # Interpolate TPR at mean FPR points
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))
        
        # Calculate confidence intervals
        tprs = np.array(tprs)
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        # Plot confidence interval
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=f'±1 std. dev.')
        
        # Calculate AUC confidence interval
        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)
        plt.plot([], [], ' ', label=f'AUC = {auc_mean:.3f} ± {auc_std:.3f}')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {class_name}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'roc_{class_name}.png'), dpi=300)
        plt.close()
    
    # Plot per-language ROC curves (for toxic class)
    plt.figure(figsize=(10, 8))
    for lang_id, lang_name in id_to_lang.items():
        # Get samples for this language
        lang_mask = langs == lang_id
        if lang_mask.sum() > 0 and len(np.unique(labels[lang_mask, 0])) > 1:
            fpr, tpr, _ = roc_curve(labels[lang_mask, 0], predictions[lang_mask, 0])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{lang_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Language - Toxic Class')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_by_language.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nROC curves have been saved to {plots_dir}")
    print("\nGenerated plots:")
    print("1. roc_all_classes.png - ROC curves for all toxicity classes")
    print("2. roc_[class_name].png - Individual ROC curves with confidence intervals for each class")
    print("3. roc_by_language.png - ROC curves for each language (toxic class)")

if __name__ == '__main__':
    # Use the latest evaluation results
    eval_dir = 'evaluation_results'
    if os.path.exists(eval_dir):
        # Find most recent evaluation directory
        eval_dirs = sorted([d for d in os.listdir(eval_dir) if d.startswith('eval_')], reverse=True)
        if eval_dirs:
            latest_eval = os.path.join(eval_dir, eval_dirs[0])
            predictions_path = os.path.join(latest_eval, 'predictions.npz')
            if os.path.exists(predictions_path):
                plot_roc_curves(predictions_path)
            else:
                print(f"No predictions file found in {latest_eval}")
        else:
            print(f"No evaluation directories found in {eval_dir}")
    else:
        print(f"Evaluation directory {eval_dir} not found") 