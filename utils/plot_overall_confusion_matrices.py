#!/usr/bin/env python
"""
Plot overall confusion matrices for toxic comment classification model.

This script loads the evaluation results and generates overall confusion matrices
for both default (0.5) and optimized thresholds, combining all toxicity classes.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns

def load_results(eval_dir):
    """Load evaluation results from the specified directory."""
    # Load predictions
    predictions_path = os.path.join(eval_dir, 'predictions.npz')
    if not os.path.exists(predictions_path):
        print(f"Error: Predictions file not found at {predictions_path}")
        return None, None, None
    
    # Load the data
    data = np.load(predictions_path)
    predictions = data['predictions']
    labels = data['labels']
    langs = data['langs']
    
    # Load evaluation results for thresholds
    results_path = os.path.join(eval_dir, 'evaluation_results.json')
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return predictions, labels, None
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return predictions, labels, results

def build_overall_confusion_matrices(predictions, labels, results):
    """Build overall confusion matrices for default and optimized thresholds."""
    if results is None:
        # Use default threshold of 0.5 if no results available
        binary_predictions_default = (predictions > 0.5).astype(int)
        binary_predictions_opt = binary_predictions_default
    else:
        # Default threshold (0.5)
        binary_predictions_default = (predictions > 0.5).astype(int)
        
        # Optimized thresholds
        binary_predictions_opt = np.zeros_like(predictions, dtype=int)
        toxicity_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        for i, class_name in enumerate(toxicity_types):
            if i < binary_predictions_opt.shape[1]:  # Make sure we don't go out of bounds
                opt_threshold = results['thresholds']['global'][class_name]['threshold']
                binary_predictions_opt[:, i] = (predictions[:, i] > opt_threshold).astype(int)
    
    # Reshape to create overall confusion matrices
    # Flatten all predictions and labels across classes
    all_preds_default = binary_predictions_default.flatten()
    all_preds_opt = binary_predictions_opt.flatten()
    all_labels = labels.flatten()
    
    # Calculate overall confusion matrices
    cm_default = confusion_matrix(all_labels, all_preds_default)
    cm_opt = confusion_matrix(all_labels, all_preds_opt)
    
    return cm_default, cm_opt

def plot_confusion_matrices(cm_default, cm_opt, output_dir):
    """Plot and save confusion matrices."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Function to plot a single confusion matrix
    def plot_cm(cm, title, filename):
        plt.figure(figsize=(12, 10))
        
        # Normalize for percentages
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        # Plot using seaborn for better visualization
        ax = sns.heatmap(
            cm_norm, 
            annot=True, 
            fmt='.2%', 
            cmap='Blues',
            xticklabels=['Not Toxic', 'Toxic'],
            yticklabels=['Not Toxic', 'Toxic'],
            annot_kws={"size": 16}
        )
        
        # Increase font size of tick labels
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
        
        # Add raw counts as separate text with larger spacing
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # Format large numbers with commas
                count_text = f"n={cm[i, j]:,}"
                
                # Use larger font size and adjust position to avoid overlap
                plt.text(
                    j + 0.5, i + 0.7, count_text,
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > 0.5 else 'black',
                    fontsize=12,
                    fontweight='bold'
                )
        
        plt.title(title, fontsize=18)
        plt.ylabel('True Label', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close()
    
    # Plot confusion matrices
    plot_cm(
        cm_default, 
        'Overall Confusion Matrix - Default Threshold (0.5)', 
        'overall_confusion_matrix_default.png'
    )
    plot_cm(
        cm_opt, 
        'Overall Confusion Matrix - Optimized Thresholds', 
        'overall_confusion_matrix_optimized.png'
    )
    
    # Calculate metrics
    tn_default, fp_default, fn_default, tp_default = cm_default.ravel()
    tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()
    
    # Calculate precision, recall, F1
    precision_default = tp_default / (tp_default + fp_default) if (tp_default + fp_default) > 0 else 0
    recall_default = tp_default / (tp_default + fn_default) if (tp_default + fn_default) > 0 else 0
    f1_default = 2 * (precision_default * recall_default) / (precision_default + recall_default) if (precision_default + recall_default) > 0 else 0
    
    precision_opt = tp_opt / (tp_opt + fp_opt) if (tp_opt + fp_opt) > 0 else 0
    recall_opt = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0
    f1_opt = 2 * (precision_opt * recall_opt) / (precision_opt + recall_opt) if (precision_opt + recall_opt) > 0 else 0
    
    # Create a markdown report
    with open(os.path.join(output_dir, 'overall_confusion_matrix_report.md'), 'w') as f:
        f.write('# Overall Confusion Matrix Report\n\n')
        
        # Default threshold
        f.write('## Default Threshold (0.5)\n\n')
        f.write('|                 | Predicted Not Toxic | Predicted Toxic |\n')
        f.write('|-----------------|--------------------:|--------------:|\n')
        f.write(f'| True Not Toxic | {tn_default:,} | {fp_default:,} |\n')
        f.write(f'| True Toxic     | {fn_default:,} | {tp_default:,} |\n\n')
        f.write(f'- Accuracy: {(tp_default + tn_default) / cm_default.sum():.4f}\n')
        f.write(f'- Precision: {precision_default:.4f}\n')
        f.write(f'- Recall: {recall_default:.4f}\n')
        f.write(f'- F1 Score: {f1_default:.4f}\n\n')
        
        # Optimized thresholds
        f.write('## Optimized Thresholds\n\n')
        f.write('|                 | Predicted Not Toxic | Predicted Toxic |\n')
        f.write('|-----------------|--------------------:|--------------:|\n')
        f.write(f'| True Not Toxic | {tn_opt:,} | {fp_opt:,} |\n')
        f.write(f'| True Toxic     | {fn_opt:,} | {tp_opt:,} |\n\n')
        f.write(f'- Accuracy: {(tp_opt + tn_opt) / cm_opt.sum():.4f}\n')
        f.write(f'- Precision: {precision_opt:.4f}\n')
        f.write(f'- Recall: {recall_opt:.4f}\n')
        f.write(f'- F1 Score: {f1_opt:.4f}\n\n')
        
        # Improvement
        f.write('## Improvement from Optimized Thresholds\n\n')
        f.write(f'- Accuracy: {((tp_opt + tn_opt) / cm_opt.sum()) - ((tp_default + tn_default) / cm_default.sum()):.4f}\n')
        f.write(f'- Precision: {precision_opt - precision_default:.4f}\n')
        f.write(f'- Recall: {recall_opt - recall_default:.4f}\n')
        f.write(f'- F1 Score: {f1_opt - f1_default:.4f}\n')
    
    # Print summary to console
    print("\n========== Overall Performance Summary ==========")
    print("Default Threshold (0.5):")
    print(f"  - Accuracy: {(tp_default + tn_default) / cm_default.sum():.4f}")
    print(f"  - Precision: {precision_default:.4f}")
    print(f"  - Recall: {recall_default:.4f}")
    print(f"  - F1 Score: {f1_default:.4f}")
    
    print("\nOptimized Thresholds:")
    print(f"  - Accuracy: {(tp_opt + tn_opt) / cm_opt.sum():.4f}")
    print(f"  - Precision: {precision_opt:.4f}")
    print(f"  - Recall: {recall_opt:.4f}")
    print(f"  - F1 Score: {f1_opt:.4f}")
    
    print("\nImprovement:")
    print(f"  - Accuracy: {((tp_opt + tn_opt) / cm_opt.sum()) - ((tp_default + tn_default) / cm_default.sum()):.4f}")
    print(f"  - Precision: {precision_opt - precision_default:.4f}")
    print(f"  - Recall: {recall_opt - recall_default:.4f}")
    print(f"  - F1 Score: {f1_opt - f1_default:.4f}")
    
    # Plot improvements in bar chart - increased figure size and font sizes
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    default_values = [
        (tp_default + tn_default) / cm_default.sum(),
        precision_default, 
        recall_default,
        f1_default
    ]
    optimized_values = [
        (tp_opt + tn_opt) / cm_opt.sum(),
        precision_opt,
        recall_opt,
        f1_opt
    ]
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, default_values, width, label='Default Threshold (0.5)')
    plt.bar(x + width/2, optimized_values, width, label='Optimized Thresholds')
    
    plt.ylabel('Score', fontsize=16)
    plt.title('Performance Improvement from Threshold Optimization', fontsize=18)
    plt.xticks(x, metrics, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_metrics_comparison.png'), dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot overall confusion matrices from evaluation results')
    parser.add_argument('--eval_dir', type=str, default=None,
                        help='Path to the evaluation results directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the plots (defaults to eval_dir/plots)')
    
    args = parser.parse_args()
    
    # If no eval_dir provided, find the most recent one
    if args.eval_dir is None:
        base_eval_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluation_results')
        eval_dirs = sorted([d for d in os.listdir(base_eval_dir) if d.startswith('eval_')])
        
        if not eval_dirs:
            print("Error: No evaluation results found")
            sys.exit(1)
        
        args.eval_dir = os.path.join(base_eval_dir, eval_dirs[-1])  # Most recent one
    
    print(f"Using evaluation results from: {args.eval_dir}")
    
    # If no output_dir provided, use eval_dir/plots
    if args.output_dir is None:
        args.output_dir = os.path.join(args.eval_dir, 'overall_plots')
    
    # Load results
    predictions, labels, results = load_results(args.eval_dir)
    
    if predictions is None:
        print("Error: Could not load evaluation results")
        sys.exit(1)
    
    # Build confusion matrices
    cm_default, cm_opt = build_overall_confusion_matrices(predictions, labels, results)
    
    # Plot and save confusion matrices
    plot_confusion_matrices(cm_default, cm_opt, args.output_dir)
    
    print(f"\nConfusion matrix plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()