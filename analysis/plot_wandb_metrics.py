import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import os

def plot_wandb_metrics(run_path: str = None, save_dir: str = 'images'):
    """
    Plot training metrics from wandb logs, showing both epoch-level and batch-level statistics.
    
    Args:
        run_path (str): Optional path to specific wandb run
        save_dir (str): Directory to save the plots
    """
    # Initialize wandb
    api = wandb.Api()
    
    # Get run
    if run_path:
        run = api.run(run_path)
    else:
        runs = api.runs("toxic-comment-classification")
        if not runs:
            raise ValueError("No runs found in the project")
        run = runs[0]
    
    # Get history
    history = pd.DataFrame(run.scan_history())
    
    # Create two figures: one for training metrics and one for class-specific metrics
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (20, 15),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11
    })
    
    # Figure 1: Training Metrics
    fig1, axes1 = plt.subplots(3, 3)
    fig1.suptitle('Training Metrics (Epoch and Batch Level)', fontsize=16, y=0.95)
    
    # Figure 2: Class-specific Metrics
    fig2, axes2 = plt.subplots(3, 2, figsize=(15, 20))
    fig2.suptitle('Class-specific and Language Metrics', fontsize=16, y=0.95)
    
    # Define colors
    colors = {
        'main_line': '#1f77b4',    # Blue
        'avg_line': '#ff7f0e',     # Orange
        'batch_line': '#2ca02c',   # Green
        'annotation': '#d62728',    # Red
        'grid': '#cccccc',         # Light gray
        'critical': '#9467bd',     # Purple for critical classes
        'confidence': '#17becf'     # Cyan for confidence intervals
    }
    
    # === Figure 1: Training Metrics ===
    
    # 1. Epoch-Level Validation Loss (top left)
    ax = axes1[0, 0]
    if 'val/loss' in history.columns:
        val_loss = history['val/loss'].dropna()
        epochs = range(1, len(val_loss) + 1)
        ax.plot(epochs, val_loss, color=colors['main_line'], label='Validation Loss')
        
        # Add moving average
        window_size = min(3, len(val_loss))
        if window_size > 1:
            moving_avg = val_loss.rolling(window=window_size).mean()
            ax.plot(epochs, moving_avg, '--', color=colors['avg_line'], 
                   alpha=0.7, label='Moving Average')
        
        # Add min point annotation
        min_loss = val_loss.min()
        min_epoch = val_loss.idxmin() + 1
        ax.annotate(f'Min: {min_loss:.4f}',
                   xy=(min_epoch, min_loss),
                   xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=colors['annotation'], alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color=colors['annotation']))
        
        ax.set_title('Epoch Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    
    # 2. Batch-Level Training Loss (top middle)
    ax = axes1[0, 1]
    if 'train/step_loss' in history.columns:
        step_loss = history['train/step_loss'].dropna()
        steps = range(1, len(step_loss) + 1)
        ax.plot(steps, step_loss, color=colors['batch_line'], alpha=0.3, label='Batch Loss')
        
        # Add moving average for smoothing
        window_size = min(50, len(step_loss))
        if window_size > 1:
            moving_avg = step_loss.rolling(window=window_size).mean()
            ax.plot(steps, moving_avg, color=colors['main_line'], 
                   label=f'{window_size}-batch Moving Avg')
        
        ax.set_title('Batch-Level Training Loss')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.legend()
    
    # 3. Gradient Statistics (top right)
    ax = axes1[0, 2]
    grad_cols = ['grad/max_norm', 'grad/mean_norm', 'grad/min_norm']
    for col in grad_cols:
        if col in history.columns:
            grad_data = history[col].dropna()
            steps = range(1, len(grad_data) + 1)
            ax.plot(steps, grad_data, alpha=0.5, label=col.split('/')[-1])
    ax.set_title('Gradient Statistics')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')
    ax.legend()
    
    # 4. Learning Rate & Training Progress (middle left)
    ax = axes1[1, 0]
    if 'train/learning_rate/base' in history.columns:
        lr = history['train/learning_rate/base'].dropna()
        steps = range(1, len(lr) + 1)
        ax.plot(steps, lr, color=colors['main_line'], label='Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.legend()
    
    # 5. Overall Metrics (middle middle)
    ax = axes1[1, 1]
    metrics_to_plot = ['val/precision', 'val/recall', 'val/f1', 'val/auc']
    for metric in metrics_to_plot:
        if metric in history.columns:
            data = history[metric].dropna()
            epochs = range(1, len(data) + 1)
            ax.plot(epochs, data, label=metric.split('/')[-1].upper())
    ax.set_title('Overall Validation Metrics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.legend()
    
    # 6. Critical Class Performance (middle right)
    ax = axes1[1, 2]
    critical_classes = ['threat', 'identity_hate', 'severe_toxicity']
    for cls in critical_classes:
        col = f'val/class_metrics/{cls}/f1'
        if col in history.columns:
            data = history[col].dropna()
            epochs = range(1, len(data) + 1)
            ax.plot(epochs, data, label=f'{cls} F1')
    ax.set_title('Critical Class Performance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.legend()
    
    # 7. Language-specific AUC (bottom row)
    languages = ['en', 'ru', 'tr', 'es', 'fr', 'it', 'pt']
    for i, lang in enumerate(languages[:3]):
        ax = axes1[2, i]
        col = f'val/{lang}/auc'
        if col in history.columns:
            data = history[col].dropna()
            epochs = range(1, len(data) + 1)
            ax.plot(epochs, data, color=colors['main_line'], label=f'{lang.upper()} AUC')
            
            # Add confidence intervals if available
            ci_col = f'val/{lang}/auc_ci'
            if ci_col in history.columns:
                ci_data = history[ci_col].dropna()
                if len(ci_data) > 0:
                    ci_lower = [ci[0] for ci in ci_data]
                    ci_upper = [ci[1] for ci in ci_data]
                    ax.fill_between(epochs, ci_lower, ci_upper, 
                                  color=colors['confidence'], alpha=0.2)
            
            ax.set_title(f'{lang.upper()} Performance')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('AUC')
            ax.legend()
    
    # === Figure 2: Class-specific Metrics ===
    
    # Plot class-specific metrics
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    metrics_per_class = ['precision', 'recall', 'f1', 'auc']
    
    for i, cls in enumerate(class_names):
        ax = axes2[i // 2, i % 2]
        for metric in metrics_per_class:
            col = f'val/class_metrics/{cls}/{metric}'
            if col in history.columns:
                data = history[col].dropna()
                epochs = range(1, len(data) + 1)
                ax.plot(epochs, data, label=metric.upper())
                
                # Add confidence intervals if available
                ci_col = f'val/class_metrics/{cls}/{metric}_ci'
                if ci_col in history.columns:
                    ci_data = history[ci_col].dropna()
                    if len(ci_data) > 0:
                        ci_lower = [ci[0] for ci in ci_data]
                        ci_upper = [ci[1] for ci in ci_data]
                        ax.fill_between(epochs, ci_lower, ci_upper, 
                                      alpha=0.2)
        
        ax.set_title(f'{cls.replace("_", " ").title()} Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
    
    # Adjust layouts
    fig1.tight_layout()
    fig2.tight_layout()
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig1.savefig(f'{save_dir}/training_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
    fig2.savefig(f'{save_dir}/class_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"Plots saved to {save_dir}/")
    
    # Show plots
    plt.show()

if __name__ == '__main__':
    try:
        plot_wandb_metrics()
    except Exception as e:
        print(f"Error plotting metrics: {str(e)}")
        print("\nPlease ensure that:")
        print("1. You have wandb installed (pip install wandb)")
        print("2. You are logged in to wandb (wandb login)")
        print("3. You have at least one completed training run")
        print("4. The project name matches your wandb project") 