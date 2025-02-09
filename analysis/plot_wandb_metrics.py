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
    
    # Set up matplotlib style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (20, 15),  # Larger figure for more plots
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11
    })
    
    # Create figure with subplots - 3x3 grid
    fig, axes = plt.subplots(3, 3)
    fig.suptitle('Training Metrics (Epoch and Batch Level)', fontsize=16, y=0.95)
    
    # Define colors
    colors = {
        'main_line': '#1f77b4',    # Blue
        'avg_line': '#ff7f0e',     # Orange
        'batch_line': '#2ca02c',   # Green
        'annotation': '#d62728',    # Red
        'grid': '#cccccc'          # Light gray
    }
    
    # 1. Epoch-Level Validation Loss (top left)
    ax = axes[0, 0]
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
    ax = axes[0, 1]
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
    
    # 3. Epoch Training Loss (top right)
    ax = axes[0, 2]
    if 'train/epoch_loss' in history.columns:
        train_loss = history['train/epoch_loss'].dropna()
        epochs = range(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, color=colors['main_line'], label='Training Loss')
        
        window_size = min(3, len(train_loss))
        if window_size > 1:
            moving_avg = train_loss.rolling(window=window_size).mean()
            ax.plot(epochs, moving_avg, '--', color=colors['avg_line'], 
                   alpha=0.7, label='Moving Average')
        
        ax.set_title('Epoch Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    
    # 4. Validation AUC (middle left)
    ax = axes[1, 0]
    if 'val/auc' in history.columns:
        auc = history['val/auc'].dropna()
        epochs = range(1, len(auc) + 1)
        ax.plot(epochs, auc, color=colors['main_line'], label='Validation AUC')
        
        # Add max point annotation
        max_auc = auc.max()
        max_epoch = auc.idxmax() + 1
        ax.annotate(f'Max: {max_auc:.4f}',
                   xy=(max_epoch, max_auc),
                   xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=colors['annotation'], alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color=colors['annotation']))
        
        ax.set_title('Validation AUC')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.legend()
    
    # 5. Gradient Norm (middle middle)
    ax = axes[1, 1]
    if 'grad/max_norm' in history.columns:
        grad_norm = history['grad/max_norm'].dropna()
        steps = range(1, len(grad_norm) + 1)
        ax.plot(steps, grad_norm, color=colors['batch_line'], alpha=0.3, label='Gradient Norm')
        
        # Add moving average
        window_size = min(50, len(grad_norm))
        if window_size > 1:
            moving_avg = grad_norm.rolling(window=window_size).mean()
            ax.plot(steps, moving_avg, color=colors['main_line'], 
                   label=f'{window_size}-batch Moving Avg')
        
        ax.set_title('Gradient Norm')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Norm')
        ax.legend()
    
    # 6. Learning Rate (middle right)
    ax = axes[1, 2]
    if 'train/learning_rate/base' in history.columns:
        lr = history['train/learning_rate/base'].dropna()
        steps = range(1, len(lr) + 1)
        ax.plot(steps, lr, color=colors['main_line'], label='Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.legend()
    
    # 7. Per-Language Metrics (bottom row)
    # Left: English metrics
    ax = axes[2, 0]
    if 'val/en/auc' in history.columns:
        en_auc = history['val/en/auc'].dropna()
        epochs = range(1, len(en_auc) + 1)
        ax.plot(epochs, en_auc, color=colors['main_line'], label='English AUC')
        ax.set_title('English Performance')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.legend()
    
    # Middle: Russian metrics
    ax = axes[2, 1]
    if 'val/ru/auc' in history.columns:
        ru_auc = history['val/ru/auc'].dropna()
        epochs = range(1, len(ru_auc) + 1)
        ax.plot(epochs, ru_auc, color=colors['main_line'], label='Russian AUC')
        ax.set_title('Russian Performance')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.legend()
    
    # Right: Other languages average
    ax = axes[2, 2]
    other_lang_cols = [col for col in history.columns if col.startswith('val/') and 
                      col.endswith('/auc') and col not in ['val/en/auc', 'val/ru/auc']]
    if other_lang_cols:
        other_lang_aucs = history[other_lang_cols].mean(axis=1).dropna()
        epochs = range(1, len(other_lang_aucs) + 1)
        ax.plot(epochs, other_lang_aucs, color=colors['main_line'], label='Other Languages Avg AUC')
        ax.set_title('Other Languages Performance')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average AUC')
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'{save_dir}/training_metrics_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {save_path}")
    
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