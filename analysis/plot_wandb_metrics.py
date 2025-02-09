import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import os

def plot_wandb_metrics(run_path: str = None, save_dir: str = 'images'):
    """
    Plot training metrics from wandb logs.
    
    Args:
        run_path (str): Optional path to specific wandb run (e.g., 'your-username/toxic-comment-classification/run_id')
                       If None, will use the most recent run
        save_dir (str): Directory to save the plots
    """
    # Initialize wandb
    api = wandb.Api()
    
    # Get run
    if run_path:
        run = api.run(run_path)
    else:
        # Get most recent run
        runs = api.runs("toxic-comment-classification")
        if not runs:
            raise ValueError("No runs found in the project")
        run = runs[0]  # Most recent run
    
    # Get history
    history = pd.DataFrame(run.scan_history())
    
    # Set up matplotlib style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (15, 12),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11
    })
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Training Metrics', fontsize=16, y=0.95)
    
    # Define colors
    colors = {
        'main_line': '#1f77b4',  # Blue
        'avg_line': '#ff7f0e',   # Orange
        'annotation': '#2ca02c',  # Green
        'grid': '#cccccc'        # Light gray
    }
    
    # 1. Plot Validation Loss
    if 'val/loss' in history.columns:
        val_loss = history['val/loss'].dropna()
        epochs = range(1, len(val_loss) + 1)
        ax1.plot(epochs, val_loss, color=colors['main_line'], label='Validation Loss')
        
        # Add moving average
        window_size = min(3, len(val_loss))
        if window_size > 1:
            moving_avg = val_loss.rolling(window=window_size).mean()
            ax1.plot(epochs, moving_avg, '--', color=colors['avg_line'], 
                    alpha=0.7, label='Moving Average')
        
        # Add min point annotation
        min_loss = val_loss.min()
        min_epoch = val_loss.idxmin() + 1
        ax1.annotate(f'Min: {min_loss:.4f}',
                    xy=(min_epoch, min_loss),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=colors['annotation'], alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color=colors['annotation']))
        
        ax1.set_title('Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
    
    # 2. Plot Training Loss
    if 'train/epoch_loss' in history.columns:
        train_loss = history['train/epoch_loss'].dropna()
        epochs = range(1, len(train_loss) + 1)
        ax2.plot(epochs, train_loss, color=colors['main_line'], label='Training Loss')
        
        # Add moving average
        window_size = min(3, len(train_loss))
        if window_size > 1:
            moving_avg = train_loss.rolling(window=window_size).mean()
            ax2.plot(epochs, moving_avg, '--', color=colors['avg_line'], 
                    alpha=0.7, label='Moving Average')
        
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
    
    # 3. Plot AUC
    if 'val/auc' in history.columns:
        auc = history['val/auc'].dropna()
        epochs = range(1, len(auc) + 1)
        ax3.plot(epochs, auc, color=colors['main_line'], label='Validation AUC')
        
        # Add max point annotation
        max_auc = auc.max()
        max_epoch = auc.idxmax() + 1
        ax3.annotate(f'Max: {max_auc:.4f}',
                    xy=(max_epoch, max_auc),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=colors['annotation'], alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color=colors['annotation']))
        
        ax3.set_title('Validation AUC')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('AUC')
        ax3.legend()
    
    # 4. Plot Learning Rate
    if 'train/learning_rate/base' in history.columns:
        lr = history['train/learning_rate/base'].dropna()
        steps = range(1, len(lr) + 1)
        ax4.plot(steps, lr, color=colors['main_line'], label='Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Learning Rate')
        ax4.legend()
    
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