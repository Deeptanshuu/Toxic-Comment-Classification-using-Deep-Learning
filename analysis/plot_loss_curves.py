import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_plot_style():
    """Configure plot styling"""
    plt.style.use('seaborn-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12

def fetch_wandb_data(run_path=None):
    """Fetch training metrics from wandb"""
    try:
        # Initialize wandb
        api = wandb.Api()
        
        # If no run_path provided, get the most recent run
        if run_path is None:
            runs = api.runs("your-project/toxic-comment-classification")
            if not runs:
                raise ValueError("No runs found in the project")
            run = runs[0]  # Most recent run
        else:
            run = api.run(run_path)
        
        # Get history
        history = pd.DataFrame(run.scan_history())
        
        # Filter relevant metrics
        loss_metrics = history[['train/step_loss', 'val/loss', '_step']].dropna()
        
        return loss_metrics, run.name
        
    except Exception as e:
        logger.error(f"Error fetching wandb data: {str(e)}")
        raise

def plot_loss_curves(loss_metrics, run_name):
    """Plot training and validation loss curves"""
    try:
        setup_plot_style()
        
        fig, ax = plt.subplots()
        
        # Plot training loss
        if 'train/step_loss' in loss_metrics.columns:
            sns.lineplot(
                data=loss_metrics,
                x='_step',
                y='train/step_loss',
                label='Training Loss',
                alpha=0.6,
                color='blue'
            )
        
        # Plot validation loss
        if 'val/loss' in loss_metrics.columns:
            # Validation loss might be less frequent, so we'll make it more visible
            sns.scatterplot(
                data=loss_metrics,
                x='_step',
                y='val/loss',
                label='Validation Loss',
                color='red',
                s=100,
                alpha=0.6
            )
            
            # Add trend line for validation loss
            val_steps = loss_metrics.index[loss_metrics['val/loss'].notna()]
            val_losses = loss_metrics['val/loss'].dropna()
            if len(val_losses) > 1:
                z = np.polyfit(val_steps, val_losses, 1)
                p = np.poly1d(z)
                plt.plot(val_steps, p(val_steps), "r--", alpha=0.8, label='Validation Trend')
        
        # Customize plot
        plt.title(f'Training and Validation Loss\nRun: {run_name}')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.legend()
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = Path('analysis/plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f'loss_curves_{timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting loss curves: {str(e)}")
        raise

def calculate_statistics(loss_metrics):
    """Calculate and print loss statistics"""
    try:
        stats = {
            'Training Loss': {
                'Mean': loss_metrics['train/step_loss'].mean(),
                'Std': loss_metrics['train/step_loss'].std(),
                'Min': loss_metrics['train/step_loss'].min(),
                'Max': loss_metrics['train/step_loss'].max()
            },
            'Validation Loss': {
                'Mean': loss_metrics['val/loss'].mean(),
                'Std': loss_metrics['val/loss'].std(),
                'Min': loss_metrics['val/loss'].min(),
                'Max': loss_metrics['val/loss'].max()
            }
        }
        
        # Print statistics
        print("\nLoss Statistics:")
        for loss_type, metrics in stats.items():
            print(f"\n{loss_type}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        raise

def main():
    try:
        # Fetch data from wandb
        logger.info("Fetching data from wandb...")
        loss_metrics, run_name = fetch_wandb_data()
        
        # Plot loss curves
        logger.info("Plotting loss curves...")
        plot_loss_curves(loss_metrics, run_name)
        
        # Calculate and print statistics
        logger.info("Calculating statistics...")
        calculate_statistics(loss_metrics)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise 