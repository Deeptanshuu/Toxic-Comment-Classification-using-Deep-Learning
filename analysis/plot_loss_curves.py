import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import os
import wandb

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.training_config import TrainingConfig
from model.language_aware_transformer import LanguageAwareTransformer
from model.train import ToxicDataset
from transformers import XLMRobertaTokenizer

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

def setup_wandb():
    """Initialize wandb for validation tracking"""
    try:
        wandb.init(
            project="toxic-comment-classification",
            name=f"validation-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "analysis_type": "validation_loss",
                "timestamp": datetime.now().strftime('%Y%m%d-%H%M%S')
            }
        )
        logger.info("Initialized wandb logging")
    except Exception as e:
        logger.error(f"Error initializing wandb: {str(e)}")
        raise

def load_model_and_data():
    """Load the model and validation data"""
    try:
        # Initialize config
        config = TrainingConfig(
            batch_size=32,
            num_workers=16
        )
        
        # Load validation data
        logger.info("Loading validation data...")
        val_df = pd.read_csv("dataset/split/val.csv")
        tokenizer = XLMRobertaTokenizer.from_pretrained(config.model_name)
        val_dataset = ToxicDataset(val_df, tokenizer, config, mode='val')
        
        # Create validation dataloader
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Load model
        logger.info("Loading model...")
        model = LanguageAwareTransformer(
            num_labels=len(config.toxicity_labels),
            model_name=config.model_name
        )
        
        # Load latest checkpoint
        checkpoint_path = Path('weights/toxic_classifier_xlm-roberta-large/pytorch_model.bin')
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            logger.info("Loaded model checkpoint")
        else:
            raise FileNotFoundError("No checkpoint found")
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        return model, val_loader, device
        
    except Exception as e:
        logger.error(f"Error loading model and data: {str(e)}")
        raise

def collect_validation_losses(model, val_loader, device):
    """Run validation and collect step losses"""
    try:
        model.eval()
        step_losses = []
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs['loss'].item()
                total_loss += loss
                
                # Store step and loss
                step_losses.append({
                    'step': step,
                    'loss': loss
                })
                
                # Calculate running averages
                avg_loss = total_loss / (step + 1)
                
                # Log to wandb
                wandb.log({
                    'val/step_loss': loss,
                    'val/running_avg_loss': avg_loss,
                    'val/progress': (step + 1) / num_batches * 100
                })
                
                # Log progress
                if step % 10 == 0:
                    logger.info(f"Processed {step}/{num_batches} validation steps. "
                              f"Current loss: {loss:.4f}, Avg loss: {avg_loss:.4f}")
        
        # Log final metrics to wandb
        wandb.log({
            'val/final_avg_loss': total_loss / num_batches
        })
        
        return pd.DataFrame(step_losses)
        
    except Exception as e:
        logger.error(f"Error collecting validation losses: {str(e)}")
        raise

def plot_validation_losses(loss_df):
    """Plot validation step losses"""
    try:
        setup_plot_style()
        
        fig, ax = plt.subplots()
        
        # Plot step losses
        sns.lineplot(
            data=loss_df,
            x='step',
            y='loss',
            label='Validation Step Loss',
            alpha=0.6,
            color='blue'
        )
        
        # Add trend line
        z = np.polyfit(loss_df['step'], loss_df['loss'], 1)
        p = np.poly1d(z)
        plt.plot(loss_df['step'], p(loss_df['step']), "r--", 
                alpha=0.8, label='Loss Trend')
        
        # Customize plot
        plt.title('Validation Step Losses')
        plt.xlabel('Validation Step')
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
        output_path = output_dir / f'validation_step_losses_{timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
        
        # Log plot to wandb
        wandb.log({"val/loss_plot": wandb.Image(str(output_path))})
        
        # Also save the loss data
        loss_df.to_csv(output_dir / f'validation_step_losses_{timestamp}.csv', index=False)
        logger.info(f"Loss data saved to {output_dir}/validation_step_losses_{timestamp}.csv")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting validation losses: {str(e)}")
        raise

def calculate_loss_statistics(loss_df):
    """Calculate and print loss statistics"""
    try:
        stats = {
            'Mean': loss_df['loss'].mean(),
            'Std': loss_df['loss'].std(),
            'Min': loss_df['loss'].min(),
            'Max': loss_df['loss'].max(),
            'Median': loss_df['loss'].median(),
            '25th Percentile': loss_df['loss'].quantile(0.25),
            '75th Percentile': loss_df['loss'].quantile(0.75)
        }
        
        # Log statistics to wandb
        wandb.log({
            'val/mean_loss': stats['Mean'],
            'val/std_loss': stats['Std'],
            'val/min_loss': stats['Min'],
            'val/max_loss': stats['Max'],
            'val/median_loss': stats['Median'],
            'val/25th_percentile_loss': stats['25th Percentile'],
            'val/75th_percentile_loss': stats['75th Percentile']
        })
        
        print("\nValidation Loss Statistics:")
        for metric_name, value in stats.items():
            print(f"{metric_name}: {value:.4f}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        raise

def main():
    try:
        # Initialize wandb
        setup_wandb()
        
        # Load model and data
        logger.info("Loading model and data...")
        model, val_loader, device = load_model_and_data()
        
        # Collect validation losses
        logger.info("Collecting validation losses...")
        loss_df = collect_validation_losses(model, val_loader, device)
        
        # Plot losses
        logger.info("Plotting validation losses...")
        plot_validation_losses(loss_df)
        
        # Calculate and print statistics
        logger.info("Calculating statistics...")
        calculate_loss_statistics(loss_df)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        # Clean up
        torch.cuda.empty_cache()
        # Finish wandb run
        wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise 