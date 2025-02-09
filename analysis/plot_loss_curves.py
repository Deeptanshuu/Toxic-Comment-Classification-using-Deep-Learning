import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import os
import wandb
from transformers import get_linear_schedule_with_warmup

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
    plt.rcParams['figure.figsize'] = (12, 12)
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
        # Initialize config with training settings
        config = TrainingConfig(
            batch_size=32,
            num_workers=16,
            lr=2e-5,
            weight_decay=0.01,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            label_smoothing=0.01,
            mixed_precision="fp16",
            activation_checkpointing=True,
            epochs=4  # Number of validation epochs

        )
        
        # Load validation data
        logger.info("Loading validation and test data...")
        val_df = pd.read_csv("dataset/split/val.csv")
        test_df = pd.read_csv("dataset/split/test.csv")
        combined_df = pd.concat([val_df, test_df])
        tokenizer = XLMRobertaTokenizer.from_pretrained(config.model_name)
        combined_dataset = ToxicDataset(combined_df, tokenizer, config, mode='combined')
        

        # Create combined dataloader
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=config.batch_size,
            shuffle=True,  # Enable shuffling
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False  # Keep all samples
        )
        
        # Log dataloader config to wandb
        if wandb.run is not None:
            wandb.config.update({
                'shuffle': True,
                'drop_last': False,
                'total_validation_steps': len(combined_loader),
                'total_validation_samples': len(combined_dataset)
            })
        

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
        
        # Setup optimizer
        param_groups = config.get_param_groups(model)
        optimizer = torch.optim.AdamW(param_groups)
        
        # Setup scheduler
        total_steps = len(combined_loader) * config.epochs
        warmup_steps = int(total_steps * config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Initialize gradient scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision == "fp16")
        
        # Log model configuration to wandb
        if wandb.run is not None:
            wandb.config.update({
                'model_name': config.model_name,
                'batch_size': config.batch_size,
                'learning_rate': config.lr,
                'weight_decay': config.weight_decay,
                'max_grad_norm': config.max_grad_norm,
                'warmup_ratio': config.warmup_ratio,
                'label_smoothing': config.label_smoothing,
                'mixed_precision': config.mixed_precision,
                'num_workers': config.num_workers,
                'activation_checkpointing': config.activation_checkpointing,
                'validation_epochs': config.epochs
            })
        
        return model, combined_loader, device, optimizer, scheduler, scaler, config
        

    except Exception as e:
        logger.error(f"Error loading model and data: {str(e)}")
        raise

def collect_validation_losses(model, combined_loader, device, optimizer, scheduler, scaler, config):
    """Run validation and collect step losses across multiple epochs"""
    try:
        model.eval()
        all_losses = []
        epoch_losses = []
        

        for epoch in range(config.epochs):
            logger.info(f"\nStarting validation epoch {epoch+1}/{config.epochs}")
            total_loss = 0
            num_batches = len(combined_loader)
            epoch_start_time = datetime.now()
            

            with torch.no_grad():
                for step, batch in enumerate(combined_loader):
                    # Move batch to device

                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass with mixed precision
                    with torch.cuda.amp.autocast(enabled=config.mixed_precision != "no"):
                        outputs = model(**batch)
                        loss = outputs['loss'].item()
                    
                    total_loss += loss
                    
                    # Calculate running averages
                    avg_loss = total_loss / (step + 1)
                    
                    # Get learning rates
                    lrs = [group['lr'] for group in optimizer.param_groups]
                    
                    # Log to wandb
                    wandb.log({
                        'val/step_loss': loss,
                        'val/running_avg_loss': avg_loss,
                        'val/progress': (step + 1) / num_batches * 100,
                        'val/learning_rate': lrs[0],  # Base learning rate
                        'val/batch_size': config.batch_size,
                        'val/epoch': epoch + 1,
                        'val/global_step': epoch * num_batches + step
                    })
                    
                    # Log progress
                    if step % 10 == 0:
                        elapsed_time = datetime.now() - epoch_start_time
                        steps_per_sec = (step + 1) / elapsed_time.total_seconds()
                        remaining_steps = num_batches - (step + 1)
                        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                        
                        logger.info(
                            f"Epoch [{epoch+1}/{config.epochs}] "
                            f"Step [{step+1}/{num_batches}] "
                            f"Loss: {loss:.4f} "
                            f"Avg Loss: {avg_loss:.4f} "
                            f"LR: {lrs[0]:.2e} "
                            f"ETA: {int(eta_seconds)}s"
                        )
            
            # Calculate epoch statistics
            epoch_avg_loss = total_loss / num_batches
            epoch_losses.append({
                'epoch': epoch + 1,
                'avg_loss': epoch_avg_loss,
                'elapsed_time': (datetime.now() - epoch_start_time).total_seconds()
            })
            
            # Log epoch metrics to wandb
            wandb.log({
                'val/epoch_avg_loss': epoch_avg_loss,
                'val/epoch_number': epoch + 1,
                'val/epoch_time': epoch_losses[-1]['elapsed_time']
            })
            
            # Clear GPU memory after each epoch
            torch.cuda.empty_cache()
        
        return epoch_losses
        
    except Exception as e:
        logger.error(f"Error collecting validation losses: {str(e)}")
        raise

def plot_validation_losses(epoch_losses):
    """Plot validation epoch losses"""
    try:
        setup_plot_style()
        
        # Create figure
        fig, ax = plt.subplots()
        
        # Extract data
        epochs = [d['epoch'] for d in epoch_losses]
        losses = [d['avg_loss'] for d in epoch_losses]
        
        # Plot epoch losses
        ax.plot(epochs, losses, 'go-', label='Epoch Average Loss', linewidth=2, markersize=8)
        
        # Add trend line
        z = np.polyfit(epochs, losses, 1)
        p = np.poly1d(z)
        ax.plot(epochs, p(epochs), "r--", alpha=0.8, label='Loss Trend')
        
        # Customize plot
        ax.set_title('Validation Epoch Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = Path('analysis/plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f'validation_losses_{timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
        
        # Log plot to wandb
        wandb.log({
            "val/loss_plot": wandb.Image(str(output_path))
        })
        
        # Show plot
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting validation losses: {str(e)}")
        raise

def calculate_loss_statistics(epoch_losses):
    """Calculate and print loss statistics"""
    try:
        losses = [d['avg_loss'] for d in epoch_losses]
        
        stats = {
            'Mean Loss': np.mean(losses),
            'Std Loss': np.std(losses),
            'Min Loss': np.min(losses),
            'Max Loss': np.max(losses),
            'Best Epoch': epoch_losses[np.argmin(losses)]['epoch']
        }
        
        # Log statistics to wandb
        wandb.log({
            'val/mean_loss': stats['Mean Loss'],
            'val/std_loss': stats['Std Loss'],
            'val/min_loss': stats['Min Loss'],
            'val/max_loss': stats['Max Loss'],
            'val/best_epoch': stats['Best Epoch']
        })
        
        # Print statistics
        print("\nValidation Loss Statistics:")
        for metric_name, value in stats.items():
            if metric_name == 'Best Epoch':
                print(f"{metric_name}: {int(value)}")
            else:
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
        model, combined_loader, device, optimizer, scheduler, scaler, config = load_model_and_data()
        

        # Collect validation losses
        logger.info("Collecting validation losses...")
        epoch_losses = collect_validation_losses(
            model, combined_loader, device, optimizer, scheduler, scaler, config
        )
        

        # Plot losses
        logger.info("Plotting validation losses...")
        plot_validation_losses(epoch_losses)
        
        # Calculate and print statistics
        logger.info("Calculating statistics...")
        calculate_loss_statistics(epoch_losses)
        
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