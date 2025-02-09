import wandb
import pandas as pd
from collections import defaultdict

def analyze_wandb_metrics(run_path: str = None):
    """
    Print all available metrics from the latest wandb run with validation metrics.
    
    Args:
        run_path (str): Optional path to specific wandb run
    """
    try:
        # Initialize wandb
        api = wandb.Api()
        
        # Get run
        if run_path:
            run = api.run(run_path)
        else:
            # Get all runs and sort by creation time (newest first)
            runs = api.runs("toxic-comment-classification")
            if not runs:
                raise ValueError("No runs found in the project")
            
            # Sort runs by creation time (newest first)
            sorted_runs = sorted(runs, key=lambda x: x.created_at, reverse=True)
            
            # Find the latest run with validation metrics
            selected_run = None
            for run in sorted_runs:
                history = pd.DataFrame(run.scan_history())
                if any(col.startswith('val/') for col in history.columns):
                    selected_run = run
                    break
            
            if selected_run is None:
                print("\nWarning: No runs found with validation metrics. Showing latest run instead.")
                selected_run = sorted_runs[0]
            
            run = selected_run
        
        print("\n=== Selected Run ===")
        print(f"Run ID: {run.id}")
        print(f"Run Name: {run.name}")
        print(f"Created: {run.created_at}")
        print(f"Status: {run.state}")
        
        # Get history
        history = pd.DataFrame(run.scan_history())
        
        # Group metrics by category
        metrics = defaultdict(list)
        for column in sorted(history.columns):
            parts = column.split('/')
            if len(parts) > 1:
                category = parts[0]
                metrics[category].append(column)
            else:
                metrics['other'].append(column)
        
        # Print available metrics by category
        print("\n=== Available Metrics ===")
        for category, columns in sorted(metrics.items()):
            print(f"\n{category.upper()}:")
            for metric in sorted(columns):
                print(f"  {metric}")
        
        # Print number of steps/epochs
        print("\n=== Run Statistics ===")
        print(f"Total Steps: {len(history)}")
        if 'train/epoch' in history.columns:
            max_epoch = history['train/epoch'].max()
            print(f"Total Epochs: {max_epoch}")
        
        return history
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure that:")
        print("1. You have wandb installed (pip install wandb)")
        print("2. You are logged in to wandb (wandb login)")
        print("3. You have at least one completed training run")
        return None

if __name__ == '__main__':
    analyze_wandb_metrics() 