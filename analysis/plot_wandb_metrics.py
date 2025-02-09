import wandb
import pandas as pd
from collections import defaultdict

def analyze_wandb_metrics(run_path: str = None):
    """
    Print all available metrics from the latest wandb run.
    
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
            run = sorted_runs[0]  # Get the most recent run
        
        print("\n=== Latest Run ===")
        print(f"Run ID: {run.id}")
        print(f"Run Name: {run.name}")
        print(f"Created: {run.created_at}")
        
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