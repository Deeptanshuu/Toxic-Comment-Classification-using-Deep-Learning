import wandb
import pandas as pd
from collections import defaultdict

def analyze_wandb_metrics(run_path: str = None):
    """
    Analyze and print all available metrics from the latest wandb run.
    
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
            runs = api.runs("toxic-comment-classification")
            if not runs:
                raise ValueError("No runs found in the project")
            run = runs[0]  # Get latest run
        
        print("\n=== Run Information ===")
        print(f"Run ID: {run.id}")
        print(f"Run Name: {run.name}")
        print(f"Project: {run.project}")
        print(f"Created: {run.created_at}")
        print(f"Status: {run.state}")
        print(f"Runtime: {run.runtime}")
        
        # Get history
        history = pd.DataFrame(run.scan_history())
        
        # Analyze available metrics
        print("\n=== Available Metrics ===")
        metrics = defaultdict(list)
        
        # Group metrics by category
        for column in sorted(history.columns):
            parts = column.split('/')
            if len(parts) > 1:
                category = parts[0]
                metrics[category].append(column)
            else:
                metrics['other'].append(column)
        
        # Print metrics by category
        for category, columns in metrics.items():
            print(f"\n{category.upper()} Metrics:")
            for col in sorted(columns):
                data = history[col].dropna()
                print(f"  {col}:")
                print(f"    - Data points: {len(data)}")
                if len(data) > 0:
                    try:
                        if data.dtype in ['float64', 'float32', 'int64', 'int32']:
                            print(f"    - Range: [{data.min():.4f}, {data.max():.4f}]")
                            print(f"    - Mean: {data.mean():.4f}")
                            print(f"    - Standard deviation: {data.std():.4f}")
                            # Print first and last values
                            print(f"    - First value: {data.iloc[0]:.4f}")
                            print(f"    - Last value: {data.iloc[-1]:.4f}")
                    except Exception as e:
                        print(f"    - Non-numeric or complex data type: {data.dtype}")
                        if len(data) < 5:  # If few values, print them all
                            print(f"    - Values: {list(data)}")
                        else:  # Otherwise print first few and last few
                            print(f"    - First 3 values: {list(data[:3])}")
                            print(f"    - Last 3 values: {list(data[-3:])}")
        
        # Print available languages
        print("\n=== Language-specific Metrics ===")
        languages = set()
        for col in history.columns:
            if '/' in col:
                lang = col.split('/')[1]
                if lang in ['en', 'ru', 'tr', 'es', 'fr', 'it', 'pt']:
                    languages.add(lang)
        print("Available languages:", sorted(list(languages)))
        
        # Print available class metrics
        print("\n=== Class-specific Metrics ===")
        classes = set()
        for col in history.columns:
            if 'class_metrics' in col:
                cls = col.split('/')[2]
                classes.add(cls)
        print("Available classes:", sorted(list(classes)))
        
        # Print summary of training progress
        print("\n=== Training Summary ===")
        if 'val/loss' in history.columns:
            val_loss = history['val/loss'].dropna()
            print(f"Number of epochs completed: {len(val_loss)}")
            print(f"Best validation loss: {val_loss.min():.4f} (epoch {val_loss.idxmin() + 1})")
        
        if 'val/auc' in history.columns:
            val_auc = history['val/auc'].dropna()
            print(f"Best validation AUC: {val_auc.max():.4f} (epoch {val_auc.idxmax() + 1})")
        
        return history
        
    except Exception as e:
        print(f"\nError analyzing metrics: {str(e)}")
        print("\nPlease ensure that:")
        print("1. You have wandb installed (pip install wandb)")
        print("2. You are logged in to wandb (wandb login)")
        print("3. You have at least one completed training run")
        print("4. The project name matches your wandb project")
        return None

if __name__ == '__main__':
    analyze_wandb_metrics() 