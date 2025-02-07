import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import wandb
import pandas as pd
from model.train import train, init_model, create_dataloaders, ToxicDataset
from model.training_config import TrainingConfig
from transformers import XLMRobertaTokenizer
import json
import torch

def load_dataset(file_path: str):
    """Load and prepare dataset"""
    df = pd.read_csv(file_path)
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
    config = TrainingConfig()
    return ToxicDataset(df, tokenizer, config)

class HyperparameterTuner:
    def __init__(self, train_dataset, val_dataset, n_trials=10):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.n_trials = n_trials
        
        # Make pruning more aggressive
        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(
                n_startup_trials=2,
                n_warmup_steps=2,
                interval_steps=1
            )
        )

    def objective(self, trial):
        """Objective function for Optuna optimization with optimal ranges"""
        # Define hyperparameter search space with optimal ranges
        config_params = {
            # Fixed architecture parameters
            "model_name": "xlm-roberta-large",
            "hidden_size": 1024,  # Fixed to original
            "num_attention_heads": 16,  # Fixed to original
            
            # Optimized ranges based on trials
            "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),  # Best range from trial-8/4
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),  # Top performers
            "model_dropout": trial.suggest_float("model_dropout", 0.3, 0.45),  # Trial-8's 0.445 effective
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.03),  # Best regularization
            "grad_accum_steps": trial.suggest_int("grad_accum_steps", 1, 4),  # Keep for throughput optimization
            
            # Fixed training parameters
            "epochs": 2,
            "mixed_precision": "bf16",
            "max_length": 128,
            "fp16": False,
            "distributed": False,
            "world_size": 1,
            "num_workers": 12,
            "activation_checkpointing": True,
            "tensor_float_32": True,
            "gc_frequency": 500
        }

        # Create config
        config = TrainingConfig(**config_params)

        # Initialize wandb for this trial with better metadata
        wandb.init(
            project="toxic-classification-hparam-tuning",
            name=f"trial-{trial.number}",
            config={
                **config_params,
                'trial_number': trial.number,
                'pruner': str(trial.study.pruner),
                'sampler': str(trial.study.sampler)
            },
            reinit=True,
            tags=['hyperparameter-optimization', f'trial-{trial.number}']
        )

        try:
            # Create model and dataloaders
            model = init_model(config)
            train_loader, val_loader = create_dataloaders(
                self.train_dataset, 
                self.val_dataset, 
                config
            )

            # Train and get metrics
            metrics = train(model, train_loader, val_loader, config)
            
            # Log detailed metrics
            wandb.log({
                'final_val_auc': metrics['val/auc'],
                'final_val_loss': metrics['val/loss'],
                'final_train_loss': metrics['train/loss'],
                'peak_gpu_memory': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                'trial_completed': True
            })
            
            # Report intermediate values for pruning
            trial.report(metrics['val/auc'], step=config.epochs)
            
            # Handle pruning
            if trial.should_prune():
                wandb.log({'pruned': True})
                raise optuna.TrialPruned()

            return metrics['val/auc']

        except Exception as e:
            wandb.log({
                'error': str(e),
                'trial_failed': True
            })
            print(f"Trial failed: {str(e)}")
            raise optuna.TrialPruned()

        finally:
            # Cleanup
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
            wandb.finish()

    def run_optimization(self):
        """Run the hyperparameter optimization"""
        print("Starting hyperparameter optimization...")
        print("Search space:")
        print("  - Learning rate: 1e-5 to 5e-5")
        print("  - Batch size: [32, 64]")
        print("  - Dropout: 0.3 to 0.45")
        print("  - Weight decay: 0.01 to 0.03")
        print("  - Gradient accumulation steps: 1 to 4")
        print("\nFixed parameters:")
        print("  - Hidden size: 1024 (original)")
        print("  - Attention heads: 16 (original)")
        
        try:
            self.study.optimize(
                self.objective,
                n_trials=self.n_trials,
                timeout=None,  # No timeout
                callbacks=[self._log_trial]
            )

            # Print optimization results
            print("\nBest trial:")
            best_trial = self.study.best_trial
            print(f"  Value: {best_trial.value:.4f}")
            print("  Params:")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")

            # Save study results with more details
            self._save_study_results()

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")
            self._save_study_results()  # Save results even if interrupted
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            raise

    def _log_trial(self, study, trial):
        """Callback to log trial results with enhanced metrics"""
        if trial.value is not None:
            metrics = {
                "best_auc": study.best_value,
                "trial_auc": trial.value,
                "trial_number": trial.number,
                **trial.params
            }
            
            # Add optimization progress metrics
            if len(study.trials) > 1:
                metrics.update({
                    "optimization_progress": {
                        "trials_completed": len(study.trials),
                        "improvement_rate": (study.best_value - study.trials[0].value) / len(study.trials),
                        "best_trial_number": study.best_trial.number
                    }
                })
            
            wandb.log(metrics)

    def _save_study_results(self):
        """Save optimization results with enhanced metadata"""
        import joblib
        from pathlib import Path
        from datetime import datetime
        
        # Create directory if it doesn't exist
        results_dir = Path("optimization_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save study object
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_path = results_dir / f"hparam_optimization_study_{timestamp}.pkl"
        joblib.dump(self.study, study_path)
        
        # Save comprehensive results
        results = {
            "best_trial": {
                "number": self.study.best_trial.number,
                "value": self.study.best_value,
                "params": self.study.best_trial.params
            },
            "study_statistics": {
                "n_trials": len(self.study.trials),
                "n_completed": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                "n_pruned": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                "datetime_start": self.study.trials[0].datetime_start.isoformat(),
                "datetime_complete": datetime.now().isoformat()
            },
            "search_space": {
                "lr": {"low": 1e-5, "high": 5e-5},
                "batch_size": [32, 64],
                "model_dropout": {"low": 0.3, "high": 0.45},
                "weight_decay": {"low": 0.01, "high": 0.03},
                "grad_accum_steps": {"low": 1, "high": 4}
            },
            "trial_history": [
                {
                    "number": t.number,
                    "value": t.value,
                    "state": str(t.state),
                    "params": t.params if hasattr(t, 'params') else None
                }
                for t in self.study.trials
            ]
        }
        
        results_path = results_dir / f"optimization_results_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to:")
        print(f"  - Study: {study_path}")
        print(f"  - Results: {results_path}")

def main():
    """Main function to run hyperparameter optimization"""
    # Load datasets
    train_dataset = load_dataset("dataset/split/train.csv")
    val_dataset = load_dataset("dataset/split/val.csv")

    # Initialize tuner
    tuner = HyperparameterTuner(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_trials=10
    )

    # Run optimization
    tuner.run_optimization()

if __name__ == "__main__":
    main() 