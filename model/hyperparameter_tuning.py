import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import wandb
import pandas as pd
from model.train import train, init_model, create_dataloaders, ToxicDataset
from model.training_config import TrainingConfig
from transformers import XLMRobertaTokenizer
import json

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
        """Objective function for Optuna optimization"""
        # Define hyperparameter search space
        config_params = {
            "model_name": "xlm-roberta-large",  # Fixed architecture
            "batch_size": trial.suggest_int("batch_size", 32, 64, step=16),
            "grad_accum_steps": trial.suggest_int("grad_accum_steps", 1, 4),
            "lr": trial.suggest_float("lr", 1e-6, 5e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
            "hidden_size": trial.suggest_categorical("hidden_size", [768, 1024]),
            "num_attention_heads": trial.suggest_categorical("num_attention_heads", [12, 16]),
            "model_dropout": trial.suggest_float("model_dropout", 0.1, 0.5),
            
            # Fixed parameters
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

        # Initialize wandb for this trial
        wandb.init(
            project="toxic-classification-hparam-tuning",
            name=f"trial-{trial.number}",
            config=config_params,
            reinit=True
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
            
            # Report intermediate values for pruning
            trial.report(metrics['val/auc'], step=config.epochs)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

            return metrics['val/auc']

        except Exception as e:
            print(f"Trial failed: {str(e)}")
            raise optuna.TrialPruned()

        finally:
            wandb.finish()

    def run_optimization(self):
        """Run the hyperparameter optimization"""
        print("Starting hyperparameter optimization...")
        
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

            # Save study results
            self._save_study_results()

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            raise

    def _log_trial(self, study, trial):
        """Callback to log trial results"""
        if trial.value is not None:
            wandb.log({
                "best_auc": study.best_value,
                "trial_auc": trial.value,
                **trial.params
            })

    def _save_study_results(self):
        """Save optimization results"""
        import joblib
        from pathlib import Path
        
        # Create directory if it doesn't exist
        Path("optimization_results").mkdir(exist_ok=True)
        
        # Save study object
        joblib.dump(
            self.study,
            "optimization_results/hparam_optimization_study.pkl"
        )
        
        # Save best parameters
        best_params = self.study.best_params
        with open("optimization_results/best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)

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