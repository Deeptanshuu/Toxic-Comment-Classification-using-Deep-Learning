import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import os

class ToxicityOrdinalEncoder:
    def __init__(self, n_bins=4, strategy='quantile'):
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_edges = {}
        self.ordinal_mapping = {}
        self.label_mapping = {}

    def _get_optimal_bins(self, values):
        """Dynamically determine bins using statistical analysis"""
        unique_vals = np.unique(values)
        if len(unique_vals) <= self.n_bins:
            return sorted(unique_vals)
            
        # Handle 1D data properly and check sample size
        if len(values) < 2:
            return np.linspace(0, 1, self.n_bins + 1)
            
        try:
            # Transpose for correct KDE dimensions (d, N) = (1, samples)
            kde = stats.gaussian_kde(values.T)
            x = np.linspace(0, 1, 100)
            minima = []
            for i in range(1, len(x)-1):
                if (kde(x[i]) < kde(x[i-1])) and (kde(x[i]) < kde(x[i+1])):
                    minima.append(x[i])
                    
            if minima:
                return [0] + sorted(minima) + [1]
        except np.linalg.LinAlgError:
            pass
            
        # Fallback to KBinsDiscretizer
        est = KBinsDiscretizer(n_bins=self.n_bins, 
                             encode='ordinal', 
                             strategy=self.strategy)
        est.fit(values)
        return est.bin_edges_[0]

    def fit(self, df, columns):
        """Learn optimal binning for each toxicity category"""
        for col in columns:
            # Filter and validate non-zero values
            non_zero = df[col][df[col] > 0].values.reshape(-1, 1)
            
            # Handle empty columns
            if len(non_zero) == 0:
                self.bin_edges[col] = [0, 1]
                self.ordinal_mapping[col] = {0: 0}
                continue
                
            # Handle small sample sizes
            if len(non_zero) < 2:
                self.bin_edges[col] = np.linspace(0, 1, self.n_bins + 1)
                continue
                
            bins = self._get_optimal_bins(non_zero)
            self.bin_edges[col] = bins
            
            # Create ordinal mapping
            self.ordinal_mapping[col] = {
                val: i for i, val in enumerate(sorted(np.unique(bins)))
            }
            
            # Create label mapping for interpretability
            self.label_mapping[col] = {
                0: 'Non-toxic',
                1: 'Low',
                2: 'Medium',
                3: 'High',
                4: 'Severe'
            }

        return self

    def transform(self, df, columns):
        """Apply learned ordinal mapping with safety checks"""
        transformed = df.copy()
        
        for col in columns:
            if col not in self.bin_edges:
                raise ValueError(f"Column {col} not fitted")
                
            bins = self.bin_edges[col]
            transformed[col] = pd.cut(df[col], bins=bins,
                                    labels=False, include_lowest=True)
            
            # Preserve zero as separate class
            transformed[col] = np.where(df[col] == 0, 0, transformed[col] + 1)
            transformed[col] = transformed[col].astype(int)  # Ensure integer type
            
        return transformed

def plot_toxicity_distribution(df, transformed_df, column, bin_edges, save_dir='images'):
    """Plot original vs binned distribution for a toxicity column"""
    plt.figure(figsize=(15, 6))
    
    # Original distribution
    plt.subplot(1, 2, 1)
    non_zero_vals = df[column][df[column] > 0]
    if len(non_zero_vals) > 0:
        plt.hist(non_zero_vals, bins=50, alpha=0.7)
        plt.title(f'Original {column.replace("_", " ").title()} Distribution\n(Non-zero values)')
        plt.xlabel('Toxicity Score')
        plt.ylabel('Count')
        
        # Add bin edges as vertical lines
        for edge in bin_edges[column]:
            plt.axvline(x=edge, color='r', linestyle='--', alpha=0.5)
    else:
        plt.text(0.5, 0.5, 'No non-zero values', ha='center', va='center')
    
    # Binned distribution
    plt.subplot(1, 2, 2)
    unique_bins = sorted(transformed_df[column].unique())
    plt.hist(transformed_df[column], bins=len(unique_bins), 
            range=(min(unique_bins)-0.5, max(unique_bins)+0.5),
            alpha=0.7, rwidth=0.8)
    plt.title(f'Binned {column.replace("_", " ").title()} Distribution')
    plt.xlabel('Toxicity Level')
    plt.ylabel('Count')
    
    # Add labels for toxicity levels
    plt.xticks(range(5), ['Non-toxic', 'Low', 'Medium', 'High', 'Severe'])
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{column}_distribution.png'))
    plt.close()

def main():
    # Load dataset
    print("Loading dataset...")
    input_file = 'dataset/raw/MULTILINGUAL_TOXIC_DATASET_367k_7LANG_cleaned.csv'
    df = pd.read_csv(input_file)
    
    # Define toxicity columns
    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Print initial value distributions
    print("\nInitial value distributions:")
    for col in toxicity_cols:
        print(f"\n{col.replace('_', ' ').title()}:")
        print(df[col].value_counts().sort_index())
    
    # Initialize and fit encoder
    print("\nFitting toxicity encoder...")
    encoder = ToxicityOrdinalEncoder(n_bins=4)
    encoder.fit(df, toxicity_cols)
    
    # Transform data
    print("Transforming toxicity values...")
    transformed_df = encoder.transform(df, toxicity_cols)
    
    # Plot distributions
    print("\nGenerating distribution plots...")
    for col in toxicity_cols:
        plot_toxicity_distribution(df, transformed_df, col, encoder.bin_edges)
    
    # Print binning information
    print("\nBin edges for each toxicity type:")
    for col in toxicity_cols:
        print(f"\n{col.replace('_', ' ').title()}:")
        edges = encoder.bin_edges[col]
        for i in range(len(edges)-1):
            print(f"Level {encoder.label_mapping[col][i+1]}: {edges[i]:.3f} to {edges[i+1]:.3f}")
    
    # Save transformed dataset
    output_file = 'dataset/processed/MULTILINGUAL_TOXIC_DATASET_binned.csv'
    print(f"\nSaving binned dataset to: {output_file}")
    transformed_df.to_csv(output_file, index=False)
    
    # Print final value distributions
    print("\nFinal binned distributions:")
    for col in toxicity_cols:
        print(f"\n{col.replace('_', ' ').title()}:")
        dist = transformed_df[col].value_counts().sort_index()
        for level, count in dist.items():
            print(f"{encoder.label_mapping[col][level]}: {count:,} ({count/len(df)*100:.1f}%)")

if __name__ == "__main__":
    main()


