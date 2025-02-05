import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os

def set_style():
    """Set the style for all plots"""
    # Use a basic style instead of seaborn
    plt.style.use('default')
    
    # Custom style settings
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Custom color palette
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC', '#FFB366']
    return colors

def create_language_distribution_plot(df, lang_dist, lang_percent, colors, image_dir):
    """Create and save language distribution plot"""
    plt.figure(figsize=(14, 8))
    
    # Create bar positions
    x = np.arange(len(lang_dist))
    
    # Create bars with language names as x-ticks
    bars = plt.bar(x, lang_dist.values, color=colors)
    plt.title('Language Distribution in Multilingual Toxic Comment Dataset', pad=20)
    plt.xlabel('Language', labelpad=10)
    plt.ylabel('Number of Comments', labelpad=10)
    
    # Set x-ticks to language names
    plt.xticks(x, lang_dist.index, rotation=45)
    
    # Add value labels on top of each bar with increased spacing
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(lang_dist.values) * 0.01),
                f'{int(height):,}\n({lang_percent.values[i]:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    # Add some padding to the top of the plot
    plt.margins(y=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'language_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_toxicity_heatmap(df, toxicity_cols, image_dir):
    """Create and save toxicity correlation heatmap"""
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation and sort
    correlation = df[toxicity_cols].corr()
    
    # Sort correlation matrix by mean correlation value
    mean_corr = correlation.mean()
    sorted_cols = mean_corr.sort_values(ascending=False).index
    correlation = correlation.loc[sorted_cols, sorted_cols]
    
    # Create heatmap with better styling
    im = plt.imshow(correlation, cmap='RdYlBu_r', aspect='equal', vmin=0, vmax=1)
    plt.colorbar(im, label='Correlation Coefficient')
    
    # Add text annotations with conditional formatting
    for i in range(len(correlation)):
        for j in range(len(correlation)):
            corr_value = correlation.iloc[i, j]
            # Choose text color based on background
            text_color = 'white' if abs(corr_value) > 0.7 else 'black'
            # Make diagonal elements bold
            fontweight = 'bold' if i == j else 'normal'
            plt.text(j, i, f'{corr_value:.2f}',
                    ha='center', va='center', 
                    color=text_color,
                    fontweight=fontweight,
                    fontsize=10)
    
    # Improve title and labels
    plt.title('Correlation between Different Types of Toxicity\n(Sorted by Average Correlation)', 
             pad=20, fontsize=14)
    
    # Format axis labels
    formatted_labels = [col.replace('_', ' ').title() for col in correlation.columns]
    plt.xticks(range(len(formatted_labels)), formatted_labels, rotation=45, ha='right')
    plt.yticks(range(len(formatted_labels)), formatted_labels)
    
    # Add gridlines
    plt.grid(False)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'toxicity_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_toxicity_by_language_plot(df, lang_dist, toxicity_cols, colors, image_dir):
    """Create and save toxicity distribution by language plot"""
    plt.figure(figsize=(15, 8))
    
    x = np.arange(len(lang_dist.index))
    width = 0.15
    multiplier = 0
    
    for attribute, color in zip(toxicity_cols, colors):
        # Calculate percentage of toxic comments (any value > 0)
        attribute_means = [(df[df['lang'] == lang][attribute] > 0).mean() * 100 
                         for lang in lang_dist.index]
        
        offset = width * multiplier
        rects = plt.bar(x + offset, attribute_means, width, 
                       label=attribute.replace('_', ' ').title(), 
                       color=color, alpha=0.8)
        
        # Add value labels on the bars
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        multiplier += 1
    
    plt.xlabel('Language')
    plt.ylabel('Percentage of Toxic Comments (%)')
    plt.title('Distribution of Toxicity Types by Language')
    plt.xticks(x + width * 2.5, lang_dist.index, rotation=45)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'toxicity_by_language.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_class_distribution_plot(df, lang_dist, image_dir):
    """Create and save class distribution across languages plot"""
    plt.figure(figsize=(16, 10))
    
    # Calculate class distribution for each language
    class_dist = {}
    for lang in lang_dist.index:
        lang_df = df[df['lang'] == lang]
        total = len(lang_df)
        
        # Count comments by number of toxic classes
        toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        toxic_counts = lang_df[toxicity_cols].astype(bool).sum(axis=1)
        dist = toxic_counts.value_counts().sort_index()
        
        # Convert to percentages
        class_dist[lang] = [(dist.get(i, 0) / total) * 100 for i in range(7)]  # 0 to 6 classes
    
    # Create stacked bar chart
    x = np.arange(len(lang_dist.index))
    bottom = np.zeros(len(lang_dist.index))
    
    # Use a more distinct color scheme
    colors = plt.cm.Set3(np.linspace(0, 1, 7))  # Using Set3 colormap for better distinction
    
    bars = []
    for i in range(7):
        values = [class_dist[lang][i] for lang in lang_dist.index]
        bar = plt.bar(x, values, bottom=bottom, label=f'{i} classes', color=colors[i], alpha=0.9)
        bars.append(bar)
        
        # Add percentage labels for all values > 1%
        for j, v in enumerate(values):
            if v > 1:  # Show all values above 1%
                # Calculate the center of the bar segment
                center = bottom[j] + v/2
                # Choose text color based on position
                text_color = 'black' if v > 10 else 'black'  # Use black for better visibility
                plt.text(x[j], center, f'{v:.1f}%', 
                        ha='center', va='center', 
                        color=text_color, 
                        fontweight='bold',
                        fontsize=9)
        bottom += values
    
    plt.xlabel('Language', labelpad=10, fontsize=12)
    plt.ylabel('Percentage of Comments', labelpad=10, fontsize=12)
    plt.title('Distribution of Toxicity Classes by Language', pad=20, fontsize=14)
    plt.xticks(x, lang_dist.index, rotation=45, fontsize=10)
    
    # Adjust legend
    plt.legend(title='Number of Toxic Classes', 
              bbox_to_anchor=(1.15, 1), 
              loc='upper left',
              fontsize=10,
              title_fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, axis='y', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.margins(y=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_language_distribution():
    """Analyze language distribution and toxicity patterns in the dataset"""
    # Create images directory if it doesn't exist
    image_dir = 'images'
    os.makedirs(image_dir, exist_ok=True)
    
    # Set style and get color palette
    colors = set_style()
    
    # Read the dataset
    print("Reading dataset...")
    input_file = 'dataset/processed/MULTILINGUAL_TOXIC_DATASET_360K_7LANG_FINAL.csv'
    df = pd.read_csv(input_file)
    
    # Get language distribution
    lang_dist = df['lang'].value_counts()
    lang_percent = df['lang'].value_counts(normalize=True) * 100
    
    # Print basic statistics
    print("\nDataset Overview:")
    print("-" * 50)
    print(f"Total number of comments: {len(df):,}")
    print(f"Number of languages: {df['lang'].nunique()}")
    
    print("\nLanguage Distribution:")
    print("-" * 50)
    for lang, count in lang_dist.items():
        print(f"{lang}: {count:,} comments ({lang_percent[lang]:.2f}%)")
    
    # Create language distribution plot
    create_language_distribution_plot(df, lang_dist, lang_percent, colors, image_dir)
    
    # Analyze toxicity
    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Create correlation heatmap
    create_toxicity_heatmap(df, toxicity_cols, image_dir)
    
    # Create toxicity by language plot
    create_toxicity_by_language_plot(df, lang_dist, toxicity_cols, colors, image_dir)
    
    # Create class distribution plot
    create_class_distribution_plot(df, lang_dist, image_dir)
    
    # Print class distribution statistics
    print("\nClass Distribution by Language:")
    print("-" * 50)
    
    for lang in lang_dist.index:
        lang_df = df[df['lang'] == lang]
        total = len(lang_df)
        
        print(f"\n{lang.upper()} (Total: {total:,} comments)")
        
        # Count comments by number of toxic classes
        toxic_counts = lang_df[toxicity_cols].astype(bool).sum(axis=1)
        class_dist = toxic_counts.value_counts().sort_index()
        
        for n_classes, count in class_dist.items():
            percentage = (count / total) * 100
            print(f"{n_classes} toxic classes: {count:,} ({percentage:.2f}%)")
    
    # Detailed toxicity analysis by language
    print("\nDetailed Toxicity Analysis by Language:")
    print("-" * 50)
    
    for lang in lang_dist.index:
        lang_df = df[df['lang'] == lang]
        print(f"\n{lang.upper()} (Total: {len(lang_df):,} comments)")
        
        # Calculate toxicity statistics
        for col in toxicity_cols:
            toxic_count = (lang_df[col] > 0).sum()
            toxic_percent = (toxic_count / len(lang_df)) * 100
            
            # Calculate confidence interval
            ci = stats.norm.interval(0.95, 
                                   loc=toxic_percent/100, 
                                   scale=np.sqrt((toxic_percent/100 * (1-toxic_percent/100)) / len(lang_df)))
            ci_lower, ci_upper = ci[0] * 100, ci[1] * 100
            
            print(f"- {col.replace('_', ' ').title()}:")
            print(f"  Count: {toxic_count:,} ({toxic_percent:.2f}%)")
            print(f"  95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    
    # Statistical tests
    print("\nStatistical Analysis:")
    print("-" * 50)
    
    # Chi-square test for independence between language and number of toxic classes
    toxic_class_counts = pd.crosstab(df['lang'], df[toxicity_cols].astype(bool).sum(axis=1))
    chi2, p_value, _, _ = stats.chi2_contingency(toxic_class_counts)
    print("\nChi-square test for number of toxic classes by language:")
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"p-value: {p_value:.10f}")
    print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Chi-square test for each toxicity type
    for col in toxicity_cols:
        binary_col = (df[col] > 0).astype(int)
        contingency_table = pd.crosstab(df['lang'], binary_col)
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        print(f"\nChi-square test for {col.replace('_', ' ').title()}:")
        print(f"Chi-square statistic: {chi2:.2f}")
        print(f"p-value: {p_value:.10f}")
        print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

if __name__ == "__main__":
    analyze_language_distribution() 