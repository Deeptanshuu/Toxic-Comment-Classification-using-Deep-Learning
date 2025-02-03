import pandas as pd
import os
import json
from datetime import datetime
import numpy as np

def get_file_size(file_path):
    size_bytes = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def get_column_descriptions():
    return {
        "id": "Unique identifier for each comment",
        "comment_text": "The text content of the comment to be classified",
        "toxic": "Binary label indicating if the comment is toxic",
        "severe_toxic": "Binary label for extremely toxic comments",
        "obscene": "Binary label for obscene content",
        "threat": "Binary label for threatening content",
        "insult": "Binary label for insulting content",
        "identity_hate": "Binary label for identity-based hate speech",
        "target": "Overall toxicity score (in bias dataset)",
        "identity_attack": "Binary label for identity-based attacks",
        "identity_*": "Various identity-related attributes in the bias dataset",
        "lang": "Language of the comment"
    }

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)

def create_datacard():
    dataset_path = "dataset"
    
    # Dataset Overview
    datacard = {
        "name": "Jigsaw Toxic Comment Classification Dataset",
        "version": "1.0",
        "date_created": datetime.now().strftime("%Y-%m-%d"),
        "description": """
        The Jigsaw Toxic Comment Classification Dataset is designed to help identify and classify toxic online comments.
        It contains text comments with multiple toxicity-related labels including general toxicity, severe toxicity,
        obscenity, threats, insults, and identity-based hate speech.

        The dataset includes:
        1. Main training data with binary toxicity labels
        2. Unintended bias training data with additional identity attributes
        3. Processed versions with sequence length 128 for direct model input
        4. Test and validation sets for model evaluation

        This dataset was created by Jigsaw and Google's Conversation AI team to help improve online conversation quality
        by identifying and classifying various forms of toxic comments.
        """,
        "files": [],
        "sample_data": {},
        "column_descriptions": get_column_descriptions()
    }
    
    # File Information
    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(dataset_path, file_name)
            file_info = {
                "name": file_name,
                "size": get_file_size(file_path),
                "rows": None,
                "columns": None,
                "column_descriptions": {},
                "sample_rows": None,
                "dataset_type": "Unknown"
            }
            
            # Determine dataset type
            if "unintended-bias" in file_name:
                file_info["dataset_type"] = "Unintended Bias Dataset"
            elif "toxic-comment-train" in file_name:
                file_info["dataset_type"] = "Main Training Dataset"
            elif "test" in file_name:
                file_info["dataset_type"] = "Test Dataset"
            elif "validation" in file_name:
                file_info["dataset_type"] = "Validation Dataset"
            
            try:
                # Read first few rows to get structure
                df = pd.read_csv(file_path, nrows=5)
                file_info["columns"] = list(df.columns)
                
                # Get total number of rows by reading in chunks
                total_rows = 0
                for chunk in pd.read_csv(file_path, chunksize=10000):
                    total_rows += len(chunk)
                file_info["rows"] = total_rows
                
                # Store sample rows
                file_info["sample_rows"] = df.replace({np.nan: None}).to_dict('records')
                
                # Add column descriptions
                for col in df.columns:
                    if col in datacard["column_descriptions"]:
                        file_info["column_descriptions"][col] = datacard["column_descriptions"][col]
                
                # Basic column statistics for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats_df = pd.read_csv(file_path, usecols=numeric_cols, nrows=10000)
                    file_info["numeric_stats"] = {
                        col: {
                            "mean": None if pd.isna(stats_df[col].mean()) else float(stats_df[col].mean()),
                            "std": None if pd.isna(stats_df[col].std()) else float(stats_df[col].std()),
                            "min": None if pd.isna(stats_df[col].min()) else float(stats_df[col].min()),
                            "max": None if pd.isna(stats_df[col].max()) else float(stats_df[col].max())
                        } for col in numeric_cols
                    }
                
                # Calculate label distribution for main toxicity columns
                if file_info["dataset_type"] in ["Main Training Dataset", "Unintended Bias Dataset"]:
                    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
                    available_cols = [col for col in toxicity_cols if col in df.columns]
                    if available_cols:
                        label_stats_df = pd.read_csv(file_path, usecols=available_cols)
                        file_info["label_distribution"] = {
                            col: {
                                "positive_samples": 0 if pd.isna(label_stats_df[col].sum()) else int(label_stats_df[col].sum()),
                                "positive_ratio": 0.0 if pd.isna(label_stats_df[col].mean()) else float(label_stats_df[col].mean())
                            } for col in available_cols
                        }
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue
                
            datacard["files"].append(file_info)
    
    # Save DataCard
    with open('datacard.json', 'w', encoding='utf-8') as f:
        json.dump(datacard, f, indent=2, cls=NpEncoder, ensure_ascii=False)
    
    # Create a more readable markdown version
    with open('datacard.md', 'w', encoding='utf-8') as f:
        f.write(f"# {datacard['name']}\n\n")
        f.write(f"## Overview\n")
        f.write(f"Version: {datacard['version']}\n")
        f.write(f"Date Created: {datacard['date_created']}\n\n")
        f.write(f"### Description\n")
        f.write(f"{datacard['description']}\n\n")
        
        f.write("## Column Descriptions\n\n")
        for col, desc in datacard["column_descriptions"].items():
            f.write(f"- **{col}**: {desc}\n")
        f.write("\n")
        
        f.write("## Files\n\n")
        for file_info in datacard["files"]:
            f.write(f"### {file_info['name']}\n")
            f.write(f"**Dataset Type**: {file_info['dataset_type']}\n")
            f.write(f"- Size: {file_info['size']}\n")
            f.write(f"- Number of rows: {file_info['rows']}\n")
            f.write(f"- Columns: {', '.join(file_info['columns'])}\n\n")
            
            if "label_distribution" in file_info:
                f.write("#### Label Distribution\n")
                for label, stats in file_info["label_distribution"].items():
                    f.write(f"\n**{label}**:\n")
                    f.write(f"- Positive samples: {stats['positive_samples']}\n")
                    f.write(f"- Positive ratio: {stats['positive_ratio']:.4f}\n")
            
            if "numeric_stats" in file_info:
                f.write("\n#### Numeric Column Statistics\n")
                for col, stats in file_info["numeric_stats"].items():
                    f.write(f"\n**{col}**:\n")
                    f.write(f"- Mean: {stats['mean']:.4f}\n")
                    f.write(f"- Std: {stats['std']:.4f}\n")
                    f.write(f"- Min: {stats['min']:.4f}\n")
                    f.write(f"- Max: {stats['max']:.4f}\n")
            
            f.write("\n#### Sample Rows\n")
            f.write("```\n")
            sample_df = pd.DataFrame(file_info["sample_rows"])
            f.write(sample_df.to_string())
            f.write("\n```\n\n")

if __name__ == "__main__":
    create_datacard() 