import pandas as pd
import os
from pathlib import Path
import json
from datetime import datetime

def create_dataset_card(file_path):
    """Create a dataset card with key information about the CSV file"""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Get file info
        file_stats = os.stat(file_path)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        last_modified = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        # Create dataset card
        card = {
            "filename": Path(file_path).name,
            "last_modified": last_modified,
            "file_size_mb": round(file_size_mb, 2),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "column_dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "sample_rows": df.head(3).to_dict('records')
        }
        
        # Add language distribution if 'lang' column exists
        if 'lang' in df.columns:
            card["language_distribution"] = df['lang'].value_counts().to_dict()
        
        # Add label distribution if any toxic-related columns exist
        toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        label_stats = {}
        for col in toxic_cols:
            if col in df.columns:
                label_stats[col] = df[col].value_counts().to_dict()
        if label_stats:
            card["label_distribution"] = label_stats
        
        return card
        
    except Exception as e:
        return {
            "filename": Path(file_path).name,
            "error": str(e)
        }

def scan_dataset_directory(directory="dataset"):
    """Scan directory for CSV files and create dataset cards"""
    print(f"\nScanning directory: {directory}")
    
    # Find all CSV files
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    print(f"\nFound {len(csv_files)} CSV files")
    
    # Create dataset cards
    cards = {}
    for file_path in csv_files:
        print(f"\nProcessing: {file_path}")
        cards[file_path] = create_dataset_card(file_path)
    
    # Save to JSON file
    output_file = "dataset/dataset_cards.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cards, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Dataset cards saved to: {output_file}")
    
    # Print summary for each file
    for file_path, card in cards.items():
        print(f"\n{'='*80}")
        print(f"File: {card['filename']}")
        if 'error' in card:
            print(f"Error: {card['error']}")
            continue
            
        print(f"Size: {card['file_size_mb']:.2f} MB")
        print(f"Rows: {card['num_rows']:,}")
        print(f"Columns: {', '.join(card['columns'])}")
        
        if 'language_distribution' in card:
            print("\nLanguage Distribution:")
            for lang, count in card['language_distribution'].items():
                print(f"  {lang}: {count:,}")
        
        if 'label_distribution' in card:
            print("\nLabel Distribution:")
            for label, dist in card['label_distribution'].items():
                print(f"  {label}: {dist}")

if __name__ == "__main__":
    scan_dataset_directory()