import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

def remove_english_comments(input_path, output_path=None):
    """Remove English comments from a dataset with progress tracking"""
    print(f"\nReading input file: {input_path}")
    
    # If no output path specified, use input name with _non_english suffix
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('').with_name(f"{Path(input_path).stem}_non_english.csv"))
    
    try:
        # Read input file with UTF-8 encoding
        df = pd.read_csv(input_path, encoding='utf-8')
        total_rows = len(df)
        
        print(f"\nDataset Info:")
        print(f"Initial Rows: {total_rows:,}")
        print(f"Columns: {', '.join(df.columns)}")
        
        # Filter out English comments (where lang == 'en')
        print("\nFiltering out English comments...")
        non_english_df = df[df['lang'] != 'en']
        
        # Save to CSV with UTF-8 encoding
        print(f"\nSaving to: {output_path}")
        non_english_df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Get statistics
        english_rows = total_rows - len(non_english_df)
        
        print(f"\n✓ Successfully removed English comments")
        print(f"Initial rows: {total_rows:,}")
        print(f"Remaining non-English rows: {len(non_english_df):,}")
        print(f"Removed English rows: {english_rows:,}")
        print(f"Output file: {output_path}")
        print(f"Output file size: {Path(output_path).stat().st_size / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    input_path = "dataset/raw/MULTILINGUAL_TOXIC_DATASET_347k_7LANG.csv"
    output_path = input_path.replace(".csv", "_non_english.csv")
    
    remove_english_comments(input_path, output_path)