import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

def convert_parquet_to_csv(parquet_path, csv_path=None):
    """Convert a parquet file to CSV with progress tracking"""
    print(f"\nReading parquet file: {parquet_path}")
    
    # If no CSV path specified, use the same name with .csv extension
    if csv_path is None:
        csv_path = str(Path(parquet_path).with_suffix('.csv'))
    
    try:
        # Read parquet file
        df = pd.read_parquet(parquet_path)
        total_rows = len(df)
        
        print(f"\nDataset Info:")
        print(f"Rows: {total_rows:,}")
        print(f"Columns: {', '.join(df.columns)}")
        print(f"\nSaving to CSV: {csv_path}")
        
        # Save to CSV with progress bar
        with tqdm(total=total_rows, desc="Converting") as pbar:
            # Use chunksize for memory efficiency
            chunk_size = 10000
            for i in range(0, total_rows, chunk_size):
                end_idx = min(i + chunk_size, total_rows)
                chunk = df.iloc[i:end_idx]
                
                # Write mode: 'w' for first chunk, 'a' for rest
                mode = 'w' if i == 0 else 'a'
                header = i == 0  # Only write header for first chunk
                
                chunk.to_csv(csv_path, mode=mode, header=header, index=False)
                pbar.update(len(chunk))
        
        print(f"\n✓ Successfully converted to CSV")
        print(f"Output file size: {Path(csv_path).stat().st_size / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
        
    parquet_path = "dataset/raw/jigsaw-toxic-comment-train-processed-seqlen128_original .parquet"
    csv_path = "dataset/raw/jigsaw-en-only-toxic-comment-train-processed-seqlen128_original.csv"
    
    convert_parquet_to_csv(parquet_path, csv_path)