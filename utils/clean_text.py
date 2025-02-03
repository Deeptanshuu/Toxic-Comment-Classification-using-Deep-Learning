import pandas as pd
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging
from pathlib import Path

def clean_text(text):
    """Clean text by removing URLs, HTML tags, and special characters"""
    try:
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove multiple punctuation
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        
        return text.strip()
    except Exception as e:
        logging.error(f"Error cleaning text: {str(e)}")
        return text

def try_read_csv(file_path):
    """Try different encodings to read the CSV file"""
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            print(f"Trying {encoding} encoding...")
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with {encoding}: {str(e)}")
            continue
    
    raise ValueError("Could not read file with any of the attempted encodings")

def clean_dataset(input_path, output_path=None):
    """Clean comment text in a dataset"""
    print(f"\nReading input file: {input_path}")
    
    # If no output path specified, use input name with _cleaned suffix
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('').with_name(f"{Path(input_path).stem}_cleaned.csv"))
    
    try:
        # Try reading with different encodings
        df = try_read_csv(input_path)
        total_rows = len(df)
        
        print(f"\nDataset Info:")
        print(f"Initial Rows: {total_rows:,}")
        print(f"Columns: {', '.join(df.columns)}")
        
        # Verify 'comment_text' column exists
        if 'comment_text' not in df.columns:
            # Try to find a column that might contain the comments
            text_columns = [col for col in df.columns if 'text' in col.lower() or 'comment' in col.lower()]
            if text_columns:
                print(f"\nUsing '{text_columns[0]}' as comment column")
                df['comment_text'] = df[text_columns[0]]
            else:
                raise ValueError("Could not find comment text column")
        
        # Clean comment text with progress bar
        print("\nCleaning comments...")
        tqdm.pandas()
        df['comment_text'] = df['comment_text'].progress_apply(clean_text)
        
        # Remove empty comments
        non_empty_mask = df['comment_text'].str.strip().str.len() > 0
        df = df[non_empty_mask]
        
        # Save cleaned dataset
        print(f"\nSaving to: {output_path}")
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Print statistics
        print(f"\n✓ Successfully cleaned comments")
        print(f"Initial rows: {total_rows:,}")
        print(f"Final rows: {len(df):,}")
        print(f"Removed empty rows: {total_rows - len(df):,}")
        print(f"Output file: {output_path}")
        print(f"Output file size: {Path(output_path).stat().st_size / (1024*1024):.1f} MB")
        
        # Sample of cleaned comments
        print("\nSample of cleaned comments:")
        for i, (orig, cleaned) in enumerate(zip(df['comment_text'].head(3), df['comment_text'].head(3))):
            print(f"\nExample {i+1}:")
            print(f"Original : {orig[:100]}...")
            print(f"Cleaned  : {cleaned[:100]}...")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return

if __name__ == "__main__":
    input_path = "dataset/raw/english-trash.csv"
    output_path = "dataset/raw/english-comments-cleaned.csv"
    
    clean_dataset(input_path, output_path)