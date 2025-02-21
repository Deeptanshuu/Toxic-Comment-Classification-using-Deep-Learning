import pandas as pd
import numpy as np
from text_preprocessor import TextPreprocessor
from tqdm import tqdm
import logging
from pathlib import Path
import time

def process_dataset(input_path: str, output_path: str = None, batch_size: int = 1000):
    """
    Process a dataset using the TextPreprocessor with efficient batch processing.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save processed CSV file. If None, will use input name with _processed suffix
        batch_size: Number of texts to process in each batch
    """
    # Setup output path
    if output_path is None:
        input_path = Path(input_path)
        output_path = input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    print(f"\nProcessing dataset: {input_path}")
    start_time = time.time()
    
    try:
        # Read the dataset
        print("Reading dataset...")
        df = pd.read_csv(input_path)
        total_rows = len(df)
        print(f"Total rows: {total_rows:,}")
        
        # Process in batches with progress bar
        print("\nProcessing text...")
        
        # Calculate number of batches
        num_batches = (total_rows + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, total_rows, batch_size), total=num_batches, desc="Processing batches"):
            # Get batch
            batch_start = i
            batch_end = min(i + batch_size, total_rows)
            
            # Process each text in the batch
            for idx in range(batch_start, batch_end):
                text = df.loc[idx, 'comment_text']
                lang = df.loc[idx, 'lang'] if 'lang' in df.columns else 'en'
                
                # Process text
                processed = preprocessor.preprocess_text(
                    text,
                    lang=lang,
                    clean_options={
                        'remove_stops': True,
                        'remove_numbers': True,
                        'remove_urls': True,
                        'remove_emails': True,
                        'remove_mentions': True,
                        'remove_hashtags': True,
                        'expand_contractions': True,
                        'remove_accents': False,
                        'min_word_length': 2
                    },
                    do_stemming=True
                )
                
                # Update the text directly
                df.loc[idx, 'comment_text'] = processed
            
            # Optional: Print sample from first batch
            if i == 0:
                print("\nSample processing results:")
                for j in range(min(3, batch_size)):
                    print(f"\nProcessed text {j+1}: {df.loc[j, 'comment_text'][:100]}...")
        
        # Save processed dataset
        print(f"\nSaving processed dataset to: {output_path}")
        df.to_csv(output_path, index=False)
        
        # Print statistics
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\nProcessing Complete!")
        print("-" * 50)
        print(f"Total rows processed: {total_rows:,}")
        print(f"Processing time: {processing_time/60:.2f} minutes")
        print(f"Average time per text: {processing_time/total_rows*1000:.2f} ms")
        print(f"Output file size: {Path(output_path).stat().st_size/1024/1024:.1f} MB")
        
        # Print sample of unique words before and after
        print("\nVocabulary Statistics:")
        sample_size = min(1000, total_rows)
        original_words = set(' '.join(df['comment_text'].head(sample_size).astype(str)).split())
        processed_words = set(' '.join(df['processed_text'].head(sample_size).astype(str)).split())
        print(f"Sample unique words (first {sample_size:,} rows):")
        print(f"Before processing: {len(original_words):,}")
        print(f"After processing : {len(processed_words):,}")
        print(f"Reduction: {(1 - len(processed_words)/len(original_words))*100:.1f}%")
        
    except Exception as e:
        print(f"\nError processing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    # Process training dataset
    input_file = "dataset/split/train.csv"
    output_file = "dataset/split/train_no_stopwords.csv"
    
    process_dataset(input_file, output_file) 