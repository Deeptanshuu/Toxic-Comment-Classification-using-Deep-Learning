import pandas as pd

def check_dataset():
    try:
        # Check train dataset
        print("\nChecking train dataset...")
        train_df = pd.read_csv("dataset/split/train.csv")
        print("\nTrain Dataset Columns:")
        print("-" * 50)
        for col in train_df.columns:
            print(f"- {col}")
        print(f"\nTrain Dataset Shape: {train_df.shape}")
        print("\nTrain Dataset Info:")
        print(train_df.info())
        print("\nFirst few rows of train dataset:")
        print(train_df.head())
        
        # Check validation dataset
        print("\nChecking validation dataset...")
        val_df = pd.read_csv("dataset/split/val.csv")
        print("\nValidation Dataset Columns:")
        print("-" * 50)
        for col in val_df.columns:
            print(f"- {col}")
        print(f"\nValidation Dataset Shape: {val_df.shape}")
        
        # Check test dataset
        print("\nChecking test dataset...")
        test_df = pd.read_csv("dataset/split/test.csv")
        print("\nTest Dataset Columns:")
        print("-" * 50)
        for col in test_df.columns:
            print(f"- {col}")
        print(f"\nTest Dataset Shape: {test_df.shape}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    check_dataset() 