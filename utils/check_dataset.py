import pandas as pd

def check_dataset():
    try:
        train_df = pd.read_csv("dataset/split/train.csv")
        print("\nDataset Columns:")
        print("-" * 50)
        for col in train_df.columns:
            print(f"- {col}")
            
        print("\nFirst few rows:")
        print(train_df.head())
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    check_dataset() 