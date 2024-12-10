import pandas as pd
import numpy as np

def inspect_data():
    # Load data
    df = pd.read_csv('data/enhanced_engagement_dataset.csv')
    
    # Print basic info
    print("Dataset Shape:", df.shape)
    print("\nFeature Statistics:")
    print(df.describe())
    
    print("\nValue Ranges:")
    for col in df.columns:
        print(f"\n{col}:")
        print(f"Min: {df[col].min()}")
        print(f"Max: {df[col].max()}")
        print(f"Null values: {df[col].isnull().sum()}")

if __name__ == "__main__":
    inspect_data()