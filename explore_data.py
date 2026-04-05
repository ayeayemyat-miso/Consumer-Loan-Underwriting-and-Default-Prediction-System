import pandas as pd
import numpy as np
import os

# Try multiple possible locations
possible_paths = [
    r'C:\Users\HP\OneDrive\Documents\Mortgage Default Prediction System\loan_data.csv',
    r'C:\Users\HP\OneDrive\Documents\loan_data.csv',
    r'..\loan_data.csv',
    r'..\..\Mortgage Default Prediction System\loan_data.csv'
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        print(f"Found file at: {path}")
        df = pd.read_csv(path)
        break

if df is None:
    print("ERROR: Could not find loan_data.csv")
    print("Current directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
    exit(1)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nData types:")
print(df.dtypes)
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nMissing values (first 10 columns):")
missing = df.isnull().sum()
print(missing[missing > 0].head(10))
print(f"\nTarget variable distribution (loan_condition):")
print(df['loan_condition'].value_counts())
print(f"\nDefault rate: {(df['loan_condition'] == 'Bad Loan').mean()*100:.2f}%")
print(f"\nNumerical columns statistics:")
print(df.describe())
