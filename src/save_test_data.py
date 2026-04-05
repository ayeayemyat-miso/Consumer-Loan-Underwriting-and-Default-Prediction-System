"""
Save test data for evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

print("Loading engineered dataset...")
df = pd.read_csv('data/loan_data_engineered.csv', low_memory=False)
print(f"Loaded {len(df):,} records")

# Define target
y = df['default']

# Remove non-feature columns
remove_cols = ['id', 'issue_d', 'final_d', 'loan_condition', 'loan_condition_cat', 
               'default', 'home_ownership', 'term', 'application_type', 'purpose',
               'interest_payments', 'grade']

# Keep all other columns as features
feature_cols = [col for col in df.columns if col not in remove_cols]

# Create X by converting everything to numeric
X = pd.DataFrame()

for col in feature_cols:
    if col in df.columns:
        X[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"Feature matrix shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test data
test_data = {
    'X_test': X_test,
    'y_test': y_test,
    'X_train': X_train,
    'y_train': y_train
}
joblib.dump(test_data, 'models/test_data.pkl')
print("✓ Test data saved to models/test_data.pkl")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_test shape: {y_test.shape}")