# Create this as debug_model.py
import pandas as pd
import joblib
import numpy as np

# Load model and features
feature_names = joblib.load('models/feature_names.pkl')
print("Model expects these features:")
for i, f in enumerate(feature_names):
    print(f"{i+1}. {f}")

# Load training data to understand distributions
print("\n" + "="*60)
print("Loading training data sample...")
df = pd.read_csv('data/loan_data_engineered.csv', nrows=10000)

# Check term column
print(f"\nTerm column unique values: {df['term'].unique()}")
print(f"Term_cat values: {df['term_cat'].unique()}")

# Check loan amounts vs default rates
print("\nDefault rate by loan amount range:")
df['loan_amount_bin'] = pd.cut(df['loan_amount'], bins=[0, 25000, 50000, 100000, 250000, 500000, 1000000])
default_by_loan = df.groupby('loan_amount_bin')['default'].mean()
print(default_by_loan)

# Check LTI ratio vs default rates
print("\nDefault rate by LTI ratio:")
df['lti_bin'] = pd.cut(df['lti_ratio'], bins=[0, 1, 2, 3, 4, 5, 10])
default_by_lti = df.groupby('lti_bin')['default'].mean()
print(default_by_lti)

# Check a low-risk profile in training data
print("\nSample of low-risk loans in training data:")
low_risk = df[(df['lti_ratio'] < 1.5) & (df['dti'] < 20) & (df['emp_length_int'] > 5)]
print(f"Number of low-risk loans: {len(low_risk)}")
if len(low_risk) > 0:
    print(f"Default rate for low-risk loans: {low_risk['default'].mean():.2%}")
    print("\nSample low-risk loan:")
    print(low_risk[['loan_amount', 'annual_inc', 'lti_ratio', 'dti', 'default']].head())