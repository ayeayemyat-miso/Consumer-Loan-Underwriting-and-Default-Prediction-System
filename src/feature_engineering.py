"""
Feature Engineering for Mortgage Default Prediction
Creates features for model training and CBI compliance
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df):
    """Create features for mortgage default prediction"""
    
    print("Starting feature engineering...")
    print(f"Original shape: {df.shape}")
    
    # Make a copy to avoid modifying original
    df_feat = df.copy()
    
    # 1. Create Loan-to-Income ratio (LTI) - Key for CBI compliance
    print("Creating Loan-to-Income ratio...")
    df_feat['lti_ratio'] = df_feat['loan_amount'] / (df_feat['annual_inc'] + 1)
    
    # 2. Create Debt-to-Income ratio categories
    df_feat['dti_category'] = pd.cut(df_feat['dti'], 
                                     bins=[0, 15, 30, 50, 100], 
                                     labels=['Low', 'Medium', 'High', 'Very High'])
    
    # 3. Monthly payment burden (installment as % of monthly income)
    df_feat['monthly_income'] = df_feat['annual_inc'] / 12
    df_feat['payment_burden'] = df_feat['installment'] / (df_feat['monthly_income'] + 1)
    df_feat['payment_burden_category'] = pd.cut(df_feat['payment_burden'], 
                                                bins=[0, 0.28, 0.35, 0.50, 1], 
                                                labels=['Low', 'Medium', 'High', 'Very High'])
    
    # 4. Interest rate stress test (+2% as per CBI requirements)
    df_feat['rate_stress'] = df_feat['interest_rate'] + 2.0
    df_feat['stress_payment'] = df_feat['installment'] * (df_feat['rate_stress'] / df_feat['interest_rate'])
    df_feat['stress_burden'] = df_feat['stress_payment'] / (df_feat['monthly_income'] + 1)
    
    # 5. Recovery rate (if default occurs)
    df_feat['recovery_rate'] = df_feat['recoveries'] / (df_feat['loan_amount'] + 1)
    
    # 6. Payment performance features
    df_feat['pymnt_to_loan_ratio'] = df_feat['total_pymnt'] / (df_feat['loan_amount'] + 1)
    df_feat['principal_paid_ratio'] = df_feat['total_rec_prncp'] / (df_feat['loan_amount'] + 1)
    
    # 7. Employment length categories
    df_feat['emp_length_category'] = pd.cut(df_feat['emp_length_int'], 
                                           bins=[-1, 0, 2, 5, 10, 50], 
                                           labels=['No_Experience', 'Junior', 'Mid', 'Senior', 'Expert'])
    
    # 8. Interest rate tiers
    df_feat['rate_tier'] = pd.cut(df_feat['interest_rate'], 
                                 bins=[0, 10, 15, 20, 30], 
                                 labels=['A', 'B', 'C', 'D'])
    
    # 9. Loan amount categories
    df_feat['loan_amount_category'] = pd.cut(df_feat['loan_amount'], 
                                            bins=[0, 10000, 25000, 50000, 100000, 1000000], 
                                            labels=['Micro', 'Small', 'Medium', 'Large', 'Jumbo'])
    
    # 10. Risk score based on multiple factors
    df_feat['risk_score'] = (
        (df_feat['dti'] > 30).astype(int) * 20 +
        (df_feat['lti_ratio'] > 4).astype(int) * 30 +
        (df_feat['payment_burden'] > 0.35).astype(int) * 25 +
        (df_feat['emp_length_int'] < 2).astype(int) * 15 +
        (df_feat['interest_rate'] > 15).astype(int) * 10
    )
    
    # 11. CBI Compliance flags
    df_feat['cbi_lti_compliant'] = (df_feat['lti_ratio'] <= 4).astype(int)
    df_feat['cbi_stress_compliant'] = (df_feat['stress_burden'] <= 0.35).astype(int)
    df_feat['cbi_overall_compliant'] = ((df_feat['cbi_lti_compliant'] == 1) & 
                                        (df_feat['cbi_stress_compliant'] == 1)).astype(int)
    
    # 12. Create binary target (1 for default, 0 for good loan)
    df_feat['default'] = (df_feat['loan_condition'] == 'Bad Loan').astype(int)
    
    print(f"Features created. Final shape: {df_feat.shape}")
    print(f"\nNew features added:")
    new_cols = ['lti_ratio', 'dti_category', 'monthly_income', 'payment_burden', 
                'payment_burden_category', 'rate_stress', 'stress_payment', 'stress_burden',
                'recovery_rate', 'pymnt_to_loan_ratio', 'principal_paid_ratio',
                'emp_length_category', 'rate_tier', 'loan_amount_category', 'risk_score',
                'cbi_lti_compliant', 'cbi_stress_compliant', 'cbi_overall_compliant', 'default']
    for col in new_cols:
        if col in df_feat.columns:
            print(f"  - {col}")
    
    return df_feat

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv('loan_data.csv')
    print(f"Loaded {len(df):,} records")
    
    # Engineer features
    df_engineered = engineer_features(df)
    
    # Save engineered dataset
    output_file = 'data/loan_data_engineered.csv'
    df_engineered.to_csv(output_file, index=False)
    print(f"\nSaved engineered dataset to {output_file}")
    
    # Display sample of new features
    print("\nSample of engineered features (first 5 rows):")
    display_cols = ['id', 'default', 'lti_ratio', 'payment_burden', 'risk_score', 
                    'cbi_overall_compliant', 'rate_tier', 'emp_length_category']
    print(df_engineered[display_cols].head())
    
    # Summary statistics
    print("\nDefault rate in engineered dataset:", df_engineered['default'].mean())
    print("CBI compliance rate:", df_engineered['cbi_overall_compliant'].mean())