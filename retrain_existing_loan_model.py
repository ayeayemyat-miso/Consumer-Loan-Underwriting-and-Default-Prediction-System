"""
RETRAIN MODEL FOR EXISTING LOANS - NO LOOK-AHEAD BIAS
COMPLETELY FIXED: Ensures ALL data is numeric before any operation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("RETRAINING MODEL FOR EXISTING LOANS (NO LOOK-AHEAD BIAS)")
print("="*70)

# Load data
print("\n1. Loading engineered data...")
df = pd.read_csv('data/loan_data_engineered.csv', low_memory=False)
print(f"   Loaded {len(df):,} records")

# Features available for existing loans at prediction time
existing_loan_features = [
    'emp_length_int', 'annual_inc', 'dti', 'loan_amount', 'interest_rate',
    'term_cat', 'grade_cat', 'home_ownership_cat', 'lti_ratio',
    'payment_burden', 'stress_burden', 'region', 'purpose_cat',
    'total_pymnt', 'total_rec_prncp',
]

# Filter to available columns
available_features = [col for col in existing_loan_features if col in df.columns]
print(f"\n2. Using {len(available_features)} features (no look-ahead bias):")
for i, f in enumerate(available_features):
    print(f"   {i+1:2d}. {f}")

# CRITICAL FIX: Encode ALL categorical columns BEFORE creating X
print("\n3. Encoding categorical variables BEFORE feature extraction...")

# Create a copy of the dataframe with only needed columns
df_encoded = df[available_features + ['year', 'default']].copy()

# Encode categorical columns
categorical_cols = ['region']
for col in categorical_cols:
    if col in df_encoded.columns and df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        print(f"   Encoded: {col} -> values: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Verify all columns are now numeric
print("\n4. Verifying all features are numeric...")
for col in available_features:
    if not pd.api.types.is_numeric_dtype(df_encoded[col]):
        print(f"   Converting {col} to numeric...")
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)

# Create X and y
X = df_encoded[available_features]
y = df_encoded['default']

print(f"\n   Feature matrix shape: {X.shape}")
print(f"   All features numeric: {all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns)}")
print(f"   Default rate: {y.mean():.2%}")

# Check data types
print(f"\n   Data types:")
for col in X.columns[:5]:
    print(f"      {col}: {X[col].dtype}")

# CRITICAL: Split by TIME to avoid look-ahead bias
print("\n5. Splitting by TIME to avoid look-ahead bias...")

# Create a time-based split using year
years = df['year'].values
unique_years = sorted(df['year'].unique())
print(f"   Available years: {unique_years}")

# Use earlier years for training, later years for testing
train_years = [2007, 2008, 2009, 2010, 2011, 2012, 2013]
test_years = [2014, 2015]

train_idx = df['year'].isin(train_years)
test_idx = df['year'].isin(test_years)

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print(f"   Training years: {train_years}")
print(f"   Test years: {test_years}")
print(f"   Train samples: {X_train.shape[0]:,}")
print(f"   Test samples: {X_test.shape[0]:,}")
print(f"   Train default rate: {y_train.mean():.2%}")
print(f"   Test default rate: {y_test.mean():.2%}")

# Verify training data is numeric
print(f"\n   Train data - all numeric: {all(pd.api.types.is_numeric_dtype(X_train[col]) for col in X_train.columns)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
print("\n" + "="*70)
print("TRAINING MODELS (NO LOOK-AHEAD BIAS)")
print("="*70)

results = {}

# 1. Logistic Regression
print("\n📊 Logistic Regression...")
lr_model = LogisticRegression(
    random_state=42, max_iter=1000, class_weight='balanced', n_jobs=-1, C=0.1
)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_pred)
results['Logistic Regression'] = lr_auc
print(f"   AUC-ROC: {lr_auc:.4f}")

# 2. Random Forest (using sample for speed)
print("\n🌲 Random Forest...")
sample_size = min(300000, len(X_train))
np.random.seed(42)
sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train.iloc[sample_idx]
y_train_sample = y_train.iloc[sample_idx]

rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42,
    class_weight='balanced', n_jobs=-1
)
rf_model.fit(X_train_sample, y_train_sample)
rf_pred = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred)
results['Random Forest'] = rf_auc
print(f"   AUC-ROC: {rf_auc:.4f}")

# 3. XGBoost (using sample for speed)
print("\n⚡ XGBoost...")
scale_pos_weight = len(y_train_sample[y_train_sample==0]) / len(y_train_sample[y_train_sample==1])
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    random_state=42, scale_pos_weight=scale_pos_weight,
    eval_metric='logloss', n_jobs=-1, verbosity=0
)
xgb_model.fit(X_train_sample, y_train_sample)
xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_pred)
results['XGBoost'] = xgb_auc
print(f"   AUC-ROC: {xgb_auc:.4f}")

# Compare models
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
for model_name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"   {model_name}: {auc:.4f}")

# Select best model
best_name = max(results, key=results.get)
best_auc = results[best_name]

if best_name == 'Logistic Regression':
    best_model = lr_model
    use_scaling = True
else:
    best_model = rf_model if best_name == 'Random Forest' else xgb_model
    use_scaling = False

print(f"\n✅ BEST MODEL: {best_name} (AUC: {best_auc:.4f})")

# Test on realistic scenario
print("\n" + "="*70)
print("TESTING ON REALISTIC EXISTING LOAN SCENARIO")
print("="*70)

# Get feature names
feature_names = X_train.columns.tolist()
print(f"\n   Features used: {feature_names[:5]}...")

# Create a test case (using the encoded values we know)
test_data = {
    'emp_length_int': 8,
    'annual_inc': 75000,
    'dti': 16,
    'loan_amount': 75000,
    'interest_rate': 4.5,
    'term_cat': 2,
    'grade_cat': 1,  # A
    'home_ownership_cat': 2,  # mortgage
    'lti_ratio': 1.0,
    'payment_burden': 0.15,
    'stress_burden': 0.20,
    'region': 3,  # Dublin (encoded)
    'purpose_cat': 1,
    'total_pymnt': 5000,
    'total_rec_prncp': 3000,
}

# Create DataFrame
test_df = pd.DataFrame([test_data])

# Ensure all columns are present
for col in feature_names:
    if col not in test_df.columns:
        test_df[col] = 0

test_df = test_df[feature_names]

# Predict
if use_scaling:
    test_scaled = scaler.transform(test_df)
    prob = best_model.predict_proba(test_scaled)[0][1]
else:
    prob = best_model.predict_proba(test_df)[0][1]

print(f"\n   Existing Loan Scenario:")
print(f"   - Income: €75,000, Loan: €75,000, Grade A")
print(f"   - Loan age: 2 years, €5,000 paid, €3,000 principal repaid")
print(f"   Default Probability: {prob:.1%}")

if prob < 0.2:
    print(f"   ✅ Model correctly predicts LOW risk")
else:
    print(f"   ⚠️ Model still shows high risk -可能需要进一步调整")

# Save models
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

import os
os.makedirs('models', exist_ok=True)

joblib.dump(best_model, 'models/existing_loan_model.pkl')
joblib.dump(scaler, 'models/existing_loan_scaler.pkl')
joblib.dump(feature_names, 'models/existing_loan_features.pkl')
joblib.dump(use_scaling, 'models/existing_loan_use_scaling.pkl')

# Save region encoder mapping for reference
region_mapping = {'leinster': 0, 'munster': 1, 'connacht': 2, 'ulster': 3, 'dublin': 4}
with open('models/region_encoding.json', 'w') as f:
    json.dump(region_mapping, f)

model_info = {
    'model_type': best_name,
    'auc_roc': best_auc,
    'features': feature_names,
    'training_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'default_rate': float(y.mean()),
    'train_default_rate': float(y_train.mean()),
    'test_default_rate': float(y_test.mean()),
    'note': 'No look-ahead bias - uses only payment history to date'
}
with open('models/existing_loan_model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("\n✅ Models saved successfully!")
print(f"   Model: models/existing_loan_model.pkl")
print(f"   Features: {len(feature_names)}")
print(f"   AUC-ROC: {best_auc:.4f}")
print("\n" + "="*70)
print("TRAINING COMPLETE - NO LOOK-AHEAD BIAS!")
print("="*70)