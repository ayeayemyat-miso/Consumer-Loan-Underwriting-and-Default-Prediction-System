"""
Train model for NEW LOAN APPLICATIONS - FAST VERSION
Uses sampling for tree-based models to speed up training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAINING MODEL FOR NEW LOAN APPLICATIONS (Fast Version)")
print("="*70)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('data/loan_data_engineered.csv', low_memory=False)
print(f"   Loaded {len(df):,} records")

# Define core features (no categorical bins)
features = [
    'emp_length_int', 'annual_inc', 'dti', 'loan_amount', 'interest_rate',
    'term_cat', 'grade_cat', 'home_ownership_cat', 'lti_ratio', 
    'payment_burden', 'stress_burden', 'region', 'purpose_cat'
]

print(f"\n2. Preparing data...")

# Create X with numeric data
X = pd.DataFrame()
for col in features:
    if col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(df[col].astype(str))
        else:
            X[col] = df[col]

y = df['default']

# Ensure all numeric
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

print(f"   Feature matrix: {X.shape}")
print(f"   Default rate: {y.mean():.2%}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n3. Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")

# Scale for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

results = {}

# 1. Logistic Regression (fast on all data)
print("\n📊 Logistic Regression...")
lr_model = LogisticRegression(
    random_state=42, max_iter=500, class_weight='balanced', C=1.0, n_jobs=-1
)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_pred)
results['Logistic Regression'] = lr_auc
print(f"   AUC-ROC: {lr_auc:.4f}")

# 2. Random Forest (using 10% sample for speed)
print("\n🌲 Random Forest (using 10% sample)...")
sample_size = min(300000, len(X_train))
print(f"   Sampling {sample_size:,} records...")
np.random.seed(42)
sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train.iloc[sample_idx]
y_train_sample = y_train.iloc[sample_idx]

rf_model = RandomForestClassifier(
    n_estimators=50, max_depth=8, random_state=42,
    class_weight='balanced', n_jobs=-1
)
rf_model.fit(X_train_sample, y_train_sample)
rf_pred = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred)
results['Random Forest'] = rf_auc
print(f"   AUC-ROC: {rf_auc:.4f}")

# 3. XGBoost (using 10% sample for speed)
print("\n⚡ XGBoost (using 10% sample)...")
scale_pos_weight = len(y_train_sample[y_train_sample==0]) / len(y_train_sample[y_train_sample==1])
xgb_model = xgb.XGBClassifier(
    n_estimators=50, max_depth=5, learning_rate=0.1,
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

# Quick test
print("\n" + "="*70)
print("QUICK TEST")
print("="*70)

feature_names = X_train.columns.tolist()

test_app = pd.DataFrame([{
    'emp_length_int': 10, 'annual_inc': 75000, 'dti': 15,
    'loan_amount': 250000, 'interest_rate': 3.5, 'term_cat': 2,
    'grade_cat': 1, 'home_ownership_cat': 2, 'lti_ratio': 3.33,
    'payment_burden': 0.25, 'stress_burden': 0.30, 'region': 2,
    'purpose_cat': 1
}])

for col in feature_names:
    if col not in test_app.columns:
        test_app[col] = 0

test_app = test_app[feature_names]

if use_scaling:
    test_scaled = scaler.transform(test_app)
    prob = best_model.predict_proba(test_scaled)[0][1]
else:
    prob = best_model.predict_proba(test_app)[0][1]

print(f"\n   Low-risk applicant (€75k income, €250k loan, good credit):")
print(f"   Default Probability: {prob:.1%}")
print(f"   Expected: Should be < 20% for low risk")

# Save models
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

import os
os.makedirs('models', exist_ok=True)

joblib.dump(best_model, 'models/preapproval_model.pkl')
joblib.dump(scaler, 'models/preapproval_scaler.pkl')
joblib.dump(feature_names, 'models/preapproval_features.pkl')
joblib.dump(use_scaling, 'models/preapproval_use_scaling.pkl')

print("✅ Models saved successfully!")

# Save model info
model_info = {
    'model_type': best_name,
    'auc_roc': best_auc,
    'features': feature_names,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'default_rate': float(y.mean())
}
import json
with open('models/preapproval_model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)