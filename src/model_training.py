"""
Model Training - Complete Data Cleaning Version
Ensures all features are numeric before training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("MORTGAGE DEFAULT PREDICTION - MODEL TRAINING")
print("="*60)

# Load data
print("\nLoading engineered dataset...")
df = pd.read_csv('data/loan_data_engineered.csv', low_memory=False)
print(f"✓ Loaded {len(df):,} records with {df.shape[1]} columns")

# Define target
y = df['default']
print(f"Default rate: {y.mean()*100:.2f}%")

# Remove non-feature columns
remove_cols = ['id', 'issue_d', 'final_d', 'loan_condition', 'loan_condition_cat', 
               'default', 'home_ownership', 'term', 'application_type', 'purpose',
               'interest_payments', 'grade']

# Keep only potential features
feature_cols = [col for col in df.columns if col not in remove_cols]

print(f"\nProcessing {len(feature_cols)} potential features...")

# Create feature matrix
X = pd.DataFrame()

for col in feature_cols:
    if col in df.columns:
        # Try numeric conversion
        X[col] = pd.to_numeric(df[col], errors='coerce')
        
        # If all values are NaN → encode as categorical
        if X[col].isna().all():
            le = LabelEncoder()
            X[col] = le.fit_transform(df[col].astype(str))
            print(f"  Encoded column: {col}")
        else:
            # Fill missing values
            X[col] = X[col].fillna(0)
            if df[col].dtype == 'object':
                print(f"  Converted to numeric: {col}")

print(f"\n✓ Final feature matrix shape: {X.shape}")
print(f"All features numeric: {all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns)}")

# Train-test split
print("\n" + "="*60)
print("TRAIN-TEST SPLIT")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")

# Logistic Regression
print("\n" + "="*60)
print("TRAINING LOGISTIC REGRESSION")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(
    random_state=42,
    max_iter=500,
    class_weight='balanced',
    n_jobs=-1
)

lr_model.fit(X_train_scaled, y_train)

lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_pred = lr_model.predict(X_test_scaled)
lr_auc = roc_auc_score(y_test, lr_pred_proba)

print(f"\n✓ Logistic Regression AUC-ROC: {lr_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred, target_names=['Good Loan', 'Bad Loan']))

# XGBoost
print("\n" + "="*60)
print("TRAINING XGBOOST")
print("="*60)

sample_size = min(300000, len(X_train))
print(f"Using {sample_size:,} samples for XGBoost...")

np.random.seed(42)
sample_idx = np.random.choice(len(X_train), sample_size, replace=False)

X_train_sample = X_train.iloc[sample_idx]
y_train_sample = y_train.iloc[sample_idx]

scale_pos_weight = len(y_train_sample[y_train_sample == 0]) / len(y_train_sample[y_train_sample == 1])

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    n_jobs=-1,
    verbosity=0
)

print("Training XGBoost...")
xgb_model.fit(X_train_sample, y_train_sample)

xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred = xgb_model.predict(X_test)
xgb_auc = roc_auc_score(y_test, xgb_pred_proba)

print(f"\n✓ XGBoost AUC-ROC: {xgb_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, xgb_pred, target_names=['Good Loan', 'Bad Loan']))

# Model comparison
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

print(f"Logistic Regression AUC-ROC: {lr_auc:.4f}")
print(f"XGBoost AUC-ROC: {xgb_auc:.4f}")

if xgb_auc > lr_auc:
    best_model = xgb_model
    best_name = 'XGBoost'
else:
    best_model = lr_model
    best_name = 'Logistic Regression'

print(f"\n✅ Best Model: {best_name}")

# Save models
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

import os
os.makedirs('models', exist_ok=True)

joblib.dump(lr_model, 'models/lr_model.pkl')
joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(X_train.columns.tolist(), 'models/feature_names.pkl')

print("✓ Models saved successfully!")

# Confusion Matrix
print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)

final_pred = xgb_pred if best_name == 'XGBoost' else lr_pred
cm = confusion_matrix(y_test, final_pred)

print(f"True Negatives: {cm[0,0]:,}")
print(f"False Positives: {cm[0,1]:,}")
print(f"False Negatives: {cm[1,0]:,}")
print(f"True Positives: {cm[1,1]:,}")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)