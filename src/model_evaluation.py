"""
Model Evaluation for Mortgage Default Prediction
Compatible with NumPy 1.26.4
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("MODEL EVALUATION - MORTGAGE DEFAULT PREDICTION")
print("="*60)

# Load models and data
print("\nLoading models and test data...")
best_model = joblib.load('models/best_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
lr_model = joblib.load('models/lr_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')
test_data = joblib.load('models/test_data.pkl')

X_test = test_data['X_test']
y_test = test_data['y_test']

print(f"Loaded test data: {X_test.shape[0]:,} samples")

# Get predictions
print("\nGenerating predictions...")
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
X_test_scaled = scaler.transform(X_test)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
print("\n" + "="*60)
print("MODEL METRICS")
print("="*60)

# ROC-AUC
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_pred_proba)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_pred_proba)
roc_auc_lr = auc(fpr_lr, tpr_lr)

print(f"\nXGBoost AUC-ROC: {roc_auc_xgb:.4f}")
print(f"Logistic Regression AUC-ROC: {roc_auc_lr:.4f}")

# Confusion Matrix for XGBoost
xgb_pred = (xgb_pred_proba >= 0.5).astype(int)
cm = confusion_matrix(y_test, xgb_pred)

print(f"\nXGBoost Confusion Matrix:")
print(f"  True Negatives: {cm[0,0]:,}")
print(f"  False Positives: {cm[0,1]:,}")
print(f"  False Negatives: {cm[1,0]:,}")
print(f"  True Positives: {cm[1,1]:,}")

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nXGBoost Metrics:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1_score:.4f}")
print(f"  Accuracy: {(tp + tn) / (tp + tn + fp + fn):.4f}")

# Feature Importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE (XGBoost)")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
for i in range(min(10, len(feature_importance))):
    print(f"  {i+1}. {feature_importance.iloc[i]['feature']}: {feature_importance.iloc[i]['importance']:.4f}")

# Simple calibration assessment
print("\n" + "="*60)
print("CALIBRATION ASSESSMENT")
print("="*60)

# Group predictions into bins
n_bins = 10
bins = np.linspace(0, 1, n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_indices = np.digitize(xgb_pred_proba, bins) - 1

observed_rates = []
for i in range(n_bins):
    mask = (bin_indices == i)
    if mask.any():
        observed_rates.append(y_test[mask].mean())
    else:
        observed_rates.append(np.nan)

print("\nCalibration by probability bin:")
print("  Prob Bin | Predicted Range | Actual Default Rate")
print("  " + "-"*50)
for i in range(n_bins):
    if not np.isnan(observed_rates[i]):
        print(f"  {bin_centers[i]:.2f}     | {bins[i]:.2f}-{bins[i+1]:.2f}          | {observed_rates[i]:.4f}")

# Risk Score Analysis
print("\n" + "="*60)
print("RISK SCORE ANALYSIS")
print("="*60)

# Check if risk_score exists in features
if 'risk_score' in X_test.columns:
    risk_scores = X_test['risk_score']
    
    # Create risk categories
    risk_categories = pd.cut(risk_scores, bins=[0, 30, 60, 80, 100], 
                             labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])
    
    print("\nDefault rate by risk category:")
    for risk_cat in ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']:
        mask = risk_categories == risk_cat
        if mask.any():
            default_rate = y_test[mask].mean()
            count = mask.sum()
            print(f"  {risk_cat}: {default_rate:.2%} ({count:,} loans)")

# CBI Compliance Analysis
print("\n" + "="*60)
print("CBI COMPLIANCE ANALYSIS")
print("="*60)

# Check for CBI compliance columns
cbi_cols = [col for col in feature_names if 'cbi' in col.lower()]
if cbi_cols:
    for col in cbi_cols:
        if col in X_test.columns:
            compliant = X_test[col] == 1
            compliant_rate = compliant.mean()
            default_rate_compliant = y_test[compliant].mean() if compliant.any() else 0
            default_rate_non_compliant = y_test[~compliant].mean() if (~compliant).any() else 0
            
            print(f"\n{col.upper()}:")
            print(f"  Compliant loans: {compliant_rate:.1%}")
            print(f"  Default rate (compliant): {default_rate_compliant:.2%}")
            print(f"  Default rate (non-compliant): {default_rate_non_compliant:.2%}")

# Save results to file
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results = {
    'model': 'XGBoost',
    'auc_roc': float(roc_auc_xgb),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1_score),
    'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
    'confusion_matrix': cm.tolist(),
    'top_features': feature_importance.head(10).to_dict('records')
}

import json
with open('models/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Results saved to models/evaluation_results.json")

# Save feature importance to CSV
feature_importance.to_csv('models/feature_importance.csv', index=False)
print("✓ Feature importance saved to models/feature_importance.csv")

print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)

# Summary
print("\n📊 SUMMARY:")
print(f"  ✅ Best Model: XGBoost")
print(f"  ✅ AUC-ROC: {roc_auc_xgb:.4f}")
print(f"  ✅ Recall (Default Detection): {recall:.2%}")
print(f"  ✅ Precision: {precision:.2%}")
print(f"  ✅ F1-Score: {f1_score:.4f}")
print(f"  ✅ Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2%}")