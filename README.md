<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/Dash-2.16.1-brightgreen" alt="Dash">
  <img src="https://img.shields.io/badge/XGBoost-2.0.3-red" alt="XGBoost">
  <img src="https://img.shields.io/badge/IFRS%209-Compliant-blue" alt="IFRS 9">
  <img src="https://img.shields.io/badge/Deployed-Render-purple" alt="Render">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>
markdown
# 🏦 Consumer Loan Underwriting & Default Prediction System

A production-ready dual-dashboard system for consumer loan risk assessment, combining **Machine Learning** with **IFRS 9 rules** for transparent, regulator-friendly credit decisions.

## 🚀 Live Demos

| Dashboard | Purpose | Link |
|-----------|---------|------|
| **Dashboard A** | New Loan Underwriting | [View Demo]([https://your-dashboard-a.onrender.com](https://mortgage-default-prediction-system.onrender.com/)) |
| **Dashboard B** | Existing Loan Monitoring | [View Demo]([https://your-dashboard-b.onrender.com](https://mortgage-default-prediction-system-1.onrender.com/)) |

The project includes two dashboards:

### Dashboard A: New Loan Underwriting
*Main underwriting interface with risk assessment and amortization schedule*

### Dashboard B: Existing Loan Monitoring
The system follows **Central Bank of Ireland (CBI)** lending guidelines and demonstrates real-world credit risk modelling.

---

## 🎯 Key Features

### Dashboard A: New Loan Underwriting
- Predict default probability using XGBoost
- Grade recommendation (A–G)
- CBI compliance checks (LTI, stress test)
- Loan amortization schedule 
- SHAP-based risk explanation
- Optional LTV (collateral) analysis
- Supports multiple repayment types

### Dashboard B: Existing Loan Monitoring
- ✅ Hybrid AI (70% Rules / 30% ML for good borrowers)
- ✅ Full IFRS 9 staging (Stages 1, 2, 3)
- ✅ Dynamic delinquency adjustment (30/60/90+ DPD)
- ✅ Payment performance tracking (ahead/behind schedule)
- ✅ Collateral coverage & LGD adjustment
- ✅ Portfolio batch upload via CSV

---

## 🧠 Model Performance

| Model | AUC | Note |
|-------|-----|------|
| New Loan Model | 0.666 | Based on application data |
| Existing Loan Model | 0.9868 | Synthetic data; real-world would be 0.65-0.80 |

> ⚠️ Models trained on synthetically generated data (CTGAN) for demonstration.

---

## 🏛️ Regulatory Compliance

- **CBI Guidelines**: LTI ≤ 4.0x, stress test +2% with 35% cap
- **IFRS 9**: 3-stage ECL model (12-month → Lifetime → Write-off)
- **Transparent**: Rule-based overrides ensure ML never violates financial logic

---

## 📁 Project Structure

```

Mortgage Default Prediction System/
├── app.py
├── app_preapproval.py
├── train_preapproval_model.py
├── retrain_existing_loan_model.py
├── src/
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── model_evaluation.py
├── models/
├── data/
└── dashboard/

````

---

## 🚀 Installation & Setup

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost shap dash plotly dash-bootstrap-components joblib imbalanced-learn openpyxl
````

### 2. Run feature engineering

```bash
python src/feature_engineering.py
```

### 3. Train models

```bash
python train_preapproval_model.py
python retrain_existing_loan_model.py
```

### 4. Run dashboards

**New Loan Dashboard**

```bash
python app_preapproval.py
```

→ [http://localhost:8051](http://localhost:8051)

**Existing Loan Dashboard**

```bash
python app.py
```

→ [http://localhost:8050](http://localhost:8050)

---

## 📖 How to Use

### New Loan Underwriting

1. Input borrower details (income, loan amount, etc.)
2. Select repayment type and term
3. Click **Evaluate Application**
4. View:

   * Default risk
   * Compliance status
   * Loan schedule
   * Decision summary

### Existing Loan Monitoring

* Single loan prediction OR

---

## 📊 Data Source

Irish Loan Data (Kaggle)
*Note: Synthetic dataset generated using CTGAN — for educational use only.*

---

## 🛠️ Technologies

* Python
* XGBoost
* Scikit-learn
* Dash / Plotly
* Pandas / NumPy
* SHAP
* Joblib

---

## 📈 Key Metrics

* **DTI**: Debt-to-Income ratio
* **LTI**: Loan-to-Income ratio
* **LTV**: Loan-to-Value ratio
* **Stress Test**: Payment at +2% interest
* **Payment Burden**: % of income used for repayments

---

## ⚠️ Limitations

* Uses synthetic data (not real borrowers)
* For demonstration purposes only
* Moderate predictive performance (AUC ~0.67)

---

## 🔧 Troubleshooting

**Port issue**

```bash
app.run(port=8052)
```

**Model not found**

```bash
python train_preapproval_model.py
```

---

## 📝 Future Improvements

* API integration
* Credit bureau data
* PDF reporting
* Macroeconomic variables
* Stress scenario analysis

---

## 📄 License

Educational use only.

---

## 👨‍💻 Author
Developed by [Aye Aye Myat]  
MSc Finance Graduate | Aspiring Banking & Credit Risk Analyst
