# Mortgage Default Prediction System

## 🏦 Overview
A mortgage default prediction system designed for the Irish banking market (AIB, Bank of Ireland, Permanent TSB).  
## 🚀 Live Demo

**🌐 Deployed on Render:** [https://mortgage-default-prediction-system.onrender.com/](https://mortgage-default-prediction-system-1.onrender.com/)


## 📸 Screenshots
![alt text](image.png)

![alt text](image-1.png)
The project includes two dashboards:

### Dashboard A: New Loan Underwriting
![New Loan Underwriting Dashboard](screenshots/dashboard_a_new_loan.png)
*Main underwriting interface with risk assessment and amortization schedule*

### Dashboard B: Existing Loan Monitoring


The system follows **Central Bank of Ireland (CBI)** lending guidelines and demonstrates real-world credit risk modelling.

---

## 🎯 Key Features

### Dashboard A: New Loan Underwriting
- Predict default probability using XGBoost
- Grade recommendation (A–G)
- CBI compliance checks (LTI, stress test)
- Loan amortization schedule (1–40 years)
- SHAP-based risk explanation
- Optional LTV (collateral) analysis
- Supports multiple repayment types

### Dashboard B: Existing Loan Monitoring
- Portfolio (batch) upload via CSV
- Payment history analysis
- Delinquency tracking (late payments, days past due)
- Portfolio risk summary
- Export results to CSV
- No look-ahead bias (realistic modelling)

---

## 📊 Model Performance

| Metric | New Loan Model | Existing Loan Model |
|--------|---------------|---------------------|
| AUC-ROC | 0.666 | 0.606 |
| Recall | 97.4% | - |
| Training Samples | ~2.7M | ~2.7M |
| Features | 17 | 15 |

---

## 🏛️ Regulatory Compliance (CBI)

- LTI ≤ 4.0 × annual income  
- Interest rate stress test: +2%  
- Stress burden ≤ 35% of income  
- Maximum loan term: 30 years  

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
* Upload CSV for portfolio analysis

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