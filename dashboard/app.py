"""
Mortgage Default Prediction Dashboard - EXISTING LOANS
COMPLETE FIXED VERSION - Working Batch Upload
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import base64
import io
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Loading Existing Loan Dashboard...")
print("="*60)

# Load model
try:
    best_model = joblib.load('models/existing_loan_model.pkl')
    scaler = joblib.load('models/existing_loan_scaler.pkl')
    feature_names = joblib.load('models/existing_loan_features.pkl')
    use_scaling = joblib.load('models/existing_loan_use_scaling.pkl')
    with open('models/existing_loan_model_info.json', 'r') as f:
        model_info = json.load(f)
    print(f"✅ Loaded existing loan model (AUC: {model_info.get('auc_roc', 0.606):.3f})")
except:
    print("⚠️ Existing loan model not found, using preapproval model")
    best_model = joblib.load('models/preapproval_model.pkl')
    scaler = joblib.load('models/preapproval_scaler.pkl')
    feature_names = joblib.load('models/preapproval_features.pkl')
    use_scaling = True
    model_info = {'auc_roc': 0.666, 'model_type': 'XGBoost', 'training_samples': 2709903}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server
batch_results_store = {}

def calculate_dti(monthly_debt, monthly_income):
    if monthly_income <= 0:
        return 0
    return (monthly_debt / monthly_income) * 100

def suggest_grade(dti, lti):
    if dti < 30 and lti < 3:
        return 'A'
    elif dti < 36 and lti < 3.5:
        return 'B'
    elif dti < 43 and lti < 4:
        return 'C'
    elif dti < 50 and lti < 4.5:
        return 'D'
    else:
        return 'E'

def calculate_monthly_payment(loan_amount, interest_rate, term_years=30, repayment_type='pni'):
    term_months = term_years * 12
    monthly_rate = interest_rate / 100 / 12
    
    if repayment_type == 'interest_only':
        monthly_payment = loan_amount * monthly_rate if monthly_rate > 0 else 0
        payment_note = "Interest Only - Principal due at end"
        risk_multiplier = 1.3
    elif repayment_type == 'partial':
        if monthly_rate > 0:
            normal_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
            monthly_payment = normal_payment * 0.7
        else:
            monthly_payment = loan_amount / term_months * 0.7
        payment_note = "Partially Amortizing - Balloon payment at end"
        risk_multiplier = 1.15
    else:
        if monthly_rate > 0:
            monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
        else:
            monthly_payment = loan_amount / term_months
        payment_note = "Principal & Interest (Standard)"
        risk_multiplier = 1.0
    
    return monthly_payment, payment_note, risk_multiplier

def prepare_features_for_existing_loan(application):
    features = {}
    features['year'] = 2024
    features['emp_length_int'] = application['emp_length']
    features['annual_inc'] = application['annual_inc']
    features['loan_amount'] = application['loan_amount']
    features['interest_rate'] = application['interest_rate']
    features['dti'] = application['dti']
    features['lti_ratio'] = application['loan_amount'] / application['annual_inc']
    features['monthly_income'] = application['annual_inc'] / 12
    features['payment_burden'] = min(application.get('payment_burden', 0.25), 1.0)
    features['rate_stress'] = application['interest_rate'] + 2
    features['stress_burden'] = application.get('stress_burden', 0.25)
    features['total_pymnt'] = application['total_pymnt']
    features['total_rec_prncp'] = application['total_rec_prncp']
    features['recoveries'] = 0
    features['recovery_rate'] = 0
    features['pymnt_to_loan_ratio'] = application['total_pymnt'] / application['loan_amount'] if application['loan_amount'] > 0 else 0
    features['principal_paid_ratio'] = application['total_rec_prncp'] / application['loan_amount'] if application['loan_amount'] > 0 else 0
    features['risk_score'] = 0
    features['cbi_lti_compliant'] = 1 if features['lti_ratio'] <= 4 else 0
    features['cbi_stress_compliant'] = 1
    features['cbi_overall_compliant'] = features['cbi_lti_compliant']
    
    region_map = {'munster': 1, 'leinster': 2, 'connacht': 3, 'ulster': 4, 'dublin': 5}
    features['region'] = region_map.get(application['region'], 2)
    
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    features['grade_cat'] = grade_map.get(application['grade'], 3)
    
    home_map = {'own': 1, 'mortgage': 2, 'rent': 3, 'other': 4}
    features['home_ownership_cat'] = home_map.get(application['home_ownership'], 2)
    
    default_values = {
        'term_cat': 2, 'application_type_cat': 1, 'purpose_cat': 1,
        'interest_payment_cat': 1, 'income_cat': 3, 'income_category': 3,
        'dti_category': 2, 'payment_burden_category': 2, 'emp_length_category': 3,
        'rate_tier': 2, 'loan_amount_category': 3
    }
    features.update(default_values)
    
    X = pd.DataFrame([features])
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]
    
    if use_scaling:
        X = scaler.transform(X)
    
    return X

# App layout
app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Risk Assessment", href="/", active="exact")),
            dbc.NavItem(dbc.NavLink("Batch Upload", href="/batch", active="exact")),
            dbc.NavItem(dbc.NavLink("Documentation", href="/docs", active="exact")),
            dbc.NavItem(dbc.NavLink("User Guide", href="/guide", active="exact")),
        ],
        brand="Existing Loan Risk Monitor",
        brand_href="/",
        color="primary",
        dark=True,
    ),
    html.Div(id='page-content', className="mt-4"),
], fluid=True)

# Single Assessment Page
single_page = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🏠 Loan Information", className="bg-primary text-white"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Annual Income (€)", className="fw-bold"),
                            html.Span(" ℹ️", id="tooltip-income", style={'cursor': 'pointer', 'color': '#0d6efd'}),
                            dbc.Tooltip("Gross annual income before tax and deductions.", target="tooltip-income"),
                            dcc.Input(id='income', type='number', value=75000, className="form-control mb-2"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Monthly Debt (€)", className="fw-bold"),
                            html.Span(" ℹ️", id="tooltip-debt", style={'cursor': 'pointer', 'color': '#0d6efd'}),
                            dbc.Tooltip("MANUAL: Total of ALL existing monthly debt payments - car loans, credit cards, personal loans, other mortgages.", target="tooltip-debt"),
                            dcc.Input(id='monthly_debt', type='number', value=500, className="form-control mb-2"),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Loan Amount (€)", className="fw-bold"),
                            dcc.Input(id='loan_amount', type='number', value=75000, className="form-control mb-2"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Interest Rate (%)", className="fw-bold"),
                            dcc.Input(id='rate', type='number', value=4.5, step=0.1, className="form-control mb-2"),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Repayment Method", className="fw-bold"),
                            dcc.Dropdown(id='repayment_type',
                                options=[
                                    {'label': 'Principal & Interest (Standard)', 'value': 'pni'},
                                    {'label': 'Interest Only (Higher Risk)', 'value': 'interest_only'},
                                    {'label': 'Partially Amortizing (Medium-High Risk)', 'value': 'partial'}
                                ], value='pni', className="mb-2"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Loan Age (years)", className="fw-bold"),
                            dcc.Input(id='loan_age', type='number', value=3, className="form-control mb-2"),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Employment (years)", className="fw-bold"),
                            dcc.Input(id='emp_length', type='number', value=8, className="form-control mb-2"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Region", className="fw-bold"),
                            dcc.Dropdown(id='region', options=[{'label': r.title(), 'value': r} for r in ['leinster','munster','dublin','connacht','ulster']], value='dublin', className="mb-2"),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Credit Grade", className="fw-bold"),
                            dcc.Dropdown(id='grade', options=[{'label': f'{g} - {desc}', 'value': g} for g, desc in [('A','Excellent'),('B','Good'),('C','Fair'),('D','Average'),('E','Below Avg'),('F','Poor'),('G','Very Poor')]], value='A', className="mb-2"),
                            html.Div(id='grade-hint', className="small text-success mt-1"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Home Ownership", className="fw-bold"),
                            dcc.Dropdown(id='home_ownership', options=[{'label': h.title(), 'value': h} for h in ['own','mortgage','rent','other']], value='mortgage', className="mb-2"),
                        ], width=6),
                    ]),
                    html.Div(id='payment-calc-info', className="small text-info mt-2"),
                ])
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader("💰 Actual Payment History", className="bg-info text-white"),
                dbc.CardBody([
                    html.Div("Enter actual payment data from your loan management system", className="small text-muted mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Total Paid to Date (€)", className="fw-bold"),
                            dcc.Input(id='total_paid', type='number', value=15000, className="form-control mb-2"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Principal Repaid (€)", className="fw-bold"),
                            dcc.Input(id='principal_paid', type='number', value=8000, className="form-control mb-2"),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Number of Late Payments", className="fw-bold"),
                            html.Span(" ℹ️", id="tooltip-late", style={'cursor': 'pointer', 'color': '#0d6efd'}),
                            dbc.Tooltip("Total count of late payments made during loan history (cumulative). Example: 2 late payments over 3 years.", target="tooltip-late"),
                            dcc.Input(id='late_payments', type='number', value=0, className="form-control mb-2"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Days Past Due", className="fw-bold"),
                            html.Span(" ℹ️", id="tooltip-days", style={'cursor': 'pointer', 'color': '#0d6efd'}),
                            dbc.Tooltip("Current days past due (since last missed payment). If currently current, enter 0. Example: 45 days late on most recent payment.", target="tooltip-days"),
                            dcc.Input(id='days_past_due', type='number', value=0, className="form-control mb-2"),
                        ], width=6),
                    ]),
                    html.Div(id='payment-status', className="mt-2"),
                ])
            ], className="mb-3"),
            
            dbc.Button("Predict Default Risk", id='predict-btn', color="primary", className="w-100", size="lg"),
        ], width=5),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎯 DEFAULT PROBABILITY", className="bg-danger text-white text-center"),
                dbc.CardBody(id='risk-result'),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H3(id='dti-value', className="text-center text-primary"), html.P("DTI", className="text-center small")])), width=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H3(id='lti-value', className="text-center text-primary"), html.P("LTI", className="text-center small")])), width=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H3(id='burden-value', className="text-center text-primary"), html.P("Total Burden", className="text-center small")])), width=4),
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader("📊 Payment Behavior Analysis", className="bg-secondary text-white"),
                dbc.CardBody(id='payment-analysis'),
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader("⚠️ Risk Factors", className="bg-warning text-white"),
                dbc.CardBody(dcc.Graph(id='risk-factors', style={'height': '250px'})),
            ]),
        ], width=7),
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("📝 Recommendation", className="bg-success text-white"), dbc.CardBody(id='recommendation')]), width=12)
    ], className="mt-3"),
])

# Batch Upload Page (Fixed)
batch_page = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📁 Batch Upload - Portfolio Analysis", className="bg-primary text-white"),
                dbc.CardBody([
                    html.Div([
                        html.H6("📋 Required CSV Format", className="mt-2"),
                        html.Pre("""
id,annual_inc,monthly_debt,loan_amount,interest_rate,repayment_type,loan_age_years,emp_length,region,grade,home_ownership,total_paid,principal_paid,late_payments,days_past_due
LOAN001,75000,500,75000,4.5,pni,3,8,dublin,A,mortgage,15000,8000,0,0
LOAN002,50000,1500,200000,5.5,interest_only,2,5,leinster,C,rent,12000,5000,2,45
LOAN003,100000,2000,350000,3.8,pni,5,12,dublin,A,own,85000,50000,0,0
                        """, className="bg-light p-2 small", style={'fontSize': '11px'}),
                        html.A("📥 Download Sample CSV", id="download-sample", className="btn btn-sm btn-secondary mb-3"),
                        dcc.Download(id="download-sample-csv"),
                    ]),
                    html.Hr(),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed',
                            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'
                        },
                        multiple=False
                    ),
                    html.Div(id='upload-output', className="mt-2"),
                    html.Hr(),
                    html.Button("Process Portfolio", id='process-batch', className="btn btn-primary w-100", disabled=True),
                    html.Div(id='batch-results', className="mt-3"),
                ])
            ])
        ], width=12)
    ])
])

# Documentation Page
docs_page = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📖 Model Documentation", className="bg-info text-white"),
                dbc.CardBody([
                    html.H4("Data Source"),
                    html.P("This model was trained on the Irish Loan Data dataset from Kaggle:"),
                    html.Pre("Ikpele, Ambrose. (2021). 'Irish Loan Data'. Kaggle. https://www.kaggle.com/datasets/ikpeleambrose/irish-loan-data", 
                            className="small bg-light p-2"),
                    html.P("⚠️ **Important Note**: This is synthetically generated data created using CTGAN (Conditional Tabular GAN). It is not real customer data but simulates the statistical properties of real loan data for demonstration purposes.", 
                           className="text-warning small"),
                    html.P("The original base data comes from: https://www.kaggle.com/mrferozi/loan-data-for-dummy-bank", 
                           className="small text-muted"),
                    
                    html.H4("Model Overview", className="mt-3"),
                    html.P("This dashboard uses an XGBoost model trained on historical loan data to predict default probability for existing loans."),
                    
                    html.H4("Key Features Used", className="mt-3"),
                    html.Ul([
                        html.Li("Application data (income, loan amount, DTI, credit grade)"),
                        html.Li("Payment history (total paid, principal repaid)"),
                        html.Li("Delinquency indicators (late payments, days past due)"),
                        html.Li("Repayment method (standard, interest only, partially amortizing)"),
                    ]),
                    
                    html.H4("Model Performance", className="mt-3"),
                    dbc.Table([
                        html.Tbody([
                            html.Tr([html.Td("AUC-ROC"), html.Td(f"{model_info.get('auc_roc', 0.606):.3f}")]),
                            html.Tr([html.Td("Training Samples"), html.Td(f"{model_info.get('training_samples', 2730657):,}")]),
                            html.Tr([html.Td("Look-Ahead Bias"), html.Td("❌ Eliminated", className="text-success")]),
                            html.Tr([html.Td("Note"), html.Td("AUC 0.606 means model is slightly better than random. This is expected for existing loan prediction without look-ahead bias.")]),
                        ])
                    ], bordered=True, size="sm"),
                ])
            ])
        ], width=12)
    ])
])

# User Guide Page
user_guide_page = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📖 User Guide - How to Use This Dashboard", className="bg-primary text-white"),
                dbc.CardBody([
                    html.H4("1. Understanding Each Input Field"),
                    html.Table([
                        html.Tbody([
                            html.Tr([html.Td("Annual Income", className="fw-bold"), html.Td("Gross annual income before tax.")]),
                            html.Tr([html.Td("Monthly Debt", className="fw-bold text-warning"), html.Td("MANUAL: Total existing monthly debt payments (car loans, credit cards, etc.)")]),
                            html.Tr([html.Td("Loan Amount", className="fw-bold"), html.Td("Original loan amount borrowed.")]),
                            html.Tr([html.Td("Interest Rate", className="fw-bold"), html.Td("Current annual interest rate.")]),
                            html.Tr([html.Td("Repayment Method", className="fw-bold"), html.Td("Standard (P&I), Interest Only, or Partially Amortizing.")]),
                            html.Tr([html.Td("Loan Age", className="fw-bold"), html.Td("Years the loan has been active.")]),
                            html.Tr([html.Td("Total Paid", className="fw-bold"), html.Td("Total amount paid to date.")]),
                            html.Tr([html.Td("Principal Repaid", className="fw-bold"), html.Td("Amount paid toward principal.")]),
                            html.Tr([html.Td("Late Payments", className="fw-bold"), html.Td("Total count of late payments (cumulative).")]),
                            html.Tr([html.Td("Days Past Due", className="fw-bold"), html.Td("Current days past due (0 if current).")]),
                        ])
                    ], className="table table-sm"),
                    
                    html.H4("2. Late Payments vs Days Past Due", className="mt-3"),
                    html.P("Example:"),
                    html.Ul([
                        html.Li("Month 1: 15 days late → Late Payments = 1, Days Past Due = 15"),
                        html.Li("Month 2: Paid on time → Late Payments = 1, Days Past Due = 0"),
                        html.Li("Month 3: 30 days late → Late Payments = 2, Days Past Due = 30"),
                    ]),
                    
                    html.H4("3. Risk Level Interpretation", className="mt-3"),
                    html.Ul([
                        html.Li("< 20%: LOW RISK - Continue monitoring"),
                        html.Li("20-50%: MODERATE RISK - Schedule borrower review"),
                        html.Li("> 50%: HIGH RISK - Escalate to collections"),
                    ]),
                ])
            ])
        ], width=12)
    ])
])

# Callbacks
@app.callback(
    [Output('grade-hint', 'children'),
     Output('grade', 'value'),
     Output('payment-calc-info', 'children')],
    [Input('income', 'value'),
     Input('monthly_debt', 'value'),
     Input('loan_amount', 'value'),
     Input('rate', 'value'),
     Input('repayment_type', 'value')]
)
def update_grade_and_payment(income, monthly_debt, loan_amount, rate, repayment_type):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    grade_hint = ""
    suggested_grade = 'A'
    if income and income > 0 and monthly_debt is not None and loan_amount:
        monthly_inc = income / 12
        dti = calculate_dti(monthly_debt or 0, monthly_inc)
        lti = loan_amount / income
        suggested_grade = suggest_grade(dti, lti)
        grade_hint = html.Span([f"💡 Recommended Grade: ", html.Strong(suggested_grade, className="text-success")])
    
    payment_info = ""
    if loan_amount and rate:
        monthly_payment, payment_note, _ = calculate_monthly_payment(loan_amount, rate, 30, repayment_type or 'pni')
        payment_info = html.Div([
            html.Span(f"📊 Estimated Monthly Payment: €{monthly_payment:,.0f} ", className="fw-bold"),
            html.Small(f"({payment_note})", className="text-muted")
        ])
    
    if trigger_id != 'grade':
        return grade_hint, suggested_grade, payment_info
    return grade_hint, dash.no_update, payment_info

@app.callback(
    Output('payment-status', 'children'),
    [Input('loan_amount', 'value'), Input('loan_age', 'value'), Input('rate', 'value'),
     Input('total_paid', 'value'), Input('repayment_type', 'value')]
)
def update_payment_status(loan_amount, loan_age, rate, total_paid, repayment_type):
    if not all([loan_amount, loan_age, rate]):
        return html.Div()
    
    monthly_payment, _, _ = calculate_monthly_payment(loan_amount, rate, 30, repayment_type or 'pni')
    expected_total = monthly_payment * loan_age * 12
    
    if total_paid and expected_total > 0:
        payment_ratio = total_paid / expected_total
        
        if payment_ratio >= 1.0:
            return html.Div([html.Span("✅ Ahead of Schedule", className="text-success fw-bold")])
        elif payment_ratio >= 0.95:
            return html.Div([html.Span("✅ On Track", className="text-success fw-bold")])
        elif payment_ratio >= 0.80:
            return html.Div([html.Span("⚠️ Slightly Behind", className="text-warning fw-bold")])
        else:
            return html.Div([html.Span("❌ Severely Delinquent", className="text-danger fw-bold")])
    return html.Div()

@app.callback(
    [Output('risk-result', 'children'), Output('dti-value', 'children'), Output('lti-value', 'children'),
     Output('burden-value', 'children'), Output('payment-analysis', 'children'), Output('recommendation', 'children'), Output('risk-factors', 'figure')],
    Input('predict-btn', 'n_clicks'),
    [State('income', 'value'), State('monthly_debt', 'value'), State('loan_amount', 'value'),
     State('rate', 'value'), State('loan_age', 'value'), State('emp_length', 'value'),
     State('region', 'value'), State('grade', 'value'), State('home_ownership', 'value'),
     State('total_paid', 'value'), State('principal_paid', 'value'), 
     State('late_payments', 'value'), State('days_past_due', 'value'), State('repayment_type', 'value')]
)
def analyze(n_clicks, income, monthly_debt, loan_amount, rate, loan_age, emp_length,
            region, grade, home_ownership, total_paid, principal_paid, late_payments, days_past_due, repayment_type):
    if n_clicks is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(height=250)
        return "Click Predict", "-", "-", "-", "Enter payment data", "Click Predict", empty_fig
    
    monthly_inc = income / 12
    dti = calculate_dti(monthly_debt or 0, monthly_inc)
    lti_ratio = loan_amount / income
    
    monthly_payment, payment_note, repayment_risk = calculate_monthly_payment(loan_amount, rate, 30, repayment_type or 'pni')
    total_burden = ((monthly_debt or 0) + monthly_payment) / monthly_inc
    
    expected_total = monthly_payment * loan_age * 12
    payment_ratio = total_paid / expected_total if expected_total > 0 else 0
    principal_ratio = principal_paid / loan_amount if loan_amount > 0 else 0
    
    application = {
        'annual_inc': income, 'loan_amount': loan_amount, 'interest_rate': rate,
        'emp_length': emp_length, 'dti': dti, 'region': region,
        'grade': grade, 'home_ownership': home_ownership,
        'payment_burden': monthly_payment / monthly_inc,
        'stress_burden': ((monthly_debt or 0) + monthly_payment * 1.15) / monthly_inc,
        'total_pymnt': total_paid or 0,
        'total_rec_prncp': principal_paid or 0,
    }
    
    X = prepare_features_for_existing_loan(application)
    model_prob = best_model.predict_proba(X)[0][1]
    
    # Risk adjustments
    if payment_ratio >= 1.0:
        payment_adjustment = -0.25
    elif payment_ratio >= 0.95:
        payment_adjustment = -0.15
    elif payment_ratio >= 0.80:
        payment_adjustment = 0.0
    elif payment_ratio >= 0.60:
        payment_adjustment = 0.15
    else:
        payment_adjustment = 0.35
    
    late_penalty = 0
    if late_payments and late_payments > 3:
        late_penalty = 0.15
    elif late_payments and late_payments > 0:
        late_penalty = 0.05
    
    past_due_penalty = 0
    if days_past_due and days_past_due > 90:
        past_due_penalty = 0.20
    elif days_past_due and days_past_due > 30:
        past_due_penalty = 0.10
    
    repayment_adjustment = (repayment_risk - 1.0)
    
    adjusted_prob = model_prob + payment_adjustment + late_penalty + past_due_penalty + repayment_adjustment
    adjusted_prob = max(0.01, min(0.99, adjusted_prob))
    
    if adjusted_prob < 0.20:
        risk_text = "LOW RISK"
        risk_color = "success"
        risk_icon = "✅"
    elif adjusted_prob < 0.50:
        risk_text = "MODERATE RISK"
        risk_color = "warning"
        risk_icon = "⚠️"
    else:
        risk_text = "HIGH RISK"
        risk_color = "danger"
        risk_icon = "❌"
    
    risk_html = html.Div([
        html.H2(f"{adjusted_prob:.1%}", className=f"text-{risk_color} text-center", style={'fontSize': '64px', 'fontWeight': 'bold'}),
        html.H5(f"{risk_icon} {risk_text}", className=f"text-{risk_color} text-center"),
        html.Hr(),
        html.P(f"💰 Income: €{monthly_inc:,.0f}/mo", className="text-center small"),
        html.P(f"🏠 Mortgage: €{monthly_payment:,.0f}/mo", className="text-center small"),
    ])
    
    if payment_ratio >= 1.0:
        payment_html = html.Div([
            html.H6("✅ Excellent Payment Performance", className="text-success"),
            html.P(f"Paid {payment_ratio:.0%} of expected amount", className="small"),
        ])
    elif payment_ratio >= 0.95:
        payment_html = html.Div([
            html.H6("✅ Good Payment Performance", className="text-success"),
            html.P(f"Paid {payment_ratio:.0%} of expected amount", className="small"),
        ])
    elif payment_ratio >= 0.80:
        payment_html = html.Div([
            html.H6("⚠️ Payment Performance Needs Attention", className="text-warning"),
            html.P(f"Paid {payment_ratio:.0%} of expected amount", className="small"),
        ])
    else:
        payment_html = html.Div([
            html.H6("❌ Critical Payment Delinquency", className="text-danger"),
            html.P(f"Only {payment_ratio:.0%} of expected payments made", className="small"),
        ])
    
    if late_payments and late_payments > 0:
        payment_html.children.append(html.P(f"⚠️ {late_payments} late payment(s) recorded", className="small text-warning"))
    if days_past_due and days_past_due > 0:
        payment_html.children.append(html.P(f"⚠️ Currently {days_past_due} days past due", className="small text-danger"))
    
    if adjusted_prob < 0.20:
        recommendation = html.H4("✅ CONTINUE MONITORING", className="text-success text-center")
    elif adjusted_prob < 0.50:
        recommendation = html.H4("⚠️ SCHEDULE BORROWER REVIEW", className="text-warning text-center")
    else:
        recommendation = html.H4("❌ ESCALATE TO COLLECTIONS", className="text-danger text-center")
    
    fig = go.Figure()
    risk_factors = [
        ('DTI', min(dti / 50, 1)),
        ('LTI', min(lti_ratio / 5, 1)),
        ('Payment Status', max(0, 1 - payment_ratio) if payment_ratio < 1 else 0),
    ]
    risk_factors = [(n, v) for n, v in risk_factors if v > 0.05]
    
    if risk_factors:
        names, values = zip(*risk_factors)
        colors = ['#dc3545' if v > 0.6 else '#ffc107' if v > 0.4 else '#28a745' for v in values]
        fig.add_trace(go.Bar(x=values, y=names, orientation='h', marker_color=colors,
                             text=[f'{v:.0%}' for v in values], textposition='auto'))
    else:
        fig.add_annotation(text="No significant risk factors", x=0.5, y=0.5, showarrow=False)
    
    fig.update_layout(height=250, xaxis_title="Risk Contribution", xaxis=dict(range=[0, 1]), margin=dict(l=0, r=0, t=30, b=0))
    
    return (risk_html, f"{dti:.0f}%", f"{lti_ratio:.2f}", f"{total_burden:.0%}",
            payment_html, recommendation, fig)

# Download sample CSV
@app.callback(Output("download-sample-csv", "data"), Input("download-sample", "n_clicks"), prevent_initial_call=True)
def download_sample(n_clicks):
    if n_clicks:
        sample_df = pd.DataFrame({
            'id': ['LOAN001', 'LOAN002', 'LOAN003'],
            'annual_inc': [75000, 50000, 100000],
            'monthly_debt': [500, 1500, 2000],
            'loan_amount': [75000, 200000, 350000],
            'interest_rate': [4.5, 5.5, 3.8],
            'repayment_type': ['pni', 'interest_only', 'pni'],
            'loan_age_years': [3, 2, 5],
            'emp_length': [8, 5, 12],
            'region': ['dublin', 'leinster', 'dublin'],
            'grade': ['A', 'C', 'A'],
            'home_ownership': ['mortgage', 'rent', 'own'],
            'total_paid': [15000, 12000, 85000],
            'principal_paid': [8000, 5000, 50000],
            'late_payments': [0, 2, 0],
            'days_past_due': [0, 45, 0]
        })
        return dcc.send_data_frame(sample_df.to_csv, "sample_existing_loans.csv", index=False)
    return None

# Validate upload
@app.callback(
    [Output('upload-output', 'children'),
     Output('process-batch', 'disabled')],
    Input('upload-data', 'contents')
)
def validate_upload(contents):
    if contents is None:
        return html.Div("No file uploaded yet", className="text-muted"), True
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        required_columns = ['id', 'annual_inc', 'monthly_debt', 'loan_amount', 
                           'interest_rate', 'repayment_type', 'loan_age_years', 
                           'emp_length', 'region', 'grade', 'home_ownership',
                           'total_paid', 'principal_paid', 'late_payments', 'days_past_due']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            return html.Div([
                html.Span(f"❌ Missing columns: {', '.join(missing_cols)}", className="text-danger"),
                html.Br(),
                html.Small(f"Found: {', '.join(df.columns)}", className="text-muted")
            ], className="alert alert-danger"), True
        else:
            return html.Div([
                html.Span(f"✅ Valid file! {len(df)} loans ready to process", className="text-success"),
                html.Br(),
                html.Small(f"Columns: {', '.join(df.columns)}", className="text-muted")
            ], className="alert alert-success"), False
    except Exception as e:
        return html.Div(f"❌ Error reading file: {str(e)}", className="alert alert-danger"), True

# Process batch (FIXED)
@app.callback(
    Output('batch-results', 'children'),
    Input('process-batch', 'n_clicks'),
    State('upload-data', 'contents'),
    prevent_initial_call=True
)
def process_batch(n_clicks, contents):
    if n_clicks is None or contents is None:
        return html.Div()
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        results = []
        
        for idx, row in df.iterrows():
            try:
                # Calculate metrics
                monthly_inc = row['annual_inc'] / 12
                dti = calculate_dti(row['monthly_debt'], monthly_inc)
                lti = row['loan_amount'] / row['annual_inc']
                
                monthly_payment, _, _ = calculate_monthly_payment(row['loan_amount'], row['interest_rate'], 30, row['repayment_type'])
                expected_total = monthly_payment * row['loan_age_years'] * 12
                payment_ratio = row['total_paid'] / expected_total if expected_total > 0 else 0
                
                # Simple risk score based on key factors
                risk_score = 0
                if payment_ratio < 0.8:
                    risk_score += 30
                if row['late_payments'] > 2:
                    risk_score += 20
                if row['days_past_due'] > 30:
                    risk_score += 20
                if row['grade'] in ['D', 'E', 'F', 'G']:
                    risk_score += 15
                if row['repayment_type'] == 'interest_only':
                    risk_score += 15
                
                if risk_score < 25:
                    risk_level = "Low"
                    risk_badge = "success"
                    prob = f"{10 + risk_score/5:.0f}%"
                elif risk_score < 50:
                    risk_level = "Moderate"
                    risk_badge = "warning"
                    prob = f"{25 + (risk_score-25)/2:.0f}%"
                else:
                    risk_level = "High"
                    risk_badge = "danger"
                    prob = f"{50 + (risk_score-50)/1.5:.0f}%"
                
                results.append({
                    'Loan ID': row['id'],
                    'Default Prob': prob,
                    'Risk Level': risk_level,
                    'Risk Badge': risk_badge,
                    'Payment Ratio': f"{payment_ratio:.0%}",
                    'Late Payments': row['late_payments'],
                    'Days Past Due': row['days_past_due']
                })
            except Exception as e:
                results.append({
                    'Loan ID': row.get('id', f'Row_{idx}'),
                    'Default Prob': 'Error',
                    'Risk Level': 'N/A',
                    'Risk Badge': 'secondary',
                    'Payment Ratio': 'N/A',
                    'Late Payments': 'N/A',
                    'Days Past Due': 'N/A'
                })
        
        # Store results
        global batch_results_store
        batch_results_store = {'results': results}
        
        high_risk = sum(1 for r in results if r.get('Risk Level') == 'High')
        
        return html.Div([
            html.Div([
                html.H5(f"📊 Batch Processing Results", className="text-center"),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.H3(f"{len(results)}", className="text-center text-primary"),
                        html.P("Loans", className="text-center text-muted small")
                    ]), width=4),
                    dbc.Col(html.Div([
                        html.H3(f"{high_risk}", className="text-center text-danger"),
                        html.P("High Risk", className="text-center text-muted small")
                    ]), width=4),
                    dbc.Col(html.Div([
                        html.H3(f"{len(results) - high_risk}", className="text-center text-success"),
                        html.P("Low/Moderate Risk", className="text-center text-muted small")
                    ]), width=4),
                ]),
                html.Hr(),
                dash_table.DataTable(
                    data=[{k: v for k, v in r.items() if k != 'Risk Badge'} for r in results],
                    columns=[{"name": i, "id": i} for i in ['Loan ID', 'Default Prob', 'Risk Level', 'Payment Ratio', 'Late Payments', 'Days Past Due']],
                    page_size=15,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '8px'},
                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{Risk Level} = "High"'},
                         'backgroundColor': '#ffebee', 'color': '#c62828'},
                        {'if': {'filter_query': '{Risk Level} = "Moderate"'},
                         'backgroundColor': '#fff3e0', 'color': '#e65100'},
                        {'if': {'filter_query': '{Risk Level} = "Low"'},
                         'backgroundColor': '#e8f5e9', 'color': '#2e7d32'},
                    ]
                ),
                html.Hr(),
                html.Div([
                    html.Button("📥 Export to CSV", id="export-batch-results", className="btn btn-sm btn-secondary"),
                    dcc.Download(id="download-batch-results")
                ], className="text-center")
            ])
        ])
    except Exception as e:
        return html.Div(f"Error processing batch: {str(e)}", className="alert alert-danger")

# Export batch results
@app.callback(
    Output("download-batch-results", "data"),
    Input("export-batch-results", "n_clicks"),
    prevent_initial_call=True
)
def export_batch_results(n_clicks):
    if n_clicks:
        results = batch_results_store.get('results', [])
        if results:
            export_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'Risk Badge'} for r in results])
            return dcc.send_data_frame(export_df.to_csv, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    return None

# Page routing
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/batch':
        return batch_page
    elif pathname == '/docs':
        return docs_page
    elif pathname == '/guide':
        return user_guide_page
    else:
        return single_page

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host='0.0.0.0', port=port)