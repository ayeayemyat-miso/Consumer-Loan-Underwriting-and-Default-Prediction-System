"""
Dashboard A: New Loan Application Underwriting
Central Bank of Ireland (CBI) Compliant
WITH LOAN TERM SLIDER & AMORTIZATION SCHEDULE
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
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Loading Underwriting Dashboard...")
print("="*60)

# Load pre-approval model
best_model = joblib.load('models/preapproval_model.pkl')
scaler = joblib.load('models/preapproval_scaler.pkl')
features = joblib.load('models/preapproval_features.pkl')
use_scaling = joblib.load('models/preapproval_use_scaling.pkl')

with open('models/preapproval_model_info.json', 'r') as f:
    model_info = json.load(f)

print(f"✅ Loaded {model_info['model_type']} model (AUC: {model_info['auc_roc']:.3f})")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

def calculate_amortization_schedule(loan_amount, interest_rate, term_years=30, repayment_type='pni'):
    """
    Calculate full loan amortization schedule
    Returns DataFrame with payment schedule and summary
    """
    term_months = term_years * 12
    monthly_rate = interest_rate / 100 / 12
    
    if repayment_type == 'interest_only':
        # Interest Only: Fixed interest payment, principal due at end
        monthly_interest = loan_amount * monthly_rate if monthly_rate > 0 else 0
        monthly_payment = monthly_interest
        payment_type = "Interest Only (Principal due at end)"
        
        schedule = []
        remaining_balance = loan_amount
        
        for month in range(1, term_months + 1):
            interest_paid = remaining_balance * monthly_rate if monthly_rate > 0 else 0
            principal_paid = 0 if month < term_months else remaining_balance
            payment = monthly_interest if month < term_months else remaining_balance + monthly_interest
            
            if month == term_months:
                payment = remaining_balance + monthly_interest
                principal_paid = remaining_balance
            
            schedule.append({
                'Month': month,
                'Year': (month - 1) // 12 + 1,
                'Payment': payment,
                'Principal': principal_paid,
                'Interest': interest_paid,
                'Remaining Balance': remaining_balance - principal_paid if month < term_months else 0
            })
            remaining_balance -= principal_paid
            
    elif repayment_type == 'partial':
        # Partially Amortizing: 70% of normal payment, balloon at end
        if monthly_rate > 0:
            normal_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
        else:
            normal_payment = loan_amount / term_months
        monthly_payment = normal_payment * 0.7
        payment_type = "Partially Amortizing (Balloon Payment at End)"
        
        schedule = []
        remaining_balance = loan_amount
        
        for month in range(1, term_months + 1):
            interest_paid = remaining_balance * monthly_rate if monthly_rate > 0 else 0
            principal_paid = monthly_payment - interest_paid if monthly_payment > interest_paid else 0
            principal_paid = max(0, min(principal_paid, remaining_balance))
            
            if month == term_months:
                principal_paid = remaining_balance
                monthly_payment = remaining_balance + interest_paid
            
            schedule.append({
                'Month': month,
                'Year': (month - 1) // 12 + 1,
                'Payment': monthly_payment,
                'Principal': principal_paid,
                'Interest': interest_paid,
                'Remaining Balance': remaining_balance - principal_paid
            })
            remaining_balance -= principal_paid
            
    else:
        # Standard Principal & Interest (Fully Amortizing)
        if monthly_rate > 0:
            monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
        else:
            monthly_payment = loan_amount / term_months
        payment_type = "Principal & Interest (Fully Amortizing)"
        
        schedule = []
        remaining_balance = loan_amount
        
        for month in range(1, term_months + 1):
            interest_paid = remaining_balance * monthly_rate if monthly_rate > 0 else 0
            principal_paid = monthly_payment - interest_paid
            principal_paid = max(0, min(principal_paid, remaining_balance))
            
            schedule.append({
                'Month': month,
                'Year': (month - 1) // 12 + 1,
                'Payment': monthly_payment,
                'Principal': principal_paid,
                'Interest': interest_paid,
                'Remaining Balance': remaining_balance - principal_paid
            })
            remaining_balance -= principal_paid
    
    df_schedule = pd.DataFrame(schedule)
    
    # Summary statistics
    total_payment = df_schedule['Payment'].sum()
    total_interest = df_schedule['Interest'].sum()
    total_principal = df_schedule['Principal'].sum()
    
    summary = {
        'monthly_payment': monthly_payment,
        'total_payment': total_payment,
        'total_interest': total_interest,
        'total_principal': total_principal,
        'payment_type': payment_type,
        'term_months': term_months,
        'term_years': term_years,
        'interest_rate': interest_rate,
        'loan_amount': loan_amount
    }
    
    return df_schedule, summary

def calculate_metrics(annual_inc, loan_amount, interest_rate, monthly_debt, property_value=None, repayment_type='pni', term_years=30):
    """Calculate all loan metrics including optional LTV and repayment type"""
    monthly_inc = annual_inc / 12
    lti_ratio = loan_amount / annual_inc
    ltv_ratio = loan_amount / property_value if property_value and property_value > 0 else None
    
    # Monthly payment calculation based on repayment type AND term
    term_months = term_years * 12
    monthly_rate = interest_rate / 100 / 12
    
    if repayment_type == 'interest_only':
        monthly_payment = loan_amount * monthly_rate if monthly_rate > 0 else 0
        repayment_note = f"Interest Only - Principal due at end ({term_years} years)"
        repayment_risk = "High"
    elif repayment_type == 'partial':
        if monthly_rate > 0:
            normal_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
            monthly_payment = normal_payment * 0.7
        else:
            monthly_payment = loan_amount / term_months * 0.7
        repayment_note = f"Partially Amortizing - Balloon payment at end ({term_years} years)"
        repayment_risk = "Medium-High"
    else:
        if monthly_rate > 0:
            monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
        else:
            monthly_payment = loan_amount / term_months
        repayment_note = f"Principal & Interest - Fully Amortizing ({term_years} years)"
        repayment_risk = "Standard"
    
    payment_burden = monthly_payment / monthly_inc
    dti = (monthly_debt / monthly_inc) * 100
    total_burden = (monthly_debt + monthly_payment) / monthly_inc
    
    # Calculate surplus after all payments
    monthly_surplus = monthly_inc - monthly_debt - monthly_payment
    surplus_ratio = monthly_surplus / monthly_inc
    
    # Stress test (+2%) - CBI requirement
    stress_rate = interest_rate + 2
    stress_monthly_rate = stress_rate / 100 / 12
    
    if repayment_type == 'interest_only':
        stress_payment = loan_amount * stress_monthly_rate if stress_monthly_rate > 0 else 0
    elif repayment_type == 'partial':
        if stress_monthly_rate > 0:
            normal_stress = loan_amount * (stress_monthly_rate * (1 + stress_monthly_rate)**term_months) / ((1 + stress_monthly_rate)**term_months - 1)
            stress_payment = normal_stress * 0.7
        else:
            stress_payment = loan_amount / term_months * 0.7
    else:
        if stress_monthly_rate > 0:
            stress_payment = loan_amount * (stress_monthly_rate * (1 + stress_monthly_rate)**term_months) / ((1 + stress_monthly_rate)**term_months - 1)
        else:
            stress_payment = loan_amount / term_months
    
    stress_burden = (monthly_debt + stress_payment) / monthly_inc
    stress_limit = 0.35  # CBI limit: 35% of gross income
    
    # Grade recommendation (with term consideration - shorter terms are lower risk)
    risk_penalty = 0
    if repayment_type == 'interest_only':
        risk_penalty = 5
    elif repayment_type == 'partial':
        risk_penalty = 3
    
    # Shorter term reduces risk
    if term_years <= 15:
        risk_penalty -= 2
    elif term_years >= 30:
        risk_penalty += 1
    
    effective_dti = dti + risk_penalty
    
    if effective_dti < 30 and lti_ratio < 3:
        suggested_grade = "A"
        grade_desc = "Excellent - Very low risk"
        grade_color = "success"
    elif effective_dti < 36 and lti_ratio < 3.5:
        suggested_grade = "B"
        grade_desc = "Good - Low risk"
        grade_color = "success"
    elif effective_dti < 43 and lti_ratio < 4:
        suggested_grade = "C"
        grade_desc = "Fair - Moderate risk"
        grade_color = "warning"
    elif effective_dti < 50 and lti_ratio < 4.5:
        suggested_grade = "D"
        grade_desc = "Average - Higher risk"
        grade_color = "warning"
    else:
        suggested_grade = "E"
        grade_desc = "Below Average - High risk"
        grade_color = "danger"
    
    # LTV assessment
    ltv_status = None
    if ltv_ratio:
        if ltv_ratio > 0.90:
            ltv_status = {"level": "High", "color": "danger", "message": f"⚠️ High LTV {ltv_ratio:.0%} > 90% - High collateral risk"}
        elif ltv_ratio > 0.80:
            ltv_status = {"level": "Elevated", "color": "warning", "message": f"⚠️ LTV {ltv_ratio:.0%} > 80% - May require PMI"}
        elif ltv_ratio > 0.70:
            ltv_status = {"level": "Good", "color": "info", "message": f"✓ Acceptable LTV {ltv_ratio:.0%}"}
        else:
            ltv_status = {"level": "Low", "color": "success", "message": f"✅ Low LTV {ltv_ratio:.0%} - Strong equity position"}
    
    return {
        'lti_ratio': lti_ratio,
        'ltv_ratio': ltv_ratio,
        'ltv_status': ltv_status,
        'monthly_payment': monthly_payment,
        'stress_payment': stress_payment,
        'stress_burden': stress_burden,
        'stress_limit': stress_limit,
        'payment_burden': payment_burden,
        'dti': dti,
        'effective_dti': effective_dti,
        'total_burden': total_burden,
        'monthly_surplus': monthly_surplus,
        'surplus_ratio': surplus_ratio,
        'lti_compliant': lti_ratio <= 4,
        'stress_compliant': stress_burden <= stress_limit,
        'stress_burden_pct': stress_burden * 100,
        'suggested_grade': suggested_grade,
        'grade_desc': grade_desc,
        'grade_color': grade_color,
        'monthly_inc': monthly_inc,
        'repayment_note': repayment_note,
        'repayment_risk': repayment_risk,
        'term_years': term_years
    }

def prepare_features(annual_inc, loan_amount, interest_rate, dti, emp_length, grade, region, home_ownership, term_years=30):
    grade_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
    region_map = {'leinster':1, 'munster':2, 'dublin':3, 'connacht':4, 'ulster':5}
    home_map = {'own':1, 'mortgage':2, 'rent':3, 'other':4}
    
    monthly_inc = annual_inc / 12
    lti_ratio = loan_amount / annual_inc
    monthly_rate = interest_rate / 100 / 12
    term_months = term_years * 12
    if monthly_rate > 0:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
    else:
        monthly_payment = loan_amount / term_months
    payment_burden = monthly_payment / monthly_inc
    stress_rate = interest_rate + 2
    stress_monthly_rate = stress_rate / 100 / 12
    if stress_monthly_rate > 0:
        stress_payment = loan_amount * (stress_monthly_rate * (1 + stress_monthly_rate)**term_months) / ((1 + stress_monthly_rate)**term_months - 1)
    else:
        stress_payment = loan_amount / term_months
    stress_burden = stress_payment / monthly_inc
    
    feature_dict = {
        'emp_length_int': emp_length, 'annual_inc': annual_inc, 'dti': dti,
        'loan_amount': loan_amount, 'interest_rate': interest_rate, 'term_cat': 2,
        'grade_cat': grade_map.get(grade, 3), 'home_ownership_cat': home_map.get(home_ownership, 2),
        'lti_ratio': lti_ratio, 'payment_burden': payment_burden, 'stress_burden': stress_burden,
        'region': region_map.get(region, 3), 'purpose_cat': 1,
    }
    X = pd.DataFrame([feature_dict])
    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features]
    if use_scaling:
        X = scaler.transform(X)
    return X

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("🏦 New Loan Underwriting System", className="text-center text-primary mb-2"), width=12),
        dbc.Col(html.P("Central Bank of Ireland (CBI) Compliant", className="text-center text-muted mb-4"), width=12)
    ]),
    
    dbc.Row([
        # Left Column - Input Form
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📋 Loan Application", className="bg-primary text-white"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([html.Label("Annual Income (€)"), dcc.Input(id='income', type='number', value=75000, className="form-control mb-2")], width=6),
                        dbc.Col([html.Label("Monthly Debt (€)"), dcc.Input(id='monthly_debt', type='number', value=1000, className="form-control mb-2")], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([html.Label("Loan Amount (€)"), dcc.Input(id='loan_amount', type='number', value=250000, className="form-control mb-2")], width=6),
                        dbc.Col([html.Label("Interest Rate (%)"), dcc.Input(id='rate', type='number', value=4.5, step=0.1, className="form-control mb-2")], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([html.Label("Property Value (€) - Optional"), dcc.Input(id='property_value', type='number', placeholder="For LTV calculation", className="form-control mb-2")], width=6),
                        dbc.Col([html.Label("Employment (years)"), dcc.Input(id='emp_length', type='number', value=8, className="form-control mb-2")], width=6),
                    ]),
                    
                    # NEW: Loan Term Slider with Presets
                    dbc.Row([
                        dbc.Col([
                            html.Label("Loan Term (years)", className="fw-bold mt-2"),
                            html.Div([
                                dcc.Slider(
                                    id='loan_term_slider',
                                    min=1,
                                    max=40,
                                    step=1,
                                    value=30,
                                    marks={i: str(i) for i in [1, 5, 10, 15, 20, 25, 30, 35, 40]},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                            ], className="mb-2"),
                            html.Div([
                                dbc.Button("5", id="term-preset-5", size="sm", className="me-1"),
                                dbc.Button("10", id="term-preset-10", size="sm", className="me-1"),
                                dbc.Button("15", id="term-preset-15", size="sm", className="me-1"),
                                dbc.Button("20", id="term-preset-20", size="sm", className="me-1"),
                                dbc.Button("25", id="term-preset-25", size="sm", className="me-1"),
                                dbc.Button("30", id="term-preset-30", size="sm", className="me-1"),
                                dbc.Button("35", id="term-preset-35", size="sm", className="me-1"),
                                dbc.Button("40", id="term-preset-40", size="sm"),
                            ], className="text-center mt-2"),
                            html.Div(id='selected-term-display', className="text-center text-primary fw-bold mt-2"),
                        ], width=12),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Repayment Type"),
                            dcc.Dropdown(id='repayment_type',
                                options=[
                                    {'label': 'Principal & Interest (Standard)', 'value': 'pni'},
                                    {'label': 'Interest Only (Higher Risk)', 'value': 'interest_only'},
                                    {'label': 'Partially Amortizing (Medium-High Risk)', 'value': 'partial'}
                                ], value='pni', className="mb-2"),
                            html.Div(id='repayment-warning', className="small text-muted")
                        ], width=6),
                        dbc.Col([
                            html.Label("Credit Grade"),
                            dcc.Dropdown(id='grade', options=[{'label': f'{g} - {desc}', 'value': g} for g, desc in [('A','Excellent'),('B','Good'),('C','Fair'),('D','Average'),('E','Below Avg'),('F','Poor'),('G','Very Poor')]], value='A', className="mb-2"),
                            html.Div(id='grade-hint', className="small text-muted")
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Region"),
                            dcc.Dropdown(id='region', options=[{'label': r.title(), 'value': r} for r in ['leinster','munster','dublin','connacht','ulster']], value='dublin', className="mb-2"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Home Ownership"),
                            dcc.Dropdown(id='home_ownership', options=[{'label': h.title(), 'value': h} for h in ['own','mortgage','rent','other']], value='mortgage', className="mb-2"),
                        ], width=6),
                    ]),
                    dbc.Button("Evaluate Application", id='evaluate-btn', color="primary", className="mt-3 w-100", size="lg"),
                ])
            ])
        ], width=5),
        
        # Right Column - Results
        dbc.Col([
            dbc.Card([dbc.CardHeader("🎯 Risk Assessment", className="bg-danger text-white"), dbc.CardBody(id='risk-result')], className="mb-3"),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H3(id='dti-value', className="text-center text-primary"), html.P("DTI", className="text-center small")]))),
                dbc.Col(dbc.Card(dbc.CardBody([html.H3(id='lti-value', className="text-center text-primary"), html.P("LTI", className="text-center small")]))),
                dbc.Col(dbc.Card(dbc.CardBody([html.H3(id='burden-value', className="text-center text-primary"), html.P("Total Burden", className="text-center small")]))),
            ], className="mb-3"),
            dbc.Card([dbc.CardHeader("💡 Recommendation", className="bg-info text-white"), dbc.CardBody(id='grade-recommendation')], className="mb-3"),
            dbc.Card([dbc.CardHeader("📊 Risk Factors", className="bg-secondary text-white"), dbc.CardBody(dcc.Graph(id='risk-factors', style={'height': '250px'}))]),
        ], width=7),
    ]),
    
    # Decision Summary Card
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("📝 Decision Summary", className="bg-dark text-white"),
            dbc.CardBody(id='decision-summary')
        ]), width=12)
    ], className="mt-3"),
    
    # Loan Amortization Schedule Card
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("📅 Loan Amortization Schedule", className="bg-dark text-white"),
            dbc.CardBody([
                html.Div(id='amortization-summary', className="mb-3"),
                html.Div([
                    html.Label("Show Schedule For:", className="fw-bold me-2"),
                    dcc.RadioItems(
                        id='schedule-period',
                        options=[
                            {'label': 'First 12 months', 'value': 12},
                            {'label': 'First 60 months (5 years)', 'value': 60},
                            {'label': 'Full term', 'value': 'full'}
                        ],
                        value=12,
                        inline=True,
                        className="mb-3"
                    ),
                    html.Div(id='amortization-table', style={'maxHeight': '400px', 'overflowY': 'auto'}),
                ]),
                html.Div([
                    html.Button("📥 Download Full Schedule (CSV)", id="download-schedule-btn", className="btn btn-sm btn-secondary mt-3"),
                    dcc.Download(id="download-schedule")
                ], className="text-center")
            ])
        ]), width=12)
    ], className="mt-3"),
    
    dbc.Row([
        dbc.Col(html.Div([
            html.Hr(), 
            html.P(f"📈 Model: {model_info['model_type']} | AUC: {model_info['auc_roc']:.3f} | Trained on {model_info['training_samples']:,} loans", 
                   className="text-center text-muted small")
        ]), width=12)
    ])
], fluid=True)

# Callbacks for Term Presets
@app.callback(
    Output('loan_term_slider', 'value'),
    [Input('term-preset-5', 'n_clicks'),
     Input('term-preset-10', 'n_clicks'),
     Input('term-preset-15', 'n_clicks'),
     Input('term-preset-20', 'n_clicks'),
     Input('term-preset-25', 'n_clicks'),
     Input('term-preset-30', 'n_clicks'),
     Input('term-preset-35', 'n_clicks'),
     Input('term-preset-40', 'n_clicks')],
    prevent_initial_call=True
)
def update_term_preset(n5, n10, n15, n20, n25, n30, n35, n40):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 30
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    preset_map = {
        'term-preset-5': 5, 'term-preset-10': 10, 'term-preset-15': 15,
        'term-preset-20': 20, 'term-preset-25': 25, 'term-preset-30': 30,
        'term-preset-35': 35, 'term-preset-40': 40
    }
    return preset_map.get(button_id, 30)

@app.callback(
    Output('selected-term-display', 'children'),
    Input('loan_term_slider', 'value')
)
def update_term_display(term_years):
    return html.Span(f"Selected Term: {term_years} years ({term_years * 12} months)")

# Real-time grade hint
@app.callback(
    [Output('grade-hint', 'children'),
     Output('repayment-warning', 'children')],
    [Input('income', 'value'),
     Input('monthly_debt', 'value'),
     Input('loan_amount', 'value'),
     Input('repayment_type', 'value')]
)
def update_hints(income, monthly_debt, loan_amount, repayment_type):
    if income and income > 0 and monthly_debt and loan_amount:
        dti = (monthly_debt / (income/12)) * 100
        lti = loan_amount / income
        if dti < 30 and lti < 3:
            grade_hint = html.Span(["💡 Recommended: ", html.Strong("Grade A", className="text-success")])
        elif dti < 36 and lti < 3.5:
            grade_hint = html.Span(["💡 Recommended: ", html.Strong("Grade B", className="text-success")])
        elif dti < 43 and lti < 4:
            grade_hint = html.Span(["💡 Recommended: ", html.Strong("Grade C", className="text-warning")])
        else:
            grade_hint = html.Span(["💡 Recommended: ", html.Strong("Grade D or E", className="text-danger")])
    else:
        grade_hint = ""
    
    if repayment_type == 'interest_only':
        repayment_warning = html.Span("⚠️ Interest Only: Higher risk, principal due at end", className="text-danger")
    elif repayment_type == 'partial':
        repayment_warning = html.Span("⚠️ Partially Amortizing: Balloon payment risk", className="text-warning")
    else:
        repayment_warning = html.Span("✓ Standard repayment", className="text-success")
    
    return grade_hint, repayment_warning

# Amortization Schedule Callback
@app.callback(
    [Output('amortization-summary', 'children'),
     Output('amortization-table', 'children'),
     Output('download-schedule', 'data')],
    [Input('loan_amount', 'value'),
     Input('rate', 'value'),
     Input('repayment_type', 'value'),
     Input('loan_term_slider', 'value'),
     Input('schedule-period', 'value'),
     Input('download-schedule-btn', 'n_clicks')]
)
def update_amortization(loan_amount, interest_rate, repayment_type, term_years, schedule_period, download_clicks):
    ctx = dash.callback_context
    
    if not loan_amount or not interest_rate:
        return html.Div("Enter loan amount and interest rate to see schedule"), "", None
    
    term_years = term_years or 30
    
    # Calculate schedule
    df_schedule, summary = calculate_amortization_schedule(loan_amount, interest_rate, term_years, repayment_type or 'pni')
    
    # Summary display
    summary_html = html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.H4(f"€{summary['monthly_payment']:,.0f}", className="text-primary text-center"),
                html.P("Monthly Payment", className="text-center small text-muted")
            ]), width=3),
            dbc.Col(html.Div([
                html.H4(f"€{summary['total_principal']:,.0f}", className="text-success text-center"),
                html.P("Total Principal", className="text-center small text-muted")
            ]), width=3),
            dbc.Col(html.Div([
                html.H4(f"€{summary['total_interest']:,.0f}", className="text-warning text-center"),
                html.P("Total Interest", className="text-center small text-muted")
            ]), width=3),
            dbc.Col(html.Div([
                html.H4(f"€{summary['total_payment']:,.0f}", className="text-info text-center"),
                html.P("Total Cost", className="text-center small text-muted")
            ]), width=3),
        ]),
        html.P(f"📋 {summary['payment_type']} | {summary['term_months']} months ({summary['term_years']} years) | Interest Rate: {summary['interest_rate']}%", 
               className="text-center small text-muted mt-2"),
        html.P(f"💰 Loan Amount: €{summary['loan_amount']:,.0f}", className="text-center small text-muted")
    ])
    
    # Determine how many months to show
    if schedule_period == 'full':
        display_months = len(df_schedule)
    else:
        display_months = min(int(schedule_period), len(df_schedule))
    
    # Filter schedule for display
    display_df = df_schedule.head(display_months).copy()
    display_df['Payment'] = display_df['Payment'].apply(lambda x: f"€{x:,.0f}")
    display_df['Principal'] = display_df['Principal'].apply(lambda x: f"€{x:,.0f}")
    display_df['Interest'] = display_df['Interest'].apply(lambda x: f"€{x:,.0f}")
    display_df['Remaining Balance'] = display_df['Remaining Balance'].apply(lambda x: f"€{x:,.0f}")
    
    table_html = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in ['Month', 'Year', 'Payment', 'Principal', 'Interest', 'Remaining Balance']],
        page_size=15,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'right', 'padding': '8px', 'fontSize': '12px'},
        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'}
        ]
    )
    
    # Handle download
    if download_clicks and ctx.triggered and 'download-schedule-btn' in ctx.triggered[0]['prop_id']:
        return summary_html, table_html, dcc.send_data_frame(df_schedule.to_csv, f"amortization_schedule_{term_years}yrs_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
    
    return summary_html, table_html, None

# Main evaluation callback
@app.callback(
    [Output('risk-result', 'children'),
     Output('dti-value', 'children'),
     Output('lti-value', 'children'),
     Output('burden-value', 'children'),
     Output('grade-recommendation', 'children'),
     Output('decision-summary', 'children'),
     Output('risk-factors', 'figure')],
    Input('evaluate-btn', 'n_clicks'),
    [State('income', 'value'),
     State('monthly_debt', 'value'),
     State('loan_amount', 'value'),
     State('rate', 'value'),
     State('emp_length', 'value'),
     State('grade', 'value'),
     State('region', 'value'),
     State('home_ownership', 'value'),
     State('property_value', 'value'),
     State('repayment_type', 'value'),
     State('loan_term_slider', 'value')]
)
def evaluate(n_clicks, income, monthly_debt, loan_amount, rate, emp_length, 
             grade, region, home_ownership, property_value, repayment_type, term_years):
    if n_clicks is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(height=250)
        return "Click Evaluate", "-", "-", "-", "-", "Click Evaluate to see summary", empty_fig
    
    term_years = term_years or 30
    
    # Calculate metrics
    metrics = calculate_metrics(income, loan_amount, rate, monthly_debt or 0, property_value, repayment_type, term_years)
    
    # Prepare features and predict
    X = prepare_features(income, loan_amount, rate, metrics['dti'], emp_length, 
                         grade, region, home_ownership, term_years)
    prob_default = best_model.predict_proba(X)[0][1]
    
    # Adjust probability for non-standard repayment and term
    if repayment_type == 'interest_only':
        prob_default = min(0.99, prob_default * 1.3)
    elif repayment_type == 'partial':
        prob_default = min(0.99, prob_default * 1.15)
    
    # Shorter term reduces risk
    if term_years <= 15:
        prob_default = prob_default * 0.85
    elif term_years >= 35:
        prob_default = min(0.99, prob_default * 1.1)
    
    # Risk result
    risk_color = "danger" if prob_default > 0.5 else "success"
    
    # Determine risk level description
    if prob_default < 0.05:
        risk_level_text = "VERY LOW RISK"
    elif prob_default < 0.25:
        risk_level_text = "LOW RISK"
    elif prob_default < 0.5:
        risk_level_text = "MODERATE RISK"
    else:
        risk_level_text = "HIGH RISK"
    
    # Stress test display with quantitative value
    if metrics['stress_compliant']:
        stress_text = f"{metrics['stress_burden_pct']:.1f}% < {metrics['stress_limit']*100:.0f}% limit"
    else:
        stress_text = f"{metrics['stress_burden_pct']:.1f}% > {metrics['stress_limit']*100:.0f}% limit"
    
    stress_display = html.Span([
        html.Span("✅ Stress " if metrics['stress_compliant'] else "❌ Stress ",
                 className=f"text-{'success' if metrics['stress_compliant'] else 'danger'}"),
        html.Small(f"({stress_text})", className="text-muted")
    ])
    
    risk_html = html.Div([
        html.H2(f"{prob_default:.1%}", className=f"text-{risk_color} text-center", style={'fontSize': '48px'}),
        html.H5(f"{risk_level_text}", className=f"text-{risk_color} text-center"),
        html.Hr(),
        html.Div([
            html.P(f"💰 Income: €{metrics['monthly_inc']:,.0f}/mo", className="text-center small"),
            html.P(f"🏠 Mortgage: €{metrics['monthly_payment']:,.0f}/mo", className="text-center small"),
            html.P(f"📋 {metrics['repayment_note']}", className="text-center small text-muted"),
        ]),
        html.Div([
            html.Span("🏦 CBI: ", className="fw-bold"),
            html.Span("✅ LTI " if metrics['lti_compliant'] else "❌ LTI ",
                     className=f"text-{'success' if metrics['lti_compliant'] else 'danger'}"),
            html.Span("| "),
            stress_display
        ], className="text-center mt-2")
    ])
    
    # Grade recommendation
    grade_html = html.Div([
        html.H3(f"Suggested: Grade {metrics['suggested_grade']}", className=f"text-{metrics['grade_color']} text-center"),
        html.P(metrics['grade_desc'], className="text-center small"),
        html.Hr(),
        html.P(f"📊 DTI: {metrics['dti']:.1f}% | 📈 LTI: {metrics['lti_ratio']:.2f}", className="text-center"),
        html.P(f"⏰ Term: {term_years} years", className="text-center small text-muted"),
    ])
    
    if repayment_type != 'pni':
        grade_html.children.append(
            html.P(f"⚠️ {metrics['repayment_note']} - {metrics['repayment_risk']} risk", 
                   className="text-warning text-center small mt-2")
        )
    
    if metrics['ltv_status']:
        grade_html.children.append(
            html.P(metrics['ltv_status']['message'], 
                   className=f"text-{metrics['ltv_status']['color']} text-center small mt-2")
        )
    
    # DECISION SUMMARY
    strengths = []
    weaknesses = []
    
    if metrics['dti'] < 30:
        strengths.append("✓ Very low DTI ratio")
    elif metrics['dti'] < 36:
        strengths.append("✓ Strong DTI ratio")
    
    if metrics['lti_ratio'] < 3:
        strengths.append("✓ Very low LTI ratio")
    elif metrics['lti_ratio'] < 3.5:
        strengths.append("✓ Strong LTI ratio")
    
    if emp_length >= 5:
        strengths.append(f"✓ Stable employment ({emp_length} years)")
    
    if grade == 'A':
        strengths.append(f"✓ Excellent credit grade ({grade})")
    elif grade == 'B':
        strengths.append(f"✓ Good credit grade ({grade})")
    
    if term_years <= 20:
        strengths.append(f"✓ Short loan term ({term_years} years) - builds equity faster")
    
    if prob_default < 0.05:
        strengths.append(f"✓ Very low default probability ({prob_default:.1%})")
    elif prob_default < 0.2:
        strengths.append(f"✓ Low default probability ({prob_default:.1%})")
    
    if metrics['surplus_ratio'] > 0.3:
        strengths.append(f"✓ Strong affordability surplus (€{metrics['monthly_surplus']:,.0f}/mo)")
    
    if not metrics['stress_compliant']:
        weaknesses.append(f"⚠️ Stress test fails: {metrics['stress_burden_pct']:.1f}% > {metrics['stress_limit']*100:.0f}% CBI limit")
    
    if metrics['ltv_ratio'] and metrics['ltv_ratio'] > 0.80:
        weaknesses.append(f"⚠️ High LTV ({metrics['ltv_ratio']:.0%}) - low equity")
    
    if repayment_type != 'pni':
        weaknesses.append(f"⚠️ Non-standard repayment: {metrics['repayment_note']}")
    
    if prob_default > 0.3:
        weaknesses.append(f"⚠️ Elevated default probability ({prob_default:.1%})")
    
    if term_years > 30:
        weaknesses.append(f"⚠️ Long loan term ({term_years} years) - more interest, slower equity")
    
    # Overall default risk wording
    if prob_default < 0.05:
        default_risk_text = "Overall default risk is very low"
    elif prob_default < 0.2:
        default_risk_text = "Overall default risk is low"
    elif prob_default < 0.4:
        default_risk_text = "Overall default risk is moderate"
    else:
        default_risk_text = "Overall default risk is high"
    
    # DECISION LOGIC
    if not metrics['stress_compliant']:
        if prob_default < 0.25:
            decision = "⚠️ CONDITIONAL APPROVAL"
            decision_color = "warning"
            decision_icon = "⚠️"
            summary_text = "Loan meets primary criteria but FAILS stress test. Approval subject to conditions."
            recommendation_text = "Approve with conditions: reduce loan amount, extend term, or require larger deposit"
        else:
            decision = "⚠️ REFER TO UNDERWRITER"
            decision_color = "warning"
            decision_icon = "📋"
            summary_text = "Fails CBI stress test requirement. Manual review required before approval."
            recommendation_text = "Refer for manual underwriting review with focus on stress test mitigation"
    elif prob_default < 0.25 and metrics['lti_compliant']:
        decision = "✅ APPROVED"
        decision_color = "success"
        decision_icon = "✅"
        summary_text = f"Loan meets all primary underwriting criteria. {default_risk_text}."
        recommendation_text = "Approve with standard terms"
    elif prob_default < 0.5:
        decision = "⚠️ CONDITIONAL APPROVAL"
        decision_color = "warning"
        decision_icon = "⚠️"
        summary_text = "Loan has some risk factors that need mitigation."
        recommendation_text = "Approve with conditions: review risk factors"
    else:
        decision = "❌ DECLINED"
        decision_color = "danger"
        decision_icon = "❌"
        summary_text = "Loan does not meet risk appetite criteria."
        recommendation_text = "Decline or require significant compensating factors"
    
    summary_html = html.Div([
        html.H4(f"{decision_icon} Final Decision: {decision}", className=f"text-{decision_color} text-center"),
        html.P(summary_text, className="text-center mt-2"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H6("✅ Strengths", className="text-success"),
                html.Ul([html.Li(s) for s in strengths] if strengths else [html.Li("No significant strengths identified")])
            ], width=6),
            dbc.Col([
                html.H6("⚠️ Risk Factors", className="text-warning"),
                html.Ul([html.Li(w) for w in weaknesses] if weaknesses else [html.Li("No major risk factors identified")])
            ], width=6),
        ]),
        html.Hr(),
        html.Div([
            html.P("📌 Recommendation:", className="fw-bold mb-1"),
            html.P(recommendation_text, className="text-muted small")
        ]),
        html.Div([
            html.P("💡 Affordability:", className="fw-bold mb-1 mt-2"),
            html.P(f"Monthly surplus after all payments: €{metrics['monthly_surplus']:,.0f} ({metrics['surplus_ratio']:.0%} of income)", 
                   className="text-success small")
        ]) if metrics['surplus_ratio'] > 0.2 else html.Div(),
        html.Div([
            html.P("📋 CBI Compliance Notes:", className="fw-bold mb-1 mt-2"),
            html.Ul([
                html.Li(f"LTI Ratio: {metrics['lti_ratio']:.2f} (Limit: 4.0) - {'✓ Compliant' if metrics['lti_compliant'] else '✗ Non-Compliant'}"),
                html.Li(f"Stress Test: {metrics['stress_burden_pct']:.1f}% of income (Limit: 35%) - {'✓ Pass' if metrics['stress_compliant'] else '✗ Fail'}"),
                html.Li(f"Loan Term: {term_years} years - {'✓ Standard' if term_years <= 30 else '⚠️ Extended'}"),
            ], className="small text-muted")
        ])
    ])
    
    # Risk factors chart
    fig = go.Figure()
    risk_factors = [
        ('DTI', min(metrics['dti'] / 50, 1)),
        ('LTI', min(metrics['lti_ratio'] / 5, 1)),
        ('Payment Burden', min(metrics['payment_burden'] / 0.35, 1)),
        ('Employment', max(0, 1 - emp_length / 20)),
        ('Loan Term', min((term_years - 15) / 25, 1) if term_years > 15 else 0),
    ]
    
    if not metrics['stress_compliant']:
        risk_factors.append(('Stress Test', min(metrics['stress_burden'] / 0.35, 1)))
    
    if repayment_type != 'pni':
        risk_factors.append(('Repayment Risk', 0.6 if repayment_type == 'interest_only' else 0.4))
    
    # Filter out zero risk factors
    risk_factors = [(n, v) for n, v in risk_factors if v > 0.05]
    
    if risk_factors:
        names, values = zip(*risk_factors)
        colors = ['#dc3545' if v > 0.6 else '#ffc107' if v > 0.4 else '#28a745' for v in values]
        fig.add_trace(go.Bar(x=values, y=names, orientation='h', marker_color=colors, 
                             text=[f'{v:.0%}' for v in values], textposition='auto'))
    else:
        fig.add_annotation(text="No significant risk factors detected", x=0.5, y=0.5, showarrow=False)
    
    fig.update_layout(height=300, xaxis_title="Risk Contribution (higher = worse)", xaxis=dict(range=[0, 1]), 
                      margin=dict(l=0, r=0, t=30, b=0))
    
    return (risk_html, f"{metrics['dti']:.0f}%", f"{metrics['lti_ratio']:.2f}", 
            f"{metrics['total_burden']:.0%}", grade_html, summary_html, fig)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8051))
    app.run(debug=False, host='0.0.0.0', port=port)