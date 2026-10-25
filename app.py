"""
Loan Approval Predictor - Interactive Web Application
Built with Streamlit, Scikit-Learn, and Banking Feature Engineering
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from feature_engineering import FinancialFeatureEngineer, train_model

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- MODERN THEME & CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Header with Dual-Tone Glow Gradient */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
        font-weight: 400;
        line-height: 1.4;
    }

    /* Form Container */
    div[data-testid="stForm"] {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.35);
        padding: 22px 24px;
    }

    /* Primary Submit Button - Royal Blue Gradient */
    button[kind="primaryFormSubmit"],
    button[data-testid="stBaseButton-primaryFormSubmit"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    button[kind="primaryFormSubmit"]:hover,
    button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.55) !important;
        transform: translateY(-1px) !important;
    }

    /* Status Cards - Translucent Glassmorphism with Sharp Neon Borders */
    .status-card-approved {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.22) 100%);
        border: 1.5px solid #10B981;
        border-radius: 14px;
        padding: 24px 20px;
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    .status-card-rejected {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.22) 100%);
        border: 1.5px solid #EF4444;
        border-radius: 14px;
        padding: 24px 20px;
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
    }
    .status-title {
        font-size: 1.75rem;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }
    .approved-text {
        color: #34D399 !important;
    }
    .rejected-text {
        color: #F87171 !important;
    }
    .status-desc {
        font-size: 1.05rem;
        color: #F1F5F9 !important;
        margin: 0;
        line-height: 1.5;
    }

    /* Metric Badges - High Contrast Theme Safe */
    .metric-badge {
        background: #1E293B;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-left: 4px solid #3B82F6;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    .metric-badge-title {
        color: #F8FAFC !important;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .metric-badge-sub {
        color: #94A3B8 !important;
        display: block;
        margin-top: 5px;
        font-size: 0.85rem;
        line-height: 1.4;
    }

    /* Metric Numbers */
    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-weight: 500 !important;
    }

    /* Hide Sidebar & Toggle Control */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# --- MODEL LOADING ---
@st.cache_resource
def get_model():
    model_path = os.path.join(os.path.dirname(__file__), "loan_approval_model.pkl")
    csv_path = os.path.join(os.path.dirname(__file__), "loan_prediction.csv")

    sample_test = pd.DataFrame([{
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": 0.0,
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban",
    }])

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            # Verify the model can execute transform and predict without cross-version issues
            model.predict(sample_test)
            return model
        except Exception:
            pass

    # Fall back to training fresh on this environment
    fresh_model = train_model(csv_path)
    try:
        joblib.dump(fresh_model, model_path)
    except Exception:
        pass
    return fresh_model


pipeline = get_model()

# --- MAIN HEADER ---
st.markdown('<div class="main-header">💳 Loan Approval Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated credit evaluation powered by Random Forest & Banking Affordability Metrics</div>', unsafe_allow_html=True)


# --- INTERACTIVE PREDICTOR FORM ---
with st.form("loan_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 👤 Personal & Employment Profile")
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        married = st.selectbox("Marital Status", ["Yes", "No"], index=0)
        dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3], index=0)
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"], index=0)
        self_employed = st.selectbox("Self Employed?", ["No", "Yes"], index=0)

    with col2:
        st.markdown("##### 💵 Financial Details")
        app_income = st.number_input(
            "Applicant Monthly Income ($)",
            min_value=0, max_value=100000,
            value=5000, step=250,
            help="Primary applicant's monthly earnings"
        )
        coapp_income = st.number_input(
            "Co-applicant Monthly Income ($)",
            min_value=0, max_value=50000,
            value=1500, step=250,
            help="Spouse or guarantor monthly earnings (0 if none)"
        )
        loan_amt = st.number_input(
            "Loan Amount (in Thousands, e.g. 120 = $120,000)",
            min_value=10, max_value=1000,
            value=120, step=5,
            help="Requested loan amount in thousands of dollars"
        )

        tot_inc = app_income + coapp_income
        st.info(f"**Total Household Income:** ${tot_inc:,.0f} / month")

    with col3:
        st.markdown("##### 📜 Loan & Credit Terms")
        term = st.selectbox(
            "Loan Term (in Months)",
            [360, 180, 120, 240, 300, 480, 84, 60],
            index=0,
            help="360 months = 30 years, 180 months = 15 years"
        )
        credit = st.selectbox(
            "Credit History Status",
            ["Good (Meets guidelines - Score 1.0)", "Poor / Defaulted (Score 0.0)"],
            index=0,
            help="Historical credit repayment track record"
        )
        property_area = st.selectbox(
            "Property Area", ["Urban", "Semiurban", "Rural"],
            index=0
        )

    submit_btn = st.form_submit_button("🔍 Run Loan Approval Prediction", use_container_width=True, type="primary")

if submit_btn:
    credit_val = 1.0 if "1.0" in credit else 0.0

    # Calculate banking metrics
    total_income = float(app_income) + float(coapp_income)
    denom_income = max(total_income, 1.0)
    term_guarded = max(float(term), 1.0)
    emi_val = (float(loan_amt) * 1000.0) / term_guarded
    lti_val = (float(loan_amt) * 1000.0) / (denom_income * 12.0)
    dti_val = emi_val / denom_income

    input_data = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': float(dependents),
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': float(app_income),
        'CoapplicantIncome': float(coapp_income),
        'LoanAmount': float(loan_amt),
        'Loan_Amount_Term': float(term),
        'Credit_History': float(credit_val),
        'Property_Area': property_area
    }])

    pred = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0]
    prob_approved = prob[1]
    prob_rejected = prob[0]

    st.markdown("---")
    st.markdown("### 📋 Prediction Outcome & Affordability Breakdown")

    res_col1, res_col2 = st.columns([1.3, 1.1])

    with res_col1:
        if pred == 1:
            st.markdown(f"""
            <div class="status-card-approved">
                <div class="status-title approved-text">🎉 LOAN APPROVED</div>
                <p class="status-desc">
                    Applicant meets credit and financial eligibility criteria with an approval confidence of <b>{prob_approved:.1%}</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-card-rejected">
                <div class="status-title rejected-text">⚠️ LOAN REJECTED</div>
                <p class="status-desc">
                    Application exhibits high default or leverage risk with a rejection probability of <b>{prob_rejected:.1%}</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)

        score_color = "#34D399" if pred == 1 else "#F87171"
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 0.95rem; font-weight: 600; color: #94A3B8;">Approval Probability Score:</span>
            <span style="font-size: 1.25rem; font-weight: 800; color: {score_color};">{prob_approved * 100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(prob_approved))

    with res_col2:
        st.markdown("#### 📐 Key Banking Affordability Metrics")

        # EMI Metric Card
        st.markdown(f"""
        <div class="metric-badge">
            <span class="metric-badge-title">Monthly EMI:</span> 
            <span style="font-size: 1.15rem; color: #38BDF8; font-weight: 700;">${emi_val:,.2f}</span> / mo<br>
            <span class="metric-badge-sub">Estimated monthly repayment obligation for ${loan_amt*1000:,.0f} over {term} months.</span>
        </div>
        """, unsafe_allow_html=True)

        # DTI Metric Card
        dti_status = "Healthy (Below 40%)" if dti_val <= 0.40 else "High Debt Burden (> 40%)"
        dti_color = "#34D399" if dti_val <= 0.40 else "#F87171"
        st.markdown(f"""
        <div class="metric-badge" style="border-left-color: {dti_color};">
            <span class="metric-badge-title">DTI (Debt-to-Income):</span> 
            <span style="font-size: 1.15rem; color: {dti_color}; font-weight: 700;">{dti_val:.1%}</span> — 
            <span style="color: {dti_color}; font-weight: 600;">{dti_status}</span><br>
            <span class="metric-badge-sub">Proportion of household monthly income consumed by this loan installment.</span>
        </div>
        """, unsafe_allow_html=True)

        # LTI Metric Card
        lti_status = "Manageable Leverage (≤ 3.0x)" if lti_val <= 3.0 else "High Leverage (> 3.0x)"
        lti_color = "#34D399" if lti_val <= 3.0 else "#FBBF24"
        st.markdown(f"""
        <div class="metric-badge" style="border-left-color: {lti_color};">
            <span class="metric-badge-title">LTI (Loan-to-Annual-Income):</span> 
            <span style="font-size: 1.15rem; color: {lti_color}; font-weight: 700;">{lti_val:.2f}x</span> — 
            <span style="color: {lti_color}; font-weight: 600;">{lti_status}</span><br>
            <span class="metric-badge-sub">Total loan size relative to annual household income (${total_income*12:,.0f}/yr).</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📊 Summary of Evaluated Parameters")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Household Monthly Income", f"${total_income:,.0f}")
    m2.metric("Requested Loan Amount", f"${loan_amt * 1000:,.0f}")
    m3.metric("Tenure Length", f"{int(term / 12)} Yrs ({term}m)")
    m4.metric("Credit History Status", "Verified (1.0)" if credit_val == 1.0 else "Deficient (0.0)")

# Footer
st.markdown("---")
st.caption("ML Lab Final Project • 5th Semester • Automated Credit Evaluation Pipeline with Streamlit & Scikit-Learn")
