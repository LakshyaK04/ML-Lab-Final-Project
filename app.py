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
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-card-approved {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-bottom: 16px;
    }
    .status-card-rejected {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-bottom: 16px;
    }
    .status-title {
        font-size: 1.7rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .approved-text { color: #065F46; }
    .rejected-text { color: #991B1B; }
    .metric-badge {
        background: #F3F4F6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid #3B82F6;
    }
    .viva-box {
        background-color: #EFF6FF;
        border-left: 5px solid #2563EB;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.98rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# --- MODEL LOADING & FALLBACK ---
@st.cache_resource
def get_model():
    model_path = os.path.join(os.path.dirname(__file__), "loan_approval_model.pkl")
    csv_path = os.path.join(os.path.dirname(__file__), "loan_prediction.csv")

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            # Smoke test
            dummy = pd.DataFrame([{
                'Gender': 'Male', 'Married': 'Yes', 'Dependents': 0.0,
                'Education': 'Graduate', 'Self_Employed': 'No',
                'ApplicantIncome': 5000.0, 'CoapplicantIncome': 1500.0,
                'LoanAmount': 120.0, 'Loan_Amount_Term': 360.0,
                'Credit_History': 1.0, 'Property_Area': 'Urban'
            }])
            model.predict(dummy)
            return model
        except Exception:
            pass

    # Fallback: train fresh pipeline directly from CSV
    fresh_model = train_model(csv_path)
    try:
        joblib.dump(fresh_model, model_path)
    except Exception:
        pass
    return fresh_model


@st.cache_data
def get_comparison():
    comp_path = os.path.join(os.path.dirname(__file__), "model_comparison.csv")
    if os.path.exists(comp_path):
        return pd.read_csv(comp_path)
    return None


pipeline = get_model()
comparison_df = get_comparison()

# --- DEFAULT SESSION STATE ---
if "app_income" not in st.session_state:
    st.session_state.gender = "Male"
    st.session_state.married = "Yes"
    st.session_state.dependents = 0
    st.session_state.education = "Graduate"
    st.session_state.self_employed = "No"
    st.session_state.app_income = 5000
    st.session_state.coapp_income = 1500
    st.session_state.loan_amt = 120
    st.session_state.term = 360
    st.session_state.credit = "Good (Meets guidelines - Score 1.0)"
    st.session_state.property_area = "Urban"

# --- SIDEBAR: QUICK TEST PRESETS ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank-building.png", width=65)
    st.markdown("### 🏦 **Quick Test Presets**")
    st.caption("Click any preset to verify realistic banking behavior:")

    if st.button("✅ 1. Normal Good Applicant", use_container_width=True):
        st.session_state.gender = "Male"
        st.session_state.married = "Yes"
        st.session_state.dependents = 0
        st.session_state.education = "Graduate"
        st.session_state.self_employed = "No"
        st.session_state.app_income = 5000
        st.session_state.coapp_income = 1500
        st.session_state.loan_amt = 120
        st.session_state.term = 360
        st.session_state.credit = "Good (Meets guidelines - Score 1.0)"
        st.session_state.property_area = "Urban"
        st.rerun()

    if st.button("⚠️ 2. Stress Test: Low Income, Huge Loan ($900k)", use_container_width=True):
        st.session_state.gender = "Male"
        st.session_state.married = "Yes"
        st.session_state.dependents = 0
        st.session_state.education = "Graduate"
        st.session_state.self_employed = "No"
        st.session_state.app_income = 500
        st.session_state.coapp_income = 0
        st.session_state.loan_amt = 900
        st.session_state.term = 360
        st.session_state.credit = "Good (Meets guidelines - Score 1.0)"
        st.session_state.property_area = "Urban"
        st.rerun()

    if st.button("❌ 3. Zero Income, Large Loan ($500k)", use_container_width=True):
        st.session_state.gender = "Male"
        st.session_state.married = "No"
        st.session_state.dependents = 0
        st.session_state.education = "Graduate"
        st.session_state.self_employed = "No"
        st.session_state.app_income = 150
        st.session_state.coapp_income = 0
        st.session_state.loan_amt = 500
        st.session_state.term = 360
        st.session_state.credit = "Good (Meets guidelines - Score 1.0)"
        st.session_state.property_area = "Urban"
        st.rerun()

    if st.button("💼 4. High Earner, Bad Credit ($50k/mo)", use_container_width=True):
        st.session_state.gender = "Female"
        st.session_state.married = "Yes"
        st.session_state.dependents = 1
        st.session_state.education = "Graduate"
        st.session_state.self_employed = "Yes"
        st.session_state.app_income = 50000
        st.session_state.coapp_income = 0
        st.session_state.loan_amt = 150
        st.session_state.term = 360
        st.session_state.credit = "Poor / Defaulted (Score 0.0)"
        st.session_state.property_area = "Semiurban"
        st.rerun()

    st.markdown("---")
    st.caption("**Model Architecture:** Random Forest (300 trees, max depth 5) with Pipeline Feature Engineering (Total Income, EMI, DTI, LTI).")


# --- MAIN HEADER ---
st.markdown('<div class="main-header">💳 Loan Approval Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated credit evaluation powered by Random Forest & Banking Affordability Metrics</div>', unsafe_allow_html=True)

tab_pred, tab_perf, tab_faq = st.tabs([
    "🔮 Interactive Predictor",
    "🏆 Model Performance & Architecture",
    "🎓 Viva Guide & FAQs"
])


# ==========================================
# TAB 1: INTERACTIVE PREDICTOR
# ==========================================
with tab_pred:
    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Personal & Employment Profile")
            gender = st.selectbox(
                "Gender", ["Male", "Female"],
                index=["Male", "Female"].index(st.session_state.gender)
            )
            married = st.selectbox(
                "Marital Status", ["Yes", "No"],
                index=["Yes", "No"].index(st.session_state.married)
            )
            dependents = st.selectbox(
                "Number of Dependents", [0, 1, 2, 3],
                index=[0, 1, 2, 3].index(st.session_state.dependents)
            )
            education = st.selectbox(
                "Education Level", ["Graduate", "Not Graduate"],
                index=["Graduate", "Not Graduate"].index(st.session_state.education)
            )
            self_employed = st.selectbox(
                "Self Employed?", ["No", "Yes"],
                index=["No", "Yes"].index(st.session_state.self_employed)
            )

        with col2:
            st.markdown("##### 💵 Financial Details")
            app_income = st.number_input(
                "Applicant Monthly Income ($)",
                min_value=0, max_value=100000,
                value=int(st.session_state.app_income), step=250,
                help="Primary applicant's monthly earnings"
            )
            coapp_income = st.number_input(
                "Co-applicant Monthly Income ($)",
                min_value=0, max_value=50000,
                value=int(st.session_state.coapp_income), step=250,
                help="Spouse or guarantor monthly earnings (0 if none)"
            )
            loan_amt = st.number_input(
                "Loan Amount (in Thousands, e.g. 120 = $120,000)",
                min_value=10, max_value=1000,
                value=int(st.session_state.loan_amt), step=5,
                help="Requested loan amount in thousands of dollars"
            )

            tot_inc = app_income + coapp_income
            st.info(f"**Total Household Income:** ${tot_inc:,.0f} / month")

        with col3:
            st.markdown("##### 📜 Loan & Credit Terms")
            term = st.selectbox(
                "Loan Term (in Months)",
                [360, 180, 120, 240, 300, 480, 84, 60],
                index=[360, 180, 120, 240, 300, 480, 84, 60].index(st.session_state.term),
                help="360 months = 30 years, 180 months = 15 years"
            )
            credit = st.selectbox(
                "Credit History Status",
                ["Good (Meets guidelines - Score 1.0)", "Poor / Defaulted (Score 0.0)"],
                index=["Good (Meets guidelines - Score 1.0)", "Poor / Defaulted (Score 0.0)"].index(st.session_state.credit),
                help="Historical credit repayment track record"
            )
            property_area = st.selectbox(
                "Property Area", ["Urban", "Semiurban", "Rural"],
                index=["Urban", "Semiurban", "Rural"].index(st.session_state.property_area)
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
                    <p style="font-size: 1.15rem; color: #047857; margin-bottom: 4px;">
                        Applicant meets credit and financial eligibility criteria with an approval confidence of <b>{prob_approved:.1%}</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-card-rejected">
                    <div class="status-title rejected-text">⚠️ LOAN REJECTED</div>
                    <p style="font-size: 1.15rem; color: #B91C1C; margin-bottom: 4px;">
                        Application exhibits high default or leverage risk with a rejection probability of <b>{prob_rejected:.1%}</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.write(f"**Approval Probability Score:** `{prob_approved * 100:.1f}%`")
            st.progress(float(prob_approved))

        with res_col2:
            st.markdown("#### 📐 Key Banking Affordability Metrics")

            # EMI Metric Card
            st.markdown(f"""
            <div class="metric-badge">
                <b>Monthly EMI:</b> ${emi_val:,.2f}/month<br>
                <small style="color: #6B7280;">Estimated monthly repayment obligation for ${loan_amt*1000:,.0f} over {term} months.</small>
            </div>
            """, unsafe_allow_html=True)

            # DTI Metric Card
            dti_status = "Healthy (Below 40%)" if dti_val <= 0.40 else "High Debt Burden (> 40%)"
            dti_color = "#10B981" if dti_val <= 0.40 else "#EF4444"
            st.markdown(f"""
            <div class="metric-badge" style="border-left-color: {dti_color};">
                <b>DTI (Debt-to-Income):</b> {dti_val:.1%} — <span style="color: {dti_color}; font-weight: 600;">{dti_status}</span><br>
                <small style="color: #6B7280;">Proportion of household monthly income consumed by this loan installment.</small>
            </div>
            """, unsafe_allow_html=True)

            # LTI Metric Card
            lti_status = "Manageable Leverage (≤ 3.0x)" if lti_val <= 3.0 else "High Leverage (> 3.0x)"
            lti_color = "#10B981" if lti_val <= 3.0 else "#F59E0B"
            st.markdown(f"""
            <div class="metric-badge" style="border-left-color: {lti_color};">
                <b>LTI (Loan-to-Annual-Income):</b> {lti_val:.2f}x — <span style="color: {lti_color}; font-weight: 600;">{lti_status}</span><br>
                <small style="color: #6B7280;">Total loan size relative to annual household income (${total_income*12:,.0f}/yr).</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 📊 Summary of Evaluated Parameters")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Household Monthly Income", f"${total_income:,.0f}")
        m2.metric("Requested Loan Amount", f"${loan_amt * 1000:,.0f}")
        m3.metric("Tenure Length", f"{term / 12:.1f} years ({term} mos)")
        m4.metric("Credit History Status", "Verified (1.0)" if credit_val == 1.0 else "Deficient (0.0)")


# ==========================================
# TAB 2: MODEL PERFORMANCE & ARCHITECTURE
# ==========================================
with tab_perf:
    st.markdown("### 🏆 Machine Learning Model Evaluation & Cross-Validation")
    st.write("""
    Models were benchmarked using **Repeated Stratified 5-Fold Cross-Validation (5 repeats = 25 evaluation splits)**
    to accurately capture mean generalization accuracy and variance:
    """)

    if comparison_df is not None:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.dataframe(
                comparison_df.style.format({
                    "Mean CV Accuracy": "{:.2%}",
                    "CV Accuracy Std": "±{:.2%}",
                    "Mean CV F1": "{:.4f}",
                    "CV F1 Std": "±{:.4f}"
                }).highlight_max(axis=0, subset=["Mean CV Accuracy", "Mean CV F1"], color="#D1FAE5"),
                use_container_width=True
            )
        with c2:
            st.markdown("""
            **Honest Evaluation Summary:**
            - **Accuracy is statistically comparable (~80%):** Linear SVM (~80.9% ± 2.3%) and Random Forest with Engineered Features (~79.6% ± 2.1%) perform equally well within noise margins.
            - **Why Random Forest was chosen:** Linear SVM relied almost exclusively on `Credit_History`, approving absurd applications (e.g. $0 income, $500k loan). Random Forest balances credit track record with economic affordability (EMI, DTI, LTI).
            - **Known Tradeoff:** The Random Forest model yields a higher approval probability for bad-credit applicants (~25%) compared to Linear SVM (~9%), even though only ~8% of bad-credit applicants were approved in the training set.
            """)

    st.markdown("---")
    st.markdown("### ⚙️ End-to-End Pipeline Architecture")
    st.code("""
Raw Applicant Data (loan_prediction.csv: 614 applications)
   │
   ├─► Step 1: Banking Feature Engineering (FinancialFeatureEngineer)
   │     ├─► TotalIncome = ApplicantIncome + CoapplicantIncome
   │     ├─► EMI = (LoanAmount * 1000) / Loan_Amount_Term
   │     ├─► LTI = (LoanAmount * 1000) / (TotalIncome * 12)
   │     └─► DTI = EMI / TotalIncome
   │
   ├─► Step 2: ColumnTransformer Preprocessing
   │     ├─► Numerical Columns: Median Imputation + StandardScaler
   │     └─► Categorical Columns: Mode Imputation + OneHotEncoder
   │
   └─► Step 3: Classifier (RandomForestClassifier)
         ├─► n_estimators = 300
         ├─► max_depth = 5
         ├─► min_samples_leaf = 5
         ├─► class_weight = 'balanced'
         └─► random_state = 42
    """, language="text")

    st.markdown("---")
    st.markdown("### 📊 Final Model Feature Importances")
    st.write("Notice how feature engineering distributes decision weight across real financial capacity rather than solely Credit History:")

    feat_data = {
        "Feature": [
            "Credit_History", "LTI (Loan-to-Income)", "TotalIncome", "DTI (Debt-to-Income)",
            "EMI (Monthly Payment)", "ApplicantIncome", "LoanAmount", "CoapplicantIncome",
            "Property_Area_Semiurban", "Property_Area_Rural", "Married", "Loan_Amount_Term", "Dependents"
        ],
        "Importance": [43.02, 9.23, 7.85, 6.98, 5.64, 5.43, 5.27, 3.68, 2.93, 1.59, 2.65, 1.30, 1.21]
    }
    feat_df = pd.DataFrame(feat_data)
    st.bar_chart(feat_df.set_index("Feature"))


# ==========================================
# TAB 3: VIVA GUIDE & FAQS
# ==========================================
with tab_faq:
    st.markdown("### 🎓 Viva Voce Preparation Cheatsheet")
    st.caption("Concise, direct, and technically accurate answers for your lab examiners:")

    faqs = [
        (
            "1. What is the objective and problem formulation of this project?",
            "It is a Supervised Binary Classification task. The objective is to predict whether a loan application should be Approved (1) or Rejected (0) based on financial attributes, loan terms, and credit history."
        ),
        (
            "2. Why did you add engineered features (TotalIncome, EMI, LTI, DTI)?",
            "Raw dollar figures in isolation do not measure debt affordability. In real banking, lenders evaluate repayment capacity using ratios: Total Household Income (joint capacity), EMI (monthly obligation), DTI (monthly debt burden), and LTI (leverage). Engineering these inside the pipeline enables the model to evaluate affordability realistically."
        ),
        (
            "3. Why switch from Linear SVM to Random Forest?",
            "Linear SVM fitted a hyperplane where Credit_History received virtually all decision weight (+0.6865) while income and loan amounts were near zero. This caused severe failure in stress tests: an applicant with $0 income requesting $500,000 was approved simply because they checked good credit! Random Forest captures non-linear feature interactions between debt ratios and credit history, rejecting unaffordable loans."
        ),
        (
            "4. Did accuracy improve after adding engineered features?",
            "No. Across repeated 5-fold cross-validation (25 folds), accuracy remained around 80% (Linear SVM ~80.9% ± 2.3% vs Random Forest with engineered features ~79.6% ± 2.1%). The purpose of feature engineering was not an artificial accuracy increase on historical data, but ensuring economically sound, stress-tested predictions on real-world edge cases."
        ),
        (
            "5. What are the key limitations of the model and dataset?",
            "First, the dataset is relatively small (614 rows). Second, the new model exhibits a tradeoff: it assigns ~25% approval probability to bad-credit applicants with high income, even though historical data only approved ~8% of bad-credit applicants. Third, real banking requires external bureau data, bank statement analysis, and collateral verification not captured here."
        ),
        (
            "6. What is the true definition and role of the F1-Score here?",
            "F1-Score is the harmonic mean of Precision and Recall specifically for the positive class (Approved). It does NOT use True Negatives (correct rejections). It measures how effectively the model identifies approved loans without excessive false alarms."
        ),
        (
            "7. Why use class_weight='balanced'?",
            "The dataset has class imbalance: approximately 69% approved vs 31% rejected. Without balanced weighting, models can become biased toward approving everyone. Setting class_weight='balanced' automatically inversely weights class frequencies."
        ),
        (
            "8. Why use an end-to-end scikit-learn Pipeline?",
            "A Pipeline bundles custom feature extraction, imputation, scaling, one-hot encoding, and the model into a single atomic object. This strictly prevents Data Leakage between train and test splits during cross-validation and guarantees identical transformations in production."
        ),
    ]

    for q, a in faqs:
        with st.expander(f"📌 {q}", expanded=False):
            st.markdown(f'<div class="viva-box">{a}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("ML Lab Final Project • 5th Semester • Automated Credit Evaluation Pipeline with Streamlit & Scikit-Learn")
