"""
Loan Approval Predictor - Interactive Web Application
Built with Streamlit and Scikit-Learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR CLEAN MODERN LOOK ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
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
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .status-card-rejected {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .status-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .approved-text { color: #065F46; }
    .rejected-text { color: #991B1B; }
    .viva-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 15px;
        border-radius: 6px;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)

def train_fresh_model():
    """Train pipeline in ~0.05s to guarantee 100% compatibility with any scikit-learn version"""
    csv_path = os.path.join(os.path.dirname(__file__), "loan_prediction.csv")
    df = pd.read_csv(csv_path)
    df_clean = df.copy()
    df_clean.drop(columns=["Loan_ID"], inplace=True, errors="ignore")
    df_clean["Dependents"] = df_clean["Dependents"].replace("3+", "3")
    df_clean["Dependents"] = pd.to_numeric(df_clean["Dependents"], errors="coerce")
    df_clean["Loan_Status"] = (
        df_clean["Loan_Status"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"Y": 1, "N": 0})
    )
    df_clean.dropna(subset=["Loan_Status"], inplace=True)
    df_clean["Loan_Status"] = df_clean["Loan_Status"].astype(int)

    x = df_clean.drop(columns=["Loan_Status"])
    y = df_clean["Loan_Status"]

    numeric_features = [
        "Dependents", "ApplicantIncome", "CoapplicantIncome",
        "LoanAmount", "Loan_Amount_Term", "Credit_History"
    ]
    categorical_features = [
        "Gender", "Married", "Education", "Self_Employed", "Property_Area"
    ]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    new_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", SVC(C=1, class_weight="balanced", kernel="linear", probability=True, random_state=42))
    ])

    new_pipeline.fit(x, y)
    return new_pipeline

# --- LOAD MODEL & DATA ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "loan_approval_model.pkl")
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            # Verify compatibility by testing a dummy prediction
            test_df = pd.DataFrame([{
                'Gender': 'Male', 'Married': 'Yes', 'Dependents': 0.0,
                'Education': 'Graduate', 'Self_Employed': 'No',
                'ApplicantIncome': 5000.0, 'CoapplicantIncome': 1500.0,
                'LoanAmount': 120.0, 'Loan_Amount_Term': 360.0,
                'Credit_History': 1.0, 'Property_Area': 'Urban'
            }])
            model.predict(test_df)
            return model
        except Exception:
            # Cross-version pickle mismatch on cloud: train fresh in 0.05s
            return train_fresh_model()
    return train_fresh_model()

@st.cache_data
def load_comparison():
    comp_path = os.path.join(os.path.dirname(__file__), "model_comparison.csv")
    if os.path.exists(comp_path):
        return pd.read_csv(comp_path)
    return None

pipeline = load_model()
comparison_df = load_comparison()

# --- SIDEBAR: PRESETS & QUICK INFO ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank-building.png", width=70)
    st.markdown("### 🏦 **Quick Test Presets**")
    st.caption("Click any preset to fill the form with sample viva cases:")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("✅ Standard Approved", use_container_width=True):
            st.session_state.gender = "Male"
            st.session_state.married = "Yes"
            st.session_state.dependents = 0
            st.session_state.education = "Graduate"
            st.session_state.self_employed = "No"
            st.session_state.app_income = 5500
            st.session_state.coapp_income = 2000
            st.session_state.loan_amt = 130
            st.session_state.term = 360
            st.session_state.credit = "Good (Meets guidelines - Score 1.0)"
            st.session_state.property_area = "Urban"
            st.rerun()

    with col_p2:
        if st.button("❌ Defaulter Case", use_container_width=True):
            st.session_state.gender = "Male"
            st.session_state.married = "No"
            st.session_state.dependents = 0
            st.session_state.education = "Not Graduate"
            st.session_state.self_employed = "No"
            st.session_state.app_income = 2500
            st.session_state.coapp_income = 0
            st.session_state.loan_amt = 150
            st.session_state.term = 360
            st.session_state.credit = "Poor / Defaulted (Score 0.0)"
            st.session_state.property_area = "Rural"
            st.rerun()

    if st.button("⚖️ Self-Employed Semiurban", use_container_width=True):
        st.session_state.gender = "Female"
        st.session_state.married = "Yes"
        st.session_state.dependents = 1
        st.session_state.education = "Graduate"
        st.session_state.self_employed = "Yes"
        st.session_state.app_income = 4800
        st.session_state.coapp_income = 3200
        st.session_state.loan_amt = 180
        st.session_state.term = 360
        st.session_state.credit = "Good (Meets guidelines - Score 1.0)"
        st.session_state.property_area = "Semiurban"
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎓 **Viva Cheatsheet**")
    st.markdown("""
    - **Model:** Support Vector Classifier (Linear)
    - **Cross-Validation Accuracy:** ~79.8%
    - **Key Feature:** `Credit_History`
    - **Target:** `Loan_Status` (1 = Yes, 0 = No)
    - **Handling Imbalance:** `class_weight='balanced'`
    """)

# --- INITIALIZE SESSION STATE DEFAULTS ---
if "gender" not in st.session_state:
    st.session_state.gender = "Male"
if "married" not in st.session_state:
    st.session_state.married = "Yes"
if "dependents" not in st.session_state:
    st.session_state.dependents = 0
if "education" not in st.session_state:
    st.session_state.education = "Graduate"
if "self_employed" not in st.session_state:
    st.session_state.self_employed = "No"
if "app_income" not in st.session_state:
    st.session_state.app_income = 5000
if "coapp_income" not in st.session_state:
    st.session_state.coapp_income = 1500
if "loan_amt" not in st.session_state:
    st.session_state.loan_amt = 120
if "term" not in st.session_state:
    st.session_state.term = 360
if "credit" not in st.session_state:
    st.session_state.credit = "Good (Meets guidelines - Score 1.0)"
if "property_area" not in st.session_state:
    st.session_state.property_area = "Urban"

# --- MAIN PAGE HEADER ---
st.markdown('<div class="main-header">💳 Loan Approval Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning classification model built using Scikit-Learn Pipeline (Imputation, Scaling, One-Hot Encoding & Support Vector Machine).</div>', unsafe_allow_html=True)

# Tabs for Navigation
tab_pred, tab_perf, tab_faq = st.tabs(["🎯 Loan Application & Prediction", "📊 Model Comparison & Metrics", "💡 Viva Quick Prep"])

# ==========================================
# TAB 1: PREDICTION
# ==========================================
with tab_pred:
    st.markdown("### Enter Applicant & Loan Information")
    
    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 👤 Personal Details")
            gender = st.selectbox(
                "Gender", ["Male", "Female"],
                index=["Male", "Female"].index(st.session_state.gender)
            )
            married = st.selectbox(
                "Marital Status", ["Yes", "No"],
                index=["Yes", "No"].index(st.session_state.married)
            )
            dependents = st.selectbox(
                "Dependents (Children / Family members)", [0, 1, 2, 3],
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
                min_value=150, max_value=100000,
                value=int(st.session_state.app_income), step=250,
                help="Primary applicant's monthly earnings"
            )
            coapp_income = st.number_input(
                "Co-applicant Monthly Income ($)",
                min_value=0, max_value=50000,
                value=int(st.session_state.coapp_income), step=250,
                help="Spouse or guarantor's monthly earnings (0 if none)"
            )
            loan_amt = st.number_input(
                "Loan Amount requested (in Thousands, e.g. 120 = $120,000)",
                min_value=10, max_value=1000,
                value=int(st.session_state.loan_amt), step=5,
                help="Loan amount requested in thousands of dollars"
            )
            total_income = app_income + coapp_income
            st.info(f"**Total Household Income:** ${total_income:,.0f}/month")

        with col3:
            st.markdown("##### 📜 Loan & Credit Terms")
            term = st.selectbox(
                "Loan Term (in Months)",
                [360, 180, 120, 240, 300, 480, 84, 60],
                index=[360, 180, 120, 240, 300, 480, 84, 60].index(st.session_state.term),
                help="360 months = 30-year loan, 180 months = 15-year loan"
            )
            credit = st.selectbox(
                "Credit History Status",
                ["Good (Meets guidelines - Score 1.0)", "Poor / Defaulted (Score 0.0)"],
                index=["Good (Meets guidelines - Score 1.0)", "Poor / Defaulted (Score 0.0)"].index(st.session_state.credit),
                help="Most significant factor in loan prediction! Past credit repayment record."
            )
            property_area = st.selectbox(
                "Property Area", ["Urban", "Semiurban", "Rural"],
                index=["Urban", "Semiurban", "Rural"].index(st.session_state.property_area)
            )
        
        submit_btn = st.form_submit_button("🔍 Run Loan Approval Prediction", use_container_width=True, type="primary")

    if submit_btn:
        credit_val = 1.0 if "1.0" in credit else 0.0

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
        st.markdown("### 📋 Prediction Outcome")

        res_col1, res_col2 = st.columns([1.4, 1])

        with res_col1:
            if pred == 1:
                st.markdown(f"""
                <div class="status-card-approved">
                    <div class="status-title approved-text">🎉 LOAN APPROVED</div>
                    <p style="font-size: 1.15rem; color: #047857; margin-bottom: 4px;">
                        The applicant fulfills bank risk criteria with an estimated approval confidence of <b>{prob_approved:.1%}</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-card-rejected">
                    <div class="status-title rejected-text">⚠️ LOAN REJECTED</div>
                    <p style="font-size: 1.15rem; color: #B91C1C; margin-bottom: 4px;">
                        The loan application exhibits high default risk with a rejection probability of <b>{prob_rejected:.1%}</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.write(f"**Approval Confidence Score:** `{prob_approved * 100:.1f}%`")
            st.progress(float(prob_approved))

        with res_col2:
            st.markdown("#### 🔍 Key Deciding Factors")
            if credit_val == 1.0:
                st.success("✔️ **Credit History:** Verified (History = 1.0) — drastically elevates approval probability.")
            else:
                st.error("❌ **Credit History:** Deficient (History = 0.0) — primary reason for application denial.")
            
            loan_dollars = loan_amt * 1000
            total_inc_annual = (app_income + coapp_income) * 12
            ratio = (loan_dollars / total_inc_annual) if total_inc_annual > 0 else 999
            
            if ratio < 3.0:
                st.info(f"✔️ **Debt-to-Annual-Income:** {ratio:.2f}x (Healthy and manageable loan size)")
            else:
                st.warning(f"⚠️ **Debt-to-Annual-Income:** {ratio:.2f}x (Higher leverage ratio)")

        st.markdown("#### 📊 Summary of Evaluated Parameters")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Household Monthly Income", f"${total_income:,.0f}")
        m2.metric("Requested Loan Amount", f"${loan_amt * 1000:,.0f}")
        m3.metric("Tenure Length", f"{term / 12:.1f} years ({term} mos)")
        m4.metric("Property Zone", property_area)

# ==========================================
# TAB 2: MODEL PERFORMANCE & BENCHMARKING
# ==========================================
with tab_perf:
    st.markdown("### 🏆 Machine Learning Model Comparison")
    st.write("""
    During development, 4 different classification algorithms were trained using 
    **Stratified 5-Fold Cross-Validation** to prevent overfitting and handle class distribution:
    """)

    if comparison_df is not None:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.dataframe(
                comparison_df.style.highlight_max(axis=0, subset=["Mean CV Accuracy", "Mean CV F1"], color="#D1FAE5"),
                use_container_width=True
            )
        with c2:
            st.markdown("""
            **Why SVM (Support Vector Machine) was chosen:**
            - **Highest CV Accuracy:** ~79.83%
            - **Highest F1-Score:** ~0.870 (Strong balance of precision & recall)
            - **Lowest variance across folds:** Consistent generalization without overfitting on small datasets.
            """)

    st.markdown("---")
    st.markdown("### ⚙️ Complete End-to-End ML Pipeline Architecture")
    st.code("""
Raw Data (614 rows)
   │
   ├─► Numerical Columns: ['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
   │     ├─► Median Imputation (SimpleImputer)
   │     └─► Standard Scaling (StandardScaler - mean 0, variance 1)
   │
   ├─► Categorical Columns: ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
   │     ├─► Mode Imputation (SimpleImputer)
   │     └─► One-Hot Encoding (OneHotEncoder)
   │
   └─► Classifier: Support Vector Classifier (SVC)
         ├─► kernel = 'linear'
         ├─► C = 1.0
         ├─► class_weight = 'balanced' (Counteracts class imbalance: ~69% Approved vs 31% Rejected)
         └─► probability = True (Generates confidence percentage)
    """, language="text")

# ==========================================
# TAB 3: VIVA CHEATSHEET
# ==========================================
with tab_faq:
    st.markdown("### 🎓 Essential Viva Questions & Confident Answers")
    st.caption("Read these through before your viva presentation — answers are direct and easy to memorize!")

    faqs = [
        (
            "1. What is the objective of this project?",
            "To automate the loan eligibility assessment process based on applicant demographics, income, and past credit history using Machine Learning classification."
        ),
        (
            "2. What is the target variable and what type of problem is this?",
            "The target is 'Loan_Status' (Yes = 1, No = 0). It is a Supervised Binary Classification problem."
        ),
        (
            "3. Why did you use Median for numerical imputation and Mode for categorical?",
            "Numerical financial data like Income has outliers/skewed distributions, so the Median is robust against outliers unlike the Mean. Categorical data uses the Mode (most frequent value) because average does not apply to non-numeric categories."
        ),
        (
            "4. Why use a Pipeline instead of transforming data manually?",
            "A Pipeline bundles preprocessing (imputation + scaling + encoding) together with the model. This strictly prevents 'Data Leakage' from the test set into the training set and makes deployment bug-free."
        ),
        (
            "5. What was the single most decisive feature for loan approval?",
            "Credit_History. In banking, applicants with past defaults (Credit_History = 0) represent very high default probability, which our model captures strongly."
        ),
        (
            "6. Why did you use 'class_weight=balanced'?",
            "In this dataset, roughly 69% of loans were approved and 31% rejected. Setting class_weight='balanced' penalizes mistakes on the minority class (rejections) equally, so the model does not become biased toward approving everyone."
        ),
        (
            "7. Why is F1-score / Recall important for this problem?",
            "In banking, granting a loan to an ineligible person (False Positive / Type I error) can cause heavy financial default loss. F1-score balances Precision and Recall rather than relying solely on raw Accuracy."
        ),
        (
            "8. How does Support Vector Machine (Linear SVM) work?",
            "SVM finds an optimal hyperplane that maximizes the margin (distance) between the two classes (Approved vs Rejected) in feature space."
        )
    ]

    for q, a in faqs:
        with st.expander(f"📌 {q}", expanded=False):
            st.markdown(f'<div class="viva-box">{a}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("ML Lab Project • 5th Semester • Loan Approval Predictor using Scikit-Learn & Streamlit")
