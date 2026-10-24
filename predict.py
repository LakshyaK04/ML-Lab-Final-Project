"""
Standalone Loan Approval Predictor Script (CLI)
Allows quick terminal prediction and testing without needing a web browser.
Gracefully handles missing model files by training from CSV if needed.
"""

import os
import sys
import joblib
import pandas as pd
from feature_engineering import FinancialFeatureEngineer, train_model

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_approval_model.pkl")
CSV_PATH = os.path.join(os.path.dirname(__file__), "loan_prediction.csv")


def load_pipeline():
    """
    Load the trained model pipeline.
    If the pickle file is missing or fails to load, falls back to training from CSV.
    """
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    return train_model(CSV_PATH)


def predict_loan_status(
    gender="Male",
    married="Yes",
    dependents=0,
    education="Graduate",
    self_employed="No",
    applicant_income=5000,
    coapplicant_income=1500,
    loan_amount=120,
    loan_term=360,
    credit_history=1.0,
    property_area="Urban",
):
    """
    Predict loan status for given applicant attributes.
    Returns: (status_text, approval_probability_percentage, metrics_dict)
    """
    pipeline = load_pipeline()

    sample_df = pd.DataFrame([
        {
            "Gender": gender,
            "Married": married,
            "Dependents": float(dependents),
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": float(applicant_income),
            "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount": float(loan_amount),
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": float(credit_history),
            "Property_Area": property_area,
        }
    ])

    pred = pipeline.predict(sample_df)[0]
    prob = pipeline.predict_proba(sample_df)[0, 1]

    # Calculate affordability metrics for display
    total_income = float(applicant_income) + float(coapplicant_income)
    denom_inc = max(total_income, 1.0)
    term_guarded = max(float(loan_term), 1.0)
    emi = (float(loan_amount) * 1000.0) / term_guarded
    lti = (float(loan_amount) * 1000.0) / (denom_inc * 12.0)
    dti = emi / denom_inc

    metrics = {
        "TotalIncome": total_income,
        "EMI": emi,
        "LTI": lti,
        "DTI": dti,
    }

    status = "APPROVED" if pred == 1 else "REJECTED"
    return status, prob, metrics


def run_interactive_cli():
    print("=" * 65)
    print("      [LOAN APPROVAL PREDICTOR - CLI DEMO]")
    print("   Random Forest Classifier with Banking Feature Engineering")
    print("=" * 65)
    print("Select an option:")
    print("1. Run Sample Test 1 (Standard Eligible Applicant)")
    print("2. Run Sample Test 2 (Absurd Case: Income $500, Loan $900k, Credit 1.0)")
    print("3. Run Sample Test 3 (Ineligible Applicant: Credit History Defaulted)")
    print("4. Enter Custom Applicant Details")
    print("5. Exit")
    print("-" * 65)

    choice = input("Enter choice (1/2/3/4/5): ").strip()

    if choice == "1":
        print("\nTesting Sample 1: Standard Eligible Applicant...")
        status, prob, m = predict_loan_status(
            applicant_income=5000, coapplicant_income=1500,
            loan_amount=120, loan_term=360, credit_history=1.0
        )
        print(f"Result: >>> {status} <<< (Approval Confidence: {prob:.2%})")
        print(f"Financial Ratios: EMI = ${m['EMI']:.2f}/mo | DTI = {m['DTI']:.2%} | LTI = {m['LTI']:.2f}x")

    elif choice == "2":
        print("\nTesting Sample 2: Absurd Case (Low Income, Huge Loan, Credit 1.0)...")
        status, prob, m = predict_loan_status(
            applicant_income=500, coapplicant_income=0,
            loan_amount=900, loan_term=360, credit_history=1.0
        )
        print(f"Result: >>> {status} <<< (Approval Confidence: {prob:.2%})")
        print(f"Financial Ratios: EMI = ${m['EMI']:.2f}/mo | DTI = {m['DTI']:.2%} | LTI = {m['LTI']:.2f}x")

    elif choice == "3":
        print("\nTesting Sample 3: Ineligible Applicant (Credit History Defaulted)...")
        status, prob, m = predict_loan_status(
            applicant_income=2500, coapplicant_income=0,
            loan_amount=150, loan_term=360, credit_history=0.0
        )
        print(f"Result: >>> {status} <<< (Approval Confidence: {prob:.2%})")
        print(f"Financial Ratios: EMI = ${m['EMI']:.2f}/mo | DTI = {m['DTI']:.2%} | LTI = {m['LTI']:.2f}x")

    elif choice == "4":
        print("\n--- Enter Applicant Details ---")
        try:
            gender = input("Gender (Male/Female) [default: Male]: ").strip() or "Male"
            married = input("Married (Yes/No) [default: Yes]: ").strip() or "Yes"
            dependents = int(input("Dependents (0, 1, 2, 3) [default: 0]: ").strip() or "0")
            education = input("Education (Graduate/Not Graduate) [default: Graduate]: ").strip() or "Graduate"
            self_emp = input("Self Employed (Yes/No) [default: No]: ").strip() or "No"
            app_inc = float(input("Applicant Monthly Income ($) [default: 5000]: ").strip() or "5000")
            coapp_inc = float(input("Coapplicant Monthly Income ($) [default: 1500]: ").strip() or "1500")
            loan_amt = float(input("Loan Amount in Thousands ($) [default: 120]: ").strip() or "120")
            term = float(input("Term in Months [default: 360]: ").strip() or "360")
            credit = float(input("Credit History (1.0 for Good, 0.0 for Bad) [default: 1.0]: ").strip() or "1.0")
            area = input("Property Area (Urban/Semiurban/Rural) [default: Urban]: ").strip() or "Urban"

            status, prob, m = predict_loan_status(
                gender=gender, married=married, dependents=dependents,
                education=education, self_employed=self_emp,
                applicant_income=app_inc, coapplicant_income=coapp_inc,
                loan_amount=loan_amt, loan_term=term, credit_history=credit,
                property_area=area
            )
            print("-" * 65)
            print(f"Prediction Result: >>> {status} <<< (Approval Confidence: {prob:.2%})")
            print(f"Financial Ratios: EMI = ${m['EMI']:.2f}/mo | DTI = {m['DTI']:.2%} | LTI = {m['LTI']:.2f}x")
            print("-" * 65)
        except Exception as e:
            print(f"Invalid input: {e}")

    else:
        print("Exiting...")


if __name__ == "__main__":
    run_interactive_cli()
