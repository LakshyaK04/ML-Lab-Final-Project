"""
Standalone Loan Approval Predictor Script
Allows quick terminal prediction and testing without needing a web browser.
"""

import os
import joblib
import pandas as pd

def load_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), "loan_approval_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)

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
    property_area="Urban"
):
    """
    Predict loan status for given applicant attributes.
    Returns: (status_text, approval_probability_percentage)
    """
    pipeline = load_pipeline()

    sample_df = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': float(dependents),
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': float(applicant_income),
        'CoapplicantIncome': float(coapplicant_income),
        'LoanAmount': float(loan_amount),
        'Loan_Amount_Term': float(loan_term),
        'Credit_History': float(credit_history),
        'Property_Area': property_area
    }])

    pred = pipeline.predict(sample_df)[0]
    prob = pipeline.predict_proba(sample_df)[0, 1]

    status = "APPROVED" if pred == 1 else "REJECTED"
    return status, prob

def run_interactive_cli():
    print("=" * 60)
    print("      🏦 LOAN APPROVAL PREDICTOR (CLI DEMO) 🏦")
    print("=" * 60)
    print("Select an option:")
    print("1. Run Sample Test 1 (Eligible Applicant - Good Credit)")
    print("2. Run Sample Test 2 (Ineligible Applicant - Bad Credit History)")
    print("3. Enter Custom Applicant Details")
    print("4. Exit")
    print("-" * 60)

    choice = input("Enter choice (1/2/3/4): ").strip()

    if choice == "1":
        print("\nTesting Sample 1: Eligible Applicant (Good Credit, High Income)...")
        status, prob = predict_loan_status(
            gender="Male",
            married="Yes",
            dependents=0,
            education="Graduate",
            self_employed="No",
            applicant_income=6000,
            coapplicant_income=2500,
            loan_amount=130,
            loan_term=360,
            credit_history=1.0,
            property_area="Urban"
        )
        print(f"Result: >>> {status} <<< (Approval Confidence: {prob:.2%})")

    elif choice == "2":
        print("\nTesting Sample 2: Ineligible Applicant (Credit History Defaulted)...")
        status, prob = predict_loan_status(
            gender="Male",
            married="No",
            dependents=0,
            education="Not Graduate",
            self_employed="No",
            applicant_income=2200,
            coapplicant_income=0,
            loan_amount=150,
            loan_term=360,
            credit_history=0.0,
            property_area="Rural"
        )
        print(f"Result: >>> {status} <<< (Approval Confidence: {prob:.2%})")

    elif choice == "3":
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

            status, prob = predict_loan_status(
                gender=gender,
                married=married,
                dependents=dependents,
                education=education,
                self_employed=self_emp,
                applicant_income=app_inc,
                coapplicant_income=coapp_inc,
                loan_amount=loan_amt,
                loan_term=term,
                credit_history=credit,
                property_area=area
            )
            print("-" * 60)
            print(f"Prediction Result: >>> {status} <<< (Approval Confidence: {prob:.2%})")
            print("-" * 60)
        except Exception as e:
            print(f"Invalid input: {e}")

    else:
        print("Exiting...")

if __name__ == "__main__":
    run_interactive_cli()
