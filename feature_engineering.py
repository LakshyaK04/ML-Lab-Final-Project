"""
Feature Engineering and Pipeline Module for Loan Approval Predictor.

Defines the custom FinancialFeatureEngineer transformer and pipeline builder
used across training, web application (app.py), and CLI (predict.py).
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to compute key banking affordability metrics:
    - TotalIncome: ApplicantIncome + CoapplicantIncome
    - EMI: Monthly installment = LoanAmount * 1000 / Loan_Amount_Term
    - LTI: Loan-to-Income ratio = LoanAmount * 1000 / (TotalIncome * 12)
    - DTI: Debt-to-Income ratio = EMI / TotalIncome
    
    Includes guards against zero-division and missing values.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Handle Dependents if present (e.g. '3+' -> 3)
        if "Dependents" in X.columns:
            X["Dependents"] = (
                X["Dependents"]
                .astype(str)
                .str.replace("+", "", regex=False)
            )
            X["Dependents"] = pd.to_numeric(X["Dependents"], errors="coerce")

        app_inc = pd.to_numeric(X.get("ApplicantIncome", 0), errors="coerce").fillna(0.0)
        coapp_inc = pd.to_numeric(X.get("CoapplicantIncome", 0), errors="coerce").fillna(0.0)
        loan_amt = pd.to_numeric(X.get("LoanAmount", 128.0), errors="coerce").fillna(128.0)
        term = pd.to_numeric(X.get("Loan_Amount_Term", 360.0), errors="coerce").fillna(360.0)

        # Guard term against <= 0
        term_guarded = np.where(term <= 0, 360.0, term)

        # Total Household Income
        total_income = app_inc + coapp_inc
        # Guard income against <= 0 for division
        income_guarded = np.maximum(total_income, 1.0)

        # 1. TotalIncome
        X["TotalIncome"] = total_income

        # 2. EMI: Monthly loan installment (LoanAmount is in thousands of dollars)
        X["EMI"] = (loan_amt * 1000.0) / term_guarded

        # 3. LTI: Loan to Annual Income Ratio
        X["LTI"] = (loan_amt * 1000.0) / (income_guarded * 12.0)

        # 4. DTI: Debt to Monthly Income Ratio
        X["DTI"] = X["EMI"] / income_guarded

        return X


def build_pipeline():
    """Build the end-to-end scikit-learn pipeline."""
    numeric_features = [
        "Dependents",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "TotalIncome",
        "EMI",
        "LTI",
        "DTI",
    ]

    categorical_features = [
        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("fe", FinancialFeatureEngineer()),
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return pipeline


def load_and_preprocess_training_data(csv_path=None):
    """Load loan_prediction.csv and clean target variable."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "loan_prediction.csv")

    df = pd.read_csv(csv_path)
    df_clean = df.copy()
    df_clean.drop(columns=["Loan_ID"], inplace=True, errors="ignore")

    df_clean["Loan_Status"] = (
        df_clean["Loan_Status"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"Y": 1, "N": 0})
    )
    df_clean.dropna(subset=["Loan_Status"], inplace=True)
    df_clean["Loan_Status"] = df_clean["Loan_Status"].astype(int)

    X = df_clean.drop(columns=["Loan_Status"])
    y = df_clean["Loan_Status"]
    return X, y


def train_model(csv_path=None):
    """Train the pipeline and return the fitted pipeline."""
    X, y = load_and_preprocess_training_data(csv_path)
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    return pipeline
