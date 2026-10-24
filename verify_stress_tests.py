import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('loan_prediction.csv')
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = (df['Loan_Status'] == 'Y').astype(int)

X_raw = X.copy()
X_raw['Dependents'] = X_raw['Dependents'].astype(str).str.replace('+', '', regex=False)
X_raw['Dependents'] = pd.to_numeric(X_raw['Dependents'], errors='coerce')

raw_num = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

preprocessor_raw = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), raw_num),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

pipe_svm = Pipeline([
    ('prep', preprocessor_raw),
    ('clf', SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42))
])
pipe_svm.fit(X_raw, y)

class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if 'Dependents' in X.columns:
            X['Dependents'] = X['Dependents'].astype(str).str.replace('+', '', regex=False)
            X['Dependents'] = pd.to_numeric(X['Dependents'], errors='coerce')
        app_inc = pd.to_numeric(X['ApplicantIncome'], errors='coerce').fillna(0)
        coapp_inc = pd.to_numeric(X['CoapplicantIncome'], errors='coerce').fillna(0)
        loan_amt = pd.to_numeric(X['LoanAmount'], errors='coerce').fillna(128.0)
        term = pd.to_numeric(X['Loan_Amount_Term'], errors='coerce').fillna(360.0)
        term = np.where(term <= 0, 360.0, term)
        
        tot_inc = app_inc + coapp_inc
        denom = np.maximum(tot_inc, 1.0)
        
        X['TotalIncome'] = tot_inc
        X['EMI'] = (loan_amt * 1000.0) / term
        X['LTI'] = (loan_amt * 1000.0) / (denom * 12.0)
        X['DTI'] = X['EMI'] / denom
        return X

eng_num = raw_num + ['TotalIncome', 'EMI', 'LTI', 'DTI']
preprocessor_eng = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), eng_num),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

pipe_rf_eng = Pipeline([
    ('fe', FinancialFeatureEngineer()),
    ('prep', preprocessor_eng),
    ('clf', RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, class_weight='balanced', random_state=42))
])
pipe_rf_eng.fit(X, y)

test_cases = [
    ('1. Normal good applicant (Income=5000, Coapp=1500, Loan=120k, Credit=1)', {
        'Gender': 'Male', 'Married': 'Yes', 'Dependents': 0, 'Education': 'Graduate', 'Self_Employed': 'No',
        'ApplicantIncome': 5000, 'CoapplicantIncome': 1500, 'LoanAmount': 120, 'Loan_Amount_Term': 360,
        'Credit_History': 1.0, 'Property_Area': 'Urban'
    }),
    ('2. Absurd case: Income 500 with Loan 900k and Credit History 1', {
        'Gender': 'Male', 'Married': 'Yes', 'Dependents': 0, 'Education': 'Graduate', 'Self_Employed': 'No',
        'ApplicantIncome': 500, 'CoapplicantIncome': 0, 'LoanAmount': 900, 'Loan_Amount_Term': 360,
        'Credit_History': 1.0, 'Property_Area': 'Urban'
    }),
    ('3. Absurd case: Income 0 with Loan 500k and Credit History 1', {
        'Gender': 'Male', 'Married': 'Yes', 'Dependents': 0, 'Education': 'Graduate', 'Self_Employed': 'No',
        'ApplicantIncome': 0, 'CoapplicantIncome': 0, 'LoanAmount': 500, 'Loan_Amount_Term': 360,
        'Credit_History': 1.0, 'Property_Area': 'Urban'
    }),
    ('4. High Earner with Bad Credit: Income 50000 with Credit History 0 (Loan 150k)', {
        'Gender': 'Male', 'Married': 'Yes', 'Dependents': 0, 'Education': 'Graduate', 'Self_Employed': 'No',
        'ApplicantIncome': 50000, 'CoapplicantIncome': 0, 'LoanAmount': 150, 'Loan_Amount_Term': 360,
        'Credit_History': 0.0, 'Property_Area': 'Urban'
    }),
]

print("=" * 80)
print("STRESS TEST COMPARISON: LINEAR SVM vs RANDOM FOREST + FEATURE ENGINEERING")
print("=" * 80)
for label, d in test_cases:
    df_case = pd.DataFrame([d])
    p_svm = pipe_svm.predict_proba(df_case)[0, 1]
    p_rf = pipe_rf_eng.predict_proba(df_case)[0, 1]
    status_svm = "APPROVED" if p_svm >= 0.5 else "REJECTED"
    status_rf = "APPROVED" if p_rf >= 0.5 else "REJECTED"
    print(f"\n{label}")
    print(f"  Linear SVM:            P(Approved) = {p_svm*100:5.1f}%  ->  {status_svm}")
    print(f"  RF + Engineered Feat:  P(Approved) = {p_rf*100:5.1f}%  ->  {status_rf}")

# Feature importances
rf_model = pipe_rf_eng.named_steps['clf']
cat_enc = pipe_rf_eng.named_steps['prep'].named_transformers_['cat'].named_steps['encoder']
cat_names = list(cat_enc.get_feature_names_out(cat_cols))
all_feature_names = eng_num + cat_names
importances = pd.Series(rf_model.feature_importances_, index=all_feature_names).sort_values(ascending=False)

print("\n" + "=" * 80)
print("FEATURE IMPORTANCES (RANDOM FOREST + FEATURE ENGINEERING):")
print("=" * 80)
for feat, imp in importances.items():
    if imp >= 0.01:
        print(f"  {feat:<25} : {imp*100:5.2f}%")
