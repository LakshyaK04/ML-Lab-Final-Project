# 🏦 Loan Approval Predictor (ML Lab Final Project)

An end-to-end Machine Learning classification system that predicts bank loan approval decisions using financial affordability metrics and historical credit records.

---

## 📁 Project Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit interactive web dashboard displaying predictions, live DTI/EMI metrics, and viva cheatsheet |
| `feature_engineering.py` | Core pipeline module implementing custom Scikit-Learn transformer (`FinancialFeatureEngineer`) |
| `predict.py` | Terminal-based Python CLI for instant testing with automatic fallback retraining |
| `loan_approval_model.pkl` | Serialized Scikit-Learn Pipeline (`FinancialFeatureEngineer` + `ColumnTransformer` + `RandomForestClassifier`) |
| `loan_prediction.csv` | Standard benchmark loan dataset (614 applications) |
| `model_comparison.csv` | Benchmarking results across models using Repeated Stratified 5-Fold Cross-Validation (25 splits) |
| `loan_approval_training.ipynb` | Jupyter Notebook with EDA, distributions, pipeline training, and cross-validation |
| `VIVA_GUIDE.md` | Comprehensive Viva Voce preparation guide with elevator pitch, stress tests, and technical Q&A |

---

## ⚙️ Key Feature Engineering

Rather than relying purely on raw dollar inputs, the pipeline automatically computes standard credit underwriting metrics inside a Scikit-Learn `Pipeline`:

- **Total Household Income:** $\text{ApplicantIncome} + \text{CoapplicantIncome}$
- **Monthly EMI:** $\frac{\text{LoanAmount} \times 1000}{\text{Loan\_Amount\_Term}}$
- **Loan-to-Income (LTI):** $\frac{\text{LoanAmount} \times 1000}{\text{TotalIncome} \times 12}$ (measures leverage)
- **Debt-to-Income (DTI):** $\frac{\text{EMI}}{\text{TotalIncome}}$ (measures monthly debt burden)

---

## 📊 Summary of Evaluation Results

Models benchmarked using **Repeated Stratified 5-Fold Cross-Validation (5 repeats = 25 splits)**:

| Model | Mean CV Accuracy | CV Accuracy Std | Mean CV F1 | CV F1 Std |
|---|---|---|---|---|
| **Linear SVM (Raw Features)** | 80.95% | ±2.30% | 0.8766 | ±0.0138 |
| **Random Forest (Engineered Features)** 🏆 | **79.61%** | **±2.15%** | **0.8616** | **±0.0162** |
| **Random Forest (Raw Features)** | 77.07% | ±3.04% | 0.8444 | ±0.0238 |
| **Logistic Regression (Raw Features)** | 75.97% | ±3.07% | 0.8313 | ±0.0260 |

### Realistic Decision Behavior (Honest Assessment):
- **Accuracy is statistically comparable (~80%):** Linear SVM (~80.9%) and Random Forest + Feature Engineering (~79.6%) perform equally well within standard noise margins.
- **Why Random Forest was chosen:** A baseline Linear SVM placed almost 100% of decision weight on `Credit_History`, approving absurd applications (e.g., $0 income, $500,000 loan request with good credit). Random Forest with engineered ratios checks actual repayment affordability, correctly rejecting unaffordable loans.
- **Known Tradeoff:** The tree ensemble gives a higher approval probability (~25%) to bad-credit applicants who exhibit high income and low loan requests, compared to Linear SVM (~9%), even though only ~8% of bad-credit applicants were approved in historical data.

---

## 🚀 How to Run

### 1. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```
- Test real-world cases with the sidebar presets (Standard Approved, Stress Test Low Income / Huge Loan, Bad Credit).
- View live **EMI ($/mo)**, **DTI (%)**, and **LTI (x)** ratio badges alongside prediction probabilities.
- Inspect 25-fold cross-validation metrics and feature importances under Tab 2.

### 2. Run Terminal Predictor (Pure Python)
```bash
python predict.py
```
- Fast interactive CLI with built-in fallback: if `loan_approval_model.pkl` is ever deleted, it automatically retrains directly from `loan_prediction.csv` in under a second without crashing.
