# 🏦 Loan Approval Predictor (ML Lab Final Project)

An end-to-end Machine Learning classification project designed to predict bank loan approval status based on applicant demographics, financial health, and credit history.

---

## 📁 Project Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit interactive web application with presets and visual dashboard |
| `predict.py` | Terminal-based Python CLI for testing predictions without a browser |
| `loan_approval_model.pkl` | Trained Scikit-Learn Pipeline (Imputer + Scaler + OneHotEncoder + SVM) |
| `loan_prediction.csv` | Original loan dataset (614 records) |
| `model_comparison.csv` | Cross-validation benchmark of the 4 models trained |
| `loan_approval_training.ipynb` | Jupyter Notebook containing EDA, preprocessing, model training & cross-validation |
| `VIVA_GUIDE.md` | Complete Viva preparation guide with top 15 questions and simple answers |

---

## 🚀 How to Run

### 1. Launch the Interactive Web App (Recommended for Viva Demo)
```bash
streamlit run app.py
```
This opens the web interface in your browser:
- Use the sidebar presets: **"Standard Approved"** or **"Defaulter Case"**.
- View confidence scores, key deciding factors, and model comparison metrics.

### 2. Run Terminal Predictor (Pure Python)
```bash
python predict.py
```

---

## 📊 Summary of Results
- **Best Model:** Support Vector Machine (Linear Kernel)
- **Mean 5-Fold Cross Validation Accuracy:** 79.83%
- **Mean Cross Validation F1-Score:** 0.870
- **Most Decisive Feature:** `Credit_History`
