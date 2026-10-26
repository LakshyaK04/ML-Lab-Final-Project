# 🎓 LOAN APPROVAL PREDICTOR - VIVA SURVIVAL GUIDE

> **Tip for Viva:** Speak calmly, stick to the facts, and don't make unsubstantiated claims. Examiners appreciate when a student can clearly explain **why** they made design choices and acknowledge real-world **tradeoffs**.

---

## 1. The 60-Second Elevator Pitch (Memorize This!)

**If the examiner asks:** *"Tell me about your project in 1 minute."*

> *"Sir/Ma'am, my project is an automated **Loan Approval Prediction System** built on historical banking records. In credit underwriting, evaluating applications manually is slow, but naively trained models often make unrealistic decisions.*
>
> *Initially, a linear model placed almost all decision weight on past credit history and practically ignored income, causing it to approve absurd edge cases, like an applicant with zero income requesting a $500,000 loan simply because their credit flag was 1. To fix this, we implemented **Banking Feature Engineering** directly inside a Scikit-Learn pipeline to compute **Total Household Income**, **EMI**, **Debt-to-Income (DTI)**, and **Loan-to-Income (LTI)** ratios.*
>
> *We evaluated our pipeline using **Repeated Stratified 5-Fold Cross-Validation (25 folds)** and selected **Random Forest** (300 trees, max depth 5). The overall accuracy remained around **80%** (Linear SVM ~80.9% ± 2.3% vs Random Forest with engineered features ~79.6% ± 2.1%), but the model behavior became economically sound: the old SVM ignored income, whereas the new model actively uses income and debt burden to check affordability. We deployed this into an interactive Streamlit application with live affordability metrics."*

---

## 2. Dataset Overview

- **Source:** Analytics Vidhya Loan Prediction Dataset (`loan_prediction.csv`).
- **Total Records:** 614 applicants.
- **Problem Type:** Supervised Binary Classification.
- **Target Variable:** `Loan_Status`
  - `Y` (Approved) → Encoded as `1` (~69% of records)
  - `N` (Rejected) → Encoded as `0` (~31% of records)
- **Features Used (11 raw input features):**
  - **Categorical (5):** `Gender`, `Married`, `Education`, `Self_Employed`, `Property_Area`.
  - **Numerical (6):** `Dependents`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History` (1.0 = good credit, 0.0 = defaulted).

---

## 3. End-to-End Machine Learning Pipeline

```
[Raw Applicant Data: 11 Features]
       │
       ▼
[Step 1: Banking Feature Engineering (FinancialFeatureEngineer)]
   ├── TotalIncome = ApplicantIncome + CoapplicantIncome
   ├── EMI = (LoanAmount * 1000) / Loan_Amount_Term
   ├── LTI = (LoanAmount * 1000) / (TotalIncome * 12)
   └── DTI = EMI / TotalIncome
   (Includes guards against zero division & missing terms)
       │
       ▼
[Step 2: Preprocessing ColumnTransformer]
   ├── Numerical Features (10) ──► SimpleImputer(median) ──► StandardScaler()
   └── Categorical Features (5) ─► SimpleImputer(most_frequent) ─► OneHotEncoder()
       │
       ▼
[Step 3: Classifier]
   └── RandomForestClassifier
         - n_estimators = 300
         - max_depth = 5 (Controls tree depth to prevent deep memorization)
         - min_samples_leaf = 5
         - class_weight = 'balanced' (Counteracts 69/31 class imbalance)
         - random_state = 42
```

---

## 4. Model Evaluation & Benchmark

Models benchmarked using **Repeated Stratified 5-Fold Cross-Validation (5 repeats = 25 evaluation splits)**:

| Model | Mean CV Accuracy | Accuracy Std | Mean CV F1 | CV F1 Std | Notes |
|---|---|---|---|---|---|
| **Linear SVM (Raw Features)** | 80.95% | ±2.30% | 0.8766 | ±0.0138 | Practically ignored income; approved absurd loans with good credit. |
| **Random Forest (Engineered Features)** | **79.61%** | **±2.15%** | **0.8616** | **±0.0162** | **Accurate (~80%) and actively checks affordability; passes stress tests.** |
| Random Forest (Raw Features) | 77.07% | ±3.04% | 0.8444 | ±0.0238 | Lower accuracy without financial ratio interactions. |
| Logistic Regression (Raw Features) | 75.97% | ±3.07% | 0.8313 | ±0.0260 | Linear decision boundary with higher variance. |

---

## 5. Essential Viva Voce Questions & Answers

### Q1: Why did you add engineered features (TotalIncome, EMI, LTI, DTI)?
**Answer:** Raw income and loan amount in isolation do not measure debt affordability. In actual banking, credit risk depends on repayment capacity relative to monthly debt obligations:
1. **TotalIncome:** Combines applicant and co-applicant income (household repayment capacity).
2. **EMI:** Monthly installment = $\frac{\text{LoanAmount} \times 1000}{\text{Loan\_Amount\_Term}}$.
3. **LTI (Loan-to-Income):** Ratio of total loan amount to annual household income ($\text{TotalIncome} \times 12$).
4. **DTI (Debt-to-Income):** Fraction of monthly income consumed by the loan EMI.

Implementing these in the pipeline allowed the model to evaluate debt burden directly rather than relying almost exclusively on credit history.

---

### Q2: Why switch from Linear SVM to Random Forest?
**Answer:** In the baseline Linear SVM, `Credit_History` dominated the decision while income and loan amounts were practically ignored (coefficients near zero). When tested on edge cases, the Linear SVM approved an applicant with **$0 income requesting a $500,000 loan** simply because `Credit_History` was 1.0!

The old SVM ignored income, whereas the new Random Forest model uses income, loan amount, and affordability ratios. Random Forest captures non-linear feature interactions between debt ratios and credit history, correctly rejecting unaffordable loans.

---

### Q3: Did accuracy improve after adding engineered features?
**Answer:** **No, accuracy remained around 80%.** 
Across 25 cross-validation splits (Repeated Stratified 5-Fold), Linear SVM achieved 80.95% ± 2.30% and Random Forest with engineered features achieved 79.61% ± 2.15%. These are within statistical noise margins. 

The real motivation was not an artificial accuracy boost on historical data, but **behavioral correctness**: ensuring the model makes economically sound decisions when faced with extreme or unbalanced applicant inputs.

---

### Q4: Why were income weights so small in the old Linear SVM if data was already scaled?
**Answer:** It was **not** a raw dollar scaling issue, because the pipeline already used `StandardScaler`. The true reason is that once `Credit_History` is known in this dataset, income adds very little linearly separable signal. Because the linear model minimizes empirical loss, it converged on a separating hyperplane that practically ignored income and relied almost entirely on the single dominant binary feature (`Credit_History`).

---

### Q5: What is the known tradeoff with the new Random Forest model?
**Answer:** While Random Forest prevents approving zero-income applicants, it gives a **higher approval probability to bad-credit applicants (~25%)** if they have very high income and small loan requests, compared to Linear SVM (~9%). In the historical dataset, only ~8% of bad-credit applicants were approved. This reflects the tree ensemble's willingness to weigh extreme financial strength against bad credit.

---

### Q6: What are the primary limitations of this project?
**Answer:**
1. **Sample Size:** 614 rows is small for real-world production credit scoring.
2. **Binary Credit Feature:** `Credit_History` is only a binary 0 or 1 flag instead of a continuous FICO / CIBIL score (300 to 850).
3. **External Obligations:** The dataset lacks external debt records (e.g. existing credit cards, car loans, or utility bills).
4. **No True Negative representation in F1:** The positive class is ~69%, so high F1 does not reflect rejection specificity.

---

### Q7: What is the true definition and formula of F1-Score?
**Answer:** F1-Score is the **harmonic mean of Precision and Recall specifically for the positive class (Approved)**:
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FP} + \text{FN}}$$

Importantly, **F1 does NOT use True Negatives (TN)**. Therefore, in imbalanced datasets where the positive class is the majority (~69%), F1 can appear high even if the model performs poorly on detecting rejections.

---

### Q8: What is a False Positive vs a False Negative here, and which is worse for a bank?
**Answer:**
- **False Positive (Type I Error):** Predicting "Approved" for an applicant who defaults.
- **False Negative (Type II Error):** Rejecting a creditworthy applicant who would have repaid.
- **Which is worse:** A **False Positive** is far more damaging to a financial institution because the bank risks losing the entire loan principal, whereas a False Negative represents lost interest revenue.

---

### Q9: Why did you use `class_weight='balanced'`?
**Answer:** In `loan_prediction.csv`, ~69% of loans are Approved and ~31% are Rejected. Without balanced weighting, algorithms can achieve ~69% accuracy by trivially predicting "Approved" for every sample. Setting `class_weight='balanced'` automatically assigns weights inversely proportional to class frequencies:
$$w_j = \frac{N}{K \times n_j}$$
This penalizes misclassifying rejections more heavily.

---

### Q10: Why did you use Median imputation for numbers and Mode for categories?
**Answer:** Numerical columns like `ApplicantIncome` and `LoanAmount` have extreme positive skew and high-income outliers. The **Mean** is pulled upward by outliers, whereas the **Median** reflects the true central tendency. For categorical variables (e.g. `Gender`, `Married`), mathematical averages do not exist, so the **Mode** (most frequent value) is the standard imputation method.

---

### Q11: Why use a Scikit-Learn `Pipeline`?
**Answer:**
1. **Prevents Data Leakage:** Imputation statistics, scaling factors, and encoders are fitted strictly on training folds and applied to test folds.
2. **Clean Deployment:** Preprocessing and model inference are encapsulated into a single serializable object (`loan_approval_model.pkl`), guaranteeing identical transformations in production.

---

### Q12: How does Random Forest work, and what hyperparameters did you set?
**Answer:** Random Forest is an ensemble of decision trees trained using Bootstrap Aggregating (Bagging) and random feature selection at each split. We configured:
- `n_estimators=300`: 300 individual decision trees to reduce prediction variance.
- `max_depth=5`: Limits the maximum depth of each tree to prevent memorizing noise.
- `min_samples_leaf=5`: Requires at least 5 training samples per terminal leaf.
- `class_weight='balanced'`: Weights classes inversely to their sample frequency.

---

## 6. Stress Test Comparison (Examiner Proof Sheet)

| Test Case | Applicant Profile | Linear SVM Result | Random Forest + Feat Eng Result | Real-World Expectation |
|---|---|---|---|---|
| **1. Standard Good** | Inc: $6,500, Loan: $120k, Credit: 1.0 | Approved (79.4%) | **Approved (63.3%)** | ✅ Approved (Affordable) |
| **2. Absurd Case A** | Inc: $500, Loan: $900k, Credit: 1.0 | **Approved (79.4%) ❌** | **Rejected (23.9% prob) ✔️** | ❌ Severe unaffordability |
| **3. Absurd Case B** | Inc: $0, Loan: $500k, Credit: 1.0 | **Approved (79.4%) ❌** | **Rejected (24.2% prob) ✔️** | ❌ Cannot repay loan |
| **4. High Earner / Bad Credit** | Inc: $50,000, Loan: $150k, Credit: 0.0 | Rejected (9.4%) | **Rejected (25.1% prob)** | ⚠️ Defaulter scrutiny |

---

## 7. How to Demonstrate Live During Viva

1. **Launch Web App:**
   ```bash
   streamlit run app.py
   ```
2. **Demonstrate Stress Test:**
   - Click **"Stress Test: Low Income, Huge Loan"** on the sidebar $\rightarrow$ Click **Run Loan Approval Prediction**.
   - Show the examiner that the model **rejects** the loan and points out high **DTI** and **LTI** metrics!
3. **Demonstrate Tab 2 (Model Performance):**
   - Show the 25-fold Repeated Stratified CV table and the feature importance chart.
4. **Demonstrate Terminal CLI:**
   ```bash
   python predict.py
   ```
   - Run Option 1 (Standard) and Option 2 (Absurd Case) directly in the console.
