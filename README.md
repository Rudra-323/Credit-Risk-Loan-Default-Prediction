# 🏦 Loan Risk Prediction using Machine Learning

Predicting whether a loan applicant will **default** or **repay**, using the Credit Risk dataset from the banking domain. The project covers end-to-end exploratory data analysis (EDA), data preparation, and a comparison of five classification algorithms for credit risk scoring.

---

## 📌 Table of Contents

- [Business Context](#-business-context)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Problem Definition](#-problem-definition)
- [Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Model Building](#-model-building)
- [Results](#-results)
- [Conclusion](#-conclusion)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)

---

## 💼 Business Context

Lending is the core business of a bank, and its profit or loss depends largely on whether borrowers repay or default. **Credit risk** is the risk that a borrower fails to meet their repayment obligations, which leads to Non-Performing Assets (NPAs).

Manual screening of applicants is slow and error-prone. A machine learning model can:

- Automate and speed up loan approval decisions
- Reduce NPAs by flagging likely defaulters early
- Improve customer experience through faster, consistent risk assessment

---

## 📂 Dataset

**File:** `credit_risk_dataset.csv` (domain: Banking)

| Feature | Description |
|---|---|
| `person_age` | Age of the applicant |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership status (RENT / OWN / MORTGAGE / OTHER) |
| `person_emp_length` | Employment length (in years) |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Interest rate |
| `loan_percent_income` | Loan amount as a percentage of income |
| `cb_person_default_on_file` | Historical default on file (Y/N) |
| `cb_person_cred_hist_length` | Credit history length (in years) |
| **`loan_status`** | **Target**: `0` = non-default, `1` = default |

---

## 🔄 Project Workflow

The project follows two sprints:

**Sprint 1: Exploratory Data Analysis**
1. Explore the distribution of each column
2. Identify the variables most related to the target (`loan_status`)
3. Derive insights and data-driven recommendations for the bank

**Sprint 2: Data Preparation & Model Building**
1. Load the data
2. Define inputs, target, ML task and evaluation metric
3. Split into train/test (75:25)
4. Prepare train data (fit + transform), then test data (transform only)
5. Train KNN, Logistic Regression, SVM, Decision Tree, Random Forest
6. Evaluate each model and compare them in a plot

---

## 🎯 Problem Definition

| Item | Detail |
|---|---|
| **Input features** | Age, income, home ownership, employment length, loan intent, loan grade, loan amount, interest rate, loan-to-income ratio, historical default, credit history length |
| **Target variable** | `loan_status` (0 = non-default, 1 = default) |
| **ML task** | Supervised **binary classification** |
| **Primary metric** | Accuracy (`metrics.accuracy_score`) |
| **Supporting metrics** | Recall, F1-score, ROC-AUC, confusion matrix |

> Because the classes are imbalanced (about 22% defaults), accuracy alone can be misleading. Recall, F1 and ROC-AUC are reported alongside it, since missing a real defaulter is costly for a bank.

---

## 🧹 Data Cleaning & Preprocessing

1. **Duplicates:** removed duplicate rows (32,416 rows remained).
2. **Invalid records:**
   - Dropped rows where employment length exceeded age (2 rows)
   - Dropped rows with unrealistic age (> 100) (5 rows)
   - Final dataset: **32,409 rows**
3. **Missing values:**
   - `person_emp_length`: **median** imputation (skewed distribution)
   - `loan_int_rate`: **mean** imputation (near-symmetric, skewness within ±0.5)
4. **Train/test split:** 75:25, **stratified** on the target, `random_state=42`
5. **Transformations** (via `ColumnTransformer`, fit on train only):
   - Numerical: `StandardScaler`
   - Categorical: `OneHotEncoder(handle_unknown='ignore')`
6. **Class imbalance:** handled with `class_weight='balanced'` and, in some pipelines, **SMOTE** on the training data only.

---

## 📊 Exploratory Data Analysis


- Histograms and boxplots for every numerical column (distributions and outliers)
- Bar charts for every categorical column
- Numerical and categorical features plotted against `loan_status`
- Correlation heatmap of numerical features
- Loan status class balance

---

## 🤖 Model Building

Each model is wrapped in a scikit-learn / imbalanced-learn `Pipeline` (preprocessing → optional SMOTE → classifier) to avoid data leakage.

| Algorithm | Notes |
|---|---|
| K-Nearest Neighbors | Default parameters |
| Logistic Regression | `class_weight='balanced'`, `max_iter=1000` |
| Support Vector Machine | RBF kernel, `class_weight='balanced'`, `probability=True` |
| Decision Tree | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'`; `GridSearchCV` run over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` |

The final Random Forest pipeline is saved with `joblib` as `loan_default_predict_rf_model.pkl`.

---

## 📈 Results

Evaluated on the held-out test set (8,103 samples: 6,331 non-defaults, 1,772 defaults).

| Model | Accuracy | Recall (Default) | F1 (Default) | ROC-AUC |
|---|---|---|---|---|
| **Random Forest** | **0.935** | 0.720 | **0.829** | **0.933** |
| SVM | 0.892 | 0.757 | 0.754 | 0.910 |
| Decision Tree | 0.890 | **0.769** | 0.753 | 0.846 |
| KNN | 0.813 | 0.764 | 0.641 | n/a |
| Logistic Regression | 0.807 | 0.774 | 0.637 | 0.871 |

*Metrics come from the per-model evaluation cells in the notebook. Small differences are expected between runs because some pipelines use SMOTE and some do not (for example, a Random Forest run with SMOTE gave 0.932 accuracy and 0.931 ROC-AUC).*

**Random Forest confusion matrix:**

|  | Predicted Non-Default | Predicted Default |
|---|---|---|
| **Actual Non-Default** | 6301 | 30 |
| **Actual Default** | 497 | 1275 |

Random Forest reaches 98% precision on the default class, meaning almost every applicant it flags as a defaulter really is one.

---

## ✅ Conclusion

- **Random Forest is the best algorithm for credit risk scoring** in this project. It has the highest accuracy (~93.5%), F1-score and ROC-AUC, and very few false alarms.
- **Logistic Regression and KNN** are weaker on accuracy and precision. They catch slightly more defaulters but wrongly flag many good applicants (over 1,000 false positives).
- **Trade-off to be aware of:** Random Forest misses more defaulters (recall of about 72%) than the simpler models. If the bank prioritises catching every defaulter, a lower decision threshold or recall-focused tuning is worth exploring.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Rudra-323/Credit-Risk-Loan-Default-Prediction.git
cd Credit-Risk-Loan-Default-Prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter
```

### 3. Add the dataset

Place `credit_risk_dataset.csv` in the project root folder.

### 4. Run the notebook

```bash
jupyter notebook Loan_Risk_Prediction_Project_.ipynb
```

### 5. Use the saved model

```python
import joblib
import pandas as pd

model = joblib.load("loan_default_predict_rf_model.pkl")

applicant = pd.DataFrame([{
    "person_age": 28,
    "person_income": 55000,
    "person_home_ownership": "RENT",
    "person_emp_length": 4.0,
    "loan_intent": "EDUCATION",
    "loan_grade": "B",
    "loan_amnt": 10000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.18,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 5,
}])

print("Default" if model.predict(applicant)[0] == 1 else "Non-default")
```

---

## 📁 Project Structure

```
├── Loan_Risk_Prediction_Project_.ipynb   # EDA + model building
├── credit_risk_dataset.csv               # Dataset (add manually)
├── loan_default_predict_rf_model.pkl     # Saved Random Forest pipeline
└── README.md
```

---

## 🛠 Tech Stack

- **Language:** Python
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** scikit-learn, imbalanced-learn (SMOTE)
- **Persistence:** joblib

---

## 🔮 Future Improvements

- Tune the decision threshold and optimise for recall or a cost-sensitive metric
- Try gradient boosting models (XGBoost, LightGBM)
- Add feature importance and SHAP explanations for model transparency
- Use cross-validation for more robust model comparison
- Deploy the model as an API or a Streamlit app

---

