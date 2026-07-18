# Telco Customer Churn Prediction

A CRISP-DM based machine learning project predicting customer churn for a
telecommunications provider, using the IBM Telco Customer Churn dataset.
Built for BMDS2003 Data Science.

Four classification models (Logistic Regression, Decision Tree, Random
Forest, XGBoost) are trained and compared on a redundancy-checked,
23-predictor feature set, with a Streamlit prototype for real-time churn
risk scoring. Full methodology, findings, and discussion are in the
accompanying project report.

## Key Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 75.30% | 52.53% | 72.19% | 60.81% | 0.833 |
| Decision Tree | 73.74% | 50.35% | **77.01%** | 60.89% | 0.823 |
| **Random Forest** | **75.87%** | **53.32%** | 72.99% | **61.63%** | **0.839** |
| XGBoost | 75.44% | 52.71% | 72.73% | 61.12% | 0.831 |

Two engineered features — `ContractRiskScore` and `ChargesToTenureRatio` —
outrank every raw column in feature importance for both Random Forest and
XGBoost. See the report for the full four-round multicollinearity audit
behind the final feature set, and the eight-feature engineering audit
(five features tested, three rejected with evidence).

## Repository Structure

```
codebase/
├── Telco_Cusomer_Churn.csv    # Raw dataset (IBM Sample Data Sets, via Kaggle)
├── data_prep.py               # Cleaning, feature engineering, encoding
├── eda.py                     # Exploratory analysis, outlier/VIF checks, feature audit
├── modelling.py                # Train/test split, scaling, SMOTE, model training
├── evaluation.py               # Metrics, plots, feature importance, threshold tuning
├── analysis_notebook.ipynb     # Full analysis, calling the modules above (Sections 2-5)
└── streamlit/
    ├── app.py                  # Streamlit churn prediction prototype
    ├── train_model.py          # Trains and saves the deployed model
    └── requirement.txt         # Streamlit app dependencies
```

## Setup

```bash
git clone https://github.com/<your-username>/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction/codebase
pip install -r requirements.txt
```

## Running the Analysis

```bash
jupyter notebook analysis_notebook.ipynb
```

Runs the full pipeline — data cleaning, feature engineering, EDA, the
four-round multicollinearity audit, model training with GridSearchCV, and
all evaluation plots — using the functions in `data_prep.py`, `eda.py`,
`modelling.py`, and `evaluation.py`.

## Running the Prototype

```bash
cd streamlit
pip install -r requirement.txt
python train_model.py      # trains the final model, saves model.pkl / scaler.pkl / feature_columns.pkl
streamlit run app.py       # launches the web app at localhost:8501
```

Enter a customer's profile (contract type, tenure, charges, services, etc.)
to get a churn probability and risk category (Low / Medium / High).

## Methodology Summary

- **Data preparation**: logic-based imputation for `TotalCharges` (not
  statistical), outlier detection via mean ± 3σ, and a four-round
  Variance Inflation Factor audit that reduced the feature set from an
  initial worst-case VIF of 2,346 down to ~20.
- **Feature engineering**: 8 features engineered from the original 21
  columns; each checked directly against the data rather than assumed
  useful — 2 kept for modelling, 3 kept for description only, 3 rejected
  outright with evidence.
- **Class imbalance**: SMOTE applied to the training set only, after the
  train-test split, and not combined with algorithmic class weighting
  (which would double-correct).
- **Models**: Logistic Regression (baseline), Decision Tree, Random
  Forest, XGBoost — each regularised via GridSearchCV after an initial
  unconstrained Random Forest was found to overfit severely (train F1
  0.999 vs test F1 0.584).
- **Evaluation**: Accuracy/Precision/Recall/F1/AUC, 5-fold stratified
  cross-validation, Precision-Recall curves, Logistic Regression odds
  ratios, Decision Tree structure, and classification-threshold tuning.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, imbalanced-learn, statsmodels,
matplotlib, seaborn, Streamlit.

## Team

| Name | Student ID |
|---|---|
| [PLACEHOLDER] | [PLACEHOLDER] |
| [PLACEHOLDER] | [PLACEHOLDER] |
| [PLACEHOLDER] | [PLACEHOLDER] |
| [PLACEHOLDER] | [PLACEHOLDER] |

BMDS2003 Data Science — Tunku Abdul Rahman University of Management and Technology
