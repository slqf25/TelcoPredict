"""
train_model.py
Trains all four churn models (Logistic Regression, Decision Tree, Random Forest,
XGBoost) on the cleaned, redundancy-checked 23-predictor feature set, and
saves everything the Streamlit app needs to make predictions on new customers:
  - models.pkl       : dict of all four trained models; Random Forest is the
                       recommended default (report Section 5.9), the rest are
                       selectable in the app for comparison
  - scaler.pkl        : fitted StandardScaler (for tenure, MonthlyCharges, TotalCharges, ChargesToTenureRatio)
  - feature_columns.pkl : exact column order the models expect (after one-hot encoding)

Run this once (`python train_model.py`) before running the Streamlit app.
Expects the raw dataset at ../data/Telco_Cusomer_Churn.csv.
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

RAW_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "Telco_Cusomer_Churn.csv"
)

# Columns where the "No internet service" / "No phone service" category is
# collapsed into "No", since that information is already captured separately
# by InternetService / PhoneService (see report Section 3.3.3).
SERVICE_COLS_WITH_NO_INTERNET = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUM_COLS_TO_SCALE = ["tenure", "MonthlyCharges", "TotalCharges", "ChargesToTenureRatio"]


def engineer_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the exact feature engineering + cleaning pipeline used in the report."""
    df = df.copy()

    # --- Data cleaning (Section 3.1) ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0)
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # --- Engineered features that survived the audit (Section 2.6) ---
    contract_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["ContractRiskScore"] = df["Contract"].map(contract_map)
    df["ChargesToTenureRatio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # --- Collapse redundant categories (Section 3.3.3) ---
    for c in SERVICE_COLS_WITH_NO_INTERNET:
        df[c] = df[c].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes the cleaned dataframe into the final 23-predictor modelling matrix."""
    df = df.drop(columns=["customerID", "Contract"], errors="ignore")

    X = df.drop(columns=["Churn"], errors="ignore")

    # Binary label encoding
    X["gender"] = (X["gender"] == "Male").astype(int)
    for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling",
              "MultipleLines"] + SERVICE_COLS_WITH_NO_INTERNET:
        X[c] = (X[c] == "Yes").astype(int)

    # One-hot encoding (drop_first=True to avoid multicollinearity, Section 3.5).
    # Categories are fixed explicitly before encoding: pd.get_dummies only creates
    # columns for categories actually PRESENT in the data, so a single-row
    # prediction (only one InternetService/PaymentMethod value present) would
    # otherwise produce zero dummy columns for that feature — silently making
    # the model ignore it regardless of which option was selected.
    X["InternetService"] = pd.Categorical(X["InternetService"], categories=["DSL", "Fiber optic", "No"])
    X["PaymentMethod"] = pd.Categorical(X["PaymentMethod"], categories=[
        "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
    X = pd.get_dummies(X, columns=["InternetService", "PaymentMethod"], drop_first=True)

    return X


def main():
    df_raw = pd.read_csv(RAW_CSV_PATH)
    df_clean = engineer_and_clean(df_raw)

    y = (df_clean["Churn"] == "Yes").astype(int)
    X = build_feature_matrix(df_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[NUM_COLS_TO_SCALE] = scaler.fit_transform(X_train[NUM_COLS_TO_SCALE])

    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train)

    # Train all four models (report Section 4) with the deployed configurations,
    # so the prototype can expose every model, not only the recommended one.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=10, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42),
        "XGBoost": XGBClassifier(
            max_depth=4, learning_rate=0.1, n_estimators=150, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1,
            random_state=42, eval_metric="logloss"),
    }
    for m in models.values():
        m.fit(X_train_sm, y_train_sm)

    # Quick sanity check on the held-out test set
    X_test_scaled = X_test.copy()
    X_test_scaled[NUM_COLS_TO_SCALE] = scaler.transform(X_test[NUM_COLS_TO_SCALE])
    print(f"Feature count: {X_train.shape[1]}")
    for name, m in models.items():
        print(f"  {name:20} test accuracy: {m.score(X_test_scaled, y_test):.4f}")

    # Save everything the app needs. models.pkl is a dict of all four trained
    # models; Random Forest is the recommended default (report Section 5.9).
    with open("models.pkl", "wb") as f:
        pickle.dump(models, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)

    print("Saved models.pkl, scaler.pkl, feature_columns.pkl")


if __name__ == "__main__":
    main()