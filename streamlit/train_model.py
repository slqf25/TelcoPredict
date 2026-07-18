"""
train_model.py
Trains the final Random Forest churn model (the recommended model from Section 5.9
of the report) on the cleaned, redundancy-checked 23-predictor feature set, and
saves everything the Streamlit app needs to make predictions on new customers:
  - model.pkl        : trained RandomForestClassifier
  - scaler.pkl        : fitted StandardScaler (for tenure, MonthlyCharges, TotalCharges, ChargesToTenureRatio)
  - feature_columns.pkl : exact column order the model expects (after one-hot encoding)

Run this once (`python train_model.py`) before running the Streamlit app.
Expects the raw dataset at ./Telco_Cusomer_Churn.csv (same filename as provided).
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

RAW_CSV_PATH = "Telco_Cusomer_Churn.csv"

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

    # One-hot encoding (drop_first=True to avoid multicollinearity, Section 3.5)
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

    # Final tuned Random Forest configuration (report Section 4.2 / 5.2)
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42
    )
    model.fit(X_train_sm, y_train_sm)

    # Quick sanity check on the held-out test set
    X_test_scaled = X_test.copy()
    X_test_scaled[NUM_COLS_TO_SCALE] = scaler.transform(X_test[NUM_COLS_TO_SCALE])
    test_acc = model.score(X_test_scaled, y_test)
    print(f"Model trained. Test accuracy: {test_acc:.4f}")
    print(f"Feature count: {X_train.shape[1]}")

    # Save everything the app needs
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)

    print("Saved model.pkl, scaler.pkl, feature_columns.pkl")


if __name__ == "__main__":
    main()