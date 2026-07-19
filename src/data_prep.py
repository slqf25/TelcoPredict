"""
data_prep.py — Data loading, cleaning, feature engineering, and encoding.
Corresponds to report Sections 2.1-2.6 (feature engineering) and Section 3
(Data Preparation).

All functions are pure (take a DataFrame in, return a new DataFrame/array out)
so they can be unit-tested and reused by both the analysis notebook and the
Streamlit prototype (see streamlit/train_model.py, which mirrors the
cleaning + build_feature_matrix logic here to keep training and inference
consistent).
"""

import os
import pandas as pd
import numpy as np

# Resolve the dataset relative to this file, so imports work regardless of the
# caller's working directory (notebook in notebooks/, ad-hoc scripts, etc.).
RAW_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "Telco_Cusomer_Churn.csv"
)

# Columns where the "No internet service" / "No phone service" category is
# collapsed into "No", since that information is already captured separately
# by InternetService / PhoneService (report Section 3.3.3).
SERVICE_COLS_WITH_NO_INTERNET = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUM_COLS_TO_SCALE = ["tenure", "MonthlyCharges", "TotalCharges", "ChargesToTenureRatio"]

# The 8 originally engineered features (report Section 2.6). Kept here for
# EDA use; only ContractRiskScore and ChargesToTenureRatio survive into
# build_feature_matrix() for modelling.
ALL_ENGINEERED_FEATURES = [
    "TenureGroup", "TotalServicesSubscribed", "AvgChargePerMonth",
    "HasInternetService", "HasPartnerOrDependents", "IsAutoPay",
    "ContractRiskScore", "ChargesToTenureRatio",
]


def load_data(path: str = RAW_CSV_PATH) -> pd.DataFrame:
    """Loads the raw Telco Customer Churn CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report Section 3.1 — Handling Missing Values.
    TotalCharges is stored as text with 11 blank entries, all belonging to
    customers with tenure = 0. These are imputed as 0 (not median/mean),
    since a brand-new customer has genuinely not been billed yet.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0)
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report Section 2.6 — builds all 8 originally engineered features for the
    EDA audit. Three of these (AvgChargePerMonth, HasInternetService,
    HasPartnerOrDependents) are rejected after validation and are not used
    downstream except to demonstrate why they were rejected (see eda.py:
    validate_engineered_features).
    """
    df = df.copy()

    # TenureGroup — EDA-only (Section 2.6, kept out of modelling per 3.3.4)
    bins = [-1, 12, 24, 48, 60, 200]
    labels = ["0-12", "13-24", "25-48", "49-60", "61+"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # TotalServicesSubscribed — EDA-only (perfectly collinear with the 8
    # service one-hot columns once encoded; Section 3.3.2)
    service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                     "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["TotalServicesSubscribed"] = sum((df[c] == "Yes").astype(int) for c in service_cols)

    # AvgChargePerMonth — REJECTED (r=1.00 with MonthlyCharges, Section 3.3.1)
    df["AvgChargePerMonth"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"])

    # HasInternetService — REJECTED (hides Fiber-vs-DSL signal, Section 2.6)
    df["HasInternetService"] = (df["InternetService"] != "No").astype(int)

    # HasPartnerOrDependents — REJECTED (adds nothing beyond Partner alone, Section 2.6)
    df["HasPartnerOrDependents"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)

    # IsAutoPay — EDA-only (drives the Section 2.7 interaction finding)
    df["IsAutoPay"] = df["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)

    # ContractRiskScore — KEPT for modelling (Section 2.6.1)
    contract_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["ContractRiskScore"] = df["Contract"].map(contract_map)

    # ChargesToTenureRatio — KEPT for modelling (Section 2.6.1, strongest predictor)
    df["ChargesToTenureRatio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    return df


def collapse_redundant_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report Section 3.3.3 — 'No internet service' / 'No phone service' are
    recorded identically across multiple columns for the same customers as
    InternetService='No' / PhoneService='No'. Collapsing them into 'No'
    removes several perfectly duplicated one-hot columns without losing
    any information.
    """
    df = df.copy()
    for c in SERVICE_COLS_WITH_NO_INTERNET:
        df[c] = df[c].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report Section 3.4-3.5 — encodes the cleaned dataframe into the final
    23-predictor modelling matrix. Assumes clean_data(), engineer_all_features(),
    and collapse_redundant_categories() have already been applied.

    Excluded from modelling, per the Section 2.6 audit and Section 3.3
    multicollinearity rounds:
      - customerID (identifier, data-leakage risk, Section 3.4)
      - Contract (replaced by ContractRiskScore, Section 3.3.3)
      - TenureGroup (redundant with continuous tenure, Section 3.3.4)
      - TotalServicesSubscribed (exact linear duplicate of service dummies, Section 3.3.2)
      - AvgChargePerMonth, HasInternetService, HasPartnerOrDependents, IsAutoPay
        (rejected or redundant, Section 2.6 / 3.3.1 / 3.3.3)
    """
    df = df.drop(columns=[
        "customerID", "Contract", "TenureGroup", "TotalServicesSubscribed",
        "AvgChargePerMonth", "HasInternetService", "HasPartnerOrDependents", "IsAutoPay",
    ], errors="ignore")

    X = df.drop(columns=["Churn"], errors="ignore")

    X["gender"] = (X["gender"] == "Male").astype(int)
    for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling",
              "MultipleLines"] + SERVICE_COLS_WITH_NO_INTERNET:
        X[c] = (X[c] == "Yes").astype(int)

    X = pd.get_dummies(X, columns=["InternetService", "PaymentMethod"], drop_first=True)
    return X


def get_target(df: pd.DataFrame) -> pd.Series:
    """Returns the binary target: 1 = Churn, 0 = No churn."""
    return (df["Churn"] == "Yes").astype(int)


def full_pipeline(path: str = RAW_CSV_PATH):
    """
    Convenience function running the entire Section 2-3 pipeline in one call.
    Returns (df_eda, X, y) where df_eda retains ALL engineered features
    (for EDA/reporting) and X is the final 23-predictor modelling matrix.
    """
    df_raw = load_data(path)
    df_clean = clean_data(df_raw)
    df_eda = engineer_all_features(df_clean)
    df_eda = collapse_redundant_categories(df_eda)
    y = get_target(df_eda)
    X = build_feature_matrix(df_eda)
    return df_eda, X, y