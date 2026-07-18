"""
eda.py — Exploratory data analysis functions.
Corresponds to report Section 2 (Data Understanding), Section 2.6 (engineered
feature audit), and Section 3.2-3.3 (outlier and multicollinearity checks).

Each function returns data (DataFrame/Series/dict) rather than printing
directly, so the notebook controls how results are displayed.
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


def summary_statistics(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Report Section 2.3 — descriptive statistics for numeric columns."""
    return df[cols].describe().T


def target_distribution(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Report Section 2.5 / Figure 1 — class balance of the target variable."""
    counts = df[target_col].value_counts()
    pct = df[target_col].value_counts(normalize=True) * 100
    return pd.DataFrame({"count": counts, "pct": pct.round(2)})


def churn_rate_by_category(df: pd.DataFrame, category_col: str, target_col: str = "Churn") -> pd.DataFrame:
    """Report Section 2.5 — churn rate (%) broken down by a categorical column."""
    return (pd.crosstab(df[category_col], df[target_col], normalize="index") * 100).round(2)


def churn_rate_by_two_categories(df: pd.DataFrame, col1: str, col2: str, target_col: str = "Churn") -> pd.DataFrame:
    """Report Section 2.7 — multivariate interaction: churn rate by two categories at once."""
    return (pd.crosstab([df[col1], df[col2]], df[target_col], normalize="index") * 100).round(2)


def correlation_with_target(df: pd.DataFrame, numeric_cols: list, target_col: str = "Churn") -> pd.Series:
    """Report Section 2.6.1 / Figure 4 — correlation of numeric/engineered features with churn."""
    tmp = df.copy()
    tmp["_churn_bin"] = (tmp[target_col] == "Yes").astype(int)
    corr = tmp[numeric_cols + ["_churn_bin"]].corr()["_churn_bin"].drop("_churn_bin")
    return corr.sort_values(ascending=False)


def detect_outliers_zscore(df: pd.DataFrame, cols: list, n_std: float = 3.0) -> dict:
    """
    Report Section 3.2 — outlier detection via mean +/- n_std * std.
    Returns a dict per column: {mean, std, lower, upper, n_outliers, pct_outliers}.
    """
    results = {}
    for c in cols:
        m, s = df[c].mean(), df[c].std()
        lower, upper = m - n_std * s, m + n_std * s
        mask = (df[c] < lower) | (df[c] > upper)
        results[c] = {
            "mean": round(m, 2), "std": round(s, 2),
            "lower": round(lower, 2), "upper": round(upper, 2),
            "n_outliers": int(mask.sum()), "pct_outliers": round(mask.mean() * 100, 2),
        }
    return results


def profile_outlier_group(df: pd.DataFrame, col: str, n_std: float = 3.0,
                           profile_cols: list = None, target_col: str = "Churn") -> pd.DataFrame:
    """
    Report Section 3.2 — compares the outlier group on `col` against the full
    dataset on `profile_cols` (e.g. mean tenure, churn rate), to check whether
    outliers are noise or a genuine signal.
    """
    m, s = df[col].mean(), df[col].std()
    lower, upper = m - n_std * s, m + n_std * s
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    profile_cols = profile_cols or ["tenure"]

    rows = []
    for label, subset in [("Outliers", outliers), ("Full dataset", df)]:
        row = {"group": label, "n": len(subset)}
        for pc in profile_cols:
            row[f"mean_{pc}"] = round(subset[pc].mean(), 2)
        row["churn_rate_%"] = round((subset[target_col] == "Yes").mean() * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def check_invalid_values(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Report Section 3.2 — boolean-filter check for negative/invalid numeric values."""
    rows = []
    for c in cols:
        n_invalid = (df[c] < 0).sum()
        rows.append({"column": c, "n_negative_values": int(n_invalid), "min_value": df[c].min()})
    return pd.DataFrame(rows)


def check_duplicate_rows(df: pd.DataFrame, id_col: str = "customerID") -> dict:
    """Checks for duplicate rows (excluding the ID column) and duplicate IDs."""
    return {
        "duplicate_rows_excl_id": int(df.drop(columns=[id_col]).duplicated().sum()),
        "duplicate_ids": int(df[id_col].duplicated().sum()),
    }


def compute_vif(X: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Report Section 3.3 — Variance Inflation Factor for multicollinearity
    detection. If `cols` is None, computes VIF across all columns in X.
    """
    cols = cols or list(X.columns)
    Xv = X[cols].astype(float)
    vifs = []
    for i, c in enumerate(cols):
        try:
            v = variance_inflation_factor(Xv.values, i)
        except Exception:
            v = np.nan
        vifs.append(v)
    return pd.DataFrame({"feature": cols, "VIF": vifs}).sort_values("VIF", ascending=False)


def validate_engineered_features(df: pd.DataFrame, target_col: str = "Churn") -> dict:
    """
    Report Section 2.6 — the direct evidence checks behind the honest
    engineered-feature audit: does each rejected feature actually add
    information beyond the raw column(s) it was built from?
    """
    results = {}

    # HasInternetService vs InternetService
    results["HasInternetService_vs_InternetService"] = {
        "InternetService (raw)": churn_rate_by_category(df, "InternetService", target_col)["Yes"].to_dict(),
        "HasInternetService (engineered)": churn_rate_by_category(df, "HasInternetService", target_col)["Yes"].to_dict(),
    }

    # HasPartnerOrDependents vs Partner alone
    results["HasPartnerOrDependents_vs_Partner"] = {
        "Partner (raw)": churn_rate_by_category(df, "Partner", target_col)["Yes"].to_dict(),
        "HasPartnerOrDependents (engineered)": churn_rate_by_category(df, "HasPartnerOrDependents", target_col)["Yes"].to_dict(),
    }

    # AvgChargePerMonth vs MonthlyCharges — correlation check
    results["AvgChargePerMonth_vs_MonthlyCharges_corr"] = round(
        df["AvgChargePerMonth"].corr(df["MonthlyCharges"]), 6
    )

    # TotalServicesSubscribed vs manual recount of the 8 service one-hot flags
    service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                     "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    manual_count = sum((df[c] == "Yes").astype(int) for c in service_cols)
    results["TotalServicesSubscribed_vs_manual_sum_corr"] = round(
        df["TotalServicesSubscribed"].corr(manual_count), 6
    )

    return results