"""
eda_plots.py — Exploratory-data-analysis visualisations.

Companion to eda.py (which returns tables): every function here returns a
matplotlib Figure so the notebook controls display, and save_all_eda_figures()
writes the report's Figures 1-5 plus the supporting charts to reports/figures/.

Report cross-reference:
  Figure 1  -> plot_target_distribution           (Section 2.5)
  Figure 2  -> plot_churn_rate_by_category(Contract)(Section 2.5)
  Figure 3  -> plot_churn_rate_by_tenuregroup      (Section 2.6.1)
  Figure 4  -> plot_correlation_with_target        (Section 2.6.1)
  Figure 5  -> plot_interaction_heatmap(Contract x IsAutoPay) (Section 2.7)
Supporting charts (Section 2.5 / 3.2, for the Graphing criterion):
  plot_churn_rate_small_multiples, plot_numeric_distributions_by_churn,
  plot_numeric_boxplots_by_churn, plot_correlation_heatmap.

Colours match evaluation.py so the report reads as one visual system.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Shared palette (same navy/blue/red family as evaluation.py)
PALETTE = ["#1F4E79", "#2E75B6", "#9DC3E6", "#C00000"]
RETAIN_COLOR = "#2E75B6"   # No churn
CHURN_COLOR = "#C00000"    # Churn
SEQ_CMAP = "Blues"

# Default column groups for this dataset
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "ChargesToTenureRatio"]
CORR_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "ChargesToTenureRatio",
             "ContractRiskScore", "TotalServicesSubscribed", "SeniorCitizen"]
CATEGORICAL_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents",
                    "InternetService", "Contract", "PaperlessBilling",
                    "PaymentMethod", "TechSupport", "OnlineSecurity"]
# Fuller categorical set for the chi-square / Cramér's V association test (Section 2.8)
CHI_CAT_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "PaperlessBilling", "PaymentMethod"]


def _churn_binary(df: pd.DataFrame, target_col: str) -> pd.Series:
    """0/1 churn series regardless of whether the column is Yes/No or already 0/1.

    Handles pandas>=3.0, where string columns default to a StringDtype rather than
    object, so a plain `dtype == object` test is not reliable here.
    """
    s = df[target_col]
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    return (s == "Yes").astype(int)


def _churn_rate_by(df: pd.DataFrame, col: str, target_col: str) -> pd.Series:
    """Churn rate (%) per category of `col`, index-ordered by the column's own order."""
    tmp = df.copy()
    tmp["_c"] = _churn_binary(tmp, target_col)
    return tmp.groupby(col, observed=True)["_c"].mean().mul(100)


# ----------------------------------------------------------------------
# Figure 1 — target distribution
# ----------------------------------------------------------------------
def plot_target_distribution(df: pd.DataFrame, target_col: str = "Churn"):
    """Report Figure 1 (Section 2.5) — class balance of the target variable."""
    counts = df[target_col].value_counts()
    total = counts.sum()
    order = [c for c in ["No", "Yes"] if c in counts.index] or list(counts.index)
    counts = counts.reindex(order)
    colors = [RETAIN_COLOR if c == "No" else CHURN_COLOR for c in order]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar([{"No": "Retained (No)", "Yes": "Churned (Yes)"}.get(c, c) for c in order],
                  counts.values, color=colors)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + total * 0.01,
                f"{v:,}\n({v / total * 100:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of customers")
    ax.set_title("Figure 1 — Target Distribution (Churn)")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Figure 2 (generic) — churn rate by a single categorical
# ----------------------------------------------------------------------
def plot_churn_rate_by_category(df: pd.DataFrame, col: str, target_col: str = "Churn",
                                 order=None, title: str = None):
    """Report Figure 2 (Section 2.5) when col='Contract'; reusable for any categorical."""
    rates = _churn_rate_by(df, col, target_col)
    if order is not None:
        rates = rates.reindex([o for o in order if o in rates.index])
    rates = rates.sort_values(ascending=True) if order is None else rates

    fig, ax = plt.subplots(figsize=(7, 4.5))
    norm = rates.values / max(rates.values.max(), 1e-9)
    cmap = plt.get_cmap("coolwarm")
    colors = [cmap(0.15 + 0.7 * n) for n in norm]
    bars = ax.barh([str(i) for i in rates.index], rates.values, color=colors)
    for b, v in zip(bars, rates.values):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Churn rate (%)")
    ax.set_title(title or f"Churn Rate by {col}")
    ax.set_xlim(0, min(rates.values.max() * 1.18, 100))
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Figure 3 — churn rate by TenureGroup
# ----------------------------------------------------------------------
def plot_churn_rate_by_tenuregroup(df: pd.DataFrame, target_col: str = "Churn"):
    """Report Figure 3 (Section 2.6.1) — churn falling across tenure lifecycle stages."""
    order = ["0-12", "13-24", "25-48", "49-60", "61+"]
    rates = _churn_rate_by(df, "TenureGroup", target_col).reindex(
        [o for o in order if o in df["TenureGroup"].astype(str).unique()])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(rates.index.astype(str), rates.values, color=PALETTE[1])
    for b, v in zip(bars, rates.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9)
    ax.plot(range(len(rates)), rates.values, color=CHURN_COLOR, marker="o", linewidth=1.5)
    ax.set_xlabel("Tenure group (months)")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Figure 3 — Churn Rate by Tenure Group (descriptive)")
    ax.set_ylim(0, rates.values.max() * 1.18)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Figure 4 — correlation of numeric/engineered features with churn
# ----------------------------------------------------------------------
def plot_correlation_with_target(df: pd.DataFrame, numeric_cols: list = None,
                                  target_col: str = "Churn"):
    """Report Figure 4 (Section 2.6.1) — signed correlation of each numeric feature with churn."""
    numeric_cols = numeric_cols or [c for c in CORR_COLS if c in df.columns]
    tmp = df.copy()
    tmp["_c"] = _churn_binary(tmp, target_col)
    corr = tmp[numeric_cols + ["_c"]].corr()["_c"].drop("_c").sort_values()

    fig, ax = plt.subplots(figsize=(7, 4.8))
    colors = [CHURN_COLOR if v > 0 else RETAIN_COLOR for v in corr.values]
    bars = ax.barh(corr.index, corr.values, color=colors)
    for b, v in zip(bars, corr.values):
        ax.text(v + (0.01 if v >= 0 else -0.01), b.get_y() + b.get_height() / 2,
                f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlation with churn (point-biserial)")
    ax.set_title("Figure 4 — Feature Correlation with Churn")
    pad = max(abs(corr.min()), abs(corr.max())) * 0.25
    ax.set_xlim(corr.min() - pad, corr.max() + pad)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Figure 5 (generic) — two-way interaction heatmap of churn rate
# ----------------------------------------------------------------------
def plot_interaction_heatmap(df: pd.DataFrame, row_col: str, col_col: str,
                             target_col: str = "Churn", row_order=None, col_order=None,
                             title: str = None):
    """
    Report Figure 5 (Section 2.7) with row_col='Contract', col_col='IsAutoPay'.
    Reusable for the TenureGroup x services interaction in the same section.
    """
    tmp = df.copy()
    tmp["_c"] = _churn_binary(tmp, target_col)
    pivot = tmp.pivot_table(index=row_col, columns=col_col, values="_c", aggfunc="mean", observed=True) * 100
    if row_order is not None:
        pivot = pivot.reindex([r for r in row_order if r in pivot.index])
    if col_order is not None:
        pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(max(5.5, 1.4 * pivot.shape[1] + 3), 0.7 * pivot.shape[0] + 2.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", ax=ax,
                cbar_kws={"label": "Churn rate (%)"}, linewidths=0.5, linecolor="white")
    ax.set_title(title or f"Churn Rate (%) by {row_col} x {col_col}")
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Section 2.7 (second interaction) — TenureGroup x binned service count
# ----------------------------------------------------------------------
def plot_tenure_service_interaction(df: pd.DataFrame, target_col: str = "Churn"):
    """
    Report Section 2.7 — resolves the non-monotonic TotalServicesSubscribed pattern
    (Section 2.6.2) by holding tenure fixed. Within every tenure band churn rises
    with the number of services; the raw "more services = less churn" is a tenure
    artefact (long-tenured customers hold the most services and churn least).
    """
    d = df.copy()
    d["ServiceBin"] = pd.cut(d["TotalServicesSubscribed"], bins=[-1, 1, 3, 5, 8],
                             labels=["0-1", "2-3", "4-5", "6-8"])
    return plot_interaction_heatmap(
        d, "TenureGroup", "ServiceBin", target_col=target_col,
        row_order=["0-12", "13-24", "25-48", "49-60", "61+"],
        col_order=["0-1", "2-3", "4-5", "6-8"],
        title="Churn Rate (%) by Tenure Group x Number of Services")


# ----------------------------------------------------------------------
# Supporting — categorical churn small-multiples
# ----------------------------------------------------------------------
def plot_churn_rate_small_multiples(df: pd.DataFrame, cols: list = None,
                                    target_col: str = "Churn", overall_line: bool = True):
    """Section 2.5 — churn rate across many categoricals at once (Graphing criterion)."""
    cols = cols or [c for c in CATEGORICAL_COLS if c in df.columns]
    overall = _churn_binary(df, target_col).mean() * 100
    n = len(cols)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.0 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        rates = _churn_rate_by(df, col, target_col).sort_values()
        ax.bar([str(i) for i in rates.index], rates.values, color=PALETTE[1])
        if overall_line:
            ax.axhline(overall, color=CHURN_COLOR, linestyle="--", linewidth=1,
                       label=f"overall {overall:.1f}%")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("Churn %", fontsize=8)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Churn Rate by Categorical Feature (dashed = overall rate)", y=1.005, fontsize=12)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Supporting — numeric distributions split by churn
# ----------------------------------------------------------------------
def plot_numeric_distributions_by_churn(df: pd.DataFrame, cols: list = None,
                                        target_col: str = "Churn"):
    """Section 2.5 — KDE of each numeric feature, split by churn class."""
    cols = cols or [c for c in NUMERIC_COLS if c in df.columns]
    churn = _churn_binary(df, target_col)
    n = len(cols)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        for label, color in [(0, RETAIN_COLOR), (1, CHURN_COLOR)]:
            sns.kdeplot(df.loc[churn == label, col], ax=ax, fill=True, alpha=0.35,
                        color=color, label="Churn" if label else "No churn", warn_singular=False)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.legend(fontsize=8)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Numeric Feature Distributions by Churn Class", y=1.005, fontsize=12)
    plt.tight_layout()
    return fig


def plot_numeric_boxplots_by_churn(df: pd.DataFrame, cols: list = None,
                                   target_col: str = "Churn"):
    """Section 3.2 — boxplots of each numeric feature by churn class (spread + outliers)."""
    cols = cols or [c for c in NUMERIC_COLS if c in df.columns]
    tmp = df.copy()
    tmp["_Churn"] = np.where(_churn_binary(tmp, target_col) == 1, "Churn", "No churn")
    n = len(cols)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        sns.boxplot(data=tmp, x="_Churn", y=col, ax=ax, hue="_Churn", legend=False,
                    order=["No churn", "Churn"], palette=[RETAIN_COLOR, CHURN_COLOR])
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Numeric Feature Boxplots by Churn Class", y=1.005, fontsize=12)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Supporting — correlation heatmap among numeric features
# ----------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list = None,
                             target_col: str = "Churn"):
    """Section 2.6 — correlation heatmap among numeric features (incl. binary churn)."""
    numeric_cols = numeric_cols or [c for c in CORR_COLS if c in df.columns]
    tmp = df.copy()
    tmp["Churn"] = _churn_binary(tmp, target_col)
    corr = tmp[numeric_cols + ["Churn"]].corr()

    fig, ax = plt.subplots(figsize=(1.0 * len(corr) + 2, 0.9 * len(corr) + 1.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap — Numeric & Engineered Features")
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Section 2.8 — Cramér's V association strength (companion to eda.chi_square_tests)
# ----------------------------------------------------------------------
def plot_cramers_v(chi_df: pd.DataFrame, top_n: int = None):
    """
    Report Section 2.8 — horizontal bar of each categorical feature's Cramér's V
    with churn, coloured by effect-size band. `chi_df` is the output of
    eda.chi_square_tests(). Makes the "significant but weak vs strong" distinction
    visible at a glance.
    """
    from matplotlib.patches import Patch
    d = chi_df.sort_values("cramers_v")
    if top_n:
        d = d.tail(top_n)
    band_color = {"negligible": "#9DC3E6", "weak": "#2E75B6",
                  "moderate": "#1F4E79", "strong": "#C00000"}
    colors = [band_color.get(b, "#2E75B6") for b in d["strength"]]

    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(d) + 1.6))
    bars = ax.barh(d["feature"], d["cramers_v"], color=colors)
    for b, v in zip(bars, d["cramers_v"]):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Cramér's V (association strength with churn)")
    ax.set_title("Categorical Feature Association with Churn (Cramér's V)")
    ax.set_xlim(0, max(d["cramers_v"].max() * 1.18, 0.05))
    handles = [Patch(color=c, label=l) for l, c in band_color.items()]
    ax.legend(handles=handles, fontsize=7, loc="lower right", title="strength")
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Batch export for the report
# ----------------------------------------------------------------------
def save_all_eda_figures(df_eda: pd.DataFrame, out_dir: str, dpi: int = 150) -> list:
    """
    Writes every report figure + supporting chart to out_dir as PNGs and returns
    the list of paths written. Called by the notebook and by a standalone export.
    """
    os.makedirs(out_dir, exist_ok=True)
    jobs = [
        ("fig01_target_distribution", plot_target_distribution(df_eda)),
        ("fig02_churn_by_contract",
         plot_churn_rate_by_category(df_eda, "Contract",
                                     order=["Month-to-month", "One year", "Two year"],
                                     title="Figure 2 — Churn Rate by Contract Type")),
        ("fig03_churn_by_tenuregroup", plot_churn_rate_by_tenuregroup(df_eda)),
        ("fig04_correlation_with_churn", plot_correlation_with_target(df_eda)),
        ("fig05_contract_x_autopay",
         plot_interaction_heatmap(df_eda, "Contract", "IsAutoPay",
                                  row_order=["Month-to-month", "One year", "Two year"],
                                  title="Figure 5 — Churn Rate (%) by Contract x Payment Automation")),
        ("fig05b_tenure_x_services", plot_tenure_service_interaction(df_eda)),
        ("supp_churn_small_multiples", plot_churn_rate_small_multiples(df_eda)),
        ("supp_numeric_distributions", plot_numeric_distributions_by_churn(df_eda)),
        ("supp_numeric_boxplots", plot_numeric_boxplots_by_churn(df_eda)),
        ("supp_correlation_heatmap", plot_correlation_heatmap(df_eda)),
    ]
    # Section 2.8 — Cramér's V figure needs the chi-square table; import eda lazily
    # so this module stays usable for plotting alone.
    try:
        import eda as _eda
        chi_df = _eda.chi_square_tests(df_eda, [c for c in CHI_CAT_COLS if c in df_eda.columns])
        jobs.append(("fig06_cramers_v_association", plot_cramers_v(chi_df)))
    except Exception as exc:  # pragma: no cover - export is best-effort
        print(f"[eda_plots] skipped Cramér's V figure: {exc}")

    paths = []
    for name, fig in jobs:
        p = os.path.join(out_dir, name + ".png")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths
