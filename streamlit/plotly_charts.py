"""
plotly_charts.py — Interactive Plotly versions of six of the report's charts, for the
Streamlit prototype only.

src/eda_plots.py and src/evaluation.py stay matplotlib-only — they feed the notebook
and the report's static PNG exports (already graded/embedded), so they are NOT touched.
This module is additive and app-only: it reuses the exact same underlying tables
(eda.py / evaluation.py compute functions, or plain sklearn metric calls with the
deployed models) so the numbers are identical to the report; only the rendering layer
differs, trading static PNGs for hover/zoom + smooth transitions.

Charts covered here: correlation heatmap, VIF bars, Cramer's V bars, model comparison
bars, ROC curves, PR curves, threshold-tuning curve. Decision tree structure and the
categorical small-multiples grid are deliberately left as matplotlib (no good Plotly
equivalent for plot_tree; small-multiples gain little from animation).

Every figure sets `transition` so Streamlit's Plotly.react re-render animates smoothly
when the underlying data changes (switching the model picker, moving a slider).
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)

NAVY, BLUE, LIGHT, RED = "#1F4E79", "#2E75B6", "#9DC3E6", "#C00000"
GREEN, AMBER = "#2E8B57", "#E8A317"
PALETTE = [NAVY, BLUE, LIGHT, RED]
STRENGTH_COLOR = {"negligible": LIGHT, "weak": BLUE, "moderate": NAVY, "strong": RED}

TRANSITION = dict(duration=500, easing="cubic-in-out")


def _churn01(df, target_col="Churn"):
    """0/1 churn series (pandas>=3.0 stores string cols as StringDtype, not object)."""
    s = df[target_col]
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    return (s == "Yes").astype(int)


def _base_layout(fig, height=420, xaxis_title=None, yaxis_title=None, show_legend=True):
    """
    No internal Plotly `title` here on purpose: every call site in app.py already
    prints a markdown header (e.g. "**Figure 6 — ROC curves**") directly above the
    chart, so a second Plotly-native title would just duplicate it — and previously
    did, sitting in the exact same top-left band as the legend and overlapping it.
    The legend instead gets its own reserved strip via `margin.t`, sitting flush
    above the plotting area (y=1.0) rather than floating over it.
    """
    fig.update_layout(
        height=height, template="plotly_white",
        margin=dict(l=10, r=10, t=58 if show_legend else 34, b=34),
        transition=TRANSITION,
        font=dict(family="Segoe UI, sans-serif", size=12),
        hoverlabel=dict(bgcolor="white", font_size=12),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
    )
    if xaxis_title:
        fig.update_xaxes(title=xaxis_title)
    if yaxis_title:
        fig.update_yaxes(title=yaxis_title)
    return fig


def plot_correlation_heatmap_plotly(df_eda, numeric_cols, target_col="Churn"):
    """Section 2.6 — correlation heatmap among numeric/engineered features + churn."""
    tmp = df_eda.copy()
    s = tmp[target_col]
    tmp["Churn"] = s.astype(int) if pd.api.types.is_numeric_dtype(s) else (s == "Yes").astype(int)
    cols = list(numeric_cols) + ["Churn"]
    corr = tmp[cols].corr().round(3)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.columns),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}: %{z}<extra></extra>",
        colorbar=dict(title="r"),
    ))
    return _base_layout(fig, height=460, show_legend=False)


def plot_vif_plotly(vif_df):
    """Section 3.3 — Variance Inflation Factor per feature, hover for exact value."""
    d = vif_df.sort_values("VIF")
    colors = [RED if v > 10 else BLUE for v in d["VIF"]]
    fig = go.Figure(go.Bar(
        x=d["VIF"], y=d["feature"], orientation="h", marker_color=colors,
        text=d["VIF"].round(2), texttemplate="%{text}", textposition="outside",
        hovertemplate="%{y}: VIF %{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=10, line_dash="dash", line_color=RED,
                  annotation_text="concern threshold (10)", annotation_position="top")
    return _base_layout(fig, height=560, xaxis_title="VIF", show_legend=False)


def plot_cramers_v_plotly(chi_df):
    """Section 2.8 — Cramer's V association strength, coloured by effect-size band."""
    d = chi_df.sort_values("cramers_v")
    colors = [STRENGTH_COLOR.get(s, BLUE) for s in d["strength"]]
    fig = go.Figure(go.Bar(
        x=d["cramers_v"], y=d["feature"], orientation="h", marker_color=colors,
        text=d["cramers_v"].round(3), texttemplate="%{text}", textposition="outside",
        customdata=d["strength"],
        hovertemplate="%{y}: V=%{x:.3f} (%{customdata})<extra></extra>",
    ))
    return _base_layout(fig, height=520, xaxis_title="Cramer's V", show_legend=False)


def plot_model_comparison_plotly(results_df):
    """Section 5.2 — Accuracy/Precision/Recall/F1 grouped bars, all four models."""
    d = (results_df[["Accuracy", "Precision", "Recall", "F1"]] * 100).round(2)
    fig = go.Figure()
    for metric, color in zip(d.columns, PALETTE):
        fig.add_bar(name=metric, x=list(d.index), y=d[metric], marker_color=color,
                    text=d[metric], texttemplate="%{text}", textposition="outside",
                    hovertemplate="%{x} — " + metric + ": %{y:.2f}%<extra></extra>")
    fig.update_layout(barmode="group")
    return _base_layout(fig, height=440, yaxis_title="Score (%)")


def plot_roc_curves_plotly(models, X_test, y_test):
    """Section 5.2/5.6 — ROC curves, hover shows the exact threshold at each point."""
    fig = go.Figure()
    for (name, m), color in zip(models.items(), PALETTE):
        proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, thr = roc_curve(y_test, proba)
        a = auc(fpr, tpr)
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={a:.3f})",
                        line=dict(color=color, width=2.5), customdata=thr,
                        hovertemplate="FPR %{x:.3f}, TPR %{y:.3f}<br>thr≈%{customdata:.2f}"
                                      f"<extra>{name}</extra>")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="gray", dash="dash"),
                    name="Chance", hoverinfo="skip")
    return _base_layout(fig, height=460,
                        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")


def plot_pr_curves_plotly(models, X_test, y_test):
    """Section 5.6 — Precision-Recall curves, more informative than ROC on imbalanced data."""
    fig = go.Figure()
    for (name, m), color in zip(models.items(), PALETTE):
        proba = m.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        fig.add_scatter(x=rec, y=prec, mode="lines", name=f"{name} (AP={ap:.3f})",
                        line=dict(color=color, width=2.5),
                        hovertemplate=f"Recall %{{x:.3f}}, Precision %{{y:.3f}}<extra>{name}</extra>")
    baseline = float(np.mean(y_test))
    fig.add_hline(y=baseline, line_dash="dash", line_color="gray",
                  annotation_text=f"no-skill ({baseline:.3f})", annotation_position="top left")
    return _base_layout(fig, height=460, xaxis_title="Recall", yaxis_title="Precision")


def plot_threshold_curve_plotly(models, X_test, y_test, model_names_to_plot=None,
                                marker_threshold=None):
    """
    Section 5.7 — F1-score across classification thresholds. If `marker_threshold` is
    given (e.g. from a live slider), a solid red vertical line marks it, animating
    smoothly to its new position on rerun via the shared Plotly.react transition.
    """
    model_names_to_plot = model_names_to_plot or list(models.keys())
    thresholds = np.arange(0.1, 0.91, 0.02)
    fig = go.Figure()
    for name, color in zip(model_names_to_plot, PALETTE):
        proba = models[name].predict_proba(X_test)[:, 1]
        f1s = [f1_score(y_test, (proba >= t).astype(int), zero_division=0) for t in thresholds]
        fig.add_scatter(x=thresholds, y=f1s, mode="lines", name=name,
                        line=dict(color=color, width=2.5),
                        hovertemplate=f"thr=%{{x:.2f}}, F1=%{{y:.3f}}<extra>{name}</extra>")
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray",
                  annotation_text="default (0.5)", annotation_position="bottom",
                  annotation_yshift=-10, annotation_font_size=10)
    if marker_threshold is not None:
        fig.add_vline(x=marker_threshold, line_dash="solid", line_color=RED, line_width=2,
                      annotation_text=f"selected {marker_threshold:.2f}", annotation_position="top",
                      annotation_yshift=-6, annotation_font_size=10, annotation_font_color=RED)
    return _base_layout(fig, height=440,
                        xaxis_title="Classification Threshold", yaxis_title="F1-score")


# ------------------------------------------------------------------------
# Section 2 EDA charts — Plotly versions of the remaining matplotlib figures.
# Decision Tree structure stays matplotlib (no good Plotly equivalent for
# sklearn's plot_tree); everything else below replaces a show_fig(epl.*) or
# show_fig(ev.*) call in app.py with an interactive, hover-enabled version.
# ------------------------------------------------------------------------

def plot_target_distribution_plotly(df, target_col="Churn"):
    """Figure 1 — class balance of the target variable."""
    churn = _churn01(df, target_col)
    counts = churn.value_counts().reindex([0, 1])
    total = int(counts.sum())
    labels = ["Retained (No)", "Churned (Yes)"]
    colors = [BLUE, RED]
    fig = go.Figure(go.Bar(
        x=labels, y=counts.values, marker_color=colors,
        text=[f"{v:,} ({v/total*100:.1f}%)" for v in counts.values],
        textposition="outside",
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    return _base_layout(fig, height=420, yaxis_title="Number of customers", show_legend=False)


def plot_churn_rate_by_category_plotly(df, category_col, target_col="Churn", order=None):
    """Figure 2 / 3 — churn rate (%) by a single categorical column."""
    tmp = df.copy()
    tmp["_c"] = _churn01(tmp, target_col)
    rates = (tmp.groupby(category_col, observed=True)["_c"].mean() * 100)
    if order is not None:
        rates = rates.reindex([o for o in order if o in rates.index])
    else:
        rates = rates.sort_values()
    norm = rates.values / max(rates.values.max(), 1e-9)
    colors = [f"rgb({int(31+150*n)},{int(78+40*n)},{int(121-80*n)})" for n in norm]
    fig = go.Figure(go.Bar(
        x=[str(i) for i in rates.index], y=rates.values, marker_color=colors,
        text=rates.round(1), texttemplate="%{text}", textposition="outside",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    return _base_layout(fig, height=420, yaxis_title="Churn rate (%)", show_legend=False)


def plot_churn_rate_by_tenuregroup_plotly(df, target_col="Churn"):
    """Figure 3 — churn rate by TenureGroup, bars + connecting line (descriptive only)."""
    order = ["0-12", "13-24", "25-48", "49-60", "61+"]
    tmp = df.copy()
    tmp["_c"] = _churn01(tmp, target_col)
    rates = (tmp.groupby("TenureGroup", observed=True)["_c"].mean() * 100)
    rates = rates.reindex([o for o in order if o in rates.index])
    fig = go.Figure()
    fig.add_bar(x=[str(i) for i in rates.index], y=rates.values, marker_color=BLUE,
               text=rates.round(1), texttemplate="%{text}", textposition="outside",
               hovertemplate="%{x}: %{y:.1f}%<extra></extra>", showlegend=False)
    fig.add_scatter(x=[str(i) for i in rates.index], y=rates.values, mode="lines+markers",
                    line=dict(color=RED, width=2), marker=dict(size=6, color=RED),
                    hoverinfo="skip", showlegend=False)
    return _base_layout(fig, height=420, xaxis_title="Tenure group (months)",
                        yaxis_title="Churn rate (%)", show_legend=False)


def plot_correlation_with_target_plotly(df, numeric_cols, target_col="Churn"):
    """Figure 4 — signed correlation of numeric/engineered features with churn."""
    tmp = df.copy()
    tmp["_c"] = _churn01(tmp, target_col)
    corr = tmp[list(numeric_cols) + ["_c"]].corr()["_c"].drop("_c").sort_values()
    colors = [RED if v > 0 else BLUE for v in corr.values]
    # Negative bars get their value label INSIDE the bar (white text) instead of
    # "outside": outside-positioned text on a negative bar renders further left
    # than the bar tip, which for the longest negative bar collides with the
    # y-axis category labels sitting in the left margin (e.g. "tenure").
    text_positions = ["inside" if v < 0 else "outside" for v in corr.values]
    text_colors = ["white" if v < 0 else "#1a1a1a" for v in corr.values]
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in corr.values], textposition=text_positions,
        textfont=dict(color=text_colors),
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="black", line_width=1)
    return _base_layout(fig, height=420, xaxis_title="Correlation with churn (point-biserial)",
                        show_legend=False)


def plot_interaction_heatmap_plotly(df, row_col, col_col, target_col="Churn",
                                    row_order=None, col_order=None):
    """Figure 5 / 5b — two-way interaction heatmap of churn rate (%)."""
    tmp = df.copy()
    tmp["_c"] = _churn01(tmp, target_col)
    pivot = tmp.pivot_table("_c", row_col, col_col, aggfunc="mean", observed=True) * 100
    if row_order is not None:
        pivot = pivot.reindex([r for r in row_order if r in pivot.index])
    if col_order is not None:
        pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[str(c) for c in pivot.columns], y=[str(r) for r in pivot.index],
        colorscale="Reds", text=pivot.round(1).values, texttemplate="%{text}",
        hovertemplate=f"{row_col} %{{y}}, {col_col} %{{x}}: %{{z:.1f}}%<extra></extra>",
        colorbar=dict(title="Churn %"),
    ))
    return _base_layout(fig, height=max(320, 90 * len(pivot) + 80),
                        xaxis_title=col_col, yaxis_title=row_col, show_legend=False)


def plot_tenure_service_interaction_plotly(df, target_col="Churn"):
    """Figure 5b — resolves the non-monotonic TotalServicesSubscribed pattern by
    holding tenure fixed (see src/eda_plots.py:plot_tenure_service_interaction)."""
    d = df.copy()
    d["ServiceBin"] = pd.cut(d["TotalServicesSubscribed"], bins=[-1, 1, 3, 5, 8],
                             labels=["0-1", "2-3", "4-5", "6-8"])
    return plot_interaction_heatmap_plotly(
        d, "TenureGroup", "ServiceBin", target_col=target_col,
        row_order=["0-12", "13-24", "25-48", "49-60", "61+"],
        col_order=["0-1", "2-3", "4-5", "6-8"])


def _subplot_grid_layout(fig, n, ncols, row_height=280, has_legend=False):
    """
    Shared layout for make_subplots grids. When `has_legend=True`, the top margin
    is widened and the legend anchored well above the row-1 subplot titles
    (Plotly positions subplot_titles just above each row's own plot domain, which
    for row 1 sits close to the top margin boundary — a tight margin makes the
    legend and row-1 titles collide, as they did before this fix).
    """
    nrows = math.ceil(n / ncols)
    fig.update_layout(
        height=row_height * nrows, template="plotly_white",
        transition=TRANSITION, showlegend=has_legend,
        font=dict(family="Segoe UI, sans-serif", size=11),
        hoverlabel=dict(bgcolor="white", font_size=11),
        margin=dict(l=10, r=10, t=70 if has_legend else 30, b=10),
    )
    if has_legend:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05,
                                      xanchor="left", x=0))
    return fig, nrows


def plot_churn_rate_small_multiples_plotly(df, cols=None, target_col="Churn"):
    """Figure 2a — churn rate across many categorical features at once, with each
    subplot's dashed line marking the overall churn rate."""
    from eda_plots import CATEGORICAL_COLS
    cols = cols or CATEGORICAL_COLS
    overall = float(_churn01(df, target_col).mean() * 100)
    ncols = 3
    nrows = math.ceil(len(cols) / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols,
                        vertical_spacing=0.12, horizontal_spacing=0.06)
    tmp = df.copy()
    tmp["_c"] = _churn01(tmp, target_col)
    for i, col in enumerate(cols):
        r, c = i // ncols + 1, i % ncols + 1
        rates = (tmp.groupby(col, observed=True)["_c"].mean() * 100).sort_values()
        fig.add_bar(x=[str(v) for v in rates.index], y=rates.values, marker_color=BLUE,
                   hovertemplate="%{x}: %{y:.1f}%<extra></extra>", row=r, col=c)
        fig.add_hline(y=overall, line_dash="dash", line_color=RED, line_width=1,
                     row=r, col=c)
    fig, nrows = _subplot_grid_layout(fig, len(cols), ncols, row_height=240)
    fig.update_annotations(font_size=11)
    return fig


def plot_numeric_distributions_by_churn_plotly(df, cols=None, target_col="Churn"):
    """Figure 2b — smoothed density of each numeric feature, split by churn class."""
    from eda_plots import NUMERIC_COLS
    cols = cols or NUMERIC_COLS
    churn = _churn01(df, target_col)
    ncols = 2
    nrows = math.ceil(len(cols) / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols,
                        vertical_spacing=0.14, horizontal_spacing=0.08)
    for i, col in enumerate(cols):
        r, c = i // ncols + 1, i % ncols + 1
        for label, name, color in [(0, "No churn", BLUE), (1, "Churn", RED)]:
            vals = df.loc[churn == label, col].dropna().values
            if len(vals) < 2 or np.std(vals) == 0:
                continue
            xs = np.linspace(vals.min(), vals.max(), 200)
            kde = gaussian_kde(vals)(xs)
            fig.add_scatter(x=xs, y=kde, mode="lines", name=name, fill="tozeroy",
                            line=dict(color=color, width=1.5), opacity=0.55,
                            showlegend=(i == 0), legendgroup=name,
                            hovertemplate=f"{name}<br>{col}=%{{x:.1f}}<extra></extra>",
                            row=r, col=c)
    fig, nrows = _subplot_grid_layout(fig, len(cols), ncols, row_height=280, has_legend=True)
    fig.update_annotations(font_size=11)
    return fig


def plot_numeric_boxplots_by_churn_plotly(df, cols=None, target_col="Churn"):
    """Figure 2c — numeric/engineered features by churn class (boxplots)."""
    from eda_plots import NUMERIC_COLS
    cols = cols or NUMERIC_COLS
    churn = _churn01(df, target_col)
    ncols = 2
    nrows = math.ceil(len(cols) / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols,
                        vertical_spacing=0.14, horizontal_spacing=0.08)
    for i, col in enumerate(cols):
        r, c = i // ncols + 1, i % ncols + 1
        for label, name, color in [(0, "No churn", BLUE), (1, "Churn", RED)]:
            fig.add_box(y=df.loc[churn == label, col], name=name, marker_color=color,
                       showlegend=(i == 0), legendgroup=name, row=r, col=c)
    fig, nrows = _subplot_grid_layout(fig, len(cols), ncols, row_height=280, has_legend=True)
    fig.update_annotations(font_size=11)
    return fig


def plot_confusion_matrices_plotly(models, X_test, y_test):
    """Figure 7 — confusion matrices for all four models, test set."""
    names = list(models.keys())
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names,
                        horizontal_spacing=0.06)
    for i, name in enumerate(names):
        cm = confusion_matrix(y_test, models[name].predict(X_test))
        fig.add_heatmap(z=cm, x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
                        colorscale="Blues", text=cm, texttemplate="%{text}",
                        showscale=False, hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
                        row=1, col=i + 1)
    fig.update_layout(height=340, template="plotly_white", transition=TRANSITION,
                      showlegend=False, margin=dict(l=10, r=10, t=30, b=10),
                      font=dict(family="Segoe UI, sans-serif", size=11))
    fig.update_annotations(font_size=12)
    return fig


def plot_feature_importance_plotly(importances, top_n=10):
    """Figures 9/9b — top-N feature importances for a tree-based model."""
    top = importances.sort_values(ascending=True).tail(top_n)
    colors = [RED if ("Charges" in i or "Contract" in i or i == "tenure") else BLUE
             for i in top.index]
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h", marker_color=colors,
        text=top.round(3), texttemplate="%{text}", textposition="outside",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    return _base_layout(fig, height=380, xaxis_title="Importance", show_legend=False)


def plot_overfit_check_plotly(overfit_df):
    """Figure 10 — Train F1 (SMOTE-balanced) vs Test F1 (natural imbalance)."""
    fig = go.Figure()
    fig.add_bar(name="Train F1 (SMOTE)", x=list(overfit_df.index), y=overfit_df["Train F1"],
               marker_color=NAVY, text=overfit_df["Train F1"].round(3),
               texttemplate="%{text}", textposition="outside",
               hovertemplate="%{x} Train F1: %{y:.3f}<extra></extra>")
    fig.add_bar(name="Test F1", x=list(overfit_df.index), y=overfit_df["Test F1"],
               marker_color=RED, text=overfit_df["Test F1"].round(3),
               texttemplate="%{text}", textposition="outside",
               hovertemplate="%{x} Test F1: %{y:.3f}<extra></extra>")
    fig.update_layout(barmode="group")
    return _base_layout(fig, height=420, yaxis_title="F1-score")


def metrics_at_threshold(model, X_test, y_test, threshold):
    """Live Precision/Recall/F1 at an arbitrary threshold — powers the Threshold-tab slider."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return dict(
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
    )
