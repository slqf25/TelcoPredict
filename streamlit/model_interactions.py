"""Interactive model-evaluation visuals used only by the Streamlit prototype.

The functions in this module never fit or mutate a model.  They derive display values from an already-trained classifier's ``predict_proba`` output 
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


NAVY = "#1F4E79"
BLUE = "#2E75B6"
LIGHT = "#9DC3E6"
RED = "#C00000"
GREEN = "#2E8B57"
AMBER = "#E8A317"
TRANSITION = dict(duration=350, easing="cubic-in-out")
SANKEY_LABEL_CARDS_VERSION = 2
MODEL_COLORS = {
    "Logistic Regression": "#2E75B6",
    "Decision Tree": "#E8A317",
    "Random Forest": "#2E8B57",
    "XGBoost": "#7A5AA6",
}


def probability_vector(model, X_test) -> np.ndarray:
    """Return the positive-class probability from an already-fitted model."""
    return np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)


def performance_ranking_frame(results_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rank the existing test-set scorecard without recomputing or fitting models."""
    ranked = results_df[[metric]].copy().sort_values(metric, ascending=False)
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
    return ranked


def plot_metric_ranking(
    results_df: pd.DataFrame,
    metric: str,
    selected_model: str | None = None,
) -> go.Figure:
    """Horizontal score ranking with the global model selection clearly outlined."""
    ranked = performance_ranking_frame(results_df, metric).sort_values(metric, ascending=True)
    models = list(ranked.index)
    values = ranked[metric].to_numpy(dtype=float) * 100
    best_value = float(values.max())
    labels = [
        ("★ " if np.isclose(value, best_value) else "") + f"{value:.1f}%"
        for value in values
    ]
    colors = [MODEL_COLORS.get(model, BLUE) for model in models]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=models,
            orientation="h",
            marker=dict(
                color=colors,
                opacity=[1.0 if model == selected_model else 0.72 for model in models],
                line=dict(
                    color=[NAVY if model == selected_model else "rgba(0,0,0,0)" for model in models],
                    width=[3 if model == selected_model else 0 for model in models],
                ),
            ),
            text=labels,
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"%{{y}}<br>{metric}: %{{x:.2f}}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=365,
        template="plotly_white",
        margin=dict(l=15, r=75, t=35, b=35),
        font=dict(family="Segoe UI, sans-serif", size=12),
        transition=TRANSITION,
        xaxis_title=f"{metric} (%)",
        xaxis=dict(range=[0, min(100, best_value + 12)]),
        yaxis_title=None,
        showlegend=False,
    )
    return fig


def curve_summary_frame(models: dict, X_test, y_true) -> pd.DataFrame:
    """ROC-AUC and Average Precision derived from existing model probabilities."""
    rows = []
    for name, model in models.items():
        probabilities = probability_vector(model, X_test)
        rows.append(
            {
                "Model": name,
                "ROC-AUC": roc_auc_score(y_true, probabilities),
                "Average Precision": average_precision_score(y_true, probabilities),
            }
        )
    return pd.DataFrame(rows).set_index("Model")


def plot_discrimination_curve(
    models: dict,
    X_test,
    y_true,
    curve_type: str,
    selected_model: str | None = None,
) -> go.Figure:
    """All-model ROC or PR curve with the globally selected model emphasised."""
    fig = go.Figure()
    for name, model in models.items():
        probabilities = probability_vector(model, X_test)
        is_selected = name == selected_model
        line = dict(
            color=MODEL_COLORS.get(name, BLUE),
            width=4.2 if is_selected else 2.0,
        )
        opacity = 1.0 if is_selected else 0.42

        if curve_type == "ROC":
            x_values, y_values, thresholds = roc_curve(y_true, probabilities)
            score = roc_auc_score(y_true, probabilities)
            trace_name = f"{name} (AUC={score:.3f})"
            hover = "FPR %{x:.3f}, TPR %{y:.3f}<br>threshold≈%{customdata:.2f}<extra>" + name + "</extra>"
            customdata = thresholds
        else:
            y_values, x_values, thresholds = precision_recall_curve(y_true, probabilities)
            score = average_precision_score(y_true, probabilities)
            trace_name = f"{name} (AP={score:.3f})"
            hover = "Recall %{x:.3f}, Precision %{y:.3f}<extra>" + name + "</extra>"
            customdata = None

        fig.add_scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name=trace_name,
            line=line,
            opacity=opacity,
            customdata=customdata,
            hovertemplate=hover,
        )

    if curve_type == "ROC":
        fig.add_scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="gray", dash="dash"), name="Chance", hoverinfo="skip",
        )
        x_title, y_title = "False Positive Rate", "True Positive Rate"
    else:
        baseline = float(np.mean(y_true))
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"no-skill ({baseline:.3f})",
            annotation_position="top left",
        )
        x_title, y_title = "Recall", "Precision"

    fig.update_layout(
        height=455,
        template="plotly_white",
        margin=dict(l=15, r=15, t=45, b=35),
        font=dict(family="Segoe UI, sans-serif", size=12),
        transition=TRANSITION,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def model_explanation_frame(model, feature_names, model_name: str) -> pd.DataFrame:
    """Return coefficients or feature importances from an already-fitted model."""
    if hasattr(model, "coef_"):
        effects = np.asarray(model.coef_[0], dtype=float)
        explanation_type = "Coefficient"
    elif hasattr(model, "feature_importances_"):
        effects = np.asarray(model.feature_importances_, dtype=float)
        explanation_type = "Importance"
    else:
        raise ValueError(f"{model_name} does not expose coefficients or feature importances")

    frame = pd.DataFrame(
        {
            "Feature": list(feature_names),
            "Effect": effects,
            "Magnitude": np.abs(effects),
        }
    )
    frame["Direction"] = np.where(
        frame["Effect"] > 0,
        "Higher churn signal",
        np.where(frame["Effect"] < 0, "Lower churn signal", "Neutral"),
    )
    frame["Explanation type"] = explanation_type
    return frame.sort_values("Magnitude", ascending=False).reset_index(drop=True)


def plot_selected_model_explanation(
    frame: pd.DataFrame,
    model_name: str,
    top_n: int = 12,
) -> go.Figure:
    """Model-aware horizontal explanation plot for coefficients or importances."""
    shown = frame.head(top_n).sort_values("Magnitude", ascending=True)
    is_coefficient = shown["Explanation type"].iloc[0] == "Coefficient"
    if is_coefficient:
        x_values = shown["Effect"]
        colors = [RED if value > 0 else BLUE for value in x_values]
        text = [f"{value:+.3f}" for value in x_values]
        x_title = "Standardised logistic coefficient"
        hover = "%{y}<br>Coefficient: %{x:+.3f}<extra></extra>"
    else:
        x_values = shown["Magnitude"]
        colors = [MODEL_COLORS.get(model_name, BLUE)] * len(shown)
        text = [f"{value:.3f}" for value in x_values]
        x_title = "Feature importance"
        hover = "%{y}<br>Importance: %{x:.3f}<extra></extra>"

    fig = go.Figure(
        go.Bar(
            x=x_values,
            y=shown["Feature"],
            orientation="h",
            marker_color=colors,
            text=text,
            textposition="outside",
            cliponaxis=False,
            hovertemplate=hover,
        )
    )
    if is_coefficient:
        fig.add_vline(x=0, line_color="gray", line_width=1)
    fig.update_layout(
        height=max(430, top_n * 34 + 90),
        template="plotly_white",
        margin=dict(l=15, r=65, t=38, b=35),
        font=dict(family="Segoe UI, sans-serif", size=12),
        transition=TRANSITION,
        xaxis_title=x_title,
        yaxis_title=None,
        showlegend=False,
    )
    return fig


def tree_importance_comparison_frame(
    models: dict,
    feature_names,
    top_n: int = 12,
) -> pd.DataFrame:
    """Aligned Random Forest/XGBoost importance table for shared-driver comparison."""
    columns = {}
    for name in ["Random Forest", "XGBoost"]:
        columns[name] = pd.Series(
            np.asarray(models[name].feature_importances_, dtype=float),
            index=list(feature_names),
        )
    frame = pd.DataFrame(columns).fillna(0.0)
    frame["Mean importance"] = frame.mean(axis=1)
    return frame.sort_values("Mean importance", ascending=False).head(top_n)


def tree_importance_overlap(models: dict, feature_names, top_n: int = 10) -> dict:
    """Top-driver agreement summary used in presentation cards and narrative."""
    rankings = {}
    for name in ["Random Forest", "XGBoost"]:
        series = pd.Series(
            np.asarray(models[name].feature_importances_, dtype=float),
            index=list(feature_names),
        ).sort_values(ascending=False)
        rankings[name] = series
    shared = set(rankings["Random Forest"].head(top_n).index) & set(
        rankings["XGBoost"].head(top_n).index
    )
    return {
        "rf_leader": rankings["Random Forest"].index[0],
        "rf_value": float(rankings["Random Forest"].iloc[0]),
        "xgb_leader": rankings["XGBoost"].index[0],
        "xgb_value": float(rankings["XGBoost"].iloc[0]),
        "shared_count": len(shared),
        "shared_features": sorted(shared),
        "top_n": top_n,
    }


def plot_tree_importance_comparison(frame: pd.DataFrame) -> go.Figure:
    """Grouped comparison of the two ensemble models' strongest global drivers."""
    shown = frame.sort_values("Mean importance", ascending=True)
    fig = go.Figure()
    for model_name in ["Random Forest", "XGBoost"]:
        values = shown[model_name].to_numpy(dtype=float)
        fig.add_bar(
            name=model_name,
            x=values,
            y=shown.index,
            orientation="h",
            marker_color=MODEL_COLORS[model_name],
            text=[f"{value:.3f}" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>" + model_name + ": %{x:.4f}<extra></extra>",
        )
    fig.update_layout(
        height=max(460, len(shown) * 40 + 90),
        template="plotly_white",
        barmode="group",
        margin=dict(l=15, r=65, t=55, b=35),
        font=dict(family="Segoe UI, sans-serif", size=12),
        transition=TRANSITION,
        xaxis_title="Feature importance",
        yaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def plot_generalisation_dumbbell(
    overfit_df: pd.DataFrame,
    selected_model: str | None = None,
) -> go.Figure:
    """Train-to-test F1 dumbbell plot that makes each generalisation gap explicit."""
    frame = overfit_df.sort_values("Gap", ascending=True)
    models = list(frame.index)
    fig = go.Figure()
    for name in models:
        train = float(frame.loc[name, "Train F1"])
        test = float(frame.loc[name, "Test F1"])
        is_selected = name == selected_model
        fig.add_scatter(
            x=[test, train],
            y=[name, name],
            mode="lines",
            line=dict(
                color=NAVY if is_selected else "#B7C7D6",
                width=6 if is_selected else 3,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    fig.add_scatter(
        x=frame["Train F1"],
        y=models,
        mode="markers",
        name="Train F1 (SMOTE)",
        marker=dict(color=NAVY, size=13, symbol="circle"),
        customdata=frame["Gap"],
        hovertemplate="%{y}<br>Train F1: %{x:.3f}<br>Gap: %{customdata:.3f}<extra></extra>",
    )
    fig.add_scatter(
        x=frame["Test F1"],
        y=models,
        mode="markers",
        name="Test F1",
        marker=dict(color=BLUE, size=13, symbol="diamond"),
        customdata=frame["Gap"],
        hovertemplate="%{y}<br>Test F1: %{x:.3f}<br>Gap: %{customdata:.3f}<extra></extra>",
    )
    fig.update_layout(
        height=390,
        template="plotly_white",
        margin=dict(l=15, r=35, t=60, b=35),
        font=dict(family="Segoe UI, sans-serif", size=12),
        transition=TRANSITION,
        xaxis_title="F1-score",
        yaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def cv_metric_columns(metric: str) -> tuple[str, str]:
    """Map presentation labels to the existing cross-validation result columns."""
    key = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC-AUC": "roc_auc",
    }[metric]
    return f"{key}_mean", f"{key}_std"


def cv_ranking_frame(cv_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rank cross-validation means while retaining fold-to-fold standard deviations."""
    mean_col, std_col = cv_metric_columns(metric)
    ranked = cv_df[[mean_col, std_col]].copy().sort_values(mean_col, ascending=False)
    ranked.columns = ["Mean", "Std"]
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
    return ranked


def plot_cv_stability(
    cv_df: pd.DataFrame,
    metric: str,
    selected_model: str | None = None,
) -> go.Figure:
    """Show each validation-fold score together with its mean and ±1 SD interval."""
    ranked = cv_ranking_frame(cv_df, metric).sort_values("Mean", ascending=True)
    models = list(ranked.index)
    mean_col, _ = cv_metric_columns(metric)
    metric_key = mean_col.removesuffix("_mean")
    fold_cols = sorted(
        [column for column in cv_df.columns if column.startswith(f"{metric_key}_fold_")],
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )

    fig = go.Figure()
    all_values: list[float] = []
    leader = ranked["Mean"].idxmax()
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#64748B", width=6),
            name="±1 SD interval",
            legendrank=3,
            hoverinfo="skip",
        )
    )

    for model_name in models:
        mean = float(ranked.loc[model_name, "Mean"]) * 100
        std = float(ranked.loc[model_name, "Std"]) * 100
        folds = cv_df.loc[model_name, fold_cols].to_numpy(dtype=float) * 100
        color = MODEL_COLORS.get(model_name, BLUE)
        is_active = model_name == selected_model
        all_values.extend(folds.tolist())
        all_values.extend([mean - std, mean + std])

        # The interval sits behind the observations and communicates variation;
        # it is intentionally not a bar starting at zero.
        fig.add_trace(
            go.Scatter(
                x=[mean - std, mean + std],
                y=[model_name, model_name],
                mode="lines",
                line=dict(color=color, width=7 if is_active else 5),
                opacity=0.42 if not is_active else 0.62,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=folds,
                y=[model_name] * len(folds),
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=10 if is_active else 9,
                    color=color,
                    opacity=1.0 if is_active else 0.78,
                    line=dict(color="#FFFFFF", width=1.5),
                ),
                customdata=np.arange(1, len(folds) + 1),
                name="Fold score",
                legendgroup="fold",
                legendrank=1,
                showlegend=model_name == models[0],
                hovertemplate=(
                    f"<b>{model_name}</b><br>Fold %{{customdata}} {metric}: "
                    "%{x:.2f}%<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[mean],
                y=[model_name],
                mode="markers+text",
                marker=dict(
                    symbol="diamond",
                    size=18 if is_active else 16,
                    color=color,
                    line=dict(color=NAVY if is_active else "#FFFFFF", width=3 if is_active else 2),
                ),
                text=[f"  {mean:.1f}% ± {std:.1f}"],
                textposition="middle right",
                textfont=dict(color="#4B5563", size=11),
                customdata=[[std, "CV leader" if model_name == leader else ""]],
                name="Mean",
                legendgroup="mean",
                legendrank=2,
                showlegend=model_name == models[0],
                hovertemplate=(
                    f"<b>{model_name}</b><br>Mean {metric}: %{{x:.2f}}%"
                    "<br>Fold SD: %{customdata[0]:.2f} pp"
                    "<br>%{customdata[1]}<extra></extra>"
                ),
            )
        )

    if all_values:
        x_min = max(0.0, min(all_values) - 2.0)
        x_max = min(100.0, max(all_values) + 7.0)
    else:
        x_min, x_max = 0.0, 100.0

    fig.update_layout(
        height=400,
        template="plotly_white",
        margin=dict(l=15, r=30, t=62, b=45),
        font=dict(family="Segoe UI, sans-serif", size=12),
        xaxis_title=f"Validation-fold {metric} (%) · zoomed scale",
        xaxis=dict(
            range=[x_min, x_max],
            ticksuffix="%",
            showgrid=True,
            gridcolor="rgba(31,78,121,0.10)",
            zeroline=False,
        ),
        # Keep Plotly from briefly coercing model labels onto a numeric axis when
        # Streamlit replaces this figure after the metric control changes.
        yaxis=dict(
            title=None,
            type="category",
            categoryorder="array",
            categoryarray=models,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=0,
            title=None,
        ),
        hovermode="closest",
    )
    return fig


def threshold_snapshot(y_true, probabilities, threshold: float) -> dict:
    """Metrics and confusion counts at one user-selected operating threshold."""
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "predicted_churn": int(pred.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y)),
    }


def threshold_series(y_true, probabilities, thresholds=None) -> pd.DataFrame:
    """Precision/Recall/F1 sweep used by both the 2D and 3D views."""
    thresholds = np.asarray(
        thresholds if thresholds is not None else np.arange(0.10, 0.901, 0.02),
        dtype=float,
    )
    rows = []
    for threshold in thresholds:
        snap = threshold_snapshot(y_true, probabilities, float(threshold))
        rows.append(
            {
                "threshold": snap["threshold"],
                "precision": snap["precision"],
                "recall": snap["recall"],
                "f1": snap["f1"],
            }
        )
    return pd.DataFrame(rows)


def all_model_confusion_frame(models: dict, X_test, y_true, threshold: float = 0.50) -> pd.DataFrame:
    """Comparable confusion counts for every fitted model at one fixed threshold."""
    rows = []
    for name, model in models.items():
        snap = threshold_snapshot(y_true, probability_vector(model, X_test), threshold)
        churn_total = snap["fn"] + snap["tp"]
        retained_total = snap["tn"] + snap["fp"]
        rows.append(
            {
                "Model": name,
                "TN": snap["tn"],
                "FP": snap["fp"],
                "FN": snap["fn"],
                "TP": snap["tp"],
                "Precision": snap["precision"],
                "Recall": snap["recall"],
                "F1": snap["f1"],
                "FN rate": snap["fn"] / churn_total if churn_total else 0.0,
                "FP rate": snap["fp"] / retained_total if retained_total else 0.0,
                "TP rate": snap["tp"] / churn_total if churn_total else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("Model")


def all_model_comparison_table(frame: pd.DataFrame, display: str = "Counts") -> pd.DataFrame:
    """Compact table backing the all-model error-comparison view."""
    if display == "Percentages":
        result = (frame[["FN rate", "FP rate", "TP rate", "Precision", "Recall", "F1"]] * 100).copy()
        result.columns = [
            "Missed churners (%)",
            "False alerts (%)",
            "Caught churners (%)",
            "Precision (%)",
            "Recall (%)",
            "F1 (%)",
        ]
        return result
    return frame[["FN", "FP", "TP", "TN", "Precision", "Recall", "F1"]].copy()


def all_model_winners(frame: pd.DataFrame) -> dict:
    """Winner names and values used by both visual labels and narrative cards."""
    return {
        "lowest_fn_model": frame["FN"].idxmin(),
        "lowest_fn": int(frame["FN"].min()),
        "lowest_fp_model": frame["FP"].idxmin(),
        "lowest_fp": int(frame["FP"].min()),
        "highest_tp_model": frame["TP"].idxmax(),
        "highest_tp": int(frame["TP"].max()),
        "highest_f1_model": frame["F1"].idxmax(),
        "highest_f1": float(frame["F1"].max()),
    }


def plot_all_model_error_comparison(
    frame: pd.DataFrame,
    display: str = "Counts",
    selected_model: str | None = None,
) -> go.Figure:
    """Grouped comparison of missed churners, false alerts and caught churners."""
    models = list(frame.index)
    if display == "Percentages":
        series = [
            ("Missed churners (FN)", "FN rate", RED, "min"),
            ("False alerts (FP)", "FP rate", AMBER, "min"),
            ("Caught churners (TP)", "TP rate", GREEN, "max"),
        ]
        multiplier = 100.0
        y_title = "Rate within the relevant actual class (%)"
        text_format = ".1f"
        hover_suffix = "%"
    else:
        series = [
            ("Missed churners (FN)", "FN", RED, "min"),
            ("False alerts (FP)", "FP", AMBER, "min"),
            ("Caught churners (TP)", "TP", GREEN, "max"),
        ]
        multiplier = 1.0
        y_title = "Customers"
        text_format = ",.0f"
        hover_suffix = " customers"

    fig = go.Figure()
    max_value = 0.0
    for label, column, color, winner_rule in series:
        values = frame[column].to_numpy(dtype=float) * multiplier
        max_value = max(max_value, float(values.max()))
        winner_value = values.min() if winner_rule == "min" else values.max()
        text_values = [
            ("★ " if np.isclose(value, winner_value) else "")
            + (f"{value:.1f}%" if display == "Percentages" else f"{value:,.0f}")
            for value in values
        ]
        fig.add_bar(
            name=label,
            x=models,
            y=values,
            marker=dict(
                color=color,
                opacity=[1.0 if model == selected_model else 0.72 for model in models],
                line=dict(
                    color=[NAVY if model == selected_model else "rgba(0,0,0,0)" for model in models],
                    width=[3 if model == selected_model else 0 for model in models],
                ),
            ),
            text=text_values,
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{x}<br>" + label + ": %{y:" + text_format + "}" + hover_suffix
                + "<extra></extra>"
            ),
        )

    if selected_model in models:
        fig.add_annotation(
            x=selected_model,
            y=max_value * 1.20,
            text="Selected model",
            showarrow=True,
            arrowhead=2,
            arrowcolor=NAVY,
            font=dict(color=NAVY, size=11),
        )
    fig.update_layout(
        height=470,
        template="plotly_white",
        barmode="group",
        margin=dict(l=15, r=15, t=82, b=35),
        font=dict(family="Segoe UI, sans-serif", size=12),
        transition=TRANSITION,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title=y_title,
        yaxis=dict(range=[0, max_value * 1.32 if max_value else 1]),
    )
    return fig


def comparison_frame(default: dict, selected: dict) -> pd.DataFrame:
    """Presentation-friendly default-vs-selected threshold comparison."""
    definitions = [
        ("Precision", "precision", "percent"),
        ("Recall", "recall", "percent"),
        ("F1-score", "f1", "percent"),
        ("False positives", "fp", "count"),
        ("False negatives", "fn", "count"),
        ("Predicted churn", "predicted_churn", "count"),
    ]
    rows = []
    for label, key, kind in definitions:
        before, after = default[key], selected[key]
        if kind == "percent":
            before, after = before * 100, after * 100
            rows.append(
                {
                    "Measure": label,
                    "Default (0.50)": f"{before:.1f}%",
                    "Selected": f"{after:.1f}%",
                    "Change": f"{after - before:+.1f} pp",
                }
            )
        else:
            rows.append(
                {
                    "Measure": label,
                    "Default (0.50)": f"{int(before):,}",
                    "Selected": f"{int(after):,}",
                    "Change": f"{int(after - before):+,}",
                }
            )
    return pd.DataFrame(rows)


def plot_confusion_matrix(snapshot: dict, display: str = "Counts") -> go.Figure:
    """Large semantic-colour confusion matrix for one model and threshold."""
    counts = np.array(
        [[snapshot["tn"], snapshot["fp"]], [snapshot["fn"], snapshot["tp"]]],
        dtype=float,
    )
    row_totals = counts.sum(axis=1, keepdims=True)
    percentages = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts),
        where=row_totals != 0,
    ) * 100
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    text = []
    for row in range(2):
        text.append([])
        for col in range(2):
            primary = (
                f"{int(counts[row, col]):,}"
                if display == "Counts"
                else f"{percentages[row, col]:.1f}%"
            )
            secondary = (
                f"{percentages[row, col]:.1f}% of actual class"
                if display == "Counts"
                else f"{int(counts[row, col]):,} customers"
            )
            text[row].append(f"<b>{labels[row, col]}</b><br>{primary}<br><span>{secondary}</span>")

    # Categorical z values allow each cell to carry a stable semantic colour:
    # correct retained / false alarm / missed churner / correctly caught churner.
    semantic_z = np.array([[0, 1], [2, 3]])
    colorscale = [
        [0.00, NAVY], [0.16, NAVY],
        [0.17, AMBER], [0.49, AMBER],
        [0.50, RED], [0.82, RED],
        [0.83, GREEN], [1.00, GREEN],
    ]
    # Keep this as nested Python lists (rather than a NumPy stack) so Plotly's
    # JSON preserves numeric count/percentage fields alongside the text label.
    custom = [
        [
            [float(counts[row, col]), float(percentages[row, col]), labels[row, col]]
            for col in range(2)
        ]
        for row in range(2)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=semantic_z,
            x=["Predicted: Retained", "Predicted: Churn"],
            y=["Actual: Retained", "Actual: Churn"],
            colorscale=colorscale,
            zmin=0,
            zmax=3,
            showscale=False,
            text=text,
            texttemplate="%{text}",
            textfont=dict(color="white", size=15),
            customdata=custom,
            hovertemplate=(
                "%{customdata[2]}<br>%{customdata[0]:,.0f} customers"
                "<br>%{customdata[1]:.1f}% of actual class<extra></extra>"
            ),
            xgap=7,
            ygap=7,
        )
    )
    fig.update_layout(
        height=430,
        template="plotly_white",
        margin=dict(l=20, r=20, t=18, b=28),
        font=dict(family="Segoe UI, sans-serif", size=12),
        transition=TRANSITION,
    )
    fig.update_xaxes(side="top", tickfont=dict(size=12), fixedrange=True)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=12), fixedrange=True)
    return fig


def plot_confusion_sankey(snapshot: dict, display: str = "Counts") -> go.Figure:
    """Actual-to-predicted flow whose widths react to the selected threshold."""
    values = np.array(
        [snapshot["tn"], snapshot["fp"], snapshot["fn"], snapshot["tp"]],
        dtype=float,
    )
    if display == "Percentages":
        plot_values = values / snapshot["n"] * 100
        value_suffix = "% of test set"
        value_format = ".1f"
    else:
        plot_values = values
        value_suffix = "customers"
        value_format = ",.0f"

    node_titles = [
        "Actual retained",
        "Actual churn",
        "Predicted retained",
        "Predicted churn",
    ]
    node_counts = np.array(
        [
            snapshot["tn"] + snapshot["fp"],
            snapshot["fn"] + snapshot["tp"],
            snapshot["tn"] + snapshot["fn"],
            snapshot["fp"] + snapshot["tp"],
        ],
        dtype=float,
    )

    def annotation_text(title: str, count: float) -> str:
        if display == "Percentages":
            detail = f"{count / snapshot['n'] * 100:.1f}% of test set"
        else:
            detail = f"{count:,.0f} customers"
        return f"<b>{title}</b><br>{detail}"

    link_labels = ["TN", "FP", "FN", "TP"]
    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=28,
                thickness=22,
                line=dict(color="rgba(31,78,121,.25)", width=1),
                # Native Sankey labels sit directly on the links and become muddy at
                # presentation scale. Clear annotation cards are added below instead.
                label=["", "", "", ""],
                color=[LIGHT, "#E6A1A1", NAVY, GREEN],
                x=[0.035, 0.035, 0.965, 0.965],
                y=[0.18, 0.78, 0.18, 0.78],
            ),
            link=dict(
                source=[0, 0, 1, 1],
                target=[2, 3, 2, 3],
                value=plot_values,
                label=link_labels,
                color=[
                    "rgba(31,78,121,.48)",
                    "rgba(232,163,23,.55)",
                    "rgba(192,0,0,.60)",
                    "rgba(46,139,87,.55)",
                ],
                customdata=values,
                hovertemplate=(
                    "%{label}<br>%{value:" + value_format + "} " + value_suffix
                    + "<br>%{customdata:,.0f} customers<extra></extra>"
                ),
            ),
        )
    )
    annotation_specs = [
        (0.045, 0.82, "left", BLUE),
        (0.045, 0.22, "left", RED),
        (0.955, 0.82, "right", NAVY),
        (0.955, 0.22, "right", GREEN),
    ]
    for title, count, (x, y, xanchor, border_color) in zip(
        node_titles, node_counts, annotation_specs
    ):
        fig.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            xanchor=xanchor,
            yanchor="middle",
            text=annotation_text(title, count),
            showarrow=False,
            align="left" if xanchor == "left" else "right",
            bgcolor="rgba(255,255,255,.96)",
            bordercolor=border_color,
            borderwidth=2,
            borderpad=7,
            font=dict(family="Segoe UI, sans-serif", size=15, color="#172033"),
        )
    fig.update_layout(
        height=460,
        template="plotly_white",
        margin=dict(l=20, r=20, t=24, b=18),
        font=dict(family="Segoe UI, sans-serif", size=14, color="#172033"),
        transition=TRANSITION,
    )
    return fig


def plot_threshold_tradeoff_2d(
    sweep: pd.DataFrame, selected_threshold: float, model_name: str
) -> go.Figure:
    """Readable primary view of the Precision/Recall/F1 threshold trade-off."""
    fig = go.Figure()
    for metric, color in [("precision", BLUE), ("recall", AMBER), ("f1", GREEN)]:
        fig.add_scatter(
            x=sweep["threshold"],
            y=sweep[metric] * 100,
            mode="lines",
            name=metric.title() if metric != "f1" else "F1-score",
            line=dict(color=color, width=3 if metric == "f1" else 2.3),
            hovertemplate=(
                f"Threshold %{{x:.2f}}<br>{metric.title()} %{{y:.1f}}%"
                f"<extra>{model_name}</extra>"
            ),
        )
    fig.add_vline(
        x=0.50,
        line_dash="dot",
        line_color="#7A7A7A",
        annotation_text="default 0.50",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=selected_threshold,
        line_color=NAVY,
        line_width=3,
        annotation_text=f"selected {selected_threshold:.2f}",
        annotation_position="top left",
    )
    fig.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=15, r=15, t=58, b=30),
        font=dict(family="Segoe UI, sans-serif", size=12),
        hovermode="x unified",
        transition=TRANSITION,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Classification threshold",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
    )
    return fig


def plot_threshold_tradeoff_3d(
    sweep: pd.DataFrame, selected_threshold: float, model_name: str
) -> go.Figure:
    """Optional threshold path with default, F1-optimal and selected reference points."""
    selected_idx = int((sweep["threshold"] - selected_threshold).abs().idxmin())
    selected = sweep.loc[selected_idx]
    default_idx = int((sweep["threshold"] - 0.50).abs().idxmin())
    optimal_idx = int(sweep["f1"].idxmax())
    default_point = sweep.loc[default_idx]
    optimal_point = sweep.loc[optimal_idx]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=sweep["threshold"],
            y=sweep["precision"] * 100,
            z=sweep["recall"] * 100,
            mode="lines+markers",
            name=f"{model_name} threshold path",
            showlegend=False,
            line=dict(color=NAVY, width=6),
            marker=dict(
                size=4,
                color=sweep["f1"] * 100,
                colorscale="Viridis",
                colorbar=dict(title="F1 (%)", thickness=14),
                cmin=0,
                cmax=100,
            ),
            customdata=sweep["f1"] * 100,
            hovertemplate=(
                "Threshold %{x:.2f}<br>Precision %{y:.1f}%<br>Recall %{z:.1f}%"
                "<br>F1 %{customdata:.1f}%<extra></extra>"
            ),
        )
    )

    reference_points = [
        ("Default 0.50", default_point, "diamond-open", "#64748B", 9),
        ("Highest F1", optimal_point, "diamond", GREEN, 11),
        ("Current selection", selected, "circle", RED, 10),
    ]
    for label, point, symbol, color, size in reference_points:
        fig.add_trace(
            go.Scatter3d(
                x=[point["threshold"]],
                y=[point["precision"] * 100],
                z=[point["recall"] * 100],
                mode="markers",
                name=label,
                marker=dict(size=size, symbol=symbol, color=color, line=dict(color="white", width=2)),
                customdata=[[point["f1"] * 100]],
                hovertemplate=(
                    f"<b>{label}</b><br>Threshold %{{x:.2f}}<br>Precision %{{y:.1f}}%"
                    "<br>Recall %{z:.1f}%<br>F1 %{customdata[0]:.1f}%<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        height=570,
        template="plotly_white",
        margin=dict(l=0, r=0, t=35, b=0),
        font=dict(family="Segoe UI, sans-serif", size=11),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        scene=dict(
            xaxis_title="Threshold",
            yaxis_title="Precision (%)",
            zaxis_title="Recall (%)",
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.15)),
            aspectmode="cube",
        ),
    )
    return fig
