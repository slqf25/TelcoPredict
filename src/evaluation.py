"""
evaluation.py — Evaluation metrics, diagnostic plots, and model-interpretation
functions. Corresponds to report Section 5 (Evaluation).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.tree import plot_tree
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

PALETTE = ["#1F4E79", "#2E75B6", "#9DC3E6", "#C00000"]


def dummy_baseline(X_train_scaled, y_train, X_test_scaled, y_test) -> dict:
    """Report Section 5.1 — majority-class dummy classifier, to prove empirically
    that Accuracy alone is misleading on this imbalanced target."""
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train_scaled, y_train)
    pred = dummy.predict(X_test_scaled)
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    """Report Section 5.2 — Accuracy/Precision/Recall/F1/AUC for one model."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
    }


def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """Report Section 5.2 — Table of Accuracy/Precision/Recall/F1/AUC for all four models."""
    results = {name: evaluate_model(m, X_test, y_test) for name, m in models.items()}
    return pd.DataFrame(results).T


def overfitting_check(models: dict, X_train_sm, y_train_sm, X_test, y_test) -> pd.DataFrame:
    """Report Section 5.8 — Train F1 vs Test F1 gap for all four models."""
    rows = {}
    for name, m in models.items():
        train_f1 = f1_score(y_train_sm, m.predict(X_train_sm))
        test_f1 = f1_score(y_test, m.predict(X_test))
        rows[name] = {"Train F1": train_f1, "Test F1": test_f1, "Gap": train_f1 - test_f1}
    return pd.DataFrame(rows).T


def kfold_cv_all_models(X_train_scaled, y_train, model_builders: dict, n_splits: int = 5) -> pd.DataFrame:
    """
    Report Section 4.3 — Stratified 5-Fold CV for all four models, with SMOTE
    applied fresh inside each fold via an imblearn Pipeline (not once before
    CV) to avoid synthetic-sample leakage across folds.

    `model_builders` should be a dict of {name: sklearn_estimator_instance}
    (unfitted), e.g. {"Random Forest": RandomForestClassifier(...), ...}.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = {}
    for name, estimator in model_builders.items():
        pipe = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", estimator)])
        cv_res = cross_validate(pipe, X_train_scaled, y_train, cv=skf, scoring=scoring, n_jobs=-1)
        rows[name] = {f"{s}_mean": cv_res[f"test_{s}"].mean() for s in scoring}
        rows[name].update({f"{s}_std": cv_res[f"test_{s}"].std() for s in scoring})
        # Retain the already-computed validation score from each fold so the
        # presentation can show the distribution rather than only mean ± SD.
        for score_name in scoring:
            rows[name].update({
                f"{score_name}_fold_{fold_no}": float(score)
                for fold_no, score in enumerate(cv_res[f"test_{score_name}"], start=1)
            })
    return pd.DataFrame(rows).T


def mcnemar_test(model_a, model_b, X_test, y_test) -> dict:
    """
    Report Section 5.9 — McNemar's test on a single pair of models over the same
    test set. Tests whether two classifiers' *disagreements* are asymmetric (one
    model right where the other is wrong, more often in one direction than the
    other) rather than whether their overall accuracy differs — the correct test
    when both models are evaluated on the identical hold-out set, since their
    predictions are paired observations, not independent samples.

    Uses the exact binomial variant when b + c < 25 (the discordant-pair count is
    small enough that the chi-square approximation is unreliable), continuity-
    corrected chi-square otherwise — both via statsmodels, which is already a
    project dependency (src/eda.py uses it for VIF).
    """
    from statsmodels.stats.contingency_tables import mcnemar as _mcnemar

    pred_a = model_a.predict(X_test)
    pred_b = model_b.predict(X_test)
    correct_a = (pred_a == y_test).astype(int)
    correct_b = (pred_b == y_test).astype(int)

    # b = A right, B wrong; c = A wrong, B right (the two discordant cells)
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    table = [[0, b], [c, 0]]  # diagonal (both-right / both-wrong counts) doesn't affect the test
    result = _mcnemar(table, exact=(b + c < 25), correction=True)

    return {
        "b_A_right_B_wrong": b,
        "c_A_wrong_B_right": c,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant_(p<0.05)": bool(result.pvalue < 0.05),
    }


def mcnemar_all_pairs(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Report Section 5.9 — McNemar's test for every pair among the four models, to
    check whether Section 5.2's Random Forest ranking is statistically distinct
    from the other three or within the noise band the Discussion flagged as an
    open question. Returns one row per pair, sorted by p-value (most significant
    difference first).
    """
    names = list(models.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b_name = names[i], names[j]
            r = mcnemar_test(models[a], models[b_name], X_test, y_test)
            rows.append({"model_A": a, "model_B": b_name, **r})
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def get_feature_importance(model, feature_names) -> pd.Series:
    """Report Section 5.3 — feature importances for a tree-based model."""
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)


def get_lr_coefficients(model, feature_names) -> pd.DataFrame:
    """Report Section 5.4 — Logistic Regression coefficients and odds ratios."""
    coefs = pd.Series(model.coef_[0], index=feature_names)
    odds_ratios = np.exp(coefs)
    return pd.DataFrame({"coefficient": coefs, "odds_ratio": odds_ratios}).sort_values("coefficient", ascending=False)


def threshold_tuning(model, X_test, y_test, thresholds=None) -> pd.DataFrame:
    """Report Section 5.7 — F1/Precision/Recall swept across classification thresholds."""
    thresholds = thresholds if thresholds is not None else np.arange(0.1, 0.91, 0.02)
    proba = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "f1": f1_score(y_test, pred, zero_division=0),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def optimal_threshold(model, X_test, y_test) -> dict:
    """Returns the threshold that maximises F1 for a given model."""
    df = threshold_tuning(model, X_test, y_test)
    best = df.loc[df["f1"].idxmax()]
    return best.to_dict()


# ------------------------------------------------------------------------
# Plotting functions (all return a matplotlib Figure; notebook calls plt.show())
# ------------------------------------------------------------------------

def plot_roc_curves(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for (name, m), c in zip(models.items(), PALETTE):
        proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={a:.3f})", color=c, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    return fig


def plot_pr_curves(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for (name, m), c in zip(models.items(), PALETTE):
        proba = m.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=c, linewidth=2)
    baseline_rate = y_test.mean()
    ax.axhline(baseline_rate, color="gray", linestyle="--", alpha=0.5, label=f"No-skill baseline ({baseline_rate:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


def plot_confusion_matrices(models: dict, X_test, y_test):
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
    for ax, (name, m) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, m.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set_title(name, fontsize=10); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig


def plot_model_comparison_bar(results_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    results_df[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", ax=ax, color=PALETTE)
    ax.set_ylabel("Score"); ax.set_title("Model Performance Comparison")
    ax.legend(loc="lower right"); plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


def plot_feature_importance(importances: pd.Series, top_n: int = 10, title: str = "Feature Importance"):
    top = importances.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#C00000" if ("Charges" in i or "Contract" in i or i == "tenure") else "#2E75B6" for i in top.index]
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel("Importance"); ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_decision_tree_structure(dt_model, feature_names, max_depth_display: int = 3):
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(dt_model, feature_names=feature_names, class_names=["No Churn", "Churn"],
              filled=True, rounded=True, fontsize=7, max_depth=max_depth_display, ax=ax)
    plt.tight_layout()
    return fig


def plot_threshold_curve(models: dict, X_test, y_test, model_names_to_plot=None):
    model_names_to_plot = model_names_to_plot or list(models.keys())
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, c in zip(model_names_to_plot, PALETTE):
        df = threshold_tuning(models[name], X_test, y_test)
        ax.plot(df["threshold"], df["f1"], label=f"{name} F1", color=c, linewidth=2)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Default (0.5)")
    ax.set_xlabel("Classification Threshold"); ax.set_ylabel("F1-score")
    ax.set_title("F1-score vs Classification Threshold")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_overfit_check(overfit_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xpos = np.arange(len(overfit_df))
    ax.bar(xpos - 0.18, overfit_df["Train F1"], width=0.36, label="Train F1 (SMOTE)", color=PALETTE[0])
    ax.bar(xpos + 0.18, overfit_df["Test F1"], width=0.36, label="Test F1", color=PALETTE[3])
    ax.set_xticks(xpos); ax.set_xticklabels(overfit_df.index, rotation=15, ha="right")
    ax.set_ylabel("F1-score"); ax.set_title("Train vs Test F1 — Overfitting Check")
    ax.legend()
    plt.tight_layout()
    return fig
