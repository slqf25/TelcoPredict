"""
modelling.py — Train/test split, scaling, SMOTE resampling, and training of
the four classification models. Corresponds to report Section 3.6-3.8
(Data Preparation) and Section 4 (Modelling).
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from data_prep import NUM_COLS_TO_SCALE


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Report Section 3.8 — stratified 80/20 train-test split."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, num_cols: list = None):
    """
    Report Section 3.6 — StandardScaler fitted on the training set only,
    then applied to both train and test to prevent leakage.
    """
    num_cols = num_cols or NUM_COLS_TO_SCALE
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    return X_train_scaled, X_test_scaled, scaler


def resample_smote(X_train_scaled: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """
    Report Section 3.7 — SMOTE applied to the training set only, after the
    train-test split. Not combined with class_weight/scale_pos_weight
    (Section 3.7 explains why combining both would double-correct).
    """
    sm = SMOTE(random_state=random_state)
    return sm.fit_resample(X_train_scaled, y_train)


def train_logistic_regression(X_train_sm, y_train_sm) -> LogisticRegression:
    """Report Section 4.1 — baseline model."""
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train_sm, y_train_sm)
    return model


def train_decision_tree(X_train_sm, y_train_sm) -> DecisionTreeClassifier:
    """Report Section 4.2 — fixed shallow configuration for interpretability."""
    model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
    model.fit(X_train_sm, y_train_sm)
    return model


def train_random_forest(X_train_sm, y_train_sm, tune: bool = True) -> RandomForestClassifier:
    """
    Report Section 4.2 — Random Forest with regularising GridSearchCV.
    An initial unconstrained (max_depth=None) run overfit badly (train F1 0.999
    vs test F1 0.584), so the search grid below only contains regularising
    configurations, per the report's account of this discovery.
    """
    if not tune:
        return RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                       random_state=42).fit(X_train_sm, y_train_sm)
    param_grid = {"n_estimators": [100, 200], "max_depth": [8, 10, 12], "min_samples_leaf": [5, 10]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train_sm, y_train_sm)
    return grid.best_estimator_


def train_xgboost(X_train_sm, y_train_sm, tune: bool = True) -> XGBClassifier:
    """Report Section 4.2 — XGBoost with regularising GridSearchCV."""
    if not tune:
        return XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=150, min_child_weight=10,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1,
                              random_state=42, eval_metric="logloss").fit(X_train_sm, y_train_sm)
    param_grid = {
        "max_depth": [3, 4], "learning_rate": [0.05, 0.1], "n_estimators": [100, 150],
        "min_child_weight": [5, 10], "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [1, 5],
    }
    grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric="logloss"), param_grid,
                         cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train_sm, y_train_sm)
    return grid.best_estimator_


def train_all_models(X_train_sm, y_train_sm, tune: bool = True) -> dict:
    """Trains all four models and returns them in a dict, in report order."""
    return {
        "Logistic Regression": train_logistic_regression(X_train_sm, y_train_sm),
        "Decision Tree": train_decision_tree(X_train_sm, y_train_sm),
        "Random Forest": train_random_forest(X_train_sm, y_train_sm, tune=tune),
        "XGBoost": train_xgboost(X_train_sm, y_train_sm, tune=tune),
    }


def full_training_pipeline(X: pd.DataFrame, y: pd.Series, tune: bool = True):
    """
    Convenience function running Section 3.6-3.8 and Section 4 end to end.
    Returns (models, X_train_sm, y_train_sm, X_test_scaled, y_test, scaler).
    """
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_train_sm, y_train_sm = resample_smote(X_train_scaled, y_train)
    models = train_all_models(X_train_sm, y_train_sm, tune=tune)
    return models, X_train_sm, y_train_sm, X_test_scaled, y_test, scaler