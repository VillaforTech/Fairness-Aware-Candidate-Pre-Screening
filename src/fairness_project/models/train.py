"""Model training functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

if TYPE_CHECKING:
    import pandas as pd


ModelType = Literal["lr", "rf", "xgb"]


def train_logistic_regression(
    preprocess: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42,
    max_iter: int = 500,
    **kwargs,
) -> Pipeline:
    """
    Train a Logistic Regression model with preprocessing.

    Parameters
    ----------
    preprocess : ColumnTransformer
        Preprocessing pipeline.
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray
        Training labels.
    random_state : int
        Random state for reproducibility.
    max_iter : int
        Maximum iterations for convergence.
    **kwargs
        Additional arguments passed to LogisticRegression.

    Returns
    -------
    Pipeline
        Fitted sklearn pipeline with preprocessing and model.
    """
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_state,
                    **kwargs,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    preprocess: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 300,
    n_jobs: int = -1,
    **kwargs,
) -> Pipeline:
    """
    Train a Random Forest model with preprocessing.

    Parameters
    ----------
    preprocess : ColumnTransformer
        Preprocessing pipeline.
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray
        Training labels.
    random_state : int
        Random state for reproducibility.
    n_estimators : int
        Number of trees.
    n_jobs : int
        Number of parallel jobs.
    **kwargs
        Additional arguments passed to RandomForestClassifier.

    Returns
    -------
    Pipeline
        Fitted sklearn pipeline with preprocessing and model.
    """
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    **kwargs,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    preprocess: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    n_jobs: int = -1,
    **kwargs,
) -> Pipeline:
    """
    Train an XGBoost model with preprocessing.

    Parameters
    ----------
    preprocess : ColumnTransformer
        Preprocessing pipeline.
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray
        Training labels.
    random_state : int
        Random state for reproducibility.
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Learning rate.
    n_jobs : int
        Number of parallel jobs.
    **kwargs
        Additional arguments passed to XGBClassifier.

    Returns
    -------
    Pipeline
        Fitted sklearn pipeline with preprocessing and model.
    """
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=kwargs.pop("subsample", 0.8),
        colsample_bytree=kwargs.pop("colsample_bytree", 0.8),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
        **kwargs,
    )

    model = Pipeline([("preprocess", preprocess), ("model", xgb)])
    model.fit(X_train, y_train)
    return model


def train_model(
    model_type: ModelType,
    preprocess: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42,
    **kwargs,
) -> Pipeline:
    """
    Train a model of the specified type.

    Parameters
    ----------
    model_type : ModelType
        Type of model to train ("lr", "rf", or "xgb").
    preprocess : ColumnTransformer
        Preprocessing pipeline.
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray
        Training labels.
    random_state : int
        Random state for reproducibility.
    **kwargs
        Model-specific keyword arguments.

    Returns
    -------
    Pipeline
        Fitted sklearn pipeline.

    Raises
    ------
    ValueError
        If model_type is not recognized.
    """
    common = {
        "preprocess": preprocess,
        "X_train": X_train,
        "y_train": y_train,
        "random_state": random_state,
        **kwargs,
    }
    if model_type == "lr":
        return train_logistic_regression(**common)
    if model_type == "rf":
        return train_random_forest(**common)
    if model_type == "xgb":
        return train_xgboost(**common)
    raise ValueError(f"Unknown model type: {model_type}. Choose from ['lr', 'rf', 'xgb']")


# Aliases for backward compatibility with old module structure
train_lr = train_logistic_regression
train_rf = train_random_forest
train_xgb = train_xgboost
