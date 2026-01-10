"""Pytest configuration and fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_binary_data():
    """Generate sample binary classification data for testing."""
    np.random.seed(42)
    n_samples = 1000

    # Ground truth labels
    y_true = np.random.randint(0, 2, n_samples)

    # Predictions (slightly correlated with truth)
    noise = np.random.rand(n_samples) > 0.7
    y_pred = np.where(noise, 1 - y_true, y_true)

    # Probabilities
    y_proba = np.clip(y_true * 0.7 + np.random.rand(n_samples) * 0.3, 0, 1)

    # Sensitive attribute with bias
    # Males more likely to be predicted positive
    sensitive = np.where(np.random.rand(n_samples) > 0.5, "Male", "Female")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "sensitive": sensitive,
    }


@pytest.fixture
def sample_adult_data():
    """Generate sample Adult-like data for testing."""
    np.random.seed(42)
    n_samples = 500

    data = {
        "age": np.random.randint(18, 70, n_samples),
        "fnlwgt": np.random.randint(10000, 1000000, n_samples),
        "education_num": np.random.randint(1, 16, n_samples),
        "capital_gain": np.random.randint(0, 100000, n_samples),
        "capital_loss": np.random.randint(0, 5000, n_samples),
        "hours_per_week": np.random.randint(10, 80, n_samples),
        "workclass": np.random.choice(
            ["Private", "Self-emp", "Gov", "Other"], n_samples
        ),
        "education": np.random.choice(
            ["HS-grad", "Bachelors", "Masters", "Doctorate", "Some-college"],
            n_samples,
        ),
        "marital_status": np.random.choice(
            ["Married", "Never-married", "Divorced"], n_samples
        ),
        "occupation": np.random.choice(
            ["Tech", "Sales", "Admin", "Exec", "Other"], n_samples
        ),
        "relationship": np.random.choice(
            ["Husband", "Wife", "Own-child", "Not-in-family"], n_samples
        ),
        "native_country": np.random.choice(
            ["United-States", "Mexico", "Other"], n_samples
        ),
        "sex": np.random.choice(["Male", "Female"], n_samples),
        "race": np.random.choice(["White", "Black", "Asian", "Other"], n_samples),
        "income": np.random.choice([">50K", "<=50K"], n_samples),
        "split": np.random.choice(["train", "test"], n_samples, p=[0.8, 0.2]),
    }

    df = pd.DataFrame(data)
    df["race_binary"] = df["race"].apply(lambda x: "White" if x == "White" else "Non-White")
    return df


@pytest.fixture
def biased_predictions():
    """Generate predictions with clear fairness disparity for testing."""
    np.random.seed(42)
    n_samples = 1000

    sensitive = np.array(["Male"] * 500 + ["Female"] * 500)

    # True labels - equal base rates
    y_true = np.concatenate([
        np.random.randint(0, 2, 500),
        np.random.randint(0, 2, 500),
    ])

    # Biased predictions - higher positive rate for males
    y_pred_male = np.where(np.random.rand(500) > 0.3, 1, 0)  # 70% positive
    y_pred_female = np.where(np.random.rand(500) > 0.7, 1, 0)  # 30% positive
    y_pred = np.concatenate([y_pred_male, y_pred_female])

    # Probabilities with similar bias
    y_proba = np.concatenate([
        np.clip(0.6 + np.random.randn(500) * 0.2, 0, 1),
        np.clip(0.4 + np.random.randn(500) * 0.2, 0, 1),
    ])

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "sensitive": sensitive,
    }
