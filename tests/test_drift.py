import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from drift_detection import check_drift
from model_registry import register_model

np.random.seed(42)


@pytest.fixture
def registered_version(conn):
    fake_model = LogisticRegression()
    fake_model.fit([[0, 0], [1, 1]], [0, 1])
    metrics = {
        "n_rows": 10,
        "accuracy": 0.9,
        "precision": 0.9,
        "recall": 0.9,
        "f1": 0.9,
        "confusion_matrix": np.array([[5, 0], [0, 5]]),
    }
    return register_model(fake_model, metrics, ["feature1", "feature2"], {}, conn)


@pytest.fixture
def sample_data():
    training_data = pd.DataFrame({
        "feature1": np.random.normal(loc=50, scale=10, size=500),
        "feature2": np.random.normal(loc=5, scale=2, size=500),
    })
    # Same distribution as training -> genuinely no drift, not just "close enough"
    recent_data_no_drift = pd.DataFrame({
        "feature1": np.random.normal(loc=50, scale=10, size=100),
        "feature2": np.random.normal(loc=5, scale=2, size=100),
    })
    recent_data_drifted = pd.DataFrame({
        "feature1": np.random.normal(loc=80, scale=10, size=100),
        "feature2": np.random.normal(loc=5, scale=2, size=100),
    })
    return training_data, recent_data_no_drift, recent_data_drifted


def test_no_drift_detected(conn, registered_version, sample_data):
    training_data, recent_data_no_drift, _ = sample_data
    result = check_drift(training_data, recent_data_no_drift, ["feature1", "feature2"], conn, registered_version)
    assert result is False


def test_drift_detected(conn, registered_version, sample_data):
    training_data, _, recent_data_drifted = sample_data
    result = check_drift(training_data, recent_data_drifted, ["feature1", "feature2"], conn, registered_version)
    assert result is True
