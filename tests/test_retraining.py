import numpy as np
from sklearn.linear_model import LogisticRegression

from model_registry import register_model, promote_model, get_production_model
from retraining_job import retraining_job


def _register_fake_model(conn, score):
    fake_model = LogisticRegression()
    fake_model.fit([[0, 0], [1, 1]], [0, 1])
    metrics = {
        "n_rows": 10,
        "accuracy": score,
        "precision": score,
        "recall": score,
        "f1": score,
        "confusion_matrix": np.array([[5, 0], [0, 5]]),
    }
    return register_model(fake_model, metrics, ["feature1", "feature2"], {}, conn)


def test_promote_model_flips_exactly_one_flag(conn):
    version_a = _register_fake_model(conn, 0.70)
    promote_model(version_a, conn)

    version_b = _register_fake_model(conn, 0.85)
    promote_model(version_b, conn)

    cursor = conn.cursor()
    cursor.execute("SELECT model_version FROM model_registry WHERE is_production = TRUE")
    production_versions = cursor.fetchall()

    assert len(production_versions) == 1
    assert production_versions[0][0] == version_b


def test_get_production_model_returns_latest_promoted(conn):
    version = _register_fake_model(conn, 0.95)
    promote_model(version, conn)

    _, row = get_production_model(conn)
    assert row["model_version"] == version


def test_retraining_job_runs_and_returns_bool(conn):
    baseline_version = _register_fake_model(conn, 0.5)
    promote_model(baseline_version, conn)

    promoted = retraining_job(conn, trigger_reason="test")
    assert isinstance(promoted, bool)
