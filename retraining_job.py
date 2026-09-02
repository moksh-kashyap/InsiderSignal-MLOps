# retraining_job.py

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from model_registry import register_model, get_production_model, promote_model


FEATURES = ["feature1", "feature2"]


# --- Stubs standing in for your real pipeline (replace these later) ---

def load_training_data(conn):
    np.random.seed(1)
    X = np.random.normal(size=(300, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=FEATURES)
    df["target"] = y
    return df


def time_series_train_test_split(df):
    X = df[FEATURES]
    y = df["target"]
    split = int(len(df) * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


def train_new_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


# --- The real Phase 4 function ---

def retraining_job(conn, trigger_reason="drift_threshold_exceeded"):
    # 1. Pull current production model
    prod_model, prod_row = get_production_model(conn)

    # 2. Train a candidate on the latest full dataset
    df = load_training_data(conn)
    X_train, X_test, y_train, y_test = time_series_train_test_split(df)

    candidate = train_new_model(X_train, y_train)
    candidate_preds = candidate.predict(X_test)
    candidate_f1 = f1_score(y_test, candidate_preds)

    # 3. Evaluate the OLD model on the SAME fresh test set (fair comparison)
    prod_preds = prod_model.predict(X_test)
    prod_f1 = f1_score(y_test, prod_preds)

    # 4. Register candidate regardless of outcome (full audit trail)
    candidate_version = register_model(
        candidate,
        {"f1": candidate_f1, "n_rows": len(X_train), "accuracy": 0, "precision": 0,
         "recall": 0, "confusion_matrix": np.array([[0, 0], [0, 0]])},
        FEATURES, {}, conn
    )

    # 5. Decide whether to promote
    promoted = candidate_f1 > prod_f1 + 0.01
    reason = (f"Candidate F1 {candidate_f1:.3f} beat production F1 {prod_f1:.3f}"
              if promoted else
              f"Candidate F1 {candidate_f1:.3f} did not exceed production F1 {prod_f1:.3f} by margin")

    if promoted:
        promote_model(candidate_version, conn)

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO retraining_events
        (trigger_reason, candidate_version, previous_version, candidate_f1, previous_f1, promoted, decision_reason)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
    """, (trigger_reason, candidate_version, prod_row["model_version"], candidate_f1, prod_f1, promoted, reason))
    conn.commit()

    print(f"Candidate: {candidate_version} | F1: {candidate_f1:.3f} vs Prod F1: {prod_f1:.3f} | Promoted: {promoted}")
    return promoted
