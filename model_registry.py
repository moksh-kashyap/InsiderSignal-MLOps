# model_registry.py

import os
import joblib
import json
from datetime import datetime


def get_next_seq(conn):
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y.%m.%d')
    cursor.execute(
        "SELECT COUNT(*) FROM model_registry WHERE model_version LIKE %s",
        (f"v{today}-%",)
    )
    count = cursor.fetchone()[0]
    return count + 1


def register_model(model, metrics, features, hyperparams, conn, artifact_dir="models/"):
    version = f"v{datetime.now().strftime('%Y.%m.%d')}-{get_next_seq(conn)}"

    os.makedirs(artifact_dir, exist_ok=True)
    artifact_path = f"{artifact_dir}{version}.joblib"
    joblib.dump(model, artifact_path)

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO model_registry
        (model_version, trained_on, training_row_count, feature_set, hyperparameters,
         accuracy, precision_score, recall_score, f1_score, confusion_matrix, artifact_path)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (version, datetime.now(), metrics["n_rows"], json.dumps(features),
          json.dumps(hyperparams), metrics["accuracy"], metrics["precision"],
          metrics["recall"], metrics["f1"], json.dumps(metrics["confusion_matrix"].tolist()),
          artifact_path))
    conn.commit()
    return version


def get_production_model(conn):
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM model_registry WHERE is_production = TRUE LIMIT 1")
    row = cursor.fetchone()
    return joblib.load(row["artifact_path"]), row


def promote_model(version, conn):
    cursor = conn.cursor()
    cursor.execute("UPDATE model_registry SET is_production = FALSE WHERE is_production = TRUE")
    cursor.execute(
        """UPDATE model_registry SET is_production = TRUE, promoted_at = NOW()
           WHERE model_version = %s""", (version,))
    conn.commit()
