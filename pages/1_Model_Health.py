import os
import numpy as np
import pandas as pd
import mysql.connector
import streamlit as st

from drift_detection import check_drift

st.set_page_config(page_title="Model Health", layout="wide")
st.title("Model Health")

FEATURES = ["feature1", "feature2"]


@st.cache_resource
def get_connection():
    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST", "localhost"),
        port=int(os.environ.get("MYSQL_PORT", 3306)),
        user=os.environ.get("MYSQL_USER", "root"),
        password=os.environ["MYSQL_ROOT_PASSWORD"],
        database=os.environ.get("MYSQL_DATABASE", "Insidersignal"),
    )


conn = get_connection()

st.header("Current Production Model")
prod_df = pd.read_sql(
    "SELECT model_version, trained_on, promoted_at, f1_score "
    "FROM model_registry WHERE is_production = TRUE LIMIT 1",
    conn,
)
if not prod_df.empty:
    row = prod_df.iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Version", row["model_version"])
    col2.metric("Trained On", str(row["trained_on"]))
    col3.metric("F1 Score", f"{row['f1_score']:.4f}")
    st.caption(f"Promoted at: {row['promoted_at']}")
    production_version = row["model_version"]
else:
    st.warning("No production model found.")
    production_version = None

st.header("F1 Score Across All Registered Models")
history_df = pd.read_sql(
    "SELECT model_version, trained_on, f1_score FROM model_registry ORDER BY trained_on",
    conn,
)
if not history_df.empty:
    chart_df = history_df.set_index("model_version")[["f1_score"]]
    st.line_chart(chart_df)
    st.dataframe(
        history_df,
        use_container_width=True,
        column_config={
            "f1_score": st.column_config.NumberColumn("F1 Score", format="%.4f"),
        },
        hide_index=True,
    )
else:
    st.info("No models registered yet.")

st.header("Run a Drift Check (demo)")
st.caption(
    "Simulates a weekly drift check using synthetic feature data, "
    "the same way the Airflow DAG does on its schedule."
)
demo_col1, demo_col2 = st.columns(2)

if demo_col1.button("Simulate check: no drift", use_container_width=True):
    if production_version:
        rng_seed = np.random.randint(0, 100000)
        np.random.seed(rng_seed)
        training_data = pd.DataFrame({
            "feature1": np.random.normal(loc=50, scale=10, size=500),
            "feature2": np.random.normal(loc=5, scale=2, size=500),
        })
        recent_data = pd.DataFrame({
            "feature1": np.random.normal(loc=50, scale=10, size=100),
            "feature2": np.random.normal(loc=5, scale=2, size=100),
        })
        any_drift = check_drift(training_data, recent_data, FEATURES, conn, production_version)
        st.success(f"Check complete. Drift detected: {any_drift}")
        st.rerun()
    else:
        st.error("No production model to check against.")

if demo_col2.button("Simulate check: drift present", use_container_width=True):
    if production_version:
        rng_seed = np.random.randint(0, 100000)
        np.random.seed(rng_seed)
        training_data = pd.DataFrame({
            "feature1": np.random.normal(loc=50, scale=10, size=500),
            "feature2": np.random.normal(loc=5, scale=2, size=500),
        })
        recent_data = pd.DataFrame({
            "feature1": np.random.normal(loc=80, scale=10, size=100),
            "feature2": np.random.normal(loc=5, scale=2, size=100),
        })
        any_drift = check_drift(training_data, recent_data, FEATURES, conn, production_version)
        st.success(f"Check complete. Drift detected: {any_drift}")
        st.rerun()
    else:
        st.error("No production model to check against.")

st.header("Recent Drift Reports")
drift_df = pd.read_sql(
    "SELECT checked_at, model_version, feature_name, drift_statistic, p_value, is_drifted "
    "FROM drift_reports ORDER BY checked_at DESC LIMIT 50",
    conn,
)
if not drift_df.empty:
    drift_df["is_drifted"] = drift_df["is_drifted"].map({1: "Drifted", 0: "Stable"})
    st.dataframe(
        drift_df,
        use_container_width=True,
        column_config={
            "drift_statistic": st.column_config.NumberColumn("Drift Statistic", format="%.4f"),
            "p_value": st.column_config.NumberColumn("P-Value", format="%.4f"),
        },
        hide_index=True,
    )
else:
    st.info("No drift checks recorded yet.")

st.header("Retraining Event Log")
retrain_df = pd.read_sql(
    "SELECT triggered_at, trigger_reason, candidate_version, previous_version, "
    "candidate_f1, previous_f1, promoted, decision_reason "
    "FROM retraining_events ORDER BY triggered_at DESC",
    conn,
)
if not retrain_df.empty:
    retrain_df["promoted"] = retrain_df["promoted"].map({1: "Promoted", 0: "Rejected"})
    st.dataframe(
        retrain_df,
        use_container_width=True,
        column_config={
            "candidate_f1": st.column_config.NumberColumn("Candidate F1", format="%.4f"),
            "previous_f1": st.column_config.NumberColumn("Previous F1", format="%.4f"),
        },
        hide_index=True,
    )
else:
    st.info("No retraining events recorded yet.")
