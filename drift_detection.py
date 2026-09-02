from scipy.stats import ks_2samp
import pandas as pd
def check_drift(training_data: pd.DataFrame, recent_data: pd.DataFrame, features, conn, model_version):
    cursor = conn.cursor()
    any_drift = False

    for feature in features:
        stat, p_value = ks_2samp(training_data[feature].dropna(), recent_data[feature].dropna())
        is_drifted = p_value < 0.05

        if is_drifted:
            any_drift = True

        cursor.execute("""
            INSERT INTO drift_reports (model_version, feature_name, drift_statistic, p_value, is_drifted)
            VALUES (%s,%s,%s,%s,%s)
        """, (model_version, feature, float(stat), float(p_value), is_drifted))

    conn.commit()
    return any_drift
