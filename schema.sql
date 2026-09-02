CREATE TABLE IF NOT EXISTS model_registry (
    model_version VARCHAR(30) PRIMARY KEY,
    trained_on DATETIME,
    training_row_count INT,
    feature_set JSON,
    hyperparameters JSON,
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    confusion_matrix JSON,
    is_production BOOLEAN DEFAULT FALSE,
    promoted_at DATETIME NULL,
    artifact_path VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS drift_reports (
    report_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    checked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(30),
    feature_name VARCHAR(50),
    drift_statistic DECIMAL(8,5),
    p_value DECIMAL(8,6),
    is_drifted BOOLEAN,
    FOREIGN KEY (model_version) REFERENCES model_registry(model_version)
);

CREATE TABLE IF NOT EXISTS retraining_events (
    event_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    trigger_reason VARCHAR(100),
    candidate_version VARCHAR(30),
    previous_version VARCHAR(30),
    candidate_f1 DECIMAL(5,4),
    previous_f1 DECIMAL(5,4),
    promoted BOOLEAN,
    decision_reason VARCHAR(255),
    FOREIGN KEY (candidate_version) REFERENCES model_registry(model_version),
    FOREIGN KEY (previous_version) REFERENCES model_registry(model_version)
);
