import os
import mysql.connector
from model_registry import register_model, get_production_model, promote_model
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Connect
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password=os.environ["MYSQL_ROOT_PASSWORD"],
    database="Insidersignal"
)

# 2. Fake a trained model + metrics to test with
fake_model = LogisticRegression()
fake_model.fit([[0,0],[1,1]], [0,1])

metrics = {
    "n_rows": 100,
    "accuracy": 0.85,
    "precision": 0.80,
    "recall": 0.75,
    "f1": 0.77,
    "confusion_matrix": np.array([[40, 5], [10, 45]])
}

# 3. Register it
version = register_model(fake_model, metrics, ["feature1", "feature2"], {"C": 1.0}, conn)
print("Registered:", version)

# 4. Promote it
promote_model(version, conn)
print("Promoted:", version)

# 5. Load it back
model, row = get_production_model(conn)
print("Loaded model version:", row["model_version"], "F1:", row["f1_score"])
