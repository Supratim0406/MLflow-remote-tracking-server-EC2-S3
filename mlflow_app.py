# ==============================
# 1. Imports
# ==============================
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# 2. Set MLflow tracking server (EC2)
# ==============================
mlflow.set_tracking_uri("http://ec2-44-222-112-218.compute-1.amazonaws.com:5000/")

# Set experiment (will be created if not exists)
mlflow.set_experiment("IRIS_RF_Basline_Experiment_02")

# ==============================
# 3. Load Iris dataset
# ==============================
iris = load_iris()
X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# ==============================
# 4. Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 5. Start MLflow run
# ==============================
with mlflow.start_run(run_name="RandomForest_Baseline-03"):

    # -------- Tags --------
    mlflow.set_tag("model_type", "RandomForestClassifier")
    mlflow.set_tag("dataset", "Iris")
    mlflow.set_tag("experiment_type", "baseline")

    # -------- Parameters --------
    n_estimators = 200
    max_depth = 5

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("test_size", 0.2)

    # -------- Train model --------
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # -------- Predictions --------
    y_pred = model.predict(X_test)

    # -------- Metrics --------
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    class_report = classification_report(
        y_test, y_pred, output_dict=True
    )

    for label, metrics in class_report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)

    # ==============================
    # 6. Artifacts
    # ==============================
    os.makedirs("artifacts", exist_ok=True)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    # ---- Dataset artifact ----
    dataset_path = "artifacts/iris_dataset.csv"
    df.to_csv(dataset_path, index=False)
    mlflow.log_artifact(dataset_path)

    # ---- Model artifact ----
    mlflow.sklearn.log_model(
        model,
        artifact_path="random_forest_model"
    )

# ==============================
# 7. Print final result
# ==============================
print(f"Accuracy: {accuracy:.4f}")
