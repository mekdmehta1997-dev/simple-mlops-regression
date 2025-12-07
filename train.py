import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import mlflow
import mlflow.sklearn
import os

EXPERIMENT_NAME = "simple_regression"

def train_and_log():
    # 1. Load data
    df = pd.read_csv("data.csv")
    X = df[["size"]]
    y = df["price"]

    # 2. Train model
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    r2 = r2_score(y, preds)

    # 3. MLflow logging (local file-based tracking)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("r2", float(r2))
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Also save model separately as artifact/model.pkl
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(model, "artifacts/model.pkl")

    print("Training complete. R2:", r2)
    return float(r2)

if __name__ == "__main__":
    train_and_log()
