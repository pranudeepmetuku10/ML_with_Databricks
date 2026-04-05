"""
register_model.py

Standalone utility to register an already-trained scikit-learn model
to the Azure Databricks MLflow Model Registry.

Usage (inside Databricks or with MLflow tracking URI configured):
    python scripts/register_model.py --model-path /path/to/model.pkl --model-name credit-card-fraud-classifier
"""

import argparse
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd


def register_model(model_path: str, model_name: str, sample_data_path: str = None):
    """Load a serialized sklearn model and register it in MLflow."""

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Optionally infer signature from sample data
    signature = None
    input_example = None
    if sample_data_path:
        sample_df = pd.read_csv(sample_data_path)
        X_sample = sample_df.iloc[:5]
        y_sample = model.predict(X_sample)
        signature = infer_signature(X_sample, y_sample)
        input_example = X_sample.iloc[:3]

    with mlflow.start_run(run_name=f"register_{model_name}") as run:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=model_name,
        )
        print(f"Model registered as '{model_name}'")
        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register an sklearn model to MLflow")
    parser.add_argument(
        "--model-path", required=True, help="Path to serialized model (.pkl)"
    )
    parser.add_argument(
        "--model-name",
        default="credit-card-fraud-classifier",
        help="Name in the MLflow registry",
    )
    parser.add_argument(
        "--sample-data",
        default=None,
        help="Optional CSV for signature inference",
    )
    args = parser.parse_args()
    register_model(args.model_path, args.model_name, args.sample_data)
