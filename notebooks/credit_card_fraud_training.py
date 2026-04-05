# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Card Fraud Detection — Model Training & Registration
# MAGIC
# MAGIC This notebook trains a **Decision Tree Classifier** for credit card fraud detection,
# MAGIC tunes hyperparameters with **5-fold cross-validation**, handles class imbalance
# MAGIC with **SMOTE**, and registers the best model in the **Databricks MLflow** registry.
# MAGIC
# MAGIC > **Compatible with Databricks Free Edition.**
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Load and explore the dataset
# MAGIC 2. Preprocess features
# MAGIC 3. Handle class imbalance (SMOTE)
# MAGIC 4. Hyperparameter tuning (GridSearchCV with 5-fold CV)
# MAGIC 5. Evaluate the best model
# MAGIC 6. Log experiment to MLflow
# MAGIC 7. Register model in MLflow Model Registry

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies & Imports

# COMMAND ----------

# MAGIC %pip install scikit-learn==1.4.0 imbalanced-learn==0.12.3

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import warnings

warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Dataset
# MAGIC
# MAGIC Upload `fraud_data.csv` to DBFS via the Databricks UI before running this cell.
# MAGIC
# MAGIC **How to upload:**
# MAGIC 1. Click **"Bring in data"** on the home page (or sidebar → **Data Ingestion**)
# MAGIC 2. Upload `fraud_data.csv` — it will be stored at `/FileStore/tables/fraud_data.csv`
# MAGIC
# MAGIC Alternatively, use the **Catalog** → **Create Table** → **Upload File** flow.

# COMMAND ----------

# Load from DBFS (uploaded via UI)
df = pd.read_csv("/dbfs/FileStore/tables/fraud_data.csv")

# Alternative: if using Git Folders (connect your repo via sidebar)
# df = pd.read_csv("fraud_data.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Explore & Preprocess

# COMMAND ----------

# Check class distribution
print("Target distribution:")
print(df["is_fraud"].value_counts())
print(f"\nFraud rate: {df['is_fraud'].mean():.4%}")

# COMMAND ----------

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# COMMAND ----------

# Encode categorical features
label_encoders = {}
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print(f"Encoded {len(categorical_cols)} categorical columns: {categorical_cols}")

# COMMAND ----------

# Define features and target
TARGET = "is_fraud"
FEATURES = [col for col in df.columns if col != TARGET]

X = df[FEATURES]
y = df[TARGET]

print(f"Features: {len(FEATURES)}")
print(f"Samples: {len(X)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train/Test Split & SMOTE

# COMMAND ----------

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Train fraud rate: {y_train.mean():.4%}")
print(f"Test fraud rate:  {y_test.mean():.4%}")

# COMMAND ----------

# Apply SMOTE to handle class imbalance on training data only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"Train size: {X_train_resampled.shape[0]}")
print(f"Class distribution:\n{pd.Series(y_train_resampled).value_counts()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Hyperparameter Tuning with 5-Fold Cross-Validation

# COMMAND ----------

# Define the parameter grid
param_grid = {
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "criterion": ["gini", "entropy"],
}

# Stratified 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

print("Starting GridSearchCV (5-fold)...")
grid_search.fit(X_train_resampled, y_train_resampled)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluate Best Model on Test Set

# COMMAND ----------

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

# Calculate metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
}

print("=== Test Set Performance ===")
for metric_name, value in metrics.items():
    print(f"  {metric_name}: {value:.4f}")

print(f"\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

print(f"=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Log to MLflow & Register Model

# COMMAND ----------

# Model name in the MLflow registry
REGISTERED_MODEL_NAME = "credit-card-fraud-classifier"

# Start an MLflow run
with mlflow.start_run(run_name="decision_tree_fraud_classifier") as run:

    # Log hyperparameters
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("smote_applied", True)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring_metric", "f1")

    # Log performance metrics
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value)

    mlflow.log_metric("best_cv_f1_score", grid_search.best_score_)

    # Infer model signature
    signature = infer_signature(X_test, y_pred)

    # Log and register the model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        input_example=X_test.iloc[:3],
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    print(f"\nMLflow Run ID: {run.info.run_id}")
    print(f"Model registered as: {REGISTERED_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Verify Registration
# MAGIC
# MAGIC After running this notebook:
# MAGIC 1. Go to **Machine Learning → Models** in the sidebar to see the registered model
# MAGIC 2. Go to **Machine Learning → Experiments** to view logged parameters and metrics
# MAGIC 3. The model is now ready to be exposed for serving/inference

# COMMAND ----------

# Quick verification: load the model back from the registry
model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Sanity check
sample_preds = loaded_model.predict(X_test.iloc[:5])
print(f"Sample predictions from registry model: {sample_preds}")
print(f"Actual labels:                          {y_test.iloc[:5].values}")
print("\nModel successfully registered and verified!")
