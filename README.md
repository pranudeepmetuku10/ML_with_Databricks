# Train and Deploy ML Model on Databricks

End-to-end guide for training a machine learning model in **Databricks Free Edition** and registering it to the **MLflow** model registry.

> **Use Case:** Credit card fraud detection using a decision-tree classifier with hyperparameter tuning via 5-fold cross-validation.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup Guide](#setup-guide)
  - [1. Sign Up / Log In](#1-sign-up--log-in)
  - [2. Create a Compute Cluster](#2-create-a-compute-cluster)
  - [3. Upload the Dataset](#3-upload-the-dataset)
  - [4. Import the Notebook](#4-import-the-notebook)
- [Training the Model](#training-the-model)
  - [Import the Notebook](#import-the-notebook)
  - [Run the Notebook](#run-the-notebook)
- [Model Registration & Tracking](#model-registration--tracking)
  - [Registered Models](#registered-models)
  - [Experiments & Runs](#experiments--runs)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Next Steps](#next-steps)
- [References](#references)
- [License](#license)

---

## Overview

This project demonstrates how to:

1. **Set up** a Databricks Free Edition workspace (no credit card needed)
2. **Train** a decision-tree fraud classifier with scikit-learn inside a Databricks notebook
3. **Tune** hyperparameters using 5-fold cross-validation
4. **Handle class imbalance** with `imbalanced-learn` (SMOTE)
5. **Track** experiments, parameters, and metrics with MLflow
6. **Register** the trained model in the MLflow Model Registry

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│          Databricks Free Edition                      │
│                                                      │
│  ┌────────────────────────┐                          │
│  │   Compute Cluster       │                          │
│  │   (Single Node)         │                          │
│  └──────────┬─────────────┘                          │
│             │                                        │
│  ┌──────────▼─────────────┐                          │
│  │  Notebook               │                          │
│  │  - %pip install deps    │                          │
│  │  - Load Dataset (DBFS)  │                          │
│  │  - Preprocess (SMOTE)   │                          │
│  │  - Train (DecisionTree) │                          │
│  │  - 5-Fold CV Tuning     │                          │
│  │  - Log to MLflow        │                          │
│  └──────────┬─────────────┘                          │
│             │                                        │
│  ┌──────────▼─────────────┐                          │
│  │  MLflow Model Registry  │                          │
│  │  "credit-card-fraud-    │                          │
│  │   classifier"           │                          │
│  └────────────────────────┘                          │
└──────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Platform | Databricks Free Edition |
| ML Tracking & Registry | MLflow (built-in) |
| Catalog | Unity Catalog |
| ML Framework | scikit-learn 1.4.0 |
| Data Processing | pandas 2.2.2 |
| Class Imbalance | imbalanced-learn 0.12.3 |
| Language | Python 3.10+ |
| Notebook Runtime | Databricks Runtime 14.3 LTS |

---

## Prerequisites

- A **Databricks** account (Free Edition works — sign up with your university/work email)
- Basic familiarity with Jupyter notebooks and Python ML workflows
- **No paid cloud subscription or credit card required**

---

## Setup Guide

### 1. Sign Up / Log In

1. Go to [https://www.databricks.com/try-databricks](https://www.databricks.com/try-databricks)
2. Sign up with your university or work email to get **Free Edition**
3. Once logged in, you’ll see the home page with sidebar options: Workspace, Compute, Catalog, etc.

> See [`docs/01-sign-up-and-login.md`](docs/01-sign-up-and-login.md) for detailed walkthrough.

### 2. Create a Compute Cluster

1. In the sidebar, click **"Compute"**
2. Click **"Create Compute"** and configure:
   - **Cluster name:** `ml-fraud-cluster`
   - **Cluster Mode:** Single Node
   - **Databricks Runtime:** 14.3 LTS (or latest available)
3. Click **"Create Compute"** — takes ~3–5 minutes to start

> See [`docs/02-create-cluster.md`](docs/02-create-cluster.md) for details.

### 3. Upload the Dataset

1. Download `fraud_data.csv` from [Kaggle](https://www.kaggle.com/datasets/neharoychoudhury/credit-card-fraud-data)
2. On the home page, click **"Bring in data"** (or sidebar → **Data Ingestion**)
3. Upload `fraud_data.csv` — it will be stored at `/FileStore/tables/fraud_data.csv`

### 4. Import the Notebook

**Option A — Connect your GitHub repo (recommended):**
1. On the home page, click **"Connect to a GitHub repo"**
2. Authenticate with GitHub and select this repository
3. The entire repo is synced into your workspace as a Git Folder

**Option B — Manual import:**
1. Sidebar → **Workspace** → your user folder
2. Click **▼** → **"Import"** → upload `notebooks/credit_card_fraud_training.py`

> Libraries are installed automatically via `%pip install` in the first cell — no manual cluster library setup needed.

---

## Training the Model

### Import the Notebook

**Option A — Git Folder (recommended):**
1. Home page → **"Connect to a GitHub repo"** → authenticate and select this repo
2. Open `notebooks/credit_card_fraud_training.py` from the synced folder

**Option B — Manual import:**
1. Sidebar → **Workspace** → your user folder
2. Click **▼** → **"Import"** → upload `notebooks/credit_card_fraud_training.py`

### Run the Notebook

1. Open `credit_card_fraud_training`
2. Attach it to your cluster (`ml-fraud-cluster`) via the dropdown at the top
3. Execute each cell sequentially — the notebook will:
   - Load and preprocess the credit card fraud dataset
   - Apply SMOTE to handle class imbalance
   - Run 5-fold cross-validated grid search over decision tree hyperparameters
   - Log all parameters, metrics, and artifacts to MLflow
   - Register the best model as **`credit-card-fraud-classifier`** in MLflow

---

## Model Registration & Tracking

### Registered Models

After notebook execution, verify registration:

1. In the sidebar → **Machine Learning** (or **AI/ML**) → **Models**
2. Confirm `credit-card-fraud-classifier` appears with a version number
3. Click on the model to see version history, signature, and source run

> **Note:** Free Edition includes the MLflow Model Registry with experiment tracking, model logging, and model versioning.

### Experiments & Runs

Inspect training details:

1. Sidebar → **Machine Learning** → **Experiments**
2. Click the experiment associated with the notebook
3. Each run contains:
   - **Parameters:** hyperparameter values (e.g., `max_depth`, `min_samples_split`)
   - **Metrics:** performance scores (e.g., `f1_score`, `precision`, `recall`, `accuracy`)
   - **Artifacts:** serialized model files

---

## Project Structure

```
ML_with_Databricks/
├── README.md                          # This file
├── LICENSE                            # Apache 2.0
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── notebooks/
│   └── credit_card_fraud_training.py  # Databricks notebook (Python)
│
├── scripts/
│   └── register_model.py             # Standalone model registration utility
│
└── docs/
    ├── 01-sign-up-and-login.md
    ├── 02-create-cluster.md
    └── 03-model-tracking-walkthrough.md
```

---

## Dataset

The model is trained on the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/neharoychoudhury/credit-card-fraud-data).

**To use:**
1. Download `fraud_data.csv` from Kaggle
2. Upload it to your Databricks workspace (DBFS or workspace files)
3. Update the file path in the notebook if necessary

> The dataset is **not** included in this repo due to licensing. See the Kaggle page for terms.

---

## Next Steps

- **Model Serving:** Create a serving endpoint to expose the model via REST API for real-time predictions
- **Real-time integration:** Connect the served model to a streaming pipeline (e.g., Kafka, Azure Event Hubs)
- **CI/CD:** Add GitHub Actions workflows for automated retraining and model promotion
- **Monitoring:** Set up model drift detection and performance monitoring dashboards

---

## References

- [Databricks Free Edition](https://www.databricks.com/try-databricks)
- [Databricks Documentation](https://docs.databricks.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking/)
- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/neharoychoudhury/credit-card-fraud-data)

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
