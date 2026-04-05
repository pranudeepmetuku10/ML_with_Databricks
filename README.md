# Train and Deploy ML Model on Databricks Community Edition

End-to-end guide for training a machine learning model in **Databricks Community Edition** (free tier) and logging it to the **MLflow** experiment tracker.

> **Use Case:** Credit card fraud detection using a decision-tree classifier with hyperparameter tuning via 5-fold cross-validation.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup Guide](#setup-guide)
  - [1. Sign Up for Community Edition](#1-sign-up-for-community-edition)
  - [2. Create a Cluster](#2-create-a-cluster)
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

1. **Set up** a Databricks Community Edition workspace (free, no Azure subscription needed)
2. **Train** a decision-tree fraud classifier with scikit-learn inside a Databricks notebook
3. **Tune** hyperparameters using 5-fold cross-validation
4. **Handle class imbalance** with `imbalanced-learn` (SMOTE)
5. **Track** experiments, parameters, and metrics with MLflow
6. **Log** the trained model to MLflow for reproducibility

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│        Databricks Community Edition (Free)            │
│                                                      │
│  ┌────────────────────────┐                          │
│  │   Compute Cluster       │                          │
│  │   (Single Node, free)   │                          │
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
│  │  MLflow Experiment       │                          │
│  │  - Parameters & Metrics  │                          │
│  │  - Model Artifacts       │                          │
│  └────────────────────────┘                          │
└──────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Platform | Databricks Community Edition (free) |
| ML Tracking | MLflow (built-in) |
| ML Framework | scikit-learn 1.4.0 |
| Data Processing | pandas 2.2.2 |
| Class Imbalance | imbalanced-learn 0.12.3 |
| Language | Python 3.10+ |
| Notebook Runtime | Databricks Runtime 14.3 LTS |

---

## Prerequisites

- A free **Databricks Community Edition** account ([sign up here](https://community.cloud.databricks.com/login.html))
- Basic familiarity with Jupyter notebooks and Python ML workflows
- **No Azure subscription or credit card required**

---

## Setup Guide

### 1. Sign Up for Community Edition

1. Go to [https://community.cloud.databricks.com/login.html](https://community.cloud.databricks.com/login.html)
2. Click **"Sign Up"** and create a free account
3. Verify your email and log in

> See [`docs/01-sign-up-community-edition.md`](docs/01-sign-up-community-edition.md) for detailed walkthrough.

### 2. Create a Cluster

1. In the sidebar, click **"Compute"**
2. Click **"Create Cluster"** and configure:
   - **Cluster name:** `ml-fraud-cluster`
   - **Databricks Runtime:** 14.3 LTS (or latest available)
   - Community Edition auto-assigns a single-node cluster
3. Click **"Create Cluster"** — takes ~5 minutes to start

> See [`docs/02-create-cluster.md`](docs/02-create-cluster.md) for details.

### 3. Upload the Dataset

1. Download `fraud_data.csv` from [Kaggle](https://www.kaggle.com/datasets/neharoychoudhury/credit-card-fraud-data)
2. In the sidebar, click **"Data"** → **"Create Table"** → **"Upload File"**
3. Drag and drop `fraud_data.csv` — it will be stored at `/FileStore/tables/fraud_data.csv`

### 4. Import the Notebook

1. In the sidebar, click **"Workspace"** → navigate to your user folder
2. Click the **▼** dropdown → **"Import"**
3. Select **"File"** and upload `notebooks/credit_card_fraud_training.py` from this repo
4. The file is automatically recognized as a Databricks notebook

> **Note:** Community Edition does not support Git Folders. Import notebooks manually via file upload.

> Libraries are installed automatically via `%pip install` in the first cell of the notebook — no manual cluster library setup needed.

---

## Training the Model

### Import the Notebook

1. In the sidebar, click **Workspace** → your user folder
2. Click **▼** → **"Import"** → upload `notebooks/credit_card_fraud_training.py`
3. The file is auto-detected as a Databricks Python notebook

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

After notebook execution, verify the model was logged:

1. In the sidebar → **Machine Learning** → **Experiments**
2. Click the experiment → click the run → **Artifacts** tab
3. Confirm the `model/` artifact exists with `MLmodel`, `model.pkl`, etc.

> **Note:** Community Edition supports MLflow experiment tracking and model logging. The full Model Registry UI (stage transitions like Staging/Production) requires a paid Databricks tier.

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
    ├── 01-sign-up-community-edition.md
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

- **Upgrade to paid Databricks:** To use the full Model Registry (stage transitions), model serving endpoints, and Git Folders, upgrade to a paid Azure/AWS/GCP Databricks workspace
- **Real-time integration:** Connect the served model to a streaming pipeline (e.g., Kafka, Azure Event Hubs)
- **CI/CD:** Add GitHub Actions workflows for automated retraining and model promotion
- **Monitoring:** Set up model drift detection and performance monitoring dashboards

---

## Community Edition vs Paid — What's Different?

| Feature | Community Edition (Free) | Paid Databricks |
|---|---|---|
| Cluster | Single node only | Multi-node, autoscaling |
| MLflow Tracking | ✅ Full support | ✅ Full support |
| MLflow Model Registry | ⚠️ Limited (log only) | ✅ Full (staging, production) |
| Model Serving | ❌ Not available | ✅ REST API endpoints |
| Git Integration | ❌ Manual import only | ✅ Git Folders / Repos |
| Library Install | `%pip install` in notebook | Cluster Libraries tab or `%pip` |
| Auto-termination | 2 hours (fixed) | Configurable |

---

## References

- [Databricks Community Edition](https://community.cloud.databricks.com/)
- [Databricks Community Edition FAQ](https://www.databricks.com/product/faq/community-edition)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking/)
- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/neharoychoudhury/credit-card-fraud-data)

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
