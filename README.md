# Train and Deploy ML Model on Azure DBX

End-to-end guide for training a machine learning model in **Azure Databricks** and registering it to the managed **MLflow** model registry — ready for production inference.

> **Use Case:** Credit card fraud detection using a decision-tree classifier with hyperparameter tuning via 5-fold cross-validation.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup Guide](#setup-guide)
  - [1. Create a Resource Group](#1-create-a-resource-group)
  - [2. Create a Databricks Workspace](#2-create-a-databricks-workspace)
  - [3. Create a Compute Cluster](#3-create-a-compute-cluster)
  - [4. Install Required Libraries](#4-install-required-libraries)
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

1. **Provision** an Azure Databricks workspace with a compute cluster
2. **Train** a decision-tree fraud classifier with scikit-learn inside a Databricks notebook
3. **Tune** hyperparameters using 5-fold cross-validation
4. **Handle class imbalance** with `imbalanced-learn` (SMOTE)
5. **Track** experiments, parameters, and metrics with MLflow
6. **Register** the trained model in the Azure Databricks managed MLflow registry
7. **Expose** the model for downstream inference (e.g., real-time fraud detection pipelines)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Azure Portal                         │
│  ┌───────────────┐    ┌──────────────────────────────┐  │
│  │ Resource Group │───▶│   Azure Databricks Workspace │  │
│  └───────────────┘    │                              │  │
│                       │  ┌────────────────────────┐  │  │
│                       │  │   Compute Cluster       │  │  │
│                       │  │   (Single Node DS3_v2)  │  │  │
│                       │  └──────────┬─────────────┘  │  │
│                       │             │                 │  │
│                       │  ┌──────────▼─────────────┐  │  │
│                       │  │  Jupyter Notebook       │  │  │
│                       │  │  - Load Dataset         │  │  │
│                       │  │  - Preprocess (SMOTE)   │  │  │
│                       │  │  - Train (DecisionTree) │  │  │
│                       │  │  - 5-Fold CV Tuning     │  │  │
│                       │  │  - Log to MLflow        │  │  │
│                       │  └──────────┬─────────────┘  │  │
│                       │             │                 │  │
│                       │  ┌──────────▼─────────────┐  │  │
│                       │  │  MLflow Model Registry  │  │  │
│                       │  │  "credit-card-fraud-    │  │  │
│                       │  │   classifier"           │  │  │
│                       │  └────────────────────────┘  │  │
│                       └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Cloud Platform | Microsoft Azure |
| Analytics Platform | Azure Databricks |
| ML Tracking & Registry | MLflow (Databricks-managed) |
| ML Framework | scikit-learn 1.4.0 |
| Data Processing | pandas 2.2.2 |
| Class Imbalance | imbalanced-learn 0.12.3 |
| Language | Python 3.10+ |
| Notebook Runtime | Databricks Runtime 14.3 LTS |

---

## Prerequisites

- An active **Azure subscription** ([free trial available](https://portal.azure.com/))
- Permissions to create Resource Groups, Databricks workspaces, and Compute Clusters
- Basic familiarity with Jupyter notebooks and Python ML workflows

---

## Setup Guide

### 1. Create a Resource Group

1. Log in to the [Azure Portal](https://portal.azure.com/)
2. Navigate to **Resource Groups** → click **"Create"**
3. Select your subscription, provide a name (e.g., `rg-ml-fraud-detection`), and choose a region
4. Click **"Review + Create"** → **"Create"**

> See [`docs/01-create-resource-group.md`](docs/01-create-resource-group.md) for detailed walkthrough.

### 2. Create a Databricks Workspace

1. Navigate to **Azure Databricks** in the portal
2. Click **"Create"** → choose your subscription and the resource group from Step 1
3. Name the workspace (e.g., `dbx-fraud-detection`) and select the same region
4. Click **"Review + Create"** → **"Create"**
5. Once deployed, click **"Go to resource"** → **"Launch Workspace"**

> See [`docs/02-create-databricks-workspace.md`](docs/02-create-databricks-workspace.md) for details.

### 3. Create a Compute Cluster

1. In the Databricks workspace sidebar, click **Compute**
2. Click **"Create Compute"** and configure:
   - **Cluster Mode:** Single Node
   - **Databricks Runtime:** 14.3 LTS (Python 3.10)
   - **Node Type:** Standard_DS3_v2 (or your preference)
3. Click **"Create Cluster"** — provisioning takes ~3-5 minutes

> See [`docs/03-create-compute-cluster.md`](docs/03-create-compute-cluster.md) for details.

### 4. Install Required Libraries

In the Databricks workspace:

1. Go to **Compute** → click on your cluster → **Libraries** tab
2. Click **"Install New"** → **Library Source: PyPI**
3. Install each library one at a time:

```
scikit-learn==1.4.0
pandas==2.2.2
imbalanced-learn==0.12.3
mlflow-skinny[databricks]
mlflow==2.18.0
```

> **Note:** On Databricks Runtime ≥ 15.0, you can alternatively use the `requirements.txt` file directly.

---

## Training the Model

### Import the Notebook

1. In the Databricks sidebar, click **Workspace**
2. Click **"Create"** → **"Git Folder"**
3. Enter the repository URL:
   ```
   https://github.com/<your-username>/ML_with_Databricks
   ```
4. Click **"Create"** to clone the repo into your workspace

### Run the Notebook

1. Open `notebooks/credit_card_fraud_training.py`
2. Attach it to your compute cluster via the dropdown at the top
3. Execute each cell sequentially — the notebook will:
   - Load and preprocess the credit card fraud dataset
   - Apply SMOTE to handle class imbalance
   - Run 5-fold cross-validated grid search over decision tree hyperparameters
   - Log all parameters, metrics, and artifacts to MLflow
   - Register the best model as **`credit-card-fraud-classifier`** in the MLflow registry

---

## Model Registration & Tracking

### Registered Models

After notebook execution, verify registration:

1. In the sidebar → **Machine Learning** → **Models**
2. Confirm `credit-card-fraud-classifier` appears with a version number

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
    ├── 01-create-resource-group.md
    ├── 02-create-databricks-workspace.md
    ├── 03-create-compute-cluster.md
    └── 04-model-registry-walkthrough.md
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

- **Expose for inference:** By default, a registered model in Azure Databricks is not immediately available for serving — it must be explicitly exposed via a model serving endpoint
- **Real-time integration:** Connect the served model to a streaming pipeline (e.g., Kafka + Nussknacker, Azure Event Hubs, or a custom REST consumer)
- **CI/CD:** Add GitHub Actions workflows for automated retraining and model promotion
- **Monitoring:** Set up model drift detection and performance monitoring dashboards

---

## References

- [Azure Databricks Documentation](https://learn.microsoft.com/en-us/azure/databricks/introduction/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking/)
- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/neharoychoudhury/credit-card-fraud-data)

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
