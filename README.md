# ML with Databricks

<div align="center">

**Train and Deploy Credit Card Fraud Detection on Databricks**

![Databricks](https://img.shields.io/badge/Databricks-Free%20Edition-0078D4?style=flat-square&logo=databricks)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-00aaff?style=flat-square&logo=mlflow)
![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-f7931e?style=flat-square&logo=scikit-learn)

> End-to-end machine learning pipeline with **hyperparameter tuning**, **class imbalance handling**, and **model tracking** • **100% Free Edition** • No credit card needed

</div>

---

## Navigation

| [Overview](#overview) | [Quick Start](#quick-start) | [Architecture](#architecture) | [Tech Stack](#tech-stack) | [Project Structure](#project-structure) | [License](#license) |
|:---:|:---:|:---:|:---:|:---:|:---:|

---

## Overview

This project provides a **production-ready ML pipeline** for credit card fraud detection on Databricks.

### What You'll Do

```
Data Loading → Preprocessing → SMOTE → Model Training → 5-Fold CV → MLflow Tracking → Model Registry
```

### What You'll Learn

- Set up Databricks Free Edition (no credit card required)
- Build a fraud classifier with scikit-learn
- Handle imbalanced data with SMOTE
- Tune hyperparameters with cross-validation
- Track experiments and metrics with MLflow
- Register models for production

---

## Quick Start

### 1. Sign Up for Databricks

```bash
Visit → https://www.databricks.com/try-databricks
Sign up with your work/university email
Activate Free Edition (no credit card needed!)
```

### 2. Create a Compute Cluster

| Setting | Value |
|---------|-------|
| **Cluster Name** | `ml-fraud-cluster` |
| **Mode** | Single Node |
| **Runtime** | 14.3 LTS |
| **Time** | ~3–5 min startup |

### 3. Get the Code

**Option A: GitHub (Recommended)**
```bash
Databricks Home → Connect to GitHub → Select this repo
```

**Option B: Manual Import**
```bash
Workspace → Import → Upload notebooks/credit_card_fraud_training.py
```

### 4. Download Dataset

```bash
Kaggle → neharoychoudhury/credit-card-fraud-data
Upload to Databricks (DBFS or workspace)
```

### 5. Run the Notebook

```python
# Opens: notebooks/credit_card_fraud_training.py
# Attach to: ml-fraud-cluster
# Execute: All cells in order
```

**Done!** Your model will be registered in MLflow Model Registry

---

## Architecture

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Databricks Free Edition Workspace                 ┃
┃                                                    ┃
┃  ┌────────────────────────────────────┐           ┃
┃  │ Compute Cluster (Single Node)      │           ┃
┃  │ Runtime: 14.3 LTS                  │           ┃
┃  └──────────────┬───────────────────┘            ┃
┃                 │                                 ┃
┃  ┌──────────────▼────────────────────┐           ┃
┃  │ Databricks Notebook                │           ┃
┃  │                                    │           ┃
┃  │  ├─ Load Data (DBFS)               │           ┃
┃  │  ├─ Preprocess & SMOTE             │           ┃
┃  │  ├─ Train DecisionTree Classifier  │           ┃
┃  │  ├─ 5-Fold Cross-Validation        │           ┃
┃  │  ├─ Hyperparameter Tuning          │           ┃
┃  │  └─ Log Metrics to MLflow          │           ┃
┃  │                                    │           ┃
┃  └──────────────┬───────────────────┘            ┃
┃                 │                                 ┃
┃  ┌──────────────▼────────────────────┐           ┃
┃  │ MLflow Model Registry              │           ┃
┃  │                                    │           ┃
┃  │ Model: fraud-classifier            │           ┃
┃  │ Version: 1, 2, 3, ...              │           ┃
┃  │ Status: Production                 │           ┃
┃  │                                    │           ┃
┃  └────────────────────────────────────┘           ┃
┃                                                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## Tech Stack

| Category | Technology | Version |
|----------|-----------|---------|
| **Platform** | Databricks | Free Edition |
| **ML Framework** | scikit-learn | 1.4.0 |
| **Data Processing** | pandas | 2.2.2 |
| **Class Imbalance** | imbalanced-learn (SMOTE) | 0.12.3 |
| **Experiment Tracking** | MLflow | Built-in |
| **Model Registry** | MLflow Model Registry | Built-in |
| **Runtime** | Databricks Runtime | 14.3 LTS |
| **Language** | Python | 3.10+ |

---

## Detailed Setup

### Step 1: Sign Up & Login

See [docs/01-sign-up-and-login.md](docs/01-sign-up-and-login.md)

1. Visit [https://www.databricks.com/try-databricks](https://www.databricks.com/try-databricks)
2. Sign up with your university or work email → **Free Edition**
3. You'll see the Databricks workspace home page

### Step 2: Create a Compute Cluster

See [docs/02-create-cluster.md](docs/02-create-cluster.md)

1. Sidebar → **Compute** → **Create Compute**
2. Configure:
   - **Name:** `ml-fraud-cluster`
   - **Mode:** Single Node
   - **Runtime:** 14.3 LTS
3. Wait ~3–5 minutes for startup

### Step 3: Upload the Dataset

1. Download `fraud_data.csv` from [Kaggle CC Fraud Dataset](https://www.kaggle.com/datasets/neharoychoudhury/credit-card-fraud-data)
2. Databricks Home → **Bring in data**
3. Upload the CSV → stored at `/FileStore/tables/fraud_data.csv`

### Step 4: Import the Notebook

**Option A: Git Folder (Recommended)**
- Home → **Connect to a GitHub repo**
- Authenticate & select this repository
- The repo syncs as a Git Folder in your workspace

**Option B: Manual Upload**
- Sidebar → **Workspace** → your folder
- Click **▼** → **Import** → select `notebooks/credit_card_fraud_training.py`

### Step 5: Run the Training

1. Open `credit_card_fraud_training`
2. Top dropdown → attach to `ml-fraud-cluster`
3. Execute all cells in order

The notebook will:
- Load and preprocess the fraud dataset
- Apply SMOTE to fix class imbalance
- Run 5-fold CV grid search over decision tree hyperparams
- Log all metrics & params to MLflow
- Register the best model as `credit-card-fraud-classifier`

---

## Model Registration & Tracking

### View Registered Models

1. Sidebar → **Machine Learning** → **Models**
2. Find `credit-card-fraud-classifier` with version history
3. Click to see signature, source run, and deployment status

> Free Edition includes full MLflow Model Registry!

### Inspect Training Runs

1. Sidebar → **Machine Learning** → **Experiments**
2. Click the experiment linked to the notebook
3. Each run shows:
   - **Parameters:** `max_depth`, `min_samples_split`, etc.
   - **Metrics:** `f1_score`, `precision`, `recall`, `accuracy`
   - **Artifacts:** serialized model files & plots

---

## Project Structure

```
ML_with_Databricks/
├── README.md                            ← You are here
├── LICENSE                              Apache 2.0
├── requirements.txt                     Python dependencies
├── .gitignore                           Git rules
│
├── notebooks/
│   └── credit_card_fraud_training.py   Main Databricks notebook
│
├── scripts/
│   └── register_model.py               Standalone registration tool
│
└── docs/
    ├── 01-sign-up-and-login.md         Databricks account setup
    ├── 02-create-cluster.md            Cluster configuration guide
    └── 03-model-tracking-walkthrough.md MLflow tracking walkthrough
```

---

## Dataset

The model trains on the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/neharoychoudhury/credit-card-fraud-data).

**To use:**
1. Download `fraud_data.csv` from Kaggle
2. Upload to your Databricks workspace (DBFS or Files)
3. Update the notebook path if needed

> **Note:** Dataset not included due to licensing. See Kaggle terms.

---

## Next Steps

| Objective | Description |
|-----------|-------------|
| **Model Serving** | Create REST endpoint for real-time predictions |
| **Streaming** | Connect to Kafka/Event Hubs for real-time inference |
| **CI/CD** | GitHub Actions for automated retraining & promotion |
| **Monitoring** | Model drift detection & performance dashboards |

---

## References

- [Databricks Free Edition](https://www.databricks.com/try-databricks)
- [Databricks Documentation](https://docs.databricks.com/)
- [MLflow Official Docs](https://mlflow.org/docs/latest/)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking/)
- [Kaggle CC Fraud Dataset](https://www.kaggle.com/datasets/neharoychoudhury/credit-card-fraud-data)

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

<div align="center">

**Made for the Databricks community**

</div>
