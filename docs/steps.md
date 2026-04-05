Steps to run this in Databricks (in order):

Create Resource Group — follow 01-create-resource-group.md. Name: rg-ml-fraud-detection

Create Databricks Workspace — follow 02-create-databricks-workspace.md. Use Premium tier (required for MLflow registry)

Create Compute Cluster — follow 03-create-compute-cluster.md. Single Node, Runtime 14.3 LTS, Standard_DS3_v2

Install libraries — on the cluster's Libraries tab, install: scikit-learn==1.4.0, pandas==2.2.2, imbalanced-learn==0.12.3, mlflow==2.18.0

Push this repo to GitHub, then in Databricks: Workspace → Create → Git Folder → paste your repo URL

Download fraud_data.csv from Kaggle and upload it to your Databricks workspace or DBFS

Open credit_card_fraud_training.py, attach to your cluster, and Run All cells

Verify — go to Machine Learning → Models in the sidebar to confirm credit-card-fraud-classifier is registered. See 04-model-registry-walkthrough.md for details.