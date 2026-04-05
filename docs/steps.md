Steps to run this in Databricks Community Edition (in order):

1. Sign up at https://community.cloud.databricks.com — free, no credit card needed

2. Create a Cluster — Compute → Create Cluster → name it "ml-fraud-cluster", pick Runtime 14.3 LTS → Create Cluster (wait ~5 min)

3. Download the dataset — get fraud_data.csv from https://www.kaggle.com/datasets/neharoychoudhury/credit-card-fraud-data

4. Upload the dataset — Data → Create Table → Upload File → drag fraud_data.csv (stored at /FileStore/tables/fraud_data.csv)

5. Import the notebook — Workspace → your user folder → ▼ → Import → upload notebooks/credit_card_fraud_training.py from this repo

6. Attach & Run — open the notebook, attach to "ml-fraud-cluster" in the top dropdown, click Run All

7. Verify — Machine Learning → Experiments → click on the experiment → click the run → check Artifacts tab for the logged model

Note: Libraries are installed automatically via %pip install in the first notebook cell — no manual cluster setup needed.