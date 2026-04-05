Steps to run this in Databricks Free Edition (in order):

1. Sign up at https://www.databricks.com/try-databricks — use your university/work email, no credit card needed

2. Create a Cluster — Compute → Create Compute → name it "ml-fraud-cluster", pick Single Node, Runtime 14.3 LTS → Create Compute (wait ~3-5 min)

3. Download the dataset — get fraud_data.csv from https://www.kaggle.com/datasets/neharoychoudhury/credit-card-fraud-data

4. Upload the dataset — Home page → "Bring in data" (or Data Ingestion in sidebar) → upload fraud_data.csv (stored at /FileStore/tables/fraud_data.csv)

5. Import the notebook (pick one):
   Option A (recommended): Home page → "Connect to a GitHub repo" → authenticate → select this repo → opens as Git Folder
   Option B: Workspace → your user folder → ▼ → Import → upload notebooks/credit_card_fraud_training.py

6. Attach & Run — open the notebook, attach to "ml-fraud-cluster" in the top dropdown, click Run All

7. Verify — AI/ML → Models → confirm "credit-card-fraud-classifier" is registered. Also check AI/ML → Experiments for logged params and metrics.

Note: Libraries are installed automatically via %pip install in the first notebook cell — no manual cluster library setup needed.