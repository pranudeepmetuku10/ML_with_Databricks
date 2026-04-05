# Step 3: Create a Compute Cluster

A **compute cluster** provides the processing power to run your notebooks. For this project, a single-node cluster is sufficient.

---

## Steps

### 3.1 Open Compute Settings

1. In the Databricks workspace sidebar, click **"Compute"**
2. Click **"Create Compute"** (or **"Create cluster"**)

### 3.2 Configure the Cluster

Fill in the following settings:

| Setting | Value |
|---|---|
| **Cluster name** | `ml-fraud-cluster` |
| **Cluster mode** | **Single Node** |
| **Access mode** | Single user (select your username) |
| **Databricks Runtime** | **14.3 LTS** (includes Python 3.10, Apache Spark 3.5) |
| **Node type** | `Standard_DS3_v2` (14 GB RAM, 4 cores) |
| **Auto termination** | 30 minutes of inactivity (saves cost) |

### 3.3 Create the Cluster

1. Click **"Create Cluster"**
2. Wait for the status to change from **Pending** → **Running** (takes ~3–5 minutes)
3. The green circle indicator confirms the cluster is ready

### 3.4 Install Required Libraries

Once the cluster is running:

1. Click on your cluster name (`ml-fraud-cluster`)
2. Go to the **"Libraries"** tab
3. Click **"Install New"**
4. Select **Library Source: PyPI**
5. Install each library one at a time:

| Package | Version |
|---|---|
| `scikit-learn` | `1.4.0` |
| `pandas` | `2.2.2` |
| `imbalanced-learn` | `0.12.3` |
| `mlflow` | `2.18.0` |

6. Wait for each library status to show **"Installed"**

> **Note:** `mlflow` and `pandas` may already be pre-installed on Databricks Runtime 14.3 LTS. Check the library list before installing duplicates.

---

## Tips

- **Cost management:** Single-node clusters are the cheapest option. Enable auto-termination to avoid unnecessary charges
- **Runtime version:** Always use an **LTS** (Long Term Support) runtime for stability
- **Scaling:** For larger datasets, switch to a multi-node cluster with autoscaling (min 2, max 8 workers)
- **Alternative:** You can also use an **All-Purpose** cluster for interactive development, or a **Job Cluster** for scheduled production runs

---

## Verify

After the cluster is running and libraries are installed:

1. Go to **Workspace** → open any notebook
2. Attach the notebook to `ml-fraud-cluster` via the cluster dropdown at the top
3. Run a test cell:

```python
import sklearn, pandas, imblearn, mlflow
print(f"sklearn: {sklearn.__version__}")
print(f"pandas: {pandas.__version__}")
print(f"imblearn: {imblearn.__version__}")
print(f"mlflow: {mlflow.__version__}")
```

All imports should succeed without errors.

---

## Next Step

Proceed to [04 — Model Registry Walkthrough](04-model-registry-walkthrough.md) (after running the training notebook).
