# Step 2: Create a Cluster

A cluster provides the compute resources to run your notebooks. In Community Edition, you get a **single-node** cluster for free.

---

## Steps

### 2.1 Open Compute Page

1. In the Databricks sidebar, click **"Compute"**

### 2.2 Create the Cluster

1. Click **"Create Cluster"**
2. Configure:

| Setting | Value |
|---|---|
| **Cluster name** | `ml-fraud-cluster` |
| **Databricks Runtime** | **14.3 LTS** (or latest available LTS) |

> Community Edition auto-assigns the node type — you cannot select it manually.

3. Click **"Create Cluster"**
4. Wait for the status to change from **Pending** → **Running** (~5 minutes)

### 2.3 Verify the Cluster

- A green circle next to the cluster name means it's ready
- Click on the cluster name to see details

---

## Important Notes

- **Auto-termination:** Community Edition clusters auto-terminate after **2 hours** of inactivity — this is fixed and cannot be changed
- **Re-create:** If the cluster is terminated, you need to restart it (click the play button) or create a new one
- **One cluster at a time:** Community Edition allows only 1 active cluster
- **No Libraries tab:** Install packages using `%pip install` directly in your notebook (already included in the training notebook's first cell)

---

## Verify Python Packages

After the cluster starts, you can run this in a notebook cell to check pre-installed packages:

```python
%pip install scikit-learn==1.4.0 imbalanced-learn==0.12.3
```

Then verify:

```python
import sklearn, pandas, imblearn, mlflow
print(f"sklearn:  {sklearn.__version__}")
print(f"pandas:   {pandas.__version__}")
print(f"imblearn: {imblearn.__version__}")
print(f"mlflow:   {mlflow.__version__}")
```

---

## Next Step

1. **Upload your dataset** — see the README's [Upload the Dataset](#3-upload-the-dataset) section
2. **Import the notebook** — see the README's [Import the Notebook](#4-import-the-notebook) section
3. **Run the notebook** — attach to `ml-fraud-cluster` and click Run All
4. After running, see [03 — Model Tracking Walkthrough](03-model-tracking-walkthrough.md)
