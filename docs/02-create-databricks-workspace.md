# Step 2: Create an Azure Databricks Workspace

The **Databricks workspace** is where you'll run notebooks, manage clusters, and access the MLflow model registry.

---

## Steps

### 2.1 Navigate to Azure Databricks

- In the Azure Portal search bar, type **"Azure Databricks"**
- Click **"Azure Databricks"** from the results

### 2.2 Create a New Workspace

1. Click **"+ Create"** at the top
2. Fill in the form:

| Field | Value |
|---|---|
| **Subscription** | Select your Azure subscription |
| **Resource group** | Select `rg-ml-fraud-detection` (created in Step 1) |
| **Workspace name** | `dbx-fraud-detection` |
| **Region** | Same region as your resource group |
| **Pricing tier** | **Premium** (required for MLflow Model Registry) |

3. Click **"Review + Create"**
4. Review the details, then click **"Create"**

> **Note:** Deployment takes approximately 2–5 minutes.

### 2.3 Launch the Workspace

1. Once deployment completes, click **"Go to resource"**
2. On the resource overview page, click **"Launch Workspace"**
3. This opens the Databricks workspace UI in a new browser tab

### 2.4 Verify the Workspace

- You should see the Databricks landing page with sidebar options:
  - **Workspace** — file browser for notebooks
  - **Compute** — manage clusters
  - **Machine Learning** — experiments and models
  - **Data** — data explorer

---

## Tips

- **Pricing tier:** The **Premium** tier is required for Unity Catalog and the managed MLflow Model Registry. The **Standard** tier does not include the model registry
- **Managed resource group:** Azure automatically creates a second resource group (e.g., `databricks-rg-dbx-fraud-detection-xxxxx`) for Databricks-managed infrastructure — do not delete it
- **Bookmark** the workspace URL for quick access later

---

## Next Step

Proceed to [03 — Create a Compute Cluster](03-create-compute-cluster.md).
