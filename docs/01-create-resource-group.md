# Step 1: Create a Resource Group

A **Resource Group** is a logical container in Azure that holds related resources for your project.

---

## Steps

### 1.1 Sign in to Azure Portal

- Go to [https://portal.azure.com](https://portal.azure.com)
- Sign in with your Azure account credentials

### 1.2 Navigate to Resource Groups

- In the portal search bar at the top, type **"Resource groups"**
- Click **"Resource groups"** from the results

### 1.3 Create a New Resource Group

1. Click the **"+ Create"** button at the top
2. Fill in the form:

| Field | Value |
|---|---|
| **Subscription** | Select your Azure subscription |
| **Resource group** | `rg-ml-fraud-detection` |
| **Region** | Choose your preferred region (e.g., `East US`, `West Europe`) |

3. Click **"Review + Create"**
4. Review the details, then click **"Create"**

### 1.4 Verify Creation

- You should see a notification: **"Resource group created"**
- Navigate back to **Resource groups** and confirm `rg-ml-fraud-detection` appears in the list

---

## Tips

- **Naming convention:** Use a descriptive prefix like `rg-` for resource groups
- **Region selection:** Pick a region close to you or your team for lower latency. Ensure your chosen region supports Azure Databricks
- **Cost:** Resource groups themselves are free — you're only billed for the resources inside them

---

## Next Step

Proceed to [02 — Create a Databricks Workspace](02-create-databricks-workspace.md).
