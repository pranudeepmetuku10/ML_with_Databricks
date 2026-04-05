# Step 4: Model Registry Walkthrough

After running the training notebook, your model is registered in the **MLflow Model Registry** inside Azure Databricks. This guide shows you how to verify and explore it.

---

## 4.1 View Registered Models

1. In the Databricks sidebar, click **"Machine Learning"** (or switch to the ML persona)
2. Click **"Models"** in the sidebar
3. You should see **`credit-card-fraud-classifier`** listed
4. Click on the model name to see:
   - **Version history** — each training run creates a new version
   - **Version details** — input/output schema (signature), creation timestamp
   - **Source run** — link back to the MLflow experiment run

---

## 4.2 View Experiments & Runs

1. In the sidebar, click **"Experiments"**
2. Click on the experiment associated with your notebook (usually named after the notebook path)
3. You'll see a list of runs. Click on the latest run to inspect:

### Parameters Logged

| Parameter | Example Value |
|---|---|
| `criterion` | `gini` or `entropy` |
| `max_depth` | `10` |
| `min_samples_split` | `5` |
| `min_samples_leaf` | `2` |
| `smote_applied` | `True` |
| `cv_folds` | `5` |
| `scoring_metric` | `f1` |

### Metrics Logged

| Metric | Description |
|---|---|
| `accuracy` | Overall classification accuracy |
| `precision` | Precision for the fraud class |
| `recall` | Recall for the fraud class |
| `f1_score` | F1 score for the fraud class |
| `best_cv_f1_score` | Best cross-validation F1 from grid search |

### Artifacts

- `model/` — the serialized scikit-learn model (MLmodel, conda.yaml, model.pkl, requirements.txt)
- Input example data

---

## 4.3 Model Version Management

### Transition Model Stage (Classic Registry)

If using the classic MLflow registry (non-Unity Catalog):

1. Go to **Models** → click `credit-card-fraud-classifier`
2. Click on the version you want to promote
3. In the **Stage** dropdown, transition to:
   - **Staging** — for validation/testing
   - **Production** — for serving
   - **Archived** — to retire old versions

### Alias-Based Management (Unity Catalog)

If using Unity Catalog:

1. Go to **Catalog** → navigate to your model
2. Assign aliases (e.g., `champion`, `challenger`) to specific versions
3. Reference models by alias: `models:/credit-card-fraud-classifier@champion`

---

## 4.4 Load Model for Inference

You can load the registered model from anywhere with access to the MLflow tracking server:

```python
import mlflow

# Load by version number
model = mlflow.sklearn.load_model("models:/credit-card-fraud-classifier/1")

# Load latest version
model = mlflow.sklearn.load_model("models:/credit-card-fraud-classifier/latest")

# Load by alias (Unity Catalog)
model = mlflow.sklearn.load_model("models:/credit-card-fraud-classifier@champion")

# Make predictions
predictions = model.predict(new_data)
```

---

## 4.5 Compare Runs

To compare multiple training runs side by side:

1. Go to **Experiments** → select the experiment
2. Check the boxes next to 2+ runs
3. Click **"Compare"**
4. View parameter and metric comparisons in table and chart format

This is useful for understanding how hyperparameter changes affect model performance.

---

## Summary

| What | Where to Find It |
|---|---|
| Registered model | Sidebar → Machine Learning → Models |
| Experiment runs | Sidebar → Machine Learning → Experiments |
| Parameters & Metrics | Inside each experiment run |
| Model artifact | Artifacts tab within a run |
| Model serving | Models → version → "Serve this model" (if enabled) |

---

## Next Steps

- **Enable Model Serving:** Create a serving endpoint to expose the model via REST API
- **Automate:** Set up a Databricks Job to retrain and register new model versions on a schedule
- **Monitor:** Track model performance over time with Databricks Lakehouse Monitoring
