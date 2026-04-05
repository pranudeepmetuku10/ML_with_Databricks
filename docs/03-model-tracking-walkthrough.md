# Step 3: Model Tracking Walkthrough

After running the training notebook, your model and metrics are logged to **MLflow Experiments** inside Databricks Community Edition.

---

## 3.1 View Experiments & Runs

1. In the Databricks sidebar, click **"Machine Learning"** (or switch to the ML persona via the dropdown at top-left)
2. Click **"Experiments"** in the sidebar
3. Find the experiment associated with your notebook (named after the notebook path)
4. Click on it to see a list of runs

### Inside Each Run

Click on a run to inspect:

#### Parameters Logged

| Parameter | Example Value |
|---|---|
| `criterion` | `gini` or `entropy` |
| `max_depth` | `10` |
| `min_samples_split` | `5` |
| `min_samples_leaf` | `2` |
| `smote_applied` | `True` |
| `cv_folds` | `5` |
| `scoring_metric` | `f1` |

#### Metrics Logged

| Metric | Description |
|---|---|
| `accuracy` | Overall classification accuracy |
| `precision` | Precision for the fraud class |
| `recall` | Recall for the fraud class |
| `f1_score` | F1 score for the fraud class |
| `best_cv_f1_score` | Best cross-validation F1 from grid search |

#### Artifacts

Click the **"Artifacts"** tab to see:
- `model/` — the serialized scikit-learn model
  - `MLmodel` — MLflow model metadata
  - `model.pkl` — the trained DecisionTreeClassifier
  - `conda.yaml` — Conda environment spec
  - `requirements.txt` — pip dependencies
  - `input_example.json` — sample input data

---

## 3.2 Compare Runs

If you run the notebook multiple times (e.g., with different parameters):

1. Go to **Experiments** → select the experiment
2. Check the boxes next to 2+ runs
3. Click **"Compare"**
4. View parameter and metric comparisons side by side

---

## 3.3 Load the Model Back

You can load the logged model from any notebook attached to the same workspace:

```python
import mlflow

# Load from a specific run
model = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")

# Make predictions
predictions = model.predict(new_data)
```

To get the run ID:
1. Go to the experiment → click the run
2. The run ID is shown at the top of the run page

---

## 3.4 Community Edition Limitations

| Feature | Available? | Notes |
|---|---|---|
| MLflow Tracking | ✅ Yes | Full logging of params, metrics, artifacts |
| View Experiments | ✅ Yes | Sidebar → Machine Learning → Experiments |
| Compare Runs | ✅ Yes | Select multiple runs → Compare |
| Load Model | ✅ Yes | Use `mlflow.sklearn.load_model()` |
| Model Registry (sidebar) | ⚠️ Limited | Models are logged but the full registry UI with stage transitions (Staging/Production) requires paid Databricks |
| Model Serving | ❌ No | REST API serving endpoints require paid Databricks |

---

## Summary

| What | Where to Find It |
|---|---|
| Experiment runs | Sidebar → Machine Learning → Experiments |
| Parameters & Metrics | Inside each experiment run |
| Model artifact | Artifacts tab within a run |
| Run ID (for loading) | Top of the run detail page |

---

## Next Steps (if upgrading later)

- **Model Registry:** Upgrade to paid Databricks for full Model Registry with Staging → Production transitions
- **Model Serving:** Create REST API endpoints for real-time inference
- **Scheduled Jobs:** Automate retraining on a schedule with Databricks Jobs
