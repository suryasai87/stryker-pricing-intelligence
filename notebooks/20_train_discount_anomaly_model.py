# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 20 - Train Discount Anomaly Detection Model (Isolation Forest)
# MAGIC
# MAGIC **Purpose**: Train a scikit-learn Isolation Forest model to detect anomalous discounting
# MAGIC patterns across sales representatives.  The model identifies reps whose discount behaviour
# MAGIC deviates significantly from the population -- potential indicators of unauthorised discounting,
# MAGIC data entry errors, or contract mis-application.
# MAGIC
# MAGIC **Input**: `hls_amer_catalog.silver.ficm_pricing_master`
# MAGIC
# MAGIC **Features** (per sales rep):
# MAGIC | Feature | Description |
# MAGIC |---------|-------------|
# MAGIC | avg_discount_pct | Mean discount percentage across all transactions |
# MAGIC | discount_stddev | Standard deviation of discount percentage |
# MAGIC | max_discount | Maximum single-transaction discount percentage |
# MAGIC | volume_weighted_discount | Volume-weighted average discount |
# MAGIC | transaction_count | Number of transactions for the rep |
# MAGIC | unique_customers | Number of distinct customers served |
# MAGIC
# MAGIC **Model**: scikit-learn `IsolationForest` with `contamination=0.1`
# MAGIC
# MAGIC **Output**:
# MAGIC - MLflow experiment run with logged model, parameters, and metrics
# MAGIC - Model registered in Unity Catalog as `hls_amer_catalog.models.discount_anomaly_detector`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# ---------------------------------------------------------------------------
# Deterministic seed
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Unity Catalog references
# ---------------------------------------------------------------------------
CATALOG: str = "hls_amer_catalog"
SOURCE_TABLE: str = f"{CATALOG}.silver.ficm_pricing_master"
MODEL_SCHEMA: str = "models"
MODEL_NAME: str = f"{CATALOG}.{MODEL_SCHEMA}.discount_anomaly_detector"

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
CONTAMINATION: float = 0.1
N_ESTIMATORS: int = 200
MAX_SAMPLES: str = "auto"
MAX_FEATURES: float = 1.0

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "avg_discount_pct",
    "discount_stddev",
    "max_discount",
    "volume_weighted_discount",
    "transaction_count",
    "unique_customers",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Read FICM Pricing Master

# COMMAND ----------

df_ficm = spark.table(SOURCE_TABLE)
ficm_count = df_ficm.count()
print(f"FICM pricing master: {ficm_count:,} rows, {len(df_ficm.columns)} columns")
df_ficm.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Compute Per-Rep Discount Features
# MAGIC
# MAGIC Aggregate transaction-level data to the sales rep grain, computing the six
# MAGIC features that characterise each rep's discounting behaviour.

# COMMAND ----------

# ---------------------------------------------------------------------------
# Derive discount_pct if not already present.
# Discount = (list_price - pocket_price) / list_price
# ---------------------------------------------------------------------------
if "discount_pct" not in df_ficm.columns:
    df_ficm = df_ficm.withColumn(
        "discount_pct",
        F.when(
            F.col("list_price") > 0,
            (F.col("list_price") - F.col("pocket_price")) / F.col("list_price") * 100,
        ).otherwise(F.lit(0.0).cast(DoubleType())),
    )
    print("Derived discount_pct column from list_price and pocket_price")

# ---------------------------------------------------------------------------
# Identify the sales rep column.
# The FICM master may use different column names; try common patterns.
# ---------------------------------------------------------------------------
rep_col_candidates = ["sales_rep_id", "rep_id", "salesperson_id", "sales_rep"]
rep_col = None
for candidate in rep_col_candidates:
    if candidate in df_ficm.columns:
        rep_col = candidate
        break

if rep_col is None:
    raise ValueError(
        f"Could not find a sales rep column. Available columns: {df_ficm.columns}"
    )

print(f"Using rep column: {rep_col}")

# ---------------------------------------------------------------------------
# Identify the customer column
# ---------------------------------------------------------------------------
cust_col_candidates = ["customer_id", "customer_key", "account_id", "ship_to_id"]
cust_col = None
for candidate in cust_col_candidates:
    if candidate in df_ficm.columns:
        cust_col = candidate
        break

if cust_col is None:
    # Fall back: use a dummy if no customer column found
    print("WARNING: No customer column found; using constant for unique_customers")
    cust_col = None

# ---------------------------------------------------------------------------
# Identify units/volume column
# ---------------------------------------------------------------------------
vol_col_candidates = ["units_sold", "quantity", "volume", "units"]
vol_col = None
for candidate in vol_col_candidates:
    if candidate in df_ficm.columns:
        vol_col = candidate
        break

if vol_col is None:
    print("WARNING: No volume column found; using count as proxy for volume_weighted_discount")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Aggregate to Rep Grain

# COMMAND ----------

# Build the aggregation expressions
agg_exprs = [
    F.avg("discount_pct").alias("avg_discount_pct"),
    F.stddev("discount_pct").alias("discount_stddev"),
    F.max("discount_pct").alias("max_discount"),
    F.count("*").alias("transaction_count"),
]

# Volume-weighted discount
if vol_col:
    agg_exprs.append(
        (
            F.sum(F.col("discount_pct") * F.col(vol_col))
            / F.sum(F.col(vol_col))
        ).alias("volume_weighted_discount")
    )
else:
    agg_exprs.append(F.avg("discount_pct").alias("volume_weighted_discount"))

# Unique customers
if cust_col:
    agg_exprs.append(F.countDistinct(cust_col).alias("unique_customers"))
else:
    agg_exprs.append(F.lit(1).alias("unique_customers"))

df_rep_features = df_ficm.groupBy(rep_col).agg(*agg_exprs)

# Fill nulls in stddev (reps with 1 transaction have null stddev)
df_rep_features = df_rep_features.fillna({"discount_stddev": 0.0})

rep_count = df_rep_features.count()
print(f"Aggregated to {rep_count} sales reps")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b. Feature Summary Statistics

# COMMAND ----------

print("Feature summary statistics:")
df_rep_features.select(FEATURE_COLUMNS).describe().show()

# Show distribution of transaction counts
print(f"\nTransaction count distribution:")
df_rep_features.select(
    F.min("transaction_count").alias("min"),
    F.percentile_approx("transaction_count", 0.25).alias("p25"),
    F.percentile_approx("transaction_count", 0.50).alias("median"),
    F.percentile_approx("transaction_count", 0.75).alias("p75"),
    F.max("transaction_count").alias("max"),
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Prepare Training Data

# COMMAND ----------

# Convert to Pandas for sklearn
pdf_features = df_rep_features.select(rep_col, *FEATURE_COLUMNS).toPandas()

# Store rep IDs separately
rep_ids = pdf_features[rep_col].values
X = pdf_features[FEATURE_COLUMNS].values.astype(np.float64)

# Handle any remaining NaN/inf values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Training data shape: {X.shape}")
print(f"Feature matrix stats:")
print(f"  Min:  {X.min(axis=0)}")
print(f"  Mean: {X.mean(axis=0).round(4)}")
print(f"  Max:  {X.max(axis=0)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train Isolation Forest Model
# MAGIC
# MAGIC The model is wrapped in a scikit-learn Pipeline with StandardScaler for feature
# MAGIC normalisation.  The Isolation Forest assigns an anomaly score to each rep; scores
# MAGIC below zero indicate anomalous behaviour.

# COMMAND ----------

# ---------------------------------------------------------------------------
# Set MLflow experiment
# ---------------------------------------------------------------------------
experiment_name = f"/Users/{spark.conf.get('spark.databricks.workspaceUrl', 'local')}/discount_anomaly_detection"
try:
    mlflow.set_experiment(experiment_name)
except Exception:
    # Fall back to default experiment if workspace URL is not available
    mlflow.set_experiment("/Shared/discount_anomaly_detection")

print(f"MLflow experiment set")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Build and train the pipeline
# ---------------------------------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("isolation_forest", IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_samples=MAX_SAMPLES,
        max_features=MAX_FEATURES,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )),
])

with mlflow.start_run(run_name="discount_anomaly_isolation_forest") as run:
    # --- Log parameters ---
    mlflow.log_param("model_type", "IsolationForest")
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("contamination", CONTAMINATION)
    mlflow.log_param("max_samples", MAX_SAMPLES)
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("random_seed", RANDOM_SEED)
    mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
    mlflow.log_param("source_table", SOURCE_TABLE)
    mlflow.log_param("n_reps", len(rep_ids))

    # --- Train ---
    pipeline.fit(X)
    print("Pipeline trained successfully")

    # --- Predict ---
    predictions = pipeline.predict(X)  # 1 = normal, -1 = anomaly
    anomaly_scores = pipeline.named_steps["isolation_forest"].decision_function(
        pipeline.named_steps["scaler"].transform(X)
    )

    n_anomalies = (predictions == -1).sum()
    n_normal = (predictions == 1).sum()
    anomaly_rate = n_anomalies / len(predictions)

    print(f"\nPrediction results:")
    print(f"  Normal reps:    {n_normal}")
    print(f"  Anomalous reps: {n_anomalies}")
    print(f"  Anomaly rate:   {anomaly_rate:.2%}")

    # --- Log metrics ---
    mlflow.log_metric("n_anomalies", int(n_anomalies))
    mlflow.log_metric("n_normal", int(n_normal))
    mlflow.log_metric("anomaly_rate", float(anomaly_rate))
    mlflow.log_metric("mean_anomaly_score", float(np.mean(anomaly_scores)))
    mlflow.log_metric("min_anomaly_score", float(np.min(anomaly_scores)))
    mlflow.log_metric("max_anomaly_score", float(np.max(anomaly_scores)))

    # --- Feature importance (approximated via score correlation) ---
    feature_importance = {}
    for i, feat in enumerate(FEATURE_COLUMNS):
        corr = np.corrcoef(X[:, i], anomaly_scores)[0, 1]
        feature_importance[feat] = float(abs(corr))
        mlflow.log_metric(f"feature_corr_{feat}", float(abs(corr)))

    print(f"\nFeature correlations with anomaly score:")
    for feat, corr in sorted(feature_importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {corr:.4f}")

    # --- Log the model ---
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="discount-anomaly-detector",
        registered_model_name=MODEL_NAME,
        input_example=pd.DataFrame([X[0]], columns=FEATURE_COLUMNS),
    )

    run_id = run.info.run_id
    print(f"\nMLflow run ID: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Analyse Anomalous Reps

# COMMAND ----------

# Build results DataFrame
pdf_results = pdf_features.copy()
pdf_results["prediction"] = predictions
pdf_results["anomaly_score"] = anomaly_scores
pdf_results["is_anomaly"] = predictions == -1

# Convert back to Spark for display
df_results = spark.createDataFrame(pdf_results)

print("Top anomalous reps (lowest anomaly scores):")
(
    df_results
    .filter(F.col("is_anomaly") == True)
    .orderBy("anomaly_score")
    .select(
        rep_col,
        "avg_discount_pct",
        "discount_stddev",
        "max_discount",
        "volume_weighted_discount",
        "transaction_count",
        "unique_customers",
        "anomaly_score",
    )
    .show(20, truncate=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Compare Normal vs Anomalous Feature Distributions

# COMMAND ----------

print("Feature comparison: Normal vs Anomalous reps\n")

for feat in FEATURE_COLUMNS:
    normal_vals = pdf_results.loc[~pdf_results["is_anomaly"], feat]
    anomaly_vals = pdf_results.loc[pdf_results["is_anomaly"], feat]

    print(f"  {feat}:")
    print(f"    Normal  -- mean: {normal_vals.mean():.4f}, std: {normal_vals.std():.4f}")
    if len(anomaly_vals) > 0:
        print(f"    Anomaly -- mean: {anomaly_vals.mean():.4f}, std: {anomaly_vals.std():.4f}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Register Model in Unity Catalog
# MAGIC
# MAGIC The model was registered during `mlflow.sklearn.log_model` via the
# MAGIC `registered_model_name` parameter.  Here we verify the registration and
# MAGIC optionally set an alias.

# COMMAND ----------

# Ensure the models schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{MODEL_SCHEMA}")

# Verify model registration
client = mlflow.MlflowClient()

try:
    model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if model_versions:
        latest_version = max(model_versions, key=lambda v: int(v.version))
        print(f"Model registered: {MODEL_NAME}")
        print(f"  Latest version: {latest_version.version}")
        print(f"  Status: {latest_version.status}")
        print(f"  Run ID: {latest_version.run_id}")

        # Set alias for easy reference
        try:
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="champion",
                version=latest_version.version,
            )
            print(f"  Alias 'champion' set to version {latest_version.version}")
        except Exception as e:
            print(f"  Could not set alias: {e}")
    else:
        print(f"WARNING: No versions found for model {MODEL_NAME}")
except Exception as e:
    print(f"Model registry check: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Model Type | Isolation Forest (scikit-learn) |
# MAGIC | Contamination | 0.1 (10% expected anomaly rate) |
# MAGIC | n_estimators | 200 |
# MAGIC | Features | 6 (discount stats per rep) |
# MAGIC | Source Table | `hls_amer_catalog.silver.ficm_pricing_master` |
# MAGIC | MLflow Model | `hls_amer_catalog.models.discount_anomaly_detector` |
# MAGIC | Pipeline | StandardScaler + IsolationForest |
# MAGIC
# MAGIC **Next notebook**: `21_train_advanced_elasticity_model.py` -- GradientBoosting elasticity with uncertainty.
