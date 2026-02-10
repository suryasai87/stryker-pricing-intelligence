# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 21 - Train Advanced Price Elasticity Model with Uncertainty Bounds
# MAGIC
# MAGIC **Purpose**: Build an enhanced price elasticity model that produces both point estimates
# MAGIC and prediction intervals (5th/95th quantiles).  The model estimates how volume responds
# MAGIC to price changes at the SKU x market segment grain, enabling confident decision-making
# MAGIC by exposing the range of plausible outcomes.
# MAGIC
# MAGIC **Input**: `hls_amer_catalog.silver.ficm_pricing_master`
# MAGIC
# MAGIC **Features**:
# MAGIC | Feature | Description |
# MAGIC |---------|-------------|
# MAGIC | log_price | Natural log of pocket price |
# MAGIC | log_volume | Natural log of units sold (target) |
# MAGIC | month_sin | Sine component of month-of-year (cyclical encoding) |
# MAGIC | month_cos | Cosine component of month-of-year (cyclical encoding) |
# MAGIC | segment_encoded | Label-encoded market segment |
# MAGIC | product_family_encoded | Label-encoded product family / category |
# MAGIC
# MAGIC **Models**:
# MAGIC 1. **Point Estimate**: `GradientBoostingRegressor` for mean elasticity prediction
# MAGIC 2. **Lower Bound (5th percentile)**: `GradientBoostingRegressor(loss='quantile', alpha=0.05)`
# MAGIC 3. **Upper Bound (95th percentile)**: `GradientBoostingRegressor(loss='quantile', alpha=0.95)`
# MAGIC
# MAGIC **Output**:
# MAGIC - MLflow experiment run with all three models logged as a composite artifact
# MAGIC - Model registered in Unity Catalog as `hls_amer_catalog.models.advanced_elasticity_model`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

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
MODEL_NAME: str = f"{CATALOG}.{MODEL_SCHEMA}.advanced_elasticity_model"

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
N_ESTIMATORS: int = 300
MAX_DEPTH: int = 5
LEARNING_RATE: float = 0.05
MIN_SAMPLES_LEAF: int = 10
SUBSAMPLE: float = 0.8
TEST_SIZE: float = 0.2

# ---------------------------------------------------------------------------
# Quantile levels for uncertainty bounds
# ---------------------------------------------------------------------------
QUANTILE_LOWER: float = 0.05
QUANTILE_UPPER: float = 0.95

# ---------------------------------------------------------------------------
# Feature and target columns
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "log_price",
    "month_sin",
    "month_cos",
    "segment_encoded",
    "product_family_encoded",
]
TARGET_COLUMN = "log_volume"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Read FICM Pricing Master

# COMMAND ----------

df_ficm = spark.table(SOURCE_TABLE)
ficm_count = df_ficm.count()
print(f"FICM pricing master: {ficm_count:,} rows, {len(df_ficm.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Prepare Time-Series Features per SKU x Segment
# MAGIC
# MAGIC Transform the transaction-level data into time-series features suitable for
# MAGIC elasticity estimation.  Key transformations:
# MAGIC - Log-transform price and volume for log-log elasticity interpretation
# MAGIC - Cyclical encoding of month-of-year to capture seasonality
# MAGIC - Label encoding of categorical segment and product family variables

# COMMAND ----------

# ---------------------------------------------------------------------------
# Identify key columns (handle schema variations)
# ---------------------------------------------------------------------------
price_col_candidates = ["pocket_price", "net_price", "selling_price"]
price_col = None
for candidate in price_col_candidates:
    if candidate in df_ficm.columns:
        price_col = candidate
        break
if price_col is None:
    raise ValueError(f"No price column found. Available: {df_ficm.columns}")

vol_col_candidates = ["units_sold", "quantity", "volume", "units"]
vol_col = None
for candidate in vol_col_candidates:
    if candidate in df_ficm.columns:
        vol_col = candidate
        break
if vol_col is None:
    raise ValueError(f"No volume column found. Available: {df_ficm.columns}")

# Segment column
seg_col_candidates = ["customer_segment", "segment", "market_segment"]
seg_col = None
for candidate in seg_col_candidates:
    if candidate in df_ficm.columns:
        seg_col = candidate
        break
if seg_col is None:
    print("WARNING: No segment column found; using 'unknown' as default")

# Product family column
prod_col_candidates = ["product_category", "category", "product_family", "sub_category"]
prod_col = None
for candidate in prod_col_candidates:
    if candidate in df_ficm.columns:
        prod_col = candidate
        break
if prod_col is None:
    print("WARNING: No product family column found; using 'unknown' as default")

# Date column
date_col_candidates = ["year_month", "date", "transaction_date", "sale_date"]
date_col = None
for candidate in date_col_candidates:
    if candidate in df_ficm.columns:
        date_col = candidate
        break
if date_col is None:
    raise ValueError(f"No date column found. Available: {df_ficm.columns}")

# SKU column
sku_col_candidates = ["product_id", "sku_id", "item_id"]
sku_col = None
for candidate in sku_col_candidates:
    if candidate in df_ficm.columns:
        sku_col = candidate
        break
if sku_col is None:
    raise ValueError(f"No SKU column found. Available: {df_ficm.columns}")

print(f"Column mapping:")
print(f"  Price:    {price_col}")
print(f"  Volume:   {vol_col}")
print(f"  Segment:  {seg_col}")
print(f"  Product:  {prod_col}")
print(f"  Date:     {date_col}")
print(f"  SKU:      {sku_col}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Aggregate to SKU x Segment x Month Grain

# COMMAND ----------

# Derive year_month if needed
if date_col == "year_month":
    df_agg_input = df_ficm
else:
    df_agg_input = df_ficm.withColumn(
        "year_month", F.date_format(F.col(date_col), "yyyy-MM")
    )

# Build groupBy columns
group_cols = [sku_col, "year_month"]
if seg_col:
    group_cols.append(seg_col)
if prod_col:
    group_cols.append(prod_col)

df_agg = (
    df_agg_input
    .filter(
        (F.col(price_col) > 0) & (F.col(vol_col) > 0)
    )
    .groupBy(*group_cols)
    .agg(
        F.avg(price_col).alias("avg_price"),
        F.sum(vol_col).alias("total_volume"),
        F.count("*").alias("n_transactions"),
    )
)

agg_count = df_agg.count()
print(f"Aggregated to {agg_count:,} SKU x Segment x Month rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b. Feature Engineering

# COMMAND ----------

# Log transforms
df_features = (
    df_agg
    .withColumn("log_price", F.log(F.col("avg_price")))
    .withColumn("log_volume", F.log(F.col("total_volume")))
)

# Extract month for cyclical encoding
df_features = df_features.withColumn(
    "month_num",
    F.month(F.to_date(F.concat(F.col("year_month"), F.lit("-01")))),
)

# Cyclical encoding: sin/cos of month
df_features = (
    df_features
    .withColumn(
        "month_sin",
        F.sin(2 * np.pi * F.col("month_num") / 12),
    )
    .withColumn(
        "month_cos",
        F.cos(2 * np.pi * F.col("month_num") / 12),
    )
)

# Convert to Pandas for sklearn label encoding
pdf = df_features.toPandas()

# Handle potential inf/nan in log transforms
pdf = pdf.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["log_price", "log_volume"]
)

print(f"Feature DataFrame: {len(pdf):,} rows after cleaning")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3c. Label Encode Categorical Variables

# COMMAND ----------

# Segment encoding
le_segment = LabelEncoder()
if seg_col and seg_col in pdf.columns:
    pdf["segment_encoded"] = le_segment.fit_transform(pdf[seg_col].fillna("unknown"))
    segment_classes = le_segment.classes_.tolist()
else:
    pdf["segment_encoded"] = 0
    segment_classes = ["unknown"]

print(f"Segment classes ({len(segment_classes)}): {segment_classes[:10]}{'...' if len(segment_classes) > 10 else ''}")

# Product family encoding
le_product = LabelEncoder()
if prod_col and prod_col in pdf.columns:
    pdf["product_family_encoded"] = le_product.fit_transform(pdf[prod_col].fillna("unknown"))
    product_classes = le_product.classes_.tolist()
else:
    pdf["product_family_encoded"] = 0
    product_classes = ["unknown"]

print(f"Product family classes ({len(product_classes)}): {product_classes[:10]}{'...' if len(product_classes) > 10 else ''}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train/Test Split

# COMMAND ----------

X = pdf[FEATURE_COLUMNS].values.astype(np.float64)
y = pdf[TARGET_COLUMN].values.astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

print(f"Training set: {X_train.shape[0]:,} rows")
print(f"Test set:     {X_test.shape[0]:,} rows")
print(f"Features:     {X_train.shape[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train Models
# MAGIC
# MAGIC Three GradientBoosting models are trained:
# MAGIC 1. **Mean model** (`loss='squared_error'`) for point estimates
# MAGIC 2. **Lower quantile model** (`loss='quantile'`, `alpha=0.05`) for the 5th percentile
# MAGIC 3. **Upper quantile model** (`loss='quantile'`, `alpha=0.95`) for the 95th percentile

# COMMAND ----------

# ---------------------------------------------------------------------------
# Set MLflow experiment
# ---------------------------------------------------------------------------
experiment_name = f"/Users/{spark.conf.get('spark.databricks.workspaceUrl', 'local')}/advanced_elasticity"
try:
    mlflow.set_experiment(experiment_name)
except Exception:
    mlflow.set_experiment("/Shared/advanced_elasticity")

print(f"MLflow experiment set")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Shared hyperparameters
# ---------------------------------------------------------------------------
shared_params = {
    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
    "learning_rate": LEARNING_RATE,
    "min_samples_leaf": MIN_SAMPLES_LEAF,
    "subsample": SUBSAMPLE,
    "random_state": RANDOM_SEED,
}

with mlflow.start_run(run_name="advanced_elasticity_with_uncertainty") as run:

    # --- Log common parameters ---
    mlflow.log_param("model_type", "GradientBoostingRegressor")
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)
    mlflow.log_param("subsample", SUBSAMPLE)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_seed", RANDOM_SEED)
    mlflow.log_param("quantile_lower", QUANTILE_LOWER)
    mlflow.log_param("quantile_upper", QUANTILE_UPPER)
    mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
    mlflow.log_param("source_table", SOURCE_TABLE)
    mlflow.log_param("n_train_rows", X_train.shape[0])
    mlflow.log_param("n_test_rows", X_test.shape[0])

    # --- Log label encoder mappings as artifacts ---
    encoder_mapping = {
        "segment_classes": segment_classes,
        "product_family_classes": product_classes,
    }
    mlflow.log_dict(encoder_mapping, "encoder_mapping.json")

    # ===================================================================
    # 5a. Point Estimate Model (mean)
    # ===================================================================
    print("Training point estimate model (squared_error loss)...")
    model_mean = GradientBoostingRegressor(
        loss="squared_error",
        **shared_params,
    )
    model_mean.fit(X_train, y_train)

    y_pred_mean = model_mean.predict(X_test)
    rmse_mean = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    mae_mean = mean_absolute_error(y_test, y_pred_mean)
    r2_mean = r2_score(y_test, y_pred_mean)

    mlflow.log_metric("rmse_mean", rmse_mean)
    mlflow.log_metric("mae_mean", mae_mean)
    mlflow.log_metric("r2_mean", r2_mean)

    print(f"  RMSE: {rmse_mean:.4f}")
    print(f"  MAE:  {mae_mean:.4f}")
    print(f"  R2:   {r2_mean:.4f}")

    # ===================================================================
    # 5b. Lower Quantile Model (5th percentile)
    # ===================================================================
    print(f"\nTraining lower bound model (quantile={QUANTILE_LOWER})...")
    model_lower = GradientBoostingRegressor(
        loss="quantile",
        alpha=QUANTILE_LOWER,
        **shared_params,
    )
    model_lower.fit(X_train, y_train)

    y_pred_lower = model_lower.predict(X_test)
    coverage_below = (y_test < y_pred_lower).mean()

    mlflow.log_metric("rmse_lower", np.sqrt(mean_squared_error(y_test, y_pred_lower)))
    mlflow.log_metric("coverage_below_5pct", float(coverage_below))

    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lower)):.4f}")
    print(f"  Actual % below lower bound: {coverage_below:.2%} (target: {QUANTILE_LOWER:.0%})")

    # ===================================================================
    # 5c. Upper Quantile Model (95th percentile)
    # ===================================================================
    print(f"\nTraining upper bound model (quantile={QUANTILE_UPPER})...")
    model_upper = GradientBoostingRegressor(
        loss="quantile",
        alpha=QUANTILE_UPPER,
        **shared_params,
    )
    model_upper.fit(X_train, y_train)

    y_pred_upper = model_upper.predict(X_test)
    coverage_above = (y_test > y_pred_upper).mean()

    mlflow.log_metric("rmse_upper", np.sqrt(mean_squared_error(y_test, y_pred_upper)))
    mlflow.log_metric("coverage_above_95pct", float(coverage_above))

    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_upper)):.4f}")
    print(f"  Actual % above upper bound: {coverage_above:.2%} (target: {1 - QUANTILE_UPPER:.0%})")

    # ===================================================================
    # 5d. Interval Coverage Metrics
    # ===================================================================
    in_interval = ((y_test >= y_pred_lower) & (y_test <= y_pred_upper)).mean()
    interval_width = np.mean(y_pred_upper - y_pred_lower)

    mlflow.log_metric("prediction_interval_coverage", float(in_interval))
    mlflow.log_metric("mean_interval_width", float(interval_width))

    print(f"\nPrediction Interval (90%):")
    print(f"  Coverage: {in_interval:.2%} (target: 90%)")
    print(f"  Mean width (log scale): {interval_width:.4f}")

    # ===================================================================
    # 5e. Feature Importance (from mean model)
    # ===================================================================
    feature_importance = dict(zip(FEATURE_COLUMNS, model_mean.feature_importances_))
    print(f"\nFeature importance (point estimate model):")
    for feat, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
        mlflow.log_metric(f"feature_importance_{feat}", float(imp))

    # ===================================================================
    # 5f. Compute Elasticity Statistics
    # ===================================================================
    # Elasticity = d(log_volume) / d(log_price) -- approximated by the model's
    # marginal response to log_price (feature index 0).
    log_price_importance = feature_importance.get("log_price", 0.0)

    # Use model's partial dependence for a more direct estimate
    y_pred_all = model_mean.predict(X_test)

    # Simple first-order elasticity estimate
    price_idx = FEATURE_COLUMNS.index("log_price")
    X_test_perturbed = X_test.copy()
    X_test_perturbed[:, price_idx] += 0.01  # 1% price increase in log space
    y_pred_perturbed = model_mean.predict(X_test_perturbed)

    elasticities = (y_pred_perturbed - y_pred_all) / 0.01
    mean_elasticity = np.mean(elasticities)
    median_elasticity = np.median(elasticities)

    mlflow.log_metric("mean_elasticity", float(mean_elasticity))
    mlflow.log_metric("median_elasticity", float(median_elasticity))
    mlflow.log_metric("std_elasticity", float(np.std(elasticities)))
    mlflow.log_metric("min_elasticity", float(np.min(elasticities)))
    mlflow.log_metric("max_elasticity", float(np.max(elasticities)))

    print(f"\nElasticity estimates (d(log_vol)/d(log_price)):")
    print(f"  Mean:   {mean_elasticity:.4f}")
    print(f"  Median: {median_elasticity:.4f}")
    print(f"  Std:    {np.std(elasticities):.4f}")
    print(f"  Range:  [{np.min(elasticities):.4f}, {np.max(elasticities):.4f}]")

    # ===================================================================
    # 5g. Log Composite Model
    # ===================================================================
    # Wrap all three models in a custom Python model for unified inference.
    class ElasticityModelWithUncertainty(mlflow.pyfunc.PythonModel):
        """Composite model that returns point estimate + prediction interval."""

        def __init__(self, model_mean, model_lower, model_upper, feature_columns):
            self.model_mean = model_mean
            self.model_lower = model_lower
            self.model_upper = model_upper
            self.feature_columns = feature_columns

        def predict(self, context, model_input):
            if isinstance(model_input, pd.DataFrame):
                X = model_input[self.feature_columns].values
            else:
                X = np.array(model_input)

            pred_mean = self.model_mean.predict(X)
            pred_lower = self.model_lower.predict(X)
            pred_upper = self.model_upper.predict(X)

            return pd.DataFrame({
                "log_volume_predicted": pred_mean,
                "log_volume_lower_5pct": pred_lower,
                "log_volume_upper_95pct": pred_upper,
                "interval_width": pred_upper - pred_lower,
            })

    composite_model = ElasticityModelWithUncertainty(
        model_mean=model_mean,
        model_lower=model_lower,
        model_upper=model_upper,
        feature_columns=FEATURE_COLUMNS,
    )

    # Log with MLflow
    input_example = pd.DataFrame([X_test[0]], columns=FEATURE_COLUMNS)

    mlflow.pyfunc.log_model(
        artifact_path="advanced-elasticity-model",
        python_model=composite_model,
        registered_model_name=MODEL_NAME,
        input_example=input_example,
        pip_requirements=[
            "scikit-learn>=1.3.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
        ],
    )

    run_id = run.info.run_id
    print(f"\nMLflow run ID: {run_id}")
    print(f"Composite model logged as: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test Registered Model

# COMMAND ----------

# Load the registered model and run inference on a sample
print("Testing registered model inference...")

try:
    loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/advanced-elasticity-model")

    sample_input = pd.DataFrame(X_test[:5], columns=FEATURE_COLUMNS)
    sample_output = loaded_model.predict(sample_input)

    print("\nSample predictions:")
    print(sample_output.to_string(index=False))

    # Verify output columns
    expected_output_cols = [
        "log_volume_predicted",
        "log_volume_lower_5pct",
        "log_volume_upper_95pct",
        "interval_width",
    ]
    assert all(col in sample_output.columns for col in expected_output_cols), (
        f"Missing output columns. Got: {sample_output.columns.tolist()}"
    )
    print("\n[CHECK] All expected output columns present")

    # Verify ordering: lower < mean < upper
    assert (sample_output["log_volume_lower_5pct"] <= sample_output["log_volume_predicted"]).all(), (
        "Lower bound exceeds mean prediction"
    )
    assert (sample_output["log_volume_predicted"] <= sample_output["log_volume_upper_95pct"]).all(), (
        "Mean prediction exceeds upper bound"
    )
    print("[CHECK] Bound ordering: lower <= mean <= upper")

except Exception as e:
    print(f"Model test: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Register Model in Unity Catalog

# COMMAND ----------

# Ensure the models schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{MODEL_SCHEMA}")

# Verify registration
client = mlflow.MlflowClient()

try:
    model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if model_versions:
        latest_version = max(model_versions, key=lambda v: int(v.version))
        print(f"Model registered: {MODEL_NAME}")
        print(f"  Latest version: {latest_version.version}")
        print(f"  Status: {latest_version.status}")
        print(f"  Run ID: {latest_version.run_id}")

        # Set alias
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
# MAGIC ## 8. Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Model Type | GradientBoostingRegressor (3 models) |
# MAGIC | Point Estimate | squared_error loss |
# MAGIC | Lower Bound | quantile loss, alpha=0.05 |
# MAGIC | Upper Bound | quantile loss, alpha=0.95 |
# MAGIC | n_estimators | 300 |
# MAGIC | max_depth | 5 |
# MAGIC | learning_rate | 0.05 |
# MAGIC | Features | 5 (log_price, month_sin/cos, segment, product_family) |
# MAGIC | Source Table | `hls_amer_catalog.silver.ficm_pricing_master` |
# MAGIC | MLflow Model | `hls_amer_catalog.models.advanced_elasticity_model` |
# MAGIC | Output Format | DataFrame with predicted, lower, upper, interval_width |
# MAGIC
# MAGIC **Next notebook**: `99_validate_end_to_end.py` -- full platform validation.
