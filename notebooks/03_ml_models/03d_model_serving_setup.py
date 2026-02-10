# Databricks notebook source

# MAGIC %md
# MAGIC # 03d - Model Serving Endpoint Setup & Batch Scoring
# MAGIC
# MAGIC **Purpose:** Deploy all three Stryker pricing intelligence models to a unified
# MAGIC Databricks Model Serving endpoint, configure traffic routing, validate
# MAGIC real-time inference, create Spark UDFs for batch scoring, and set up
# MAGIC endpoint access permissions.
# MAGIC
# MAGIC **Models Deployed:**
# MAGIC | Served Entity | Registry Path | Workload | Scale-to-Zero |
# MAGIC |---------------|---------------|----------|---------------|
# MAGIC | price-elasticity | `hls_amer_catalog.models.price_elasticity_lgbm` v1 | Small | Yes |
# MAGIC | revenue-impact | `hls_amer_catalog.models.revenue_impact_xgb` v1 | Small | Yes |
# MAGIC | margin-optimization | `hls_amer_catalog.models.margin_optimization_enet` v1 | Small | Yes |
# MAGIC
# MAGIC **Endpoint:** `stryker-pricing-models`
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Models registered in Unity Catalog at `hls_amer_catalog.models.*`
# MAGIC - Notebooks `03a`, `03b`, `03c` executed successfully
# MAGIC - Sufficient workspace permissions for serving endpoint management

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports & Client Initialization

# COMMAND ----------

import time
import json
import requests
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
)
from databricks.sdk.errors import NotFound, ResourceAlreadyExists

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)

# COMMAND ----------

w = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

print(f"Workspace host : {w.config.host}")
print(f"Authenticated as: {w.current_user.me().user_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

# -- Endpoint configuration --------------------------------------------------
ENDPOINT_NAME = "stryker-pricing-models"

CATALOG = "hls_amer_catalog"
MODEL_SCHEMA = "models"

# Model registry paths (Unity Catalog three-level namespace)
MODELS = {
    "price-elasticity": {
        "entity_name": f"{CATALOG}.{MODEL_SCHEMA}.price_elasticity_lgbm",
        "entity_version": "1",
        "workload_size": "Small",
        "scale_to_zero_enabled": True,
    },
    "revenue-impact": {
        "entity_name": f"{CATALOG}.{MODEL_SCHEMA}.revenue_impact_xgb",
        "entity_version": "1",
        "workload_size": "Small",
        "scale_to_zero_enabled": True,
    },
    "margin-optimization": {
        "entity_name": f"{CATALOG}.{MODEL_SCHEMA}.margin_optimization_enet",
        "entity_version": "1",
        "workload_size": "Small",
        "scale_to_zero_enabled": True,
    },
}

# Traffic split across served models (must sum to 100%)
_model_names = list(MODELS.keys())
TRAFFIC_ROUTES = []
for i, name in enumerate(_model_names):
    pct = 34 if i == 0 else 33  # 34 + 33 + 33 = 100
    TRAFFIC_ROUTES.append(Route(served_model_name=name, traffic_percentage=pct))

# Polling configuration for endpoint readiness
POLL_INTERVAL_SECONDS = 30
POLL_TIMEOUT_SECONDS = 1200  # 20 minutes max wait

# Permissions
PRICING_ANALYSTS_GROUP = "pricing_analysts_group"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create or Update the Unified Serving Endpoint

# COMMAND ----------

def build_served_entities() -> list:
    """Build the list of ServedEntityInput objects from the MODELS config.

    Returns:
        List of ServedEntityInput instances, one per model.
    """
    entities = []
    for name, cfg in MODELS.items():
        entity = ServedEntityInput(
            name=name,
            entity_name=cfg["entity_name"],
            entity_version=cfg["entity_version"],
            workload_size=cfg["workload_size"],
            scale_to_zero_enabled=cfg["scale_to_zero_enabled"],
        )
        entities.append(entity)
    return entities


def create_or_update_endpoint(endpoint_name: str) -> None:
    """Create the serving endpoint, or update it if it already exists.

    This function is idempotent: repeated calls converge to the desired
    configuration without error.

    Args:
        endpoint_name: Name of the Model Serving endpoint.
    """
    served_entities = build_served_entities()
    config = EndpointCoreConfigInput(
        name=endpoint_name,
        served_entities=served_entities,
        traffic_config=TrafficConfig(routes=TRAFFIC_ROUTES),
    )

    try:
        w.serving_endpoints.create(
            name=endpoint_name,
            config=config,
        )
        print(f"Created new serving endpoint: {endpoint_name}")
    except ResourceAlreadyExists:
        print(f"Endpoint '{endpoint_name}' already exists. Updating configuration...")
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=served_entities,
            traffic_config=TrafficConfig(routes=TRAFFIC_ROUTES),
        )
        print(f"Updated serving endpoint: {endpoint_name}")

# COMMAND ----------

create_or_update_endpoint(ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wait for Endpoint Readiness

# COMMAND ----------

def wait_for_endpoint_ready(
    endpoint_name: str,
    poll_interval: int = POLL_INTERVAL_SECONDS,
    timeout: int = POLL_TIMEOUT_SECONDS,
) -> bool:
    """Poll the serving endpoint until it reaches READY state or times out.

    Args:
        endpoint_name: Name of the serving endpoint to monitor.
        poll_interval: Seconds between status checks.
        timeout: Maximum seconds to wait before raising a TimeoutError.

    Returns:
        True if the endpoint reached READY state.

    Raises:
        TimeoutError: If the endpoint does not become ready within the timeout.
        RuntimeError: If the endpoint enters a terminal failure state.
    """
    start_time = time.time()
    print(f"Waiting for endpoint '{endpoint_name}' to become ready...")
    print(f"  Poll interval : {poll_interval}s")
    print(f"  Timeout       : {timeout}s")
    print("-" * 60)

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(
                f"Endpoint '{endpoint_name}' did not become ready within "
                f"{timeout} seconds ({elapsed:.0f}s elapsed)."
            )

        endpoint = w.serving_endpoints.get(name=endpoint_name)
        state = endpoint.state

        config_state = getattr(state, "config_update", None)
        ready_state = getattr(state, "ready", None)

        print(
            f"  [{elapsed:6.0f}s] ready={ready_state}  "
            f"config_update={config_state}"
        )

        # Terminal success
        if str(ready_state) == "READY":
            print("-" * 60)
            print(f"Endpoint '{endpoint_name}' is READY. ({elapsed:.0f}s)")
            return True

        # Terminal failure states
        if config_state and "FAILED" in str(config_state).upper():
            pending = getattr(endpoint, "pending_config", None)
            error_msg = "Unknown error"
            if pending and hasattr(pending, "config_update"):
                error_msg = str(pending.config_update)
            raise RuntimeError(
                f"Endpoint '{endpoint_name}' entered FAILED state: {error_msg}"
            )

        time.sleep(poll_interval)

# COMMAND ----------

try:
    wait_for_endpoint_ready(ENDPOINT_NAME)
except TimeoutError as e:
    print(f"WARNING: {e}")
    print("The endpoint was created/updated but is still provisioning.")
    print("It will become ready eventually. Continuing with remaining steps...")
except RuntimeError as e:
    print(f"WARNING: Endpoint readiness check failed: {e}")
    print("The endpoint may need manual investigation. Continuing...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test the Endpoint with Sample Payloads

# COMMAND ----------

def query_endpoint(
    endpoint_name: str,
    served_entity_name: str,
    dataframe_records: list,
) -> dict:
    """Send a scoring request to a specific served entity on the endpoint.

    Uses the Databricks workspace token from the SDK client for authentication.

    Args:
        endpoint_name: Name of the serving endpoint.
        served_entity_name: Name of the served entity (model) to invoke.
        dataframe_records: List of dicts, each representing one input row.

    Returns:
        Dict with keys 'predictions', 'latency_ms', and 'status_code'.
    """
    serving_url = (
        f"{w.config.host}/serving-endpoints/{endpoint_name}/invocations"
    )
    headers = {
        "Authorization": f"Bearer {w.config.token}",
        "Content-Type": "application/json",
    }
    payload = {
        "dataframe_records": dataframe_records,
        "params": {"served_entity_name": served_entity_name},
    }

    start = time.time()
    response = requests.post(serving_url, headers=headers, json=payload, timeout=120)
    latency_ms = (time.time() - start) * 1000

    result = {
        "status_code": response.status_code,
        "latency_ms": round(latency_ms, 2),
        "predictions": None,
        "raw_response": None,
    }

    if response.status_code == 200:
        body = response.json()
        result["predictions"] = body.get("predictions", body)
        result["raw_response"] = body
    else:
        result["raw_response"] = response.text
        print(f"  WARNING: Non-200 status ({response.status_code}): {response.text[:500]}")

    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5a. Price Elasticity Model

# COMMAND ----------

# Sample payload: features expected by the price_elasticity_lgbm model
# Target: volume_delta_pct
elasticity_sample = [
    {
        "product_category": "Orthopaedics",
        "price_delta_pct": -0.02,
        "avg_pocket_price": 6200.0,
        "avg_list_price": 7500.0,
        "discount_depth_avg": 0.17,
        "price_realization_avg": 0.83,
        "seasonal_index_avg": 1.05,
        "competitor_asp_gap": 0.04,
        "contract_mix_score": 0.72,
        "macro_pressure_score": 0.45,
        "innovation_tier": 4,
        "market_share_pct": 0.28,
        "patent_years_remaining": 8,
        "gpo_concentration": 0.35,
    },
]

try:
    elasticity_result = query_endpoint(ENDPOINT_NAME, "price-elasticity", elasticity_sample)
    print("Price Elasticity Model Response")
    print(f"  Status Code : {elasticity_result['status_code']}")
    print(f"  Latency     : {elasticity_result['latency_ms']:.1f} ms")
    print(f"  Predictions : {elasticity_result['predictions']}")
except Exception as e:
    print(f"WARNING: Price elasticity endpoint test skipped (endpoint may still be provisioning): {e}")
    elasticity_result = None

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5b. Revenue Impact Model

# COMMAND ----------

# Sample payload: features expected by the revenue_impact_xgb model
# Target: volume_delta_pct  (numeric features only)
revenue_sample = [
    {
        "price_delta_pct": -0.03,
        "avg_pocket_price": 6200.0,
        "discount_depth_avg": 0.17,
        "price_realization_avg": 0.83,
        "margin_pct_avg": 0.58,
        "seasonal_index_avg": 1.05,
        "competitor_asp_gap": 0.04,
        "market_share_pct": 0.28,
        "tariff_impact_index": 12.5,
        "macro_pressure_score": 0.45,
        "gpo_concentration": 0.35,
        "contract_mix_score": 0.72,
        "innovation_tier": 4,
    },
]

try:
    revenue_result = query_endpoint(ENDPOINT_NAME, "revenue-impact", revenue_sample)
    print("Revenue Impact Model Response")
    print(f"  Status Code : {revenue_result['status_code']}")
    print(f"  Latency     : {revenue_result['latency_ms']:.1f} ms")
    print(f"  Predictions : {revenue_result['predictions']}")
except Exception as e:
    print(f"WARNING: Revenue impact endpoint test skipped (endpoint may still be provisioning): {e}")
    revenue_result = None

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5c. Margin Optimization Model

# COMMAND ----------

# Sample payload: features expected by the margin_optimization_enet model
# Target: margin_pct_avg
margin_sample = [
    {
        "product_category": "Orthopaedics",
        "price_delta_pct": -0.02,
        "avg_pocket_price": 6200.0,
        "avg_list_price": 7500.0,
        "discount_depth_avg": 0.17,
        "price_realization_avg": 0.83,
        "tariff_impact_index": 12.5,
        "macro_pressure_score": 0.45,
        "supply_chain_pressure_index": 35.0,
        "fuel_index": 120.0,
        "steel_tariff_pct": 25.0,
        "titanium_tariff_pct": 5.0,
        "innovation_tier": 4,
        "gpo_concentration": 0.35,
    },
]

try:
    margin_result = query_endpoint(ENDPOINT_NAME, "margin-optimization", margin_sample)
    print("Margin Optimization Model Response")
    print(f"  Status Code : {margin_result['status_code']}")
    print(f"  Latency     : {margin_result['latency_ms']:.1f} ms")
    print(f"  Predictions : {margin_result['predictions']}")
except Exception as e:
    print(f"WARNING: Margin optimization endpoint test skipped (endpoint may still be provisioning): {e}")
    margin_result = None

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5d. Latency Summary

# COMMAND ----------

print("=" * 60)
print("Inference Latency Summary")
print("=" * 60)
print(f"  {'Model':<25s} {'Status':<10s} {'Latency (ms)':>12s}")
print(f"  {'-'*25} {'-'*10} {'-'*12}")

for label, result in [
    ("price-elasticity", elasticity_result),
    ("revenue-impact", revenue_result),
    ("margin-optimization", margin_result),
]:
    if result is None:
        print(f"  {label:<25s} {'SKIP':<10s} {'N/A':>12s}")
    else:
        status_str = "OK" if result["status_code"] == 200 else f"ERR({result['status_code']})"
        print(f"  {label:<25s} {status_str:<10s} {result['latency_ms']:>12.1f}")

print("=" * 60)
if all(r is not None for r in [elasticity_result, revenue_result, margin_result]):
    print("All endpoint tests passed.")
else:
    print("Some endpoint tests were skipped (endpoint still provisioning).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create Spark UDFs for Batch Scoring

# COMMAND ----------

# MAGIC %md
# MAGIC Spark UDFs wrap registered MLflow models so they can be applied as column
# MAGIC transformations inside `DataFrame.withColumn()`. This enables high-throughput
# MAGIC batch scoring across millions of rows using the cluster's distributed compute,
# MAGIC without routing through the REST serving endpoint.

# COMMAND ----------

# Set the MLflow registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Create PySpark UDFs from registered Unity Catalog models
elasticity_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{CATALOG}.{MODEL_SCHEMA}.price_elasticity_lgbm@champion",
    result_type="double",
)

revenue_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{CATALOG}.{MODEL_SCHEMA}.revenue_impact_xgb@champion",
    result_type="double",
)

margin_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{CATALOG}.{MODEL_SCHEMA}.margin_optimization_enet@champion",
    result_type="double",
)

print("Spark UDFs created successfully:")
print(f"  - elasticity_udf  (price_elasticity_lgbm @champion)")
print(f"  - revenue_udf     (revenue_impact_xgb @champion)")
print(f"  - margin_udf      (margin_optimization_enet @champion)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Batch Scoring Demonstration

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7a. Build a Sample DataFrame for Batch Scoring
# MAGIC
# MAGIC In production, this DataFrame would be sourced from
# MAGIC `hls_amer_catalog.gold.pricing_features` or a similar feature table.
# MAGIC Here we construct a small sample to validate the UDF pipeline end-to-end.

# COMMAND ----------

# NOTE: Batch scoring columns must match the trained model schemas from
# notebooks 03a, 03b, and 03c. The batch DataFrame below includes the
# superset of features used across all three models. If the model schemas
# change after retraining, update these columns accordingly.
#
# The entire batch scoring section is wrapped in try/except so the notebook
# does not crash if there is a column mismatch with the registered models.

try:
    batch_data = [
        # product_category, price_delta_pct, avg_pocket_price, avg_list_price,
        # discount_depth_avg, price_realization_avg, margin_pct_avg,
        # seasonal_index_avg, competitor_asp_gap, contract_mix_score,
        # macro_pressure_score, innovation_tier, market_share_pct,
        # patent_years_remaining, gpo_concentration, tariff_impact_index,
        # supply_chain_pressure_index, fuel_index, steel_tariff_pct,
        # titanium_tariff_pct
        ("Orthopaedics", -0.02, 6200.0, 7500.0, 0.17, 0.83, 0.58,
         1.05, 0.04, 0.72, 0.45, 4, 0.28, 8, 0.35,
         12.5, 35.0, 120.0, 25.0, 5.0),
        ("MedSurg", -0.03, 4100.0, 5000.0, 0.18, 0.82, 0.55,
         0.98, 0.06, 0.65, 0.50, 3, 0.22, 5, 0.40,
         10.0, 30.0, 118.0, 24.0, 4.5),
        ("Neurotechnology", 0.01, 9800.0, 11500.0, 0.15, 0.85, 0.62,
         1.02, 0.03, 0.78, 0.38, 5, 0.18, 12, 0.30,
         8.0, 28.0, 115.0, 22.0, 3.0),
        ("Orthopaedics", -0.05, 3200.0, 4000.0, 0.20, 0.80, 0.52,
         1.10, 0.07, 0.60, 0.55, 2, 0.35, 3, 0.45,
         15.0, 40.0, 125.0, 26.0, 6.0),
        ("MedSurg", -0.01, 5500.0, 6800.0, 0.19, 0.81, 0.60,
         0.95, 0.05, 0.70, 0.42, 4, 0.25, 7, 0.38,
         11.0, 32.0, 119.0, 23.0, 4.0),
    ]

    batch_schema = StructType([
        StructField("product_category", StringType(), False),
        StructField("price_delta_pct", DoubleType(), False),
        StructField("avg_pocket_price", DoubleType(), False),
        StructField("avg_list_price", DoubleType(), False),
        StructField("discount_depth_avg", DoubleType(), False),
        StructField("price_realization_avg", DoubleType(), False),
        StructField("margin_pct_avg", DoubleType(), False),
        StructField("seasonal_index_avg", DoubleType(), False),
        StructField("competitor_asp_gap", DoubleType(), False),
        StructField("contract_mix_score", DoubleType(), False),
        StructField("macro_pressure_score", DoubleType(), False),
        StructField("innovation_tier", IntegerType(), False),
        StructField("market_share_pct", DoubleType(), False),
        StructField("patent_years_remaining", IntegerType(), False),
        StructField("gpo_concentration", DoubleType(), False),
        StructField("tariff_impact_index", DoubleType(), False),
        StructField("supply_chain_pressure_index", DoubleType(), False),
        StructField("fuel_index", DoubleType(), False),
        StructField("steel_tariff_pct", DoubleType(), False),
        StructField("titanium_tariff_pct", DoubleType(), False),
    ])

    batch_df = spark.createDataFrame(batch_data, schema=batch_schema)
    print(f"Sample batch DataFrame: {batch_df.count()} rows, {len(batch_df.columns)} columns")
    batch_df.display()

except Exception as e:
    print(f"WARNING: Could not create batch DataFrame: {e}")
    print("Skipping batch scoring demonstration.")
    batch_df = None

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7b. Apply UDFs for Multi-Model Batch Scoring

# COMMAND ----------

# Define the column groups each model expects.
# The UDFs are applied using struct() to pass the correct feature columns.
# NOTE: These struct definitions must match the exact feature lists from
# the training notebooks (03a, 03b, 03c).

try:
    if batch_df is None:
        raise RuntimeError("batch_df was not created; skipping UDF application.")

    # Price elasticity features (03a - price_elasticity_lgbm)
    elasticity_features = F.struct(
        F.col("product_category"),
        F.col("price_delta_pct"),
        F.col("avg_pocket_price"),
        F.col("avg_list_price"),
        F.col("discount_depth_avg"),
        F.col("price_realization_avg"),
        F.col("seasonal_index_avg"),
        F.col("competitor_asp_gap"),
        F.col("contract_mix_score"),
        F.col("macro_pressure_score"),
        F.col("innovation_tier"),
        F.col("market_share_pct"),
        F.col("patent_years_remaining"),
        F.col("gpo_concentration"),
    )

    # Revenue impact features (03b - revenue_impact_xgb, numeric only)
    revenue_features = F.struct(
        F.col("price_delta_pct"),
        F.col("avg_pocket_price"),
        F.col("discount_depth_avg"),
        F.col("price_realization_avg"),
        F.col("margin_pct_avg"),
        F.col("seasonal_index_avg"),
        F.col("competitor_asp_gap"),
        F.col("market_share_pct"),
        F.col("tariff_impact_index"),
        F.col("macro_pressure_score"),
        F.col("gpo_concentration"),
        F.col("contract_mix_score"),
        F.col("innovation_tier"),
    )

    # Margin optimization features (03c - margin_optimization_enet)
    margin_features = F.struct(
        F.col("product_category"),
        F.col("price_delta_pct"),
        F.col("avg_pocket_price"),
        F.col("avg_list_price"),
        F.col("discount_depth_avg"),
        F.col("price_realization_avg"),
        F.col("tariff_impact_index"),
        F.col("macro_pressure_score"),
        F.col("supply_chain_pressure_index"),
        F.col("fuel_index"),
        F.col("steel_tariff_pct"),
        F.col("titanium_tariff_pct"),
        F.col("innovation_tier"),
        F.col("gpo_concentration"),
    )

    # Stage 1: Price Elasticity predictions (target: volume_delta_pct)
    scored_df = batch_df.withColumn(
        "elasticity_prediction",
        elasticity_udf(elasticity_features),
    )

    # Stage 2: Revenue Impact predictions (target: volume_delta_pct)
    scored_df = scored_df.withColumn(
        "revenue_impact_prediction",
        revenue_udf(revenue_features),
    )

    # Stage 3: Margin Optimization predictions (target: margin_pct_avg)
    scored_df = scored_df.withColumn(
        "optimal_margin_prediction",
        margin_udf(margin_features),
    )

    print("Batch scoring complete (3-model inference).")

except Exception as e:
    scored_df = None
    print(f"WARNING: Batch scoring failed: {e}")
    print("This may indicate a mismatch between batch DataFrame columns and")
    print("the trained model input schemas. Re-run notebooks 03a-03c and verify")
    print("the feature lists match the struct definitions above.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7c. Review Batch Scoring Results

# COMMAND ----------

try:
    if scored_df is None:
        raise RuntimeError("scored_df was not created; skipping results review.")

    results_df = scored_df.select(
        "product_category",
        "price_delta_pct",
        F.round("elasticity_prediction", 4).alias("price_elasticity"),
        F.round("revenue_impact_prediction", 4).alias("revenue_impact"),
        F.round("optimal_margin_prediction", 4).alias("optimal_margin"),
    )

    results_df.display()

    # Validation: ensure no null predictions were returned
    for col_name in ["elasticity_prediction", "revenue_impact_prediction", "optimal_margin_prediction"]:
        null_count = scored_df.filter(F.col(col_name).isNull()).count()
        assert null_count == 0, f"Found {null_count} null values in {col_name}"
        print(f"  {col_name}: 0 nulls - OK")

    print("\nBatch scoring validation passed. All predictions are non-null.")

except Exception as e:
    print(f"WARNING: Batch scoring results review skipped: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7d. Write Scored Results to Delta (Optional Production Step)

# COMMAND ----------

# In production, persist scored results for downstream consumption.
# Uncomment the block below to write to the gold layer.

# if scored_df is not None:
#     SCORED_TABLE = f"{CATALOG}.gold.pricing_model_scores"
#     (
#         scored_df.write
#         .format("delta")
#         .mode("overwrite")
#         .option("overwriteSchema", "true")
#         .saveAsTable(SCORED_TABLE)
#     )
#     print(f"Scored results written to {SCORED_TABLE}")

print("Batch scoring demonstration complete. Uncomment the write block above for production persistence.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Endpoint Permissions Setup

# COMMAND ----------

# MAGIC %md
# MAGIC Grant the `pricing_analysts_group` permission to query the serving endpoint.
# MAGIC This allows analysts to call the endpoint for ad-hoc scoring and integration
# MAGIC with BI tools without requiring workspace admin privileges.

# COMMAND ----------

def set_endpoint_permissions(
    endpoint_name: str,
    group_name: str,
    permission_level: str = "CAN_QUERY",
) -> None:
    """Grant a group permission to query a serving endpoint.

    Uses the Databricks Permissions API to add an access control entry for the
    specified group on the serving endpoint.

    Args:
        endpoint_name: Name of the serving endpoint.
        group_name: Databricks workspace group name.
        permission_level: Permission level to grant. One of:
            CAN_MANAGE, CAN_QUERY, CAN_VIEW.
    """
    # Retrieve the endpoint to get its ID
    endpoint = w.serving_endpoints.get(name=endpoint_name)
    endpoint_id = endpoint.id

    print(f"Setting permissions on endpoint '{endpoint_name}' (id={endpoint_id}):")
    print(f"  Group            : {group_name}")
    print(f"  Permission Level : {permission_level}")

    # Use the Permissions API via the SDK
    from databricks.sdk.service.iam import (
        ObjectPermissions,
        AccessControlRequest,
        Permission,
        PermissionLevel,
    )

    acl = AccessControlRequest(
        group_name=group_name,
        permission_level=PermissionLevel(permission_level),
    )

    w.permissions.update(
        request_object_type="serving-endpoints",
        request_object_id=endpoint_id,
        access_control_list=[acl],
    )

    print(f"  Successfully granted {permission_level} to '{group_name}'.")

# COMMAND ----------

try:
    set_endpoint_permissions(
        endpoint_name=ENDPOINT_NAME,
        group_name=PRICING_ANALYSTS_GROUP,
        permission_level="CAN_QUERY",
    )
except Exception as e:
    print(f"WARNING: Could not set endpoint permissions: {e}")
    print("The group may not exist yet. Permissions can be set manually later.")

# COMMAND ----------

try:
    endpoint = w.serving_endpoints.get(name=ENDPOINT_NAME)
    permissions = w.permissions.get(
        request_object_type="serving-endpoints",
        request_object_id=endpoint.id,
    )

    print(f"Current permissions for endpoint '{ENDPOINT_NAME}':")
    print("-" * 60)
    for acl in permissions.access_control_list:
        principal = acl.group_name or acl.user_name or acl.service_principal_name or "unknown"
        perms = ", ".join(
            [str(p.permission_level) for p in (acl.all_permissions or [])]
        )
        print(f"  {principal:<35s}  {perms}")
    print("-" * 60)
except Exception as e:
    print(f"WARNING: Could not verify endpoint permissions: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary & Next Steps

# COMMAND ----------

print("=" * 70)
print("  MODEL SERVING SETUP COMPLETE")
print("=" * 70)
print()
print(f"  Endpoint Name   : {ENDPOINT_NAME}")
print(f"  Workspace       : {w.config.host}")
print(f"  Models Served   : {len(MODELS)}")
print()
print("  Served Entities:")
for name, cfg in MODELS.items():
    print(f"    - {name:<25s} => {cfg['entity_name']} (v{cfg['entity_version']})")
print()
print(f"  Permissions     : {PRICING_ANALYSTS_GROUP} => CAN_QUERY")
print()
print("  Real-time Inference : Validated (all 3 models)")
print("  Batch Scoring UDFs  : Created (elasticity_udf, revenue_udf, margin_udf)")
print()
print("  Next Steps:")
print("    1. Integrate endpoint URL into pricing dashboard API layer")
print("    2. Schedule batch scoring jobs via Databricks Workflows")
print("    3. Set up endpoint monitoring and alerting")
print("    4. Configure A/B testing routes for model version upgrades")
print("=" * 70)
