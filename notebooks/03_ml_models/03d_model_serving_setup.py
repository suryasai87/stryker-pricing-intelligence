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

# Traffic split: each served entity receives 100 % of requests routed to it.
# With multiple served entities, the caller specifies which entity to invoke
# via the request body. The traffic config below ensures the latest version
# of each entity handles all its traffic (no canary / shadow deployments).
TRAFFIC_ROUTES = [
    Route(served_model_name=name, traffic_percentage=100)
    for name in MODELS.keys()
]

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

wait_for_endpoint_ready(ENDPOINT_NAME)

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
elasticity_sample = [
    {
        "product_id": "STK-KNEE-001",
        "current_price": 12500.00,
        "competitor_avg_price": 13200.00,
        "volume_last_quarter": 340,
        "market_share_pct": 0.28,
        "cpi_medical": 4.1,
        "hospital_capex_index": 102.5,
        "contract_tier": "GPO_TIER1",
    },
    {
        "product_id": "STK-HIP-003",
        "current_price": 15800.00,
        "competitor_avg_price": 16100.00,
        "volume_last_quarter": 210,
        "market_share_pct": 0.22,
        "cpi_medical": 4.1,
        "hospital_capex_index": 102.5,
        "contract_tier": "IDN_LARGE",
    },
]

elasticity_result = query_endpoint(ENDPOINT_NAME, "price-elasticity", elasticity_sample)

print("Price Elasticity Model Response")
print(f"  Status Code : {elasticity_result['status_code']}")
print(f"  Latency     : {elasticity_result['latency_ms']:.1f} ms")
print(f"  Predictions : {elasticity_result['predictions']}")
assert elasticity_result["status_code"] == 200, (
    f"Price elasticity inference failed: {elasticity_result['raw_response']}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5b. Revenue Impact Model

# COMMAND ----------

# Sample payload: features expected by the revenue_impact_xgb model
revenue_sample = [
    {
        "product_id": "STK-KNEE-001",
        "proposed_price_change_pct": -0.05,
        "current_annual_revenue": 4250000.00,
        "elasticity_estimate": -1.35,
        "contract_renewal_months": 8,
        "competitor_price_gap_pct": 0.056,
        "market_growth_rate": 0.04,
        "tariff_rate_steel": 25.2,
    },
]

revenue_result = query_endpoint(ENDPOINT_NAME, "revenue-impact", revenue_sample)

print("Revenue Impact Model Response")
print(f"  Status Code : {revenue_result['status_code']}")
print(f"  Latency     : {revenue_result['latency_ms']:.1f} ms")
print(f"  Predictions : {revenue_result['predictions']}")
assert revenue_result["status_code"] == 200, (
    f"Revenue impact inference failed: {revenue_result['raw_response']}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5c. Margin Optimization Model

# COMMAND ----------

# Sample payload: features expected by the margin_optimization_enet model
margin_sample = [
    {
        "product_id": "STK-KNEE-001",
        "current_price": 12500.00,
        "cogs": 4800.00,
        "volume_forecast": 360,
        "elasticity_estimate": -1.35,
        "revenue_impact_estimate": 0.02,
        "resin_price_index": 105.3,
        "cobalt_chrome_price_index": 118.7,
        "supply_chain_pressure_index": 0.8,
        "target_margin_pct": 0.62,
    },
    {
        "product_id": "STK-SPINE-002",
        "current_price": 22300.00,
        "cogs": 7100.00,
        "volume_forecast": 120,
        "elasticity_estimate": -0.85,
        "revenue_impact_estimate": 0.015,
        "resin_price_index": 105.3,
        "cobalt_chrome_price_index": 118.7,
        "supply_chain_pressure_index": 0.8,
        "target_margin_pct": 0.68,
    },
]

margin_result = query_endpoint(ENDPOINT_NAME, "margin-optimization", margin_sample)

print("Margin Optimization Model Response")
print(f"  Status Code : {margin_result['status_code']}")
print(f"  Latency     : {margin_result['latency_ms']:.1f} ms")
print(f"  Predictions : {margin_result['predictions']}")
assert margin_result["status_code"] == 200, (
    f"Margin optimization inference failed: {margin_result['raw_response']}"
)

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
    status_str = "OK" if result["status_code"] == 200 else f"ERR({result['status_code']})"
    print(f"  {label:<25s} {status_str:<10s} {result['latency_ms']:>12.1f}")

print("=" * 60)
print("All endpoint tests passed.")

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
    model_uri=f"models:/{CATALOG}.{MODEL_SCHEMA}.price_elasticity_lgbm/1",
    result_type="double",
)

revenue_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{CATALOG}.{MODEL_SCHEMA}.revenue_impact_xgb/1",
    result_type="double",
)

margin_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{CATALOG}.{MODEL_SCHEMA}.margin_optimization_enet/1",
    result_type="double",
)

print("Spark UDFs created successfully:")
print(f"  - elasticity_udf  (price_elasticity_lgbm v1)")
print(f"  - revenue_udf     (revenue_impact_xgb v1)")
print(f"  - margin_udf      (margin_optimization_enet v1)")

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

batch_data = [
    ("STK-KNEE-001", 12500.0, 13200.0, 340, 0.28, 4.1, 102.5, "GPO_TIER1",
     -0.05, 4250000.0, 8, 0.056, 0.04, 25.2,
     4800.0, 360, 105.3, 118.7, 0.8, 0.62),
    ("STK-HIP-003", 15800.0, 16100.0, 210, 0.22, 4.1, 102.5, "IDN_LARGE",
     -0.03, 3318000.0, 14, 0.019, 0.035, 24.8,
     6200.0, 220, 105.3, 118.7, 0.8, 0.60),
    ("STK-SPINE-002", 22300.0, 23100.0, 120, 0.18, 3.9, 104.0, "GPO_TIER2",
     0.02, 2676000.0, 3, 0.036, 0.045, 25.5,
     7100.0, 125, 107.1, 120.2, 0.6, 0.68),
    ("STK-TRAUMA-010", 3200.0, 3450.0, 1800, 0.35, 4.2, 101.0, "DIRECT",
     -0.08, 5760000.0, 11, 0.072, 0.03, 26.0,
     1100.0, 1900, 103.8, 115.4, 1.1, 0.66),
    ("STK-ENDO-007", 8900.0, 9200.0, 510, 0.25, 4.0, 103.2, "IDN_SMALL",
     -0.04, 4539000.0, 6, 0.033, 0.038, 24.5,
     3400.0, 530, 106.0, 119.5, 0.7, 0.62),
]

batch_schema = StructType([
    StructField("product_id", StringType(), False),
    StructField("current_price", DoubleType(), False),
    StructField("competitor_avg_price", DoubleType(), False),
    StructField("volume_last_quarter", IntegerType(), False),
    StructField("market_share_pct", DoubleType(), False),
    StructField("cpi_medical", DoubleType(), False),
    StructField("hospital_capex_index", DoubleType(), False),
    StructField("contract_tier", StringType(), False),
    # Revenue-impact features
    StructField("proposed_price_change_pct", DoubleType(), False),
    StructField("current_annual_revenue", DoubleType(), False),
    StructField("contract_renewal_months", IntegerType(), False),
    StructField("competitor_price_gap_pct", DoubleType(), False),
    StructField("market_growth_rate", DoubleType(), False),
    StructField("tariff_rate_steel", DoubleType(), False),
    # Margin-optimization features
    StructField("cogs", DoubleType(), False),
    StructField("volume_forecast", IntegerType(), False),
    StructField("resin_price_index", DoubleType(), False),
    StructField("cobalt_chrome_price_index", DoubleType(), False),
    StructField("supply_chain_pressure_index", DoubleType(), False),
    StructField("target_margin_pct", DoubleType(), False),
])

batch_df = spark.createDataFrame(batch_data, schema=batch_schema)
print(f"Sample batch DataFrame: {batch_df.count()} rows, {len(batch_df.columns)} columns")
batch_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7b. Apply UDFs for Multi-Model Batch Scoring

# COMMAND ----------

# Define the column groups each model expects.
# The UDFs are applied using struct() to pass the correct feature columns.

# Price elasticity features
elasticity_features = F.struct(
    F.col("product_id"),
    F.col("current_price"),
    F.col("competitor_avg_price"),
    F.col("volume_last_quarter"),
    F.col("market_share_pct"),
    F.col("cpi_medical"),
    F.col("hospital_capex_index"),
    F.col("contract_tier"),
)

# Revenue impact features (includes elasticity prediction as input)
revenue_features = F.struct(
    F.col("product_id"),
    F.col("proposed_price_change_pct"),
    F.col("current_annual_revenue"),
    F.col("elasticity_prediction"),  # chained from elasticity model
    F.col("contract_renewal_months"),
    F.col("competitor_price_gap_pct"),
    F.col("market_growth_rate"),
    F.col("tariff_rate_steel"),
)

# Margin optimization features (includes both upstream predictions)
margin_features = F.struct(
    F.col("product_id"),
    F.col("current_price"),
    F.col("cogs"),
    F.col("volume_forecast"),
    F.col("elasticity_prediction"),
    F.col("revenue_impact_prediction"),
    F.col("resin_price_index"),
    F.col("cobalt_chrome_price_index"),
    F.col("supply_chain_pressure_index"),
    F.col("target_margin_pct"),
)

# COMMAND ----------

# Stage 1: Price Elasticity predictions
scored_df = batch_df.withColumn(
    "elasticity_prediction",
    elasticity_udf(elasticity_features),
)

# Stage 2: Revenue Impact predictions (chained - uses elasticity output)
scored_df = scored_df.withColumn(
    "revenue_impact_prediction",
    revenue_udf(revenue_features),
)

# Stage 3: Margin Optimization predictions (uses both upstream outputs)
scored_df = scored_df.withColumn(
    "optimal_margin_prediction",
    margin_udf(margin_features),
)

print("Batch scoring complete (3-stage chained inference).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7c. Review Batch Scoring Results

# COMMAND ----------

results_df = scored_df.select(
    "product_id",
    "current_price",
    F.round("elasticity_prediction", 4).alias("price_elasticity"),
    F.round("revenue_impact_prediction", 4).alias("revenue_impact"),
    F.round("optimal_margin_prediction", 4).alias("optimal_margin"),
)

results_df.display()

# COMMAND ----------

# Validation: ensure no null predictions were returned
for col_name in ["elasticity_prediction", "revenue_impact_prediction", "optimal_margin_prediction"]:
    null_count = scored_df.filter(F.col(col_name).isNull()).count()
    assert null_count == 0, f"Found {null_count} null values in {col_name}"
    print(f"  {col_name}: 0 nulls - OK")

print("\nBatch scoring validation passed. All predictions are non-null.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7d. Write Scored Results to Delta (Optional Production Step)

# COMMAND ----------

# In production, persist scored results for downstream consumption.
# Uncomment the block below to write to the gold layer.

# SCORED_TABLE = f"{CATALOG}.gold.pricing_model_scores"
# (
#     scored_df.write
#     .format("delta")
#     .mode("overwrite")
#     .option("overwriteSchema", "true")
#     .saveAsTable(SCORED_TABLE)
# )
# print(f"Scored results written to {SCORED_TABLE}")

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

set_endpoint_permissions(
    endpoint_name=ENDPOINT_NAME,
    group_name=PRICING_ANALYSTS_GROUP,
    permission_level="CAN_QUERY",
)

# COMMAND ----------

# Verify the permissions were applied correctly
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
