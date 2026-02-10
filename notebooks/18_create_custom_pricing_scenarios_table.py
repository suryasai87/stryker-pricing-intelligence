# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 18 - Custom Pricing Scenarios Table
# MAGIC
# MAGIC **Purpose**: Create the `hls_amer_catalog.gold.custom_pricing_scenarios` Delta table and
# MAGIC populate it with 18 synthetic pricing scenarios from 5 mock users.  This table backs the
# MAGIC "What-If Scenario Planner" feature in the Stryker Pricing Intelligence application,
# MAGIC allowing analysts and commercial teams to save, share, and track pricing scenarios
# MAGIC through a lightweight approval workflow.
# MAGIC
# MAGIC **Output**: `hls_amer_catalog.gold.custom_pricing_scenarios` (Delta, Unity Catalog)
# MAGIC
# MAGIC **Reproducibility**: All stochastic operations use `seed=42`.
# MAGIC
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | scenario_id | STRING | UUID primary key |
# MAGIC | user_id | STRING | Opaque user identifier |
# MAGIC | user_email | STRING | Creator email address |
# MAGIC | scenario_name | STRING | Human-readable scenario title |
# MAGIC | description | STRING | Free-text description of the scenario |
# MAGIC | assumptions | STRING | JSON object of modelling assumptions |
# MAGIC | target_uplift_pct | DOUBLE | Target revenue uplift percentage |
# MAGIC | selected_skus | STRING | JSON array of selected SKU identifiers |
# MAGIC | selected_segments | STRING | JSON array of selected market segments |
# MAGIC | simulation_results | STRING | JSON object with simulation outputs |
# MAGIC | status | STRING | Workflow status: Draft / Submitted / Reviewed / Approved |
# MAGIC | created_at | TIMESTAMP | Row creation timestamp |
# MAGIC | updated_at | TIMESTAMP | Row last-update timestamp |
# MAGIC | is_deleted | BOOLEAN | Soft-delete flag (default false) |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import json
import uuid
import numpy as np
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    BooleanType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Deterministic seed
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Unity Catalog target
# ---------------------------------------------------------------------------
CATALOG: str = "hls_amer_catalog"
SCHEMA: str = "gold"
TABLE_NAME: str = "custom_pricing_scenarios"
FULLY_QUALIFIED_TABLE: str = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Mock Users & Scenario Templates
# MAGIC
# MAGIC Five personas representing typical platform users with varied roles and
# MAGIC scenario creation patterns.

# COMMAND ----------

# ---------------------------------------------------------------------------
# Mock user definitions: (user_id, email, scenario_count)
# ---------------------------------------------------------------------------
MOCK_USERS = [
    ("USR-001", "admin@stryker.com", 3),
    ("USR-002", "john.pricing@stryker.com", 4),
    ("USR-003", "sarah.analyst@stryker.com", 4),
    ("USR-004", "mike.strategy@stryker.com", 3),
    ("USR-005", "lisa.commercial@stryker.com", 4),
]

# ---------------------------------------------------------------------------
# Scenario templates -- each tuple: (name, description, segments, status)
# ---------------------------------------------------------------------------
SCENARIO_TEMPLATES = [
    # admin@stryker.com (3 scenarios)
    (
        "Q1 2025 Ortho Price Increase",
        "Evaluate 3-5% list price increase across hip and knee implants for Q1 2025",
        ["Orthopaedics - Joint Replacement"],
        "Approved",
    ),
    (
        "Trauma Portfolio Margin Recovery",
        "Assess discount reduction strategy for trauma plates and screws segment",
        ["Orthopaedics - Trauma"],
        "Reviewed",
    ),
    (
        "Enterprise-Wide Tariff Impact",
        "Model pricing adjustments needed to offset projected 2025 steel/titanium tariff increases",
        ["Orthopaedics - Joint Replacement", "Orthopaedics - Trauma", "MedSurg - Instruments"],
        "Draft",
    ),
    # john.pricing@stryker.com (4 scenarios)
    (
        "MedSurg Power Tools Competitive Response",
        "Price matching analysis for System 9 vs DePuy Synthes Power Pro launch",
        ["MedSurg - Instruments"],
        "Submitted",
    ),
    (
        "Knee Replacement Volume Optimization",
        "Find optimal price point to maximize volume in competitive IDN accounts",
        ["Orthopaedics - Joint Replacement"],
        "Approved",
    ),
    (
        "GPO Contract Renewal - Premier",
        "Model pricing tiers for Premier GPO contract renewal covering 120+ hospitals",
        ["Orthopaedics - Joint Replacement", "Orthopaedics - Trauma", "MedSurg - Instruments"],
        "Reviewed",
    ),
    (
        "Biologics Margin Expansion",
        "Evaluate premium pricing strategy for next-gen bone graft substitutes",
        ["Consumables - Biologics"],
        "Draft",
    ),
    # sarah.analyst@stryker.com (4 scenarios)
    (
        "Shoulder Arthroplasty Market Share Gain",
        "Price-volume tradeoff analysis for reverse shoulder portfolio vs Zimmer Comprehensive",
        ["Orthopaedics - Joint Replacement"],
        "Submitted",
    ),
    (
        "Endoscopy Visualization Bundle Pricing",
        "Evaluate bundled pricing for 1788 camera system + disposables",
        ["MedSurg - Endoscopy", "Consumables - Disposables"],
        "Draft",
    ),
    (
        "Neurovascular Coil Pricing Refresh",
        "Annual pricing review for Target coil portfolio with competitor gap analysis",
        ["Neurotechnology - Neurovascular"],
        "Approved",
    ),
    (
        "Capital Equipment Leasing vs Purchase",
        "Compare Mako system lease pricing models for community hospital segment",
        ["Capital Equipment - Robotics"],
        "Reviewed",
    ),
    # mike.strategy@stryker.com (3 scenarios)
    (
        "Cross-Portfolio Inflation Adjustment",
        "Model CPI-indexed price adjustments across all segments for FY2025",
        ["Orthopaedics - Joint Replacement", "MedSurg - Instruments", "Neurotechnology - Neurovascular", "Consumables - Disposables"],
        "Submitted",
    ),
    (
        "Extremities Launch Pricing",
        "Set launch pricing for new small bone fixation system entering competitive market",
        ["Orthopaedics - Extremities"],
        "Draft",
    ),
    (
        "Hospital Bed Competitive Defense",
        "Defensive pricing strategy against Hill-Rom Centrella in acute care segment",
        ["MedSurg - Medical"],
        "Approved",
    ),
    # lisa.commercial@stryker.com (4 scenarios)
    (
        "IDN Standardization Discount Analysis",
        "Model volume-based discount tiers for top 20 IDN accounts standardizing on Stryker",
        ["Orthopaedics - Joint Replacement", "Orthopaedics - Trauma"],
        "Reviewed",
    ),
    (
        "Ambulance Cot Fleet Pricing",
        "Fleet pricing analysis for Power-PRO XT across EMS agencies",
        ["MedSurg - Medical"],
        "Submitted",
    ),
    (
        "Cranial CMF Value-Based Pricing",
        "Shift cranial fixation from cost-plus to value-based pricing tied to outcomes",
        ["Neurotechnology - Cranial"],
        "Draft",
    ),
    (
        "Disposables Pull-Through Strategy",
        "Model disposable pricing to maximize capital equipment pull-through revenue",
        ["Consumables - Disposables", "MedSurg - Instruments"],
        "Approved",
    ),
]

print(f"Defined {len(SCENARIO_TEMPLATES)} scenario templates across {len(MOCK_USERS)} users")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate Synthetic Scenario Rows

# COMMAND ----------

def _generate_scenario_rows(seed: int = RANDOM_SEED):
    """Generate 18 synthetic pricing scenario rows deterministically.

    Each row includes realistic JSON payloads for assumptions, selected SKUs,
    and simulation results.  Timestamps span October 2024 through January 2025.

    Returns
    -------
    list[dict]
        Each dict maps column name to scalar value.
    """
    rng = np.random.RandomState(seed)
    _uuid_ns = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

    # Date range: Oct 1, 2024 to Jan 31, 2025
    date_start = datetime(2024, 10, 1)
    date_end = datetime(2025, 1, 31)
    date_range_days = (date_end - date_start).days

    # Sample SKU IDs (deterministic UUIDs matching product master pattern)
    sample_skus = [
        str(uuid.uuid5(_uuid_ns, f"stryker-sku-{i:04d}"))
        for i in range(1, 221)
    ]

    rows = []
    template_idx = 0

    for user_id, user_email, scenario_count in MOCK_USERS:
        for _ in range(scenario_count):
            template = SCENARIO_TEMPLATES[template_idx]
            scenario_name, description, segments, status = template

            # Deterministic scenario ID
            scenario_id = str(uuid.uuid5(_uuid_ns, f"scenario-{template_idx:03d}"))

            # Target uplift: 1-12%
            target_uplift_pct = round(float(rng.uniform(1.0, 12.0)), 2)

            # Select 3-15 random SKUs
            n_skus = rng.randint(3, 16)
            selected_sku_indices = rng.choice(len(sample_skus), size=n_skus, replace=False)
            selected_skus = json.dumps([sample_skus[i] for i in selected_sku_indices])

            selected_segments = json.dumps(segments)

            # Assumptions JSON
            assumptions = json.dumps({
                "price_change_pct": round(float(rng.uniform(-5.0, 10.0)), 2),
                "volume_elasticity": round(float(rng.uniform(-2.5, -0.3)), 2),
                "competitor_response": rng.choice(["none", "partial_match", "full_match"]),
                "tariff_scenario": rng.choice(["baseline", "moderate_increase", "severe_increase"]),
                "contract_renewal": bool(rng.choice([True, False])),
                "time_horizon_months": int(rng.choice([3, 6, 12])),
            })

            # Simulation results JSON (populated for non-Draft scenarios)
            if status != "Draft":
                base_revenue = round(float(rng.uniform(5_000_000, 50_000_000)), 2)
                simulated_uplift = round(float(rng.uniform(0.5, target_uplift_pct * 1.2)), 2)
                simulation_results = json.dumps({
                    "base_revenue": base_revenue,
                    "simulated_revenue": round(base_revenue * (1 + simulated_uplift / 100), 2),
                    "revenue_delta": round(base_revenue * simulated_uplift / 100, 2),
                    "margin_impact_pct": round(float(rng.uniform(-1.0, 3.0)), 2),
                    "volume_impact_pct": round(float(rng.uniform(-8.0, 5.0)), 2),
                    "confidence_score": round(float(rng.uniform(0.65, 0.95)), 3),
                    "risk_level": rng.choice(["low", "medium", "high"]),
                    "breakeven_months": int(rng.randint(2, 9)),
                })
            else:
                simulation_results = json.dumps(None)

            # Timestamps: created_at in Oct-Jan range, updated_at >= created_at
            created_offset = timedelta(days=int(rng.randint(0, date_range_days)))
            created_at = date_start + created_offset
            update_lag = timedelta(
                days=int(rng.randint(0, 15)),
                hours=int(rng.randint(0, 24)),
                minutes=int(rng.randint(0, 60)),
            )
            updated_at = created_at + update_lag

            rows.append({
                "scenario_id": scenario_id,
                "user_id": user_id,
                "user_email": user_email,
                "scenario_name": scenario_name,
                "description": description,
                "assumptions": assumptions,
                "target_uplift_pct": target_uplift_pct,
                "selected_skus": selected_skus,
                "selected_segments": selected_segments,
                "simulation_results": simulation_results,
                "status": status,
                "created_at": created_at,
                "updated_at": updated_at,
                "is_deleted": False,
            })

            template_idx += 1

    return rows


scenario_rows = _generate_scenario_rows()
print(f"Generated {len(scenario_rows)} scenario rows")
print(f"\nStatus distribution:")
from collections import Counter
status_counts = Counter(r["status"] for r in scenario_rows)
for status, count in sorted(status_counts.items()):
    print(f"  {status}: {count}")
print(f"\nUser distribution:")
user_counts = Counter(r["user_email"] for r in scenario_rows)
for email, count in sorted(user_counts.items()):
    print(f"  {email}: {count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build PySpark DataFrame

# COMMAND ----------

SCENARIO_SCHEMA = StructType([
    StructField("scenario_id", StringType(), nullable=False),
    StructField("user_id", StringType(), nullable=False),
    StructField("user_email", StringType(), nullable=False),
    StructField("scenario_name", StringType(), nullable=False),
    StructField("description", StringType(), nullable=True),
    StructField("assumptions", StringType(), nullable=True),
    StructField("target_uplift_pct", DoubleType(), nullable=False),
    StructField("selected_skus", StringType(), nullable=True),
    StructField("selected_segments", StringType(), nullable=True),
    StructField("simulation_results", StringType(), nullable=True),
    StructField("status", StringType(), nullable=False),
    StructField("created_at", TimestampType(), nullable=False),
    StructField("updated_at", TimestampType(), nullable=False),
    StructField("is_deleted", BooleanType(), nullable=False),
])

df_scenarios = spark.createDataFrame(scenario_rows, schema=SCENARIO_SCHEMA)

print(f"DataFrame created: {df_scenarios.count()} rows, {len(df_scenarios.columns)} columns")
df_scenarios.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data Quality Checks

# COMMAND ----------

from pyspark.sql import functions as F

# --- 5a. Row count ---
row_count = df_scenarios.count()
assert row_count >= 15, f"Expected >= 15 rows, got {row_count}"
print(f"[CHECK] Row count: {row_count} (>= 15 required)")

# --- 5b. Unique scenario_ids ---
unique_ids = df_scenarios.select("scenario_id").distinct().count()
assert unique_ids == row_count, f"Duplicate scenario_ids found: {unique_ids} unique vs {row_count} total"
print(f"[CHECK] Unique scenario_ids: {unique_ids}")

# --- 5c. Valid statuses ---
valid_statuses = {"Draft", "Submitted", "Reviewed", "Approved"}
actual_statuses = set(
    row["status"]
    for row in df_scenarios.select("status").distinct().collect()
)
assert actual_statuses.issubset(valid_statuses), f"Invalid statuses found: {actual_statuses - valid_statuses}"
print(f"[CHECK] Valid statuses: {actual_statuses}")

# --- 5d. All 5 users represented ---
unique_users = df_scenarios.select("user_email").distinct().count()
assert unique_users == 5, f"Expected 5 unique users, got {unique_users}"
print(f"[CHECK] Unique users: {unique_users}")

# --- 5e. Target uplift in reasonable range ---
uplift_stats = df_scenarios.select(
    F.min("target_uplift_pct").alias("min"),
    F.max("target_uplift_pct").alias("max"),
).collect()[0]
assert uplift_stats["min"] > 0 and uplift_stats["max"] <= 15, "Uplift out of range"
print(f"[CHECK] Target uplift range: {uplift_stats['min']:.2f}% - {uplift_stats['max']:.2f}%")

# --- 5f. JSON columns are parseable ---
for col_name in ["assumptions", "selected_skus", "selected_segments", "simulation_results"]:
    sample_val = df_scenarios.filter(F.col(col_name).isNotNull()).select(col_name).first()
    if sample_val:
        json.loads(sample_val[0])
print("[CHECK] JSON columns are valid")

# --- 5g. No soft-deleted rows ---
deleted_count = df_scenarios.filter(F.col("is_deleted") == True).count()
print(f"[CHECK] Soft-deleted rows: {deleted_count}")

# --- 5h. updated_at >= created_at ---
invalid_ts = df_scenarios.filter(F.col("updated_at") < F.col("created_at")).count()
assert invalid_ts == 0, f"Found {invalid_ts} rows where updated_at < created_at"
print("[CHECK] updated_at >= created_at for all rows")

print("\n=== All data quality checks passed ===")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write to Delta Lake (Unity Catalog)

# COMMAND ----------

# Ensure schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema ready: {CATALOG}.{SCHEMA}")

# COMMAND ----------

(
    df_scenarios
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FULLY_QUALIFIED_TABLE)
)

print(f"Delta table written: {FULLY_QUALIFIED_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Post-Write Validation & Table Metadata

# COMMAND ----------

# --- 7a. Table comment ---
_comment = (
    "Custom pricing scenarios created by analysts and commercial teams for what-if "
    "planning and approval workflow. Supports Draft/Submitted/Reviewed/Approved lifecycle. "
    "Source notebook: 18_create_custom_pricing_scenarios_table.py"
)
spark.sql(f"COMMENT ON TABLE {FULLY_QUALIFIED_TABLE} IS '{_comment}'")

# --- 7b. Column comments ---
column_comments = {
    "scenario_id": "UUID primary key for the pricing scenario",
    "user_id": "Opaque user identifier (maps to identity provider)",
    "user_email": "Email address of the scenario creator",
    "scenario_name": "Human-readable title for the pricing scenario",
    "description": "Free-text description of the scenario rationale and scope",
    "assumptions": "JSON object encoding modelling assumptions (price change %, elasticity, etc.)",
    "target_uplift_pct": "Target revenue uplift percentage the scenario aims to achieve",
    "selected_skus": "JSON array of product_id UUIDs included in the scenario",
    "selected_segments": "JSON array of market segment labels included in the scenario",
    "simulation_results": "JSON object with simulation outputs (revenue, margin, risk)",
    "status": "Workflow status: Draft, Submitted, Reviewed, or Approved",
    "created_at": "Timestamp when the scenario was first created",
    "updated_at": "Timestamp of the most recent update to the scenario",
    "is_deleted": "Soft-delete flag; true = logically deleted (default false)",
}

for col_name, comment in column_comments.items():
    spark.sql(
        f"ALTER TABLE {FULLY_QUALIFIED_TABLE} ALTER COLUMN {col_name} COMMENT '{comment}'"
    )

print("Table and column comments applied")

# COMMAND ----------

# --- 7c. Read back and verify ---
df_verify = spark.table(FULLY_QUALIFIED_TABLE)
verify_count = df_verify.count()
assert verify_count == len(scenario_rows), (
    f"Post-write verification failed: expected {len(scenario_rows)}, got {verify_count}"
)

print(f"\n{'=' * 80}")
print(f"SUCCESS: {FULLY_QUALIFIED_TABLE}")
print(f"{'=' * 80}")
print(f"  Rows:       {verify_count}")
print(f"  Columns:    {len(df_verify.columns)}")
print(f"  Users:      {df_verify.select('user_email').distinct().count()}")
print(f"  Statuses:   {df_verify.select('status').distinct().count()}")
print(f"{'=' * 80}")

# Status breakdown
print("\nScenarios by status:")
df_verify.groupBy("status").count().orderBy("status").show(truncate=False)

# User breakdown
print("Scenarios by user:")
df_verify.groupBy("user_email").agg(
    F.count("*").alias("scenario_count"),
    F.collect_set("status").alias("statuses"),
).orderBy("user_email").show(truncate=False)

# Sample row
print("Sample scenario:")
df_verify.select(
    "scenario_id", "user_email", "scenario_name", "status", "target_uplift_pct", "created_at"
).show(5, truncate=40)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Total Scenarios | 18 |
# MAGIC | Unique Users | 5 |
# MAGIC | Statuses | Draft, Submitted, Reviewed, Approved |
# MAGIC | Date Range | Oct 2024 - Jan 2025 |
# MAGIC | Target Table | `hls_amer_catalog.gold.custom_pricing_scenarios` |
# MAGIC | Format | Delta (managed, Unity Catalog) |
# MAGIC | Reproducible | Yes (seed=42) |
# MAGIC
# MAGIC **Next notebook**: `19_gold_external_data_integration.py` -- external market data ingestion.
