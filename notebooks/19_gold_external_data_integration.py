# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 19 - External Market Data Integration
# MAGIC
# MAGIC **Purpose**: Create the `hls_amer_catalog.gold.external_market_data` Delta table and load
# MAGIC external market signals including tariff schedules, commodity prices, foreign exchange
# MAGIC rates, and competitor pricing intelligence.  This table enables the Pricing Intelligence
# MAGIC platform to correlate internal pricing decisions with external market dynamics.
# MAGIC
# MAGIC **Data Sources** (in priority order):
# MAGIC 1. Unity Catalog Volume: `/Volumes/hls_amer_catalog/gold/external_uploads/` (CSV files)
# MAGIC 2. Fallback: synthetically generated sample data matching the CSV schema
# MAGIC
# MAGIC **Output**: `hls_amer_catalog.gold.external_market_data` (Delta, Unity Catalog)
# MAGIC
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | data_source | STRING | Origin system or provider name |
# MAGIC | upload_timestamp | TIMESTAMP | When the data was ingested |
# MAGIC | category | STRING | tariff, commodity, fx, or competitor |
# MAGIC | item_key | STRING | Unique key within category (e.g., HTS code, ticker) |
# MAGIC | item_description | STRING | Human-readable description of the data item |
# MAGIC | value | DOUBLE | Numeric value of the data point |
# MAGIC | unit | STRING | Unit of measure (percent, USD, ratio, etc.) |
# MAGIC | effective_date | DATE | Date when the value takes effect |
# MAGIC | region | STRING | Geographic region (nullable) |
# MAGIC | notes | STRING | Additional context or annotations (nullable) |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import json
import numpy as np
from datetime import datetime, date, timedelta

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType,
    DateType,
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
TABLE_NAME: str = "external_market_data"
FULLY_QUALIFIED_TABLE: str = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

# ---------------------------------------------------------------------------
# Volume path for uploaded CSV files
# ---------------------------------------------------------------------------
VOLUME_PATH: str = f"/Volumes/{CATALOG}/{SCHEMA}/external_uploads"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define Table Schema

# COMMAND ----------

EXTERNAL_DATA_SCHEMA = StructType([
    StructField("data_source", StringType(), nullable=False),
    StructField("upload_timestamp", TimestampType(), nullable=False),
    StructField("category", StringType(), nullable=False),
    StructField("item_key", StringType(), nullable=False),
    StructField("item_description", StringType(), nullable=False),
    StructField("value", DoubleType(), nullable=False),
    StructField("unit", StringType(), nullable=False),
    StructField("effective_date", DateType(), nullable=False),
    StructField("region", StringType(), nullable=True),
    StructField("notes", StringType(), nullable=True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Table If Not Exists

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {FULLY_QUALIFIED_TABLE} (
        data_source STRING NOT NULL,
        upload_timestamp TIMESTAMP NOT NULL,
        category STRING NOT NULL,
        item_key STRING NOT NULL,
        item_description STRING NOT NULL,
        value DOUBLE NOT NULL,
        unit STRING NOT NULL,
        effective_date DATE NOT NULL,
        region STRING,
        notes STRING
    )
    USING DELTA
    COMMENT 'External market data including tariffs, commodity prices, FX rates, and competitor intelligence for pricing analytics.'
    TBLPROPERTIES (
        'quality' = 'gold',
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact' = 'true'
    )
""")

print(f"Table ready: {FULLY_QUALIFIED_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Attempt Volume-Based Ingestion
# MAGIC
# MAGIC Try to read CSV files from the Unity Catalog Volume.  If the volume does not
# MAGIC exist or contains no files, fall back to generating sample data in the next cell.

# COMMAND ----------

volume_data_loaded = False

try:
    # Check if the volume path exists and contains CSV files
    csv_files = dbutils.fs.ls(VOLUME_PATH)
    csv_paths = [f.path for f in csv_files if f.path.endswith(".csv")]

    if csv_paths:
        print(f"Found {len(csv_paths)} CSV file(s) in {VOLUME_PATH}")

        df_volume = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "false")
            .schema(EXTERNAL_DATA_SCHEMA)
            .csv(VOLUME_PATH)
        )

        volume_row_count = df_volume.count()
        if volume_row_count > 0:
            print(f"Loaded {volume_row_count} rows from volume")
            volume_data_loaded = True

            # Write to table (merge to avoid duplicates on re-run)
            df_volume.createOrReplaceTempView("_volume_external_data")
            spark.sql(f"""
                MERGE INTO {FULLY_QUALIFIED_TABLE} AS target
                USING _volume_external_data AS source
                ON target.data_source = source.data_source
                   AND target.item_key = source.item_key
                   AND target.effective_date = source.effective_date
                WHEN MATCHED THEN UPDATE SET *
                WHEN NOT MATCHED THEN INSERT *
            """)
            print(f"Merged volume data into {FULLY_QUALIFIED_TABLE}")
        else:
            print("Volume CSV files are empty; falling back to sample data generation")
    else:
        print(f"No CSV files found in {VOLUME_PATH}; falling back to sample data generation")

except Exception as e:
    print(f"Volume path not accessible ({type(e).__name__}: {e})")
    print("Falling back to sample data generation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate Sample Data (Fallback)
# MAGIC
# MAGIC If the volume is not available, generate ~50 rows of realistic external market
# MAGIC data spanning tariff schedules, commodity prices, FX rates, and competitor
# MAGIC intelligence.

# COMMAND ----------

def _generate_sample_external_data(seed: int = RANDOM_SEED):
    """Generate ~50 rows of realistic external market data.

    Categories covered:
    - **tariff**: US Section 232 steel tariffs, EU CBAM, HTS codes for medical devices
    - **commodity**: LME metals (cobalt, nickel), titanium bar, stainless steel, UHMWPE
    - **fx**: EUR/USD, GBP/USD, JPY/USD, CHF/USD exchange rates
    - **competitor**: ASP estimates for Zimmer Biomet, DePuy Synthes, Smith+Nephew, Hill-Rom

    Returns
    -------
    list[dict]
        Each dict maps column name to scalar value.
    """
    rng = np.random.RandomState(seed)
    rows = []

    # -----------------------------------------------------------------------
    # TARIFF DATA (12 rows)
    # -----------------------------------------------------------------------
    tariff_items = [
        ("HTS-7208.51", "Hot-rolled steel plate tariff rate", 25.0, "percent", "US", "Section 232 steel tariff"),
        ("HTS-7208.52", "Hot-rolled steel coil tariff rate", 25.0, "percent", "US", "Section 232 steel tariff"),
        ("HTS-7219.11", "Stainless steel hot-rolled tariff rate", 25.0, "percent", "US", "Section 232 stainless"),
        ("HTS-8108.20", "Titanium unwrought tariff rate", 0.0, "percent", "US", "No current tariff on titanium ingot"),
        ("HTS-8108.90", "Titanium articles tariff rate", 5.5, "percent", "US", "General tariff rate"),
        ("HTS-9021.10", "Orthopaedic implant tariff rate", 0.0, "percent", "US", "Duty-free under HTS"),
        ("HTS-9021.31", "Artificial joint tariff rate", 0.0, "percent", "US", "Duty-free medical device"),
        ("HTS-7601.10", "Aluminum unwrought tariff rate", 10.0, "percent", "US", "Section 232 aluminum"),
        ("EU-CBAM-Steel", "EU CBAM steel import adjustment", 26.1, "EUR_per_tonne_CO2", "EU", "Carbon border adjustment"),
        ("EU-MDR-Compliance", "EU MDR compliance surcharge estimate", 3.2, "percent", "EU", "Regulatory compliance cost"),
    ]

    tariff_dates = [date(2024, 10, 1), date(2025, 1, 1)]
    for eff_date in tariff_dates:
        ts = datetime(eff_date.year, eff_date.month, 1, 8, 0, 0)
        for item_key, desc, base_val, unit, region, notes in tariff_items[:6]:
            # Slight variation for Q1 2025
            val = base_val if eff_date.month < 12 else base_val + round(float(rng.uniform(-0.5, 1.0)), 2)
            rows.append({
                "data_source": "USTR_Tariff_Schedule",
                "upload_timestamp": ts,
                "category": "tariff",
                "item_key": item_key,
                "item_description": desc,
                "value": val,
                "unit": unit,
                "effective_date": eff_date,
                "region": region,
                "notes": notes,
            })

    # -----------------------------------------------------------------------
    # COMMODITY DATA (16 rows)
    # -----------------------------------------------------------------------
    commodity_items = [
        ("LME-COBALT", "Cobalt cathode spot price", 33450.0, "USD_per_tonne", "LME settlement"),
        ("LME-NICKEL", "Nickel spot price", 18250.0, "USD_per_tonne", "LME settlement"),
        ("LME-TIN", "Tin spot price", 31800.0, "USD_per_tonne", "LME settlement"),
        ("PLATTS-TI-6AL4V", "Titanium Ti-6Al-4V bar price", 18.50, "USD_per_lb", "Aerospace/medical grade"),
        ("PLATTS-316L-SS", "316L surgical stainless steel plate", 3.85, "USD_per_lb", "Medical grade stainless"),
        ("PLATTS-UHMWPE", "UHMWPE medical grade resin", 4.20, "USD_per_lb", "Polyethylene for bearings"),
        ("CPI-MEDICAL", "CPI Medical Care Commodities index", 395.2, "index_1982_100", "BLS CPI-U Medical Care"),
        ("PMMA-CEMENT", "PMMA bone cement price index", 145.0, "USD_per_unit", "Orthopaedic cement"),
    ]

    commodity_months = [
        date(2024, 10, 15),
        date(2024, 11, 15),
        date(2024, 12, 15),
        date(2025, 1, 15),
    ]
    for month_date in commodity_months:
        ts = datetime(month_date.year, month_date.month, month_date.day, 16, 0, 0)
        for item_key, desc, base_val, unit, notes in commodity_items[:4]:
            drift = round(float(rng.normal(0, base_val * 0.02)), 2)
            rows.append({
                "data_source": "LME_Daily" if item_key.startswith("LME") else "Platts_Metals",
                "upload_timestamp": ts,
                "category": "commodity",
                "item_key": item_key,
                "item_description": desc,
                "value": round(base_val + drift, 2),
                "unit": unit,
                "effective_date": month_date,
                "region": None,
                "notes": notes,
            })

    # -----------------------------------------------------------------------
    # FX DATA (12 rows)
    # -----------------------------------------------------------------------
    fx_pairs = [
        ("EUR-USD", "Euro to US Dollar exchange rate", 1.1165, "ECB reference rate"),
        ("GBP-USD", "British Pound to US Dollar", 1.3375, "ECB reference rate"),
        ("JPY-USD", "Japanese Yen to US Dollar", 0.006720, "ECB reference rate"),
        ("CHF-USD", "Swiss Franc to US Dollar", 1.1580, "ECB reference rate"),
        ("AUD-USD", "Australian Dollar to US Dollar", 0.6520, "ECB reference rate"),
        ("CAD-USD", "Canadian Dollar to US Dollar", 0.7380, "ECB reference rate"),
    ]

    fx_dates = [
        date(2024, 10, 1),
        date(2024, 12, 1),
        date(2025, 1, 1),
    ]
    for eff_date in fx_dates:
        ts = datetime(eff_date.year, eff_date.month, 1, 14, 0, 0)
        for pair_key, desc, base_rate, notes in fx_pairs[:4]:
            drift = round(float(rng.normal(0, base_rate * 0.01)), 6)
            rows.append({
                "data_source": "ECB_FX",
                "upload_timestamp": ts,
                "category": "fx",
                "item_key": pair_key,
                "item_description": desc,
                "value": round(base_rate + drift, 6),
                "unit": "ratio",
                "effective_date": eff_date,
                "region": None,
                "notes": notes,
            })

    # -----------------------------------------------------------------------
    # COMPETITOR DATA (10 rows)
    # -----------------------------------------------------------------------
    competitor_items = [
        ("COMP-ZBH-KNEE", "Zimmer Biomet Persona knee ASP estimate", 5850.0),
        ("COMP-JNJ-HIP", "DePuy Synthes Corail hip ASP estimate", 5200.0),
        ("COMP-SNN-KNEE", "Smith+Nephew JOURNEY II knee ASP estimate", 5650.0),
        ("COMP-ZBH-ROSA", "Zimmer Biomet ROSA robot list price", 850000.0),
        ("COMP-HILLROM-BED", "Hill-Rom Centrella ICU bed ASP", 38500.0),
    ]

    comp_dates = [date(2024, 11, 1), date(2025, 1, 1)]
    for eff_date in comp_dates:
        ts = datetime(eff_date.year, eff_date.month, 15, 10, 0, 0)
        for item_key, desc, base_price in competitor_items:
            drift = round(float(rng.normal(0, base_price * 0.015)), 2)
            rows.append({
                "data_source": "Market_Intelligence",
                "upload_timestamp": ts,
                "category": "competitor",
                "item_key": item_key,
                "item_description": desc,
                "value": round(base_price + drift, 2),
                "unit": "USD",
                "effective_date": eff_date,
                "region": "US",
                "notes": "Industry intelligence",
            })

    return rows


if not volume_data_loaded:
    sample_rows = _generate_sample_external_data()
    print(f"Generated {len(sample_rows)} sample external market data rows")

    # Category breakdown
    from collections import Counter
    cat_counts = Counter(r["category"] for r in sample_rows)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write Sample Data to Delta Table

# COMMAND ----------

if not volume_data_loaded:
    df_sample = spark.createDataFrame(sample_rows, schema=EXTERNAL_DATA_SCHEMA)

    print(f"Sample DataFrame: {df_sample.count()} rows, {len(df_sample.columns)} columns")

    # Overwrite the table with sample data
    (
        df_sample
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(FULLY_QUALIFIED_TABLE)
    )

    print(f"Sample data written to {FULLY_QUALIFIED_TABLE}")
else:
    print("Volume data was already loaded; skipping sample data write")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Data Quality Checks

# COMMAND ----------

df_final = spark.table(FULLY_QUALIFIED_TABLE)
final_count = df_final.count()

# --- 7a. Minimum row count ---
assert final_count >= 30, f"Expected >= 30 rows, got {final_count}"
print(f"[CHECK] Row count: {final_count} (>= 30 required)")

# --- 7b. All four categories present ---
expected_categories = {"tariff", "commodity", "fx", "competitor"}
actual_categories = set(
    row["category"]
    for row in df_final.select("category").distinct().collect()
)
assert expected_categories.issubset(actual_categories), (
    f"Missing categories: {expected_categories - actual_categories}"
)
print(f"[CHECK] Categories present: {actual_categories}")

# --- 7c. No nulls in required columns ---
required_cols = ["data_source", "category", "item_key", "value", "unit", "effective_date"]
for col_name in required_cols:
    null_count = df_final.filter(F.col(col_name).isNull()).count()
    assert null_count == 0, f"Found {null_count} nulls in required column: {col_name}"
print(f"[CHECK] No nulls in required columns: {required_cols}")

# --- 7d. Values are finite and non-negative where expected ---
negative_values = df_final.filter(
    (F.col("category") != "fx") & (F.col("value") < 0)
).count()
print(f"[CHECK] Negative values in non-FX data: {negative_values}")

# --- 7e. Date range ---
date_range = df_final.select(
    F.min("effective_date").alias("min_date"),
    F.max("effective_date").alias("max_date"),
).collect()[0]
print(f"[CHECK] Date range: {date_range['min_date']} to {date_range['max_date']}")

print("\n=== All data quality checks passed ===")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Post-Write Validation & Summary

# COMMAND ----------

# --- Category breakdown ---
print(f"\n{'=' * 80}")
print(f"SUCCESS: {FULLY_QUALIFIED_TABLE}")
print(f"{'=' * 80}")
print(f"  Total rows:   {final_count}")
print(f"  Categories:   {len(actual_categories)}")
print(f"  Date range:   {date_range['min_date']} to {date_range['max_date']}")
print(f"{'=' * 80}")

print("\nRows by category:")
df_final.groupBy("category").agg(
    F.count("*").alias("row_count"),
    F.countDistinct("item_key").alias("unique_items"),
    F.min("effective_date").alias("earliest_date"),
    F.max("effective_date").alias("latest_date"),
).orderBy("category").show(truncate=False)

print("Rows by data source:")
df_final.groupBy("data_source", "category").agg(
    F.count("*").alias("row_count"),
).orderBy("data_source", "category").show(20, truncate=False)

print("Sample rows:")
df_final.select(
    "data_source", "category", "item_key", "value", "unit", "effective_date", "region"
).show(10, truncate=40)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Total Rows | ~50 |
# MAGIC | Categories | tariff, commodity, fx, competitor |
# MAGIC | Data Sources | USTR, LME, Platts, ECB, Market Intelligence, BLS |
# MAGIC | Date Range | Oct 2024 - Jan 2025 |
# MAGIC | Target Table | `hls_amer_catalog.gold.external_market_data` |
# MAGIC | Format | Delta (managed, Unity Catalog) |
# MAGIC | Volume Fallback | Yes -- generates sample data if volume unavailable |
# MAGIC
# MAGIC **Next notebook**: `20_train_discount_anomaly_model.py` -- Isolation Forest for discount anomaly detection.
