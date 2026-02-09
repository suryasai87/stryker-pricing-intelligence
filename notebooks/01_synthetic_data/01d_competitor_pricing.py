# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01d - Competitor Pricing & Market Intelligence
# MAGIC
# MAGIC **Purpose**: Generate synthetic competitor ASP (Average Selling Price) data for Stryker's
# MAGIC primary competitors across all major product categories. This notebook models realistic
# MAGIC competitive dynamics including robotic surgery adoption curves, ASP trends, market share
# MAGIC shifts, and innovation scoring.
# MAGIC
# MAGIC **Competitors Modeled**:
# MAGIC - Zimmer Biomet (Hip/Knee reconstruction, S.E.T.)
# MAGIC - Medtronic (Spine, Neurovascular, Power Tools)
# MAGIC - J&J DePuy Synthes (Trauma, Hip/Knee, Spine)
# MAGIC - Globus Medical (Spine, enabling technologies)
# MAGIC - Smith+Nephew (Trauma, Hip/Knee, Sports Medicine)
# MAGIC
# MAGIC **Output**: `hls_amer_catalog.bronze.competitor_pricing` (Delta table)
# MAGIC
# MAGIC **Schema**:
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | competitor | string | Competitor company name |
# MAGIC | product_category | string | Major product category |
# MAGIC | sub_category | string | Product sub-category |
# MAGIC | avg_asp | double | Average selling price in USD |
# MAGIC | asp_trend_pct | double | Quarter-over-quarter ASP change (%) |
# MAGIC | market_share | double | Estimated market share (0-1) |
# MAGIC | robot_installed_base | int | Cumulative robotic systems installed |
# MAGIC | innovation_score | double | R&D and innovation index (1-10) |
# MAGIC | quarter | int | Fiscal quarter (1-4) |
# MAGIC | year | int | Fiscal year (2023-2025) |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration & Imports

# COMMAND ----------

import random
import math
from itertools import product as iter_product
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)
import pyspark.sql.functions as F

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Target catalog / schema / table
# ---------------------------------------------------------------------------
CATALOG = "hls_amer_catalog"
SCHEMA = "bronze"
TABLE_NAME = "competitor_pricing"
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

# ---------------------------------------------------------------------------
# Temporal range: 12 quarters (Q1 2023 - Q4 2025)
# ---------------------------------------------------------------------------
QUARTERS = [(y, q) for y in range(2023, 2026) for q in range(1, 5)]

print(f"Target table : {FULL_TABLE}")
print(f"Quarters     : {len(QUARTERS)} ({QUARTERS[0]} -> {QUARTERS[-1]})")
print(f"Random seed  : {RANDOM_SEED}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Competitor & Category Definitions
# MAGIC
# MAGIC Each competitor entry encodes:
# MAGIC - **base_asp**: Starting ASP at Q1 2023 (USD)
# MAGIC - **asp_drift**: Annualised ASP increase (fractional, e.g. 0.03 = 3%)
# MAGIC - **base_share**: Market share at Q1 2023
# MAGIC - **share_drift**: Annual market-share delta (positive = gaining share)
# MAGIC - **base_robots**: Installed robotic base at Q1 2023
# MAGIC - **robot_qtr_add**: Net new robots per quarter (can accelerate)
# MAGIC - **innovation**: Base innovation score (1-10)
# MAGIC - **innov_drift**: Annual innovation score change

# COMMAND ----------

def _build_competitor_catalog():
    """
    Return a nested dictionary defining every (competitor, category, sub_category)
    combination with calibrated base metrics.

    Competitive dynamics encoded:
    -----------------------------------------------------------------
    Hip / Knee Reconstruction
      Stryker leads via Mako robotic premium (~8-12% above competitors).
      Zimmer Biomet is close #2 (ROSA robot), J&J DePuy #3.
      Smith+Nephew participates but smaller share.
    -----------------------------------------------------------------
    Trauma & Extremities
      J&J DePuy Synthes is the market leader. Stryker is #2 (within 3-5%).
      Smith+Nephew is also competitive.
    -----------------------------------------------------------------
    Spine
      Stryker divested spine. Medtronic and Globus Medical dominate.
      J&J DePuy Synthes also has presence.
    -----------------------------------------------------------------
    Beds & Stretchers (MedSurg)
      Stryker is the overall leader. Hill-Rom (Baxter) is closest competitor.
      Modeled under 'Zimmer Biomet' placeholder for MedSurg peers.
    -----------------------------------------------------------------
    Neurovascular
      Medtronic leads. Stryker is a strong #2.
    -----------------------------------------------------------------
    Power Tools (MedSurg)
      Stryker dominant (~40% share). Others trail significantly.
    -----------------------------------------------------------------
    Endoscopy
      Stryker competitive. Olympus / Karl Storz strong.
      Modeled under Smith+Nephew (sports-med/endo overlap).
    -----------------------------------------------------------------
    """

    catalog = {}

    # ---------------------------------------------------------------
    # Helper to add an entry
    # ---------------------------------------------------------------
    def _add(competitor, category, sub_category, base_asp, asp_drift,
             base_share, share_drift, base_robots, robot_qtr_add,
             innovation, innov_drift):
        key = (competitor, category, sub_category)
        catalog[key] = {
            "base_asp": base_asp,
            "asp_drift": asp_drift,
            "base_share": base_share,
            "share_drift": share_drift,
            "base_robots": base_robots,
            "robot_qtr_add": robot_qtr_add,
            "innovation": innovation,
            "innov_drift": innov_drift,
        }

    # =================================================================
    # HIP RECONSTRUCTION
    # =================================================================
    # Stryker Mako ASP reference: ~$8,500 (implant) -> competitors lower
    _add("Zimmer Biomet", "Hip Reconstruction", "Primary Hip",
         7600, 0.030, 0.24, 0.002, 320, 18, 7.8, 0.15)
    _add("Zimmer Biomet", "Hip Reconstruction", "Revision Hip",
         9200, 0.025, 0.22, 0.001, 320, 18, 7.5, 0.10)

    _add("J&J DePuy Synthes", "Hip Reconstruction", "Primary Hip",
         7400, 0.028, 0.20, -0.003, 150, 10, 7.2, 0.08)
    _add("J&J DePuy Synthes", "Hip Reconstruction", "Revision Hip",
         8900, 0.022, 0.18, -0.002, 150, 10, 7.0, 0.05)

    _add("Smith+Nephew", "Hip Reconstruction", "Primary Hip",
         7100, 0.025, 0.08, -0.001, 60, 5, 6.5, 0.10)
    _add("Smith+Nephew", "Hip Reconstruction", "Revision Hip",
         8500, 0.020, 0.07, -0.001, 60, 5, 6.3, 0.08)

    # =================================================================
    # KNEE RECONSTRUCTION
    # =================================================================
    # Stryker Mako TKA reference: ~$9,200 -> competitors ~8-12% below
    _add("Zimmer Biomet", "Knee Reconstruction", "Primary Knee",
         8200, 0.032, 0.25, 0.003, 420, 22, 8.0, 0.18)
    _add("Zimmer Biomet", "Knee Reconstruction", "Revision Knee",
         11500, 0.028, 0.23, 0.002, 420, 22, 7.6, 0.12)
    _add("Zimmer Biomet", "Knee Reconstruction", "Partial Knee",
         6800, 0.030, 0.18, 0.001, 420, 22, 7.4, 0.10)

    _add("J&J DePuy Synthes", "Knee Reconstruction", "Primary Knee",
         8000, 0.025, 0.18, -0.004, 180, 12, 7.3, 0.10)
    _add("J&J DePuy Synthes", "Knee Reconstruction", "Revision Knee",
         11000, 0.022, 0.16, -0.003, 180, 12, 7.0, 0.06)
    _add("J&J DePuy Synthes", "Knee Reconstruction", "Partial Knee",
         6500, 0.020, 0.12, -0.002, 180, 12, 6.8, 0.05)

    _add("Smith+Nephew", "Knee Reconstruction", "Primary Knee",
         7800, 0.022, 0.09, 0.000, 80, 6, 6.8, 0.12)
    _add("Smith+Nephew", "Knee Reconstruction", "Revision Knee",
         10500, 0.018, 0.08, -0.001, 80, 6, 6.5, 0.08)
    _add("Smith+Nephew", "Knee Reconstruction", "Partial Knee",
         6200, 0.020, 0.06, 0.000, 80, 6, 6.3, 0.06)

    # =================================================================
    # TRAUMA & EXTREMITIES
    # =================================================================
    # J&J DePuy leads; Stryker #2 (~3-5% below on ASP)
    _add("J&J DePuy Synthes", "Trauma", "Internal Fixation",
         3800, 0.028, 0.28, 0.002, 0, 0, 8.2, 0.12)
    _add("J&J DePuy Synthes", "Trauma", "External Fixation",
         2900, 0.025, 0.26, 0.001, 0, 0, 7.8, 0.08)
    _add("J&J DePuy Synthes", "Trauma", "Intramedullary Nails",
         4200, 0.030, 0.30, 0.003, 0, 0, 8.5, 0.15)

    _add("Smith+Nephew", "Trauma", "Internal Fixation",
         3500, 0.022, 0.14, 0.001, 0, 0, 7.0, 0.10)
    _add("Smith+Nephew", "Trauma", "External Fixation",
         2700, 0.020, 0.12, 0.000, 0, 0, 6.8, 0.08)
    _add("Smith+Nephew", "Trauma", "Intramedullary Nails",
         3900, 0.025, 0.12, 0.001, 0, 0, 7.2, 0.10)

    _add("Zimmer Biomet", "Trauma", "Internal Fixation",
         3400, 0.024, 0.10, -0.001, 0, 0, 6.5, 0.06)
    _add("Zimmer Biomet", "Trauma", "External Fixation",
         2600, 0.020, 0.08, -0.001, 0, 0, 6.2, 0.04)
    _add("Zimmer Biomet", "Trauma", "Intramedullary Nails",
         3700, 0.022, 0.09, 0.000, 0, 0, 6.4, 0.05)

    # =================================================================
    # SPINE (Stryker divested -- Globus & Medtronic dominate)
    # =================================================================
    _add("Medtronic", "Spine", "Spinal Fusion",
         12500, 0.035, 0.30, -0.005, 350, 20, 8.5, 0.10)
    _add("Medtronic", "Spine", "Spinal Implants",
         8500, 0.030, 0.28, -0.004, 350, 20, 8.2, 0.08)
    _add("Medtronic", "Spine", "Biologics",
         6200, 0.025, 0.25, -0.003, 0, 0, 7.8, 0.06)

    _add("Globus Medical", "Spine", "Spinal Fusion",
         11800, 0.040, 0.18, 0.012, 280, 25, 9.0, 0.25)
    _add("Globus Medical", "Spine", "Spinal Implants",
         8000, 0.038, 0.16, 0.010, 280, 25, 8.8, 0.22)
    _add("Globus Medical", "Spine", "Biologics",
         5800, 0.030, 0.12, 0.005, 0, 0, 8.0, 0.15)
    _add("Globus Medical", "Spine", "Enabling Technologies",
         45000, 0.035, 0.22, 0.015, 280, 25, 9.2, 0.30)

    _add("J&J DePuy Synthes", "Spine", "Spinal Fusion",
         11200, 0.025, 0.15, -0.005, 80, 5, 7.5, 0.05)
    _add("J&J DePuy Synthes", "Spine", "Spinal Implants",
         7600, 0.022, 0.14, -0.004, 80, 5, 7.2, 0.04)
    _add("J&J DePuy Synthes", "Spine", "Biologics",
         5500, 0.020, 0.12, -0.003, 0, 0, 7.0, 0.03)

    # =================================================================
    # BEDS & STRETCHERS (MedSurg Equipment)
    # =================================================================
    # Stryker leads; Hill-Rom (Baxter) is closest -- modeled as separate entries
    _add("Zimmer Biomet", "Beds & Stretchers", "Hospital Beds",
         18000, 0.020, 0.12, -0.002, 0, 0, 6.0, 0.05)
    _add("Zimmer Biomet", "Beds & Stretchers", "Stretchers & Transport",
         12000, 0.018, 0.10, -0.002, 0, 0, 5.8, 0.04)

    _add("Medtronic", "Beds & Stretchers", "Hospital Beds",
         17500, 0.018, 0.08, -0.001, 0, 0, 5.5, 0.03)
    _add("Medtronic", "Beds & Stretchers", "Stretchers & Transport",
         11500, 0.015, 0.06, -0.001, 0, 0, 5.3, 0.02)

    # =================================================================
    # NEUROVASCULAR
    # =================================================================
    # Medtronic leads; Stryker strong #2
    _add("Medtronic", "Neurovascular", "Flow Diverters",
         8500, 0.035, 0.35, -0.008, 0, 0, 8.8, 0.12)
    _add("Medtronic", "Neurovascular", "Coils & Stents",
         3200, 0.030, 0.32, -0.006, 0, 0, 8.5, 0.10)
    _add("Medtronic", "Neurovascular", "Thrombectomy Devices",
         6800, 0.038, 0.38, -0.005, 0, 0, 9.0, 0.15)

    _add("J&J DePuy Synthes", "Neurovascular", "Flow Diverters",
         7800, 0.025, 0.08, 0.002, 0, 0, 6.5, 0.08)
    _add("J&J DePuy Synthes", "Neurovascular", "Coils & Stents",
         2900, 0.020, 0.10, 0.001, 0, 0, 6.2, 0.05)

    # =================================================================
    # POWER TOOLS (MedSurg)
    # =================================================================
    # Stryker dominant (~40% share); others trail
    _add("Medtronic", "Power Tools", "Surgical Drills",
         4500, 0.020, 0.15, -0.005, 0, 0, 6.5, 0.05)
    _add("Medtronic", "Power Tools", "Saws & Blades",
         3800, 0.018, 0.14, -0.004, 0, 0, 6.2, 0.04)
    _add("Medtronic", "Power Tools", "Battery Systems",
         2200, 0.022, 0.12, -0.003, 0, 0, 6.0, 0.03)

    _add("Zimmer Biomet", "Power Tools", "Surgical Drills",
         4200, 0.018, 0.12, -0.003, 0, 0, 6.0, 0.04)
    _add("Zimmer Biomet", "Power Tools", "Saws & Blades",
         3500, 0.015, 0.10, -0.002, 0, 0, 5.8, 0.03)
    _add("Zimmer Biomet", "Power Tools", "Battery Systems",
         2000, 0.020, 0.08, -0.002, 0, 0, 5.5, 0.02)

    _add("J&J DePuy Synthes", "Power Tools", "Surgical Drills",
         4000, 0.015, 0.10, -0.002, 0, 0, 5.8, 0.03)
    _add("J&J DePuy Synthes", "Power Tools", "Saws & Blades",
         3400, 0.012, 0.09, -0.002, 0, 0, 5.5, 0.02)

    # =================================================================
    # ENDOSCOPY / VISUALIZATION
    # =================================================================
    # Stryker competitive; Olympus / Karl Storz strong (modeled via S+N overlap)
    _add("Smith+Nephew", "Endoscopy", "Visualization Systems",
         32000, 0.025, 0.18, 0.003, 0, 0, 7.8, 0.15)
    _add("Smith+Nephew", "Endoscopy", "Arthroscopic Instruments",
         1800, 0.022, 0.16, 0.002, 0, 0, 7.5, 0.12)
    _add("Smith+Nephew", "Endoscopy", "Resection & Ablation",
         2500, 0.028, 0.15, 0.003, 0, 0, 7.6, 0.14)

    _add("Medtronic", "Endoscopy", "Visualization Systems",
         30000, 0.020, 0.12, 0.001, 0, 0, 7.0, 0.08)
    _add("Medtronic", "Endoscopy", "Arthroscopic Instruments",
         1600, 0.018, 0.10, 0.000, 0, 0, 6.8, 0.06)

    return catalog


COMPETITOR_CATALOG = _build_competitor_catalog()
print(f"Competitor-category combinations: {len(COMPETITOR_CATALOG)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Quarterly Data Generation Engine
# MAGIC
# MAGIC The generator applies the following temporal dynamics per quarter:
# MAGIC
# MAGIC 1. **ASP Growth** -- Compound quarterly increase derived from annual drift,
# MAGIC    with small random jitter (+/- 0.3%) to simulate real-world pricing variability.
# MAGIC 2. **Market Share Evolution** -- Linear drift with bounded noise. Shares are
# MAGIC    clamped to [0.01, 0.60] to avoid degenerate values.
# MAGIC 3. **Robot Installed Base** -- Additive growth each quarter with a 5% quarterly
# MAGIC    acceleration factor to model adoption curve inflection (especially Mako vs ROSA).
# MAGIC 4. **Innovation Score** -- Gradual drift reflecting R&D pipeline maturity.
# MAGIC    Clamped to [1.0, 10.0].
# MAGIC 5. **ASP Trend %** -- Computed as the realised quarter-over-quarter ASP change,
# MAGIC    capturing both the structural drift and stochastic component.

# COMMAND ----------

def generate_competitor_rows(catalog, quarters, seed=42):
    """
    Generate a list of row-tuples for every (competitor, category, sub_category)
    across all quarters.

    Parameters
    ----------
    catalog : dict
        Output of ``_build_competitor_catalog()`` keyed by
        (competitor, category, sub_category) with metric dictionaries.
    quarters : list[tuple[int, int]]
        Ordered list of (year, quarter) tuples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[tuple]
        Each tuple contains:
        (competitor, product_category, sub_category, avg_asp, asp_trend_pct,
         market_share, robot_installed_base, innovation_score, quarter, year)
    """
    rng = random.Random(seed)
    rows = []

    for (competitor, category, sub_cat), params in catalog.items():
        # Unpack base parameters
        asp = float(params["base_asp"])
        annual_asp_drift = params["asp_drift"]
        quarterly_asp_drift = annual_asp_drift / 4.0

        share = params["base_share"]
        annual_share_drift = params["share_drift"]
        quarterly_share_drift = annual_share_drift / 4.0

        robots = params["base_robots"]
        robot_qtr_add = params["robot_qtr_add"]

        innov = params["innovation"]
        annual_innov_drift = params["innov_drift"]
        quarterly_innov_drift = annual_innov_drift / 4.0

        prev_asp = None

        for idx, (year, quarter) in enumerate(quarters):
            # --- ASP with jitter ---
            asp_jitter = rng.gauss(0, 0.003)  # +/- ~0.3% noise
            asp_multiplier = 1.0 + quarterly_asp_drift + asp_jitter
            if idx > 0:
                asp = asp * asp_multiplier
            current_asp = round(asp, 2)

            # --- ASP trend (QoQ %) ---
            if prev_asp is not None and prev_asp > 0:
                asp_trend = round(((current_asp - prev_asp) / prev_asp) * 100, 3)
            else:
                asp_trend = 0.0
            prev_asp = current_asp

            # --- Market share with noise ---
            if idx > 0:
                share_noise = rng.gauss(0, 0.002)
                share = share + quarterly_share_drift + share_noise
            share = max(0.01, min(0.60, share))
            current_share = round(share, 4)

            # --- Robot installed base (accelerating growth) ---
            if robot_qtr_add > 0 and idx > 0:
                # 5% quarterly acceleration in adoption rate
                accel_factor = 1.0 + 0.05 * (idx / len(quarters))
                new_robots = int(robot_qtr_add * accel_factor)
                # Small random variation (+/- 15%)
                new_robots = max(1, int(new_robots * (1 + rng.gauss(0, 0.15))))
                robots += new_robots
            current_robots = int(robots)

            # --- Innovation score with drift ---
            if idx > 0:
                innov_noise = rng.gauss(0, 0.05)
                innov = innov + quarterly_innov_drift + innov_noise
            innov = max(1.0, min(10.0, innov))
            current_innov = round(innov, 2)

            rows.append((
                competitor,
                category,
                sub_cat,
                current_asp,
                asp_trend,
                current_share,
                current_robots,
                current_innov,
                quarter,
                year,
            ))

    return rows


all_rows = generate_competitor_rows(COMPETITOR_CATALOG, QUARTERS, seed=RANDOM_SEED)
print(f"Total rows generated: {len(all_rows)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build PySpark DataFrame

# COMMAND ----------

schema = StructType([
    StructField("competitor", StringType(), False),
    StructField("product_category", StringType(), False),
    StructField("sub_category", StringType(), False),
    StructField("avg_asp", DoubleType(), False),
    StructField("asp_trend_pct", DoubleType(), False),
    StructField("market_share", DoubleType(), False),
    StructField("robot_installed_base", IntegerType(), False),
    StructField("innovation_score", DoubleType(), False),
    StructField("quarter", IntegerType(), False),
    StructField("year", IntegerType(), False),
])

df_competitor_pricing = spark.createDataFrame(all_rows, schema=schema)

print(f"DataFrame row count : {df_competitor_pricing.count()}")
print(f"DataFrame columns   : {df_competitor_pricing.columns}")
df_competitor_pricing.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data Quality Validation
# MAGIC
# MAGIC Run assertion-style checks to catch generation defects before writing to Delta.

# COMMAND ----------

from pyspark.sql import functions as F

# -----------------------------------------------------------------------
# 5a. Null checks -- every column must be fully populated
# -----------------------------------------------------------------------
null_counts = df_competitor_pricing.select(
    *[F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
      for c in df_competitor_pricing.columns]
)
null_row = null_counts.collect()[0]
for col_name in df_competitor_pricing.columns:
    assert null_row[col_name] == 0, f"Null values detected in column: {col_name}"
print("[PASS] No null values in any column.")

# -----------------------------------------------------------------------
# 5b. Value range checks
# -----------------------------------------------------------------------
stats = df_competitor_pricing.agg(
    F.min("avg_asp").alias("min_asp"),
    F.max("avg_asp").alias("max_asp"),
    F.min("market_share").alias("min_share"),
    F.max("market_share").alias("max_share"),
    F.min("innovation_score").alias("min_innov"),
    F.max("innovation_score").alias("max_innov"),
    F.min("robot_installed_base").alias("min_robots"),
    F.min("quarter").alias("min_q"),
    F.max("quarter").alias("max_q"),
    F.min("year").alias("min_y"),
    F.max("year").alias("max_y"),
).collect()[0]

assert stats["min_asp"] > 0, "ASP must be positive"
assert 0 < stats["min_share"] <= stats["max_share"] <= 1.0, "Market share out of [0,1]"
assert 1.0 <= stats["min_innov"] and stats["max_innov"] <= 10.0, "Innovation score out of [1,10]"
assert stats["min_robots"] >= 0, "Robot installed base cannot be negative"
assert stats["min_q"] == 1 and stats["max_q"] == 4, "Quarters must span 1-4"
assert stats["min_y"] == 2023 and stats["max_y"] == 2025, "Years must span 2023-2025"
print("[PASS] All value ranges within expected bounds.")

# -----------------------------------------------------------------------
# 5c. Coverage checks -- verify all 5 competitors present
# -----------------------------------------------------------------------
competitors = sorted(
    [r["competitor"] for r in df_competitor_pricing.select("competitor").distinct().collect()]
)
expected_competitors = sorted([
    "Globus Medical",
    "J&J DePuy Synthes",
    "Medtronic",
    "Smith+Nephew",
    "Zimmer Biomet",
])
assert competitors == expected_competitors, (
    f"Competitor mismatch. Got: {competitors}, Expected: {expected_competitors}"
)
print(f"[PASS] All {len(expected_competitors)} competitors present.")

# -----------------------------------------------------------------------
# 5d. Row count -- each combo should have exactly 12 quarters
# -----------------------------------------------------------------------
combo_counts = (
    df_competitor_pricing
    .groupBy("competitor", "product_category", "sub_category")
    .count()
)
bad_combos = combo_counts.filter(F.col("count") != 12)
assert bad_combos.count() == 0, (
    f"Found combos without exactly 12 quarters:\n{bad_combos.show(truncate=False)}"
)
print("[PASS] Every (competitor, category, sub_category) has exactly 12 quarters.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Exploratory Summary Statistics

# COMMAND ----------

# -----------------------------------------------------------------------
# 6a. Average ASP by competitor and category
# -----------------------------------------------------------------------
display(
    df_competitor_pricing
    .groupBy("competitor", "product_category")
    .agg(
        F.round(F.avg("avg_asp"), 2).alias("mean_asp"),
        F.round(F.avg("asp_trend_pct"), 3).alias("mean_asp_trend_pct"),
        F.round(F.avg("market_share"), 4).alias("mean_share"),
        F.round(F.avg("innovation_score"), 2).alias("mean_innovation"),
        F.max("robot_installed_base").alias("max_robot_base"),
    )
    .orderBy("product_category", "competitor")
)

# COMMAND ----------

# -----------------------------------------------------------------------
# 6b. Robot installed base growth trajectory (for robotics-enabled categories)
# -----------------------------------------------------------------------
display(
    df_competitor_pricing
    .filter(F.col("robot_installed_base") > 0)
    .groupBy("competitor", "product_category", "year", "quarter")
    .agg(F.max("robot_installed_base").alias("robots"))
    .orderBy("competitor", "product_category", "year", "quarter")
)

# COMMAND ----------

# -----------------------------------------------------------------------
# 6c. ASP trend distribution
# -----------------------------------------------------------------------
display(
    df_competitor_pricing
    .filter(F.col("asp_trend_pct") != 0.0)
    .groupBy("competitor")
    .agg(
        F.round(F.avg("asp_trend_pct"), 3).alias("avg_asp_trend_pct"),
        F.round(F.min("asp_trend_pct"), 3).alias("min_asp_trend_pct"),
        F.round(F.max("asp_trend_pct"), 3).alias("max_asp_trend_pct"),
        F.round(F.stddev("asp_trend_pct"), 3).alias("std_asp_trend_pct"),
    )
    .orderBy("competitor")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Write to Delta Table

# COMMAND ----------

# -----------------------------------------------------------------------
# Ensure catalog and schema exist
# -----------------------------------------------------------------------
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# -----------------------------------------------------------------------
# Write as managed Delta table (overwrite for idempotency)
# -----------------------------------------------------------------------
(
    df_competitor_pricing
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FULL_TABLE)
)

print(f"Successfully wrote {df_competitor_pricing.count()} rows to {FULL_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Post-Write Verification

# COMMAND ----------

# -----------------------------------------------------------------------
# Read back and verify round-trip integrity
# -----------------------------------------------------------------------
df_verify = spark.table(FULL_TABLE)
write_count = df_competitor_pricing.count()
read_count = df_verify.count()

assert read_count == write_count, (
    f"Row count mismatch: wrote {write_count}, read {read_count}"
)
print(f"[PASS] Round-trip verification: {read_count} rows in {FULL_TABLE}")

# Show sample rows
display(df_verify.orderBy("competitor", "product_category", "year", "quarter").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Table Metadata & Comments

# COMMAND ----------

spark.sql(f"""
    COMMENT ON TABLE {FULL_TABLE} IS
    'Synthetic competitor ASP and market intelligence data for Stryker pricing analysis. '
    'Covers Zimmer Biomet, Medtronic, J&J DePuy Synthes, Globus Medical, and Smith+Nephew '
    'across Hip/Knee Reconstruction, Trauma, Spine, Beds & Stretchers, Neurovascular, '
    'Power Tools, and Endoscopy. Quarterly data from Q1 2023 through Q4 2025 (12 quarters). '
    'Generated with random seed 42 for reproducibility. '
    'Source notebook: 01d_competitor_pricing'
""")

spark.sql(f"ALTER TABLE {FULL_TABLE} SET TBLPROPERTIES ('quality' = 'bronze', 'source' = 'synthetic', 'seed' = '42', 'generator' = '01d_competitor_pricing')")

print(f"Table metadata applied to {FULL_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Notebook complete.** Output table `hls_amer_catalog.bronze.competitor_pricing` is ready
# MAGIC for downstream joins with Stryker ASP data in the silver-layer pricing intelligence pipeline.
