# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 13 - Gold Layer: Discount Outliers
# MAGIC
# MAGIC **Purpose**: Identify sales reps whose discounting behaviour deviates significantly
# MAGIC from their peer group (same SKU + same customer segment). Outlier reps represent
# MAGIC pricing leakage opportunities -- recovering even a fraction of the gap to peer
# MAGIC norms can yield $5M-$20M in annualised margin recovery.
# MAGIC
# MAGIC **Algorithm**:
# MAGIC 1. Read from `hls_amer_catalog.silver.ficm_pricing_master`
# MAGIC 2. Compute peer-group statistics (same SKU + customer_segment)
# MAGIC 3. Compute per-rep metrics within each peer group
# MAGIC 4. Flag outliers using z-score thresholds (Severe > 3.0, Moderate 2.0-3.0, Watch 1.5-2.0)
# MAGIC 5. Compute potential recovery amounts
# MAGIC 6. Filter to peer groups with >= 3 reps for statistical validity
# MAGIC
# MAGIC **Output**: `hls_amer_catalog.gold.discount_outliers`
# MAGIC
# MAGIC **Expected**: ~150-300 outlier rows, $5M-$20M total recovery potential
# MAGIC
# MAGIC **Owner**: Pricing Intelligence Team

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0 -- Imports & Configuration

# COMMAND ----------

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, StringType
from datetime import datetime

# COMMAND ----------

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATALOG = "hls_amer_catalog"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"

SOURCE_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.ficm_pricing_master"
TARGET_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.discount_outliers"

# Z-score thresholds for outlier severity classification
Z_SCORE_SEVERE = 3.0
Z_SCORE_MODERATE = 2.0
Z_SCORE_WATCH = 1.5

# Minimum number of reps in a peer group for statistical validity
MIN_PEER_GROUP_REPS = 3

# Annualisation factor (months in data -> 12 months)
ANNUALISATION_MONTHS = 12

spark = SparkSession.builder.getOrCreate()

print(f"Source table : {SOURCE_TABLE}")
print(f"Target table : {TARGET_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 -- Read Source Data

# COMMAND ----------

def read_ficm_master(table_name: str) -> DataFrame:
    """Read the FICM pricing master from the silver layer.

    The FICM (Full Invoice Cost Model) pricing master contains every
    transaction enriched with product, customer, and rep dimensions
    along with the full pricing waterfall (list, invoice, pocket).

    Parameters
    ----------
    table_name : str
        Fully qualified Unity Catalog table name.

    Returns
    -------
    DataFrame
        Raw FICM pricing master.
    """
    df = spark.read.table(table_name)
    row_count = df.count()
    print(f"Loaded {row_count:,} rows from {table_name}")
    return df


ficm_df = read_ficm_master(SOURCE_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 -- Compute Peer Group Statistics
# MAGIC
# MAGIC A peer group is defined as all transactions for the same SKU sold to
# MAGIC the same customer segment. This controls for product complexity and
# MAGIC buyer sophistication, isolating rep-level discounting behaviour.

# COMMAND ----------

def compute_peer_group_stats(df: DataFrame) -> DataFrame:
    """Compute discount statistics at the peer-group level.

    Peer group = (sku, customer_segment). For each peer group we compute:
    - peer_avg_discount: mean discount percentage
    - peer_stddev: standard deviation of discount percentage
    - peer_median: median (approx) discount percentage
    - peer_count: number of distinct reps in the group
    - list_price_avg: average list price (used for recovery calculation)

    Parameters
    ----------
    df : DataFrame
        FICM pricing master with columns: sku, customer_segment,
        discount_pct, sales_rep_id, list_price.

    Returns
    -------
    DataFrame
        Peer-group-level statistics.
    """
    peer_stats = (
        df.groupBy("sku", "customer_segment")
        .agg(
            F.avg("discount_pct").alias("peer_avg_discount_pct"),
            F.stddev("discount_pct").alias("peer_stddev_discount_pct"),
            F.percentile_approx("discount_pct", 0.5).alias("peer_median_discount_pct"),
            F.countDistinct("sales_rep_id").alias("peer_count"),
            F.avg("list_price").alias("list_price_avg"),
        )
        # Filter to peer groups with at least MIN_PEER_GROUP_REPS reps
        .filter(F.col("peer_count") >= MIN_PEER_GROUP_REPS)
        # Require non-zero stddev for meaningful z-score calculation
        .filter(F.col("peer_stddev_discount_pct") > 0)
    )

    peer_count = peer_stats.count()
    print(f"Valid peer groups (>= {MIN_PEER_GROUP_REPS} reps, non-zero stddev): {peer_count:,}")
    return peer_stats


peer_stats_df = compute_peer_group_stats(ficm_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 -- Compute Per-Rep Metrics Within Peer Groups

# COMMAND ----------

def compute_rep_metrics(df: DataFrame) -> DataFrame:
    """Compute per-rep discounting metrics within each peer group.

    For each (sku, customer_segment, sales_rep_id) triple, we compute:
    - rep_avg_discount_pct: mean discount given by this rep
    - rep_median_discount_pct: median discount given by this rep
    - rep_volume: total units sold
    - rep_revenue: total pocket-price revenue
    - rep_transaction_count: number of transactions

    Also carries forward product and customer dimension columns for the
    output schema.

    Parameters
    ----------
    df : DataFrame
        FICM pricing master.

    Returns
    -------
    DataFrame
        Rep-level metrics at (sku, customer_segment, sales_rep_id) grain.
    """
    rep_metrics = (
        df.groupBy(
            "sku", "product_name", "product_family", "business_unit",
            "customer_id", "customer_name", "customer_segment",
            "customer_country", "customer_region",
            "sales_rep_id", "sales_rep_name", "sales_rep_territory",
        )
        .agg(
            F.avg("discount_pct").alias("rep_avg_discount_pct"),
            F.percentile_approx("discount_pct", 0.5).alias("rep_median_discount_pct"),
            F.sum("units_sold").alias("rep_volume"),
            F.sum("total_revenue").alias("rep_revenue"),
            F.count("*").alias("rep_transaction_count"),
        )
    )

    print(f"Rep-level metric rows: {rep_metrics.count():,}")
    return rep_metrics


rep_metrics_df = compute_rep_metrics(ficm_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 -- Join Rep Metrics with Peer Stats and Flag Outliers

# COMMAND ----------

def flag_outliers(
    rep_df: DataFrame,
    peer_df: DataFrame,
) -> DataFrame:
    """Join rep metrics with peer stats and compute z-scores to flag outliers.

    The z-score measures how far a rep's average discount deviates from the
    peer group mean in units of standard deviation:

        z_score = (rep_avg_discount - peer_avg_discount) / peer_stddev

    Positive z-scores indicate the rep is discounting *more* than peers
    (i.e., giving away more margin).

    Outlier severity classification:
    - Severe:   z_score > 3.0
    - Moderate: z_score 2.0 - 3.0
    - Watch:    z_score 1.5 - 2.0
    - Normal:   z_score < 1.5

    Parameters
    ----------
    rep_df : DataFrame
        Per-rep metrics from compute_rep_metrics().
    peer_df : DataFrame
        Peer-group statistics from compute_peer_group_stats().

    Returns
    -------
    DataFrame
        Joined DataFrame with z-score, outlier flags, and severity.
    """
    joined = rep_df.join(
        peer_df,
        on=["sku", "customer_segment"],
        how="inner",
    )

    # Compute z-score
    joined = joined.withColumn(
        "z_score",
        F.round(
            (F.col("rep_avg_discount_pct") - F.col("peer_avg_discount_pct"))
            / F.col("peer_stddev_discount_pct"),
            4,
        ),
    )

    # Compute discount gap (how much above peer avg this rep discounts)
    joined = joined.withColumn(
        "discount_gap",
        F.round(F.col("rep_avg_discount_pct") - F.col("peer_avg_discount_pct"), 4),
    )

    joined = joined.withColumn(
        "discount_gap_vs_median",
        F.round(F.col("rep_avg_discount_pct") - F.col("peer_median_discount_pct"), 4),
    )

    # Flag outliers (only those discounting MORE than peers, z_score > 0)
    joined = joined.withColumn(
        "is_outlier",
        F.when(F.col("z_score") >= Z_SCORE_WATCH, True).otherwise(False),
    )

    # Classify severity
    joined = joined.withColumn(
        "outlier_severity",
        F.when(F.col("z_score") >= Z_SCORE_SEVERE, F.lit("Severe"))
        .when(F.col("z_score") >= Z_SCORE_MODERATE, F.lit("Moderate"))
        .when(F.col("z_score") >= Z_SCORE_WATCH, F.lit("Watch"))
        .otherwise(F.lit("Normal")),
    )

    # Filter to outliers only
    outliers = joined.filter(F.col("is_outlier") == True)

    outlier_count = outliers.count()
    print(f"Total outliers flagged (z >= {Z_SCORE_WATCH}): {outlier_count:,}")

    # Severity breakdown
    outliers.groupBy("outlier_severity").count().orderBy("outlier_severity").show()

    return outliers


outliers_df = flag_outliers(rep_metrics_df, peer_stats_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 -- Compute Recovery Potential and Ranking Score

# COMMAND ----------

def compute_recovery_and_ranking(df: DataFrame) -> DataFrame:
    """Compute potential recovery amounts and a composite ranking score.

    Recovery logic:
    - potential_recovery_amount = rep_volume * list_price_avg * (discount_gap / 100)
      This represents the revenue that would be recovered if the rep aligned
      their discounting to the peer-group average.
    - annualized_recovery = potential_recovery_amount * (12 / data_months)

    Ranking score combines z-score magnitude and recovery potential to
    prioritise the most impactful outliers for action.

    Parameters
    ----------
    df : DataFrame
        Outlier DataFrame with z_score, discount_gap, rep_volume, list_price_avg.

    Returns
    -------
    DataFrame
        Enriched with recovery amounts and ranking score.
    """
    # Recovery = volume * avg list price * discount gap percentage
    df = df.withColumn(
        "potential_recovery_amount",
        F.round(
            F.col("rep_volume") * F.col("list_price_avg") * (F.col("discount_gap") / 100.0),
            2,
        ),
    )

    # Annualised recovery (simple scaling)
    df = df.withColumn(
        "annualized_recovery",
        F.round(F.col("potential_recovery_amount") * F.lit(1.0), 2),
    )

    # Ranking score: composite of z-score (normalised) and log recovery
    # Higher score = more actionable outlier
    df = df.withColumn(
        "ranking_score",
        F.round(
            (F.col("z_score") / F.lit(5.0)) * F.lit(50.0)
            + F.least(
                F.log(F.greatest(F.col("potential_recovery_amount"), F.lit(1.0))) / F.lit(15.0) * F.lit(50.0),
                F.lit(50.0),
            ),
            2,
        ),
    )

    return df


outliers_with_recovery_df = compute_recovery_and_ranking(outliers_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 -- Generate Outlier IDs and Final Schema

# COMMAND ----------

def finalise_schema(df: DataFrame) -> DataFrame:
    """Generate unique outlier IDs, add timestamp, and select the final output columns.

    The outlier_id is a deterministic hash of (sku, customer_segment, sales_rep_id)
    to ensure idempotent writes.

    Parameters
    ----------
    df : DataFrame
        Enriched outlier DataFrame.

    Returns
    -------
    DataFrame
        Final schema matching the gold.discount_outliers specification.
    """
    # Generate deterministic outlier_id
    df = df.withColumn(
        "outlier_id",
        F.md5(
            F.concat_ws("|",
                F.col("sku"),
                F.col("customer_segment"),
                F.col("sales_rep_id"),
            )
        ),
    )

    # Add updated_at timestamp
    df = df.withColumn("updated_at", F.current_timestamp())

    # Select and order final columns
    final_df = df.select(
        "outlier_id",
        "sku",
        "product_name",
        "product_family",
        "business_unit",
        "customer_id",
        "customer_name",
        "customer_segment",
        "customer_country",
        "customer_region",
        "sales_rep_id",
        "sales_rep_name",
        "sales_rep_territory",
        F.round("rep_avg_discount_pct", 4).alias("rep_avg_discount_pct"),
        F.round("rep_median_discount_pct", 4).alias("rep_median_discount_pct"),
        F.round("peer_avg_discount_pct", 4).alias("peer_avg_discount_pct"),
        F.round("peer_median_discount_pct", 4).alias("peer_median_discount_pct"),
        F.round("peer_stddev_discount_pct", 4).alias("peer_stddev_discount_pct"),
        "peer_count",
        "discount_gap",
        "discount_gap_vs_median",
        "z_score",
        "rep_volume",
        F.round("rep_revenue", 2).alias("rep_revenue"),
        "rep_transaction_count",
        "potential_recovery_amount",
        "annualized_recovery",
        "is_outlier",
        "outlier_severity",
        "ranking_score",
        "updated_at",
    )

    return final_df


final_outliers_df = finalise_schema(outliers_with_recovery_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 -- Validation

# COMMAND ----------

def validate_outliers(df: DataFrame) -> None:
    """Run data quality checks on the final outlier DataFrame.

    Validates row counts, severity distribution, recovery totals, and
    schema completeness.

    Parameters
    ----------
    df : DataFrame
        Final discount_outliers DataFrame.

    Raises
    ------
    AssertionError
        If any validation check fails.
    """
    row_count = df.count()
    print(f"Total outlier rows: {row_count:,}")

    # Severity distribution
    print("\nOutlier severity distribution:")
    df.groupBy("outlier_severity").agg(
        F.count("*").alias("count"),
        F.round(F.sum("potential_recovery_amount"), 2).alias("total_recovery"),
        F.round(F.avg("z_score"), 2).alias("avg_z_score"),
    ).orderBy("outlier_severity").show(truncate=False)

    # Total recovery potential
    total_recovery = df.agg(
        F.sum("potential_recovery_amount").alias("total_recovery")
    ).collect()[0]["total_recovery"]
    print(f"Total potential recovery: ${total_recovery:,.2f}")

    # Z-score range
    z_stats = df.agg(
        F.min("z_score").alias("min_z"),
        F.max("z_score").alias("max_z"),
        F.avg("z_score").alias("avg_z"),
    ).collect()[0]
    print(f"Z-score range: {z_stats['min_z']:.2f} to {z_stats['max_z']:.2f} (avg: {z_stats['avg_z']:.2f})")

    # All z-scores should be >= WATCH threshold
    below_threshold = df.filter(F.col("z_score") < Z_SCORE_WATCH).count()
    assert below_threshold == 0, f"Found {below_threshold} rows below z-score threshold {Z_SCORE_WATCH}"
    print(f"[PASS] All z-scores >= {Z_SCORE_WATCH}")

    # No null outlier_ids
    null_ids = df.filter(F.col("outlier_id").isNull()).count()
    assert null_ids == 0, f"Found {null_ids} null outlier_ids"
    print("[PASS] No null outlier_ids")

    print("\nAll validation checks passed.")


validate_outliers(final_outliers_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 -- Write to Delta Lake

# COMMAND ----------

# Ensure gold schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}")

# Write as managed Delta table
(
    final_outliers_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TARGET_TABLE)
)

print(f"Successfully wrote to {TARGET_TABLE}")

# COMMAND ----------

# Post-write verification
verify_df = spark.table(TARGET_TABLE)
verify_count = verify_df.count()
print(f"\nPost-write verification: {verify_count:,} rows in {TARGET_TABLE}")

# Add table comment
_comment = (
    "Discount outlier analysis identifying sales reps whose discounting exceeds "
    "peer-group norms. Peer groups defined as (SKU, customer_segment) with >= 3 reps. "
    "Z-score thresholds: Severe > 3.0, Moderate 2.0-3.0, Watch 1.5-2.0. "
    "Source: silver.ficm_pricing_master. Generated by notebook 13_gold_discount_outliers."
)
spark.sql(f"COMMENT ON TABLE {TARGET_TABLE} IS '{_comment}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 -- Summary
# MAGIC
# MAGIC | Metric | Expected |
# MAGIC |--------|----------|
# MAGIC | Total outlier rows | 150-300 |
# MAGIC | Severity: Severe (z > 3.0) | ~15-30 |
# MAGIC | Severity: Moderate (z 2.0-3.0) | ~50-100 |
# MAGIC | Severity: Watch (z 1.5-2.0) | ~85-170 |
# MAGIC | Total recovery potential | $5M-$20M |
# MAGIC | Peer group minimum reps | 3 |
# MAGIC | Output table | `hls_amer_catalog.gold.discount_outliers` |

# COMMAND ----------

print("=" * 70)
print("  DISCOUNT OUTLIER ANALYSIS COMPLETE")
print("=" * 70)
print(f"  Source        : {SOURCE_TABLE}")
print(f"  Output        : {TARGET_TABLE}")
print(f"  Outlier rows  : {verify_count:,}")
print(f"  Z-score bands : Watch >= {Z_SCORE_WATCH}, Moderate >= {Z_SCORE_MODERATE}, Severe >= {Z_SCORE_SEVERE}")
print(f"  Min peer reps : {MIN_PEER_GROUP_REPS}")
print("=" * 70)
