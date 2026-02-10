# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 15 - Gold Layer: Uplift Simulation
# MAGIC
# MAGIC **Purpose**: Score every (SKU, customer, rep) combination for price-uplift
# MAGIC potential, rank them by a composite uplift score, compute the suggested
# MAGIC price increase, and determine how many actions are needed to achieve the
# MAGIC target portfolio uplift (default 1.0%).
# MAGIC
# MAGIC **Algorithm**:
# MAGIC 1. Join FICM master with elasticity and discount outlier data
# MAGIC 2. Compute uplift_score = 0.30 * inelasticity + 0.25 * discount_gap +
# MAGIC    0.20 * revenue_weight + 0.15 * margin_headroom - 0.10 * competitive_risk
# MAGIC 3. Suggested increase = MIN(safe_increase_3pct, discount_gap * 0.5, 5.0)
# MAGIC 4. Expected impacts using elasticity coefficients
# MAGIC 5. Cumulative uplift: sort by score, running sum of portfolio contribution
# MAGIC 6. Mark rows where cumulative uplift >= target
# MAGIC 7. Generate rationale text for each row
# MAGIC
# MAGIC **Output**: `hls_amer_catalog.gold.uplift_simulation`
# MAGIC
# MAGIC **Expected**: 2000-5000 rows, target achievable within top 80-200 actions
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

# COMMAND ----------

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATALOG = "hls_amer_catalog"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"

SOURCE_FICM = f"{CATALOG}.{SILVER_SCHEMA}.ficm_pricing_master"
SOURCE_ELASTICITY = f"{CATALOG}.{GOLD_SCHEMA}.price_elasticity"
SOURCE_OUTLIERS = f"{CATALOG}.{GOLD_SCHEMA}.discount_outliers"

TARGET_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.uplift_simulation"

# Widget for parameterisation (can be overridden at runtime)
try:
    dbutils.widgets.text("target_uplift_pct", "1.0")
    TARGET_UPLIFT_PCT = float(dbutils.widgets.get("target_uplift_pct"))
except Exception:
    TARGET_UPLIFT_PCT = 1.0

# Uplift score weights
W_INELASTICITY = 0.30
W_DISCOUNT_GAP = 0.25
W_REVENUE = 0.20
W_MARGIN = 0.15
W_COMPETITIVE_RISK = 0.10

# Maximum suggested price increase (cap)
MAX_SUGGESTED_INCREASE_PCT = 5.0

spark = SparkSession.builder.getOrCreate()

print(f"Source FICM      : {SOURCE_FICM}")
print(f"Source Elasticity : {SOURCE_ELASTICITY}")
print(f"Source Outliers   : {SOURCE_OUTLIERS}")
print(f"Target table     : {TARGET_TABLE}")
print(f"Target uplift    : {TARGET_UPLIFT_PCT}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 -- Read Source Data

# COMMAND ----------

def read_source_tables() -> tuple:
    """Read all source tables needed for the uplift simulation.

    Returns
    -------
    tuple of (DataFrame, DataFrame, DataFrame)
        (ficm_agg, elasticity, outliers) DataFrames.
    """
    # Read and aggregate FICM to (sku, customer_id, sales_rep_id) grain
    ficm_raw = spark.read.table(SOURCE_FICM)
    print(f"FICM raw rows: {ficm_raw.count():,}")

    ficm_agg = (
        ficm_raw.groupBy(
            "sku", "product_id", "product_name", "product_family", "business_unit",
            "customer_id", "customer_name", "customer_segment",
            "customer_country", "customer_region",
            "sales_rep_id", "sales_rep_name",
        )
        .agg(
            F.avg("pocket_price").alias("current_avg_pocket_price"),
            F.avg("list_price").alias("current_avg_list_price"),
            F.avg("discount_pct").alias("current_discount_pct"),
            F.avg("margin_pct").alias("current_margin_pct"),
            F.sum("total_revenue").alias("current_annual_revenue"),
            F.sum("units_sold").alias("current_annual_volume"),
            F.count("*").alias("transaction_count"),
        )
    )
    print(f"FICM aggregated rows: {ficm_agg.count():,}")

    # Read elasticity
    elasticity = spark.read.table(SOURCE_ELASTICITY)
    print(f"Elasticity rows: {elasticity.count():,}")

    # Read discount outliers
    outliers = spark.read.table(SOURCE_OUTLIERS)
    print(f"Outlier rows: {outliers.count():,}")

    return ficm_agg, elasticity, outliers


ficm_agg_df, elasticity_df, outliers_df = read_source_tables()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 -- Join Data Sources

# COMMAND ----------

def join_data_sources(
    ficm_df: DataFrame,
    elasticity_df: DataFrame,
    outliers_df: DataFrame,
) -> DataFrame:
    """Join FICM aggregation with elasticity and outlier data.

    The join brings in:
    - Elasticity coefficients and safe increase thresholds (by sku + segment)
    - Discount gap and peer benchmarks (by sku + segment + rep)

    Not all (sku, customer, rep) combinations will have outlier data --
    left joins are used so all FICM rows are retained.

    Parameters
    ----------
    ficm_df : DataFrame
        Aggregated FICM data at (sku, customer, rep) grain.
    elasticity_df : DataFrame
        Elasticity coefficients by (sku, customer_segment).
    outliers_df : DataFrame
        Discount outlier flags by (sku, customer_segment, sales_rep_id).

    Returns
    -------
    DataFrame
        Joined DataFrame with all inputs combined.
    """
    # Join elasticity on (sku, customer_segment)
    joined = ficm_df.join(
        elasticity_df.select(
            "sku", "customer_segment",
            "elasticity_coefficient", "elasticity_classification",
            "elasticity_abs", "confidence_level",
            "safe_increase_3pct_vol_loss",
            "safe_increase_5pct_vol_loss",
        ),
        on=["sku", "customer_segment"],
        how="left",
    )

    # Join outlier data on (sku, customer_segment, sales_rep_id)
    joined = joined.join(
        outliers_df.select(
            "sku", "customer_segment", "sales_rep_id",
            F.col("peer_avg_discount_pct").alias("outlier_peer_avg_discount_pct"),
            F.col("discount_gap").alias("outlier_discount_gap"),
            "z_score", "outlier_severity",
        ),
        on=["sku", "customer_segment", "sales_rep_id"],
        how="left",
    )

    # Fill nulls for rows without outlier or elasticity data
    joined = (
        joined
        .withColumn("elasticity_coefficient",
                     F.coalesce(F.col("elasticity_coefficient"), F.lit(-0.5)))
        .withColumn("elasticity_abs",
                     F.coalesce(F.col("elasticity_abs"), F.lit(0.5)))
        .withColumn("elasticity_classification",
                     F.coalesce(F.col("elasticity_classification"), F.lit("Unknown")))
        .withColumn("confidence_level",
                     F.coalesce(F.col("confidence_level"), F.lit("Low")))
        .withColumn("safe_increase_3pct_vol_loss",
                     F.coalesce(F.col("safe_increase_3pct_vol_loss"), F.lit(3.0)))
        .withColumn("outlier_peer_avg_discount_pct",
                     F.coalesce(F.col("outlier_peer_avg_discount_pct"), F.col("current_discount_pct")))
        .withColumn("outlier_discount_gap",
                     F.coalesce(F.col("outlier_discount_gap"), F.lit(0.0)))
    )

    print(f"Joined rows: {joined.count():,}")
    return joined


joined_df = join_data_sources(ficm_agg_df, elasticity_df, outliers_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 -- Compute Uplift Score

# COMMAND ----------

def compute_uplift_score(df: DataFrame) -> DataFrame:
    """Compute the composite uplift score for each row.

    The uplift score combines five normalised factors:

    uplift_score = 0.30 * inelasticity_score
                 + 0.25 * discount_gap_score
                 + 0.20 * revenue_weight_score
                 + 0.15 * margin_headroom_score
                 - 0.10 * competitive_risk_score

    Each component is normalised to [0, 1] range.

    Parameters
    ----------
    df : DataFrame
        Joined data with elasticity and outlier information.

    Returns
    -------
    DataFrame
        With uplift_score column added.
    """
    # Compute total portfolio revenue for weighting
    total_revenue = df.agg(F.sum("current_annual_revenue")).collect()[0][0]
    if total_revenue is None or total_revenue == 0:
        total_revenue = 1.0

    # 1. Inelasticity score: lower |elasticity| = higher score
    # Normalise: 1 / (1 + |elasticity|), capped at 1.0
    df = df.withColumn(
        "_inelasticity_score",
        F.lit(1.0) / (F.lit(1.0) + F.col("elasticity_abs")),
    )

    # 2. Discount gap score: higher gap = more opportunity
    # Normalise using sigmoid-like function: gap / (gap + 5)
    df = df.withColumn(
        "_discount_gap_score",
        F.when(
            F.col("outlier_discount_gap") > 0,
            F.col("outlier_discount_gap") / (F.col("outlier_discount_gap") + F.lit(5.0)),
        ).otherwise(F.lit(0.0)),
    )

    # 3. Revenue weight: larger revenue = higher impact priority
    # Log-normalised to avoid extreme skew
    df = df.withColumn(
        "_revenue_score",
        F.log(F.greatest(F.col("current_annual_revenue"), F.lit(1.0)))
        / F.log(F.lit(total_revenue)),
    )

    # 4. Margin headroom: higher current margin = more room for price action
    # Normalise margin_pct to [0, 1]
    df = df.withColumn(
        "_margin_score",
        F.least(F.greatest(F.col("current_margin_pct"), F.lit(0.0)), F.lit(1.0)),
    )

    # 5. Competitive risk: proxy from discount gap direction
    # If discounting below peer avg, competitive pressure may be high
    df = df.withColumn(
        "_competitive_risk_score",
        F.when(
            F.col("outlier_discount_gap") < 0,
            F.abs(F.col("outlier_discount_gap")) / F.lit(10.0),
        ).otherwise(F.lit(0.1)),
    )

    # Composite score
    df = df.withColumn(
        "uplift_score",
        F.round(
            F.lit(W_INELASTICITY) * F.col("_inelasticity_score")
            + F.lit(W_DISCOUNT_GAP) * F.col("_discount_gap_score")
            + F.lit(W_REVENUE) * F.col("_revenue_score")
            + F.lit(W_MARGIN) * F.col("_margin_score")
            - F.lit(W_COMPETITIVE_RISK) * F.col("_competitive_risk_score"),
            6,
        ),
    )

    # Drop intermediate columns
    df = df.drop(
        "_inelasticity_score", "_discount_gap_score",
        "_revenue_score", "_margin_score", "_competitive_risk_score",
    )

    return df


scored_df = compute_uplift_score(joined_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 -- Compute Suggested Price Increase and Expected Impacts

# COMMAND ----------

def compute_suggested_increase(df: DataFrame) -> DataFrame:
    """Compute the suggested price increase and expected volume/revenue impacts.

    Suggested increase = MIN(safe_increase_3pct, discount_gap * 0.5, MAX_CAP)

    Expected impacts are derived from the elasticity coefficient:
    - volume_change_pct = elasticity_coefficient * price_increase_pct
    - volume_change_units = current_volume * volume_change_pct / 100
    - revenue_impact = new_price * new_volume - current_revenue
    - margin_impact = revenue_impact * contribution_margin_ratio

    Parameters
    ----------
    df : DataFrame
        Scored DataFrame with uplift_score.

    Returns
    -------
    DataFrame
        With suggested increase and impact columns.
    """
    # Suggested price increase percentage
    df = df.withColumn(
        "suggested_price_increase_pct",
        F.round(
            F.least(
                F.col("safe_increase_3pct_vol_loss"),
                F.greatest(F.col("outlier_discount_gap") * F.lit(0.5), F.lit(0.0)),
                F.lit(MAX_SUGGESTED_INCREASE_PCT),
            ),
            4,
        ),
    )

    # Ensure non-negative suggestion
    df = df.withColumn(
        "suggested_price_increase_pct",
        F.greatest(F.col("suggested_price_increase_pct"), F.lit(0.0)),
    )

    # Suggested new pocket price
    df = df.withColumn(
        "suggested_new_pocket_price",
        F.round(
            F.col("current_avg_pocket_price")
            * (1.0 + F.col("suggested_price_increase_pct") / 100.0),
            2,
        ),
    )

    # Expected volume change (using elasticity)
    df = df.withColumn(
        "expected_volume_change_pct",
        F.round(
            F.col("elasticity_coefficient") * F.col("suggested_price_increase_pct"),
            4,
        ),
    )

    df = df.withColumn(
        "expected_volume_change_units",
        F.round(
            F.col("current_annual_volume") * F.col("expected_volume_change_pct") / 100.0,
            0,
        ),
    )

    # Expected revenue impact
    new_price = F.col("current_avg_pocket_price") * (1 + F.col("suggested_price_increase_pct") / 100.0)
    new_volume = F.col("current_annual_volume") * (1 + F.col("expected_volume_change_pct") / 100.0)

    df = df.withColumn(
        "expected_revenue_impact",
        F.round(
            new_price * new_volume - F.col("current_annual_revenue"),
            2,
        ),
    )

    # Expected margin impact (~65% contribution margin)
    df = df.withColumn(
        "expected_margin_impact",
        F.round(F.col("expected_revenue_impact") * F.lit(0.65), 2),
    )

    return df


impact_df = compute_suggested_increase(scored_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 -- Compute Portfolio Weight and Cumulative Uplift

# COMMAND ----------

def compute_cumulative_uplift(df: DataFrame, target_pct: float) -> DataFrame:
    """Sort by uplift score and compute cumulative portfolio uplift.

    Portfolio weight = row's revenue / total portfolio revenue.
    Marginal uplift contribution = portfolio_weight * suggested_increase_pct.
    Cumulative uplift = running sum of marginal contributions.

    Rows where cumulative uplift >= target are marked as part of the
    target action set.

    Parameters
    ----------
    df : DataFrame
        Scored and impact-calculated DataFrame.
    target_pct : float
        Target portfolio uplift percentage (e.g. 1.0 for 1%).

    Returns
    -------
    DataFrame
        With rank, portfolio_weight, cumulative_uplift, and is_within_target_set.
    """
    # Compute total portfolio revenue
    total_revenue = df.agg(F.sum("current_annual_revenue")).collect()[0][0]
    if total_revenue is None or total_revenue == 0:
        total_revenue = 1.0

    # Portfolio weight
    df = df.withColumn(
        "portfolio_weight_pct",
        F.round(
            F.col("current_annual_revenue") / F.lit(total_revenue) * 100.0,
            6,
        ),
    )

    # Marginal uplift contribution = weight * suggested increase
    df = df.withColumn(
        "marginal_uplift_contribution_pct",
        F.round(
            F.col("portfolio_weight_pct") / 100.0 * F.col("suggested_price_increase_pct"),
            6,
        ),
    )

    # Rank by uplift score descending
    rank_window = Window.orderBy(F.col("uplift_score").desc())
    df = df.withColumn("rank", F.row_number().over(rank_window))

    # Cumulative uplift (running sum ordered by rank)
    cumulative_window = Window.orderBy("rank").rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )
    df = df.withColumn(
        "cumulative_portfolio_uplift_pct",
        F.round(
            F.sum("marginal_uplift_contribution_pct").over(cumulative_window),
            6,
        ),
    )

    # Mark target set
    df = df.withColumn(
        "is_within_target_set",
        F.when(
            F.col("cumulative_portfolio_uplift_pct") <= target_pct,
            True,
        ).otherwise(False),
    )

    # Target uplift parameter for reference
    df = df.withColumn("target_uplift_pct", F.lit(target_pct))

    # Count how many actions to reach target
    target_actions = df.filter(F.col("is_within_target_set") == True).count()
    print(f"Actions needed to reach {target_pct}% uplift: {target_actions:,}")

    return df


cumulative_df = compute_cumulative_uplift(impact_df, TARGET_UPLIFT_PCT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 -- Generate Rationale Text and Risk Factors

# COMMAND ----------

def generate_rationale(df: DataFrame) -> DataFrame:
    """Generate human-readable rationale and risk factor text for each row.

    The rationale summarises why this SKU/customer/rep combination is a
    good candidate for price uplift. Risk factors highlight potential
    concerns.

    Parameters
    ----------
    df : DataFrame
        Cumulative uplift DataFrame.

    Returns
    -------
    DataFrame
        With rationale and risk_factors columns.
    """
    # Rationale: combine key factors into a readable string
    df = df.withColumn(
        "rationale",
        F.concat(
            F.lit("Elasticity: "), F.col("elasticity_classification"),
            F.lit(" (coef="), F.round("elasticity_coefficient", 3).cast("string"),
            F.lit("). "),
            F.when(
                F.col("outlier_discount_gap") > 0,
                F.concat(
                    F.lit("Discount gap: +"),
                    F.round("outlier_discount_gap", 2).cast("string"),
                    F.lit("pp vs peer avg. "),
                ),
            ).otherwise(F.lit("Discount aligned with peers. ")),
            F.lit("Suggested increase: "),
            F.round("suggested_price_increase_pct", 2).cast("string"),
            F.lit("% yielding est. $"),
            F.format_number("expected_revenue_impact", 0),
            F.lit(" revenue impact. "),
            F.when(
                F.col("confidence_level") == "High",
                F.lit("High-confidence elasticity estimate."),
            ).when(
                F.col("confidence_level") == "Medium",
                F.lit("Medium-confidence elasticity estimate."),
            ).otherwise(F.lit("Low-confidence estimate -- validate with field data.")),
        ),
    )

    # Risk factors
    df = df.withColumn(
        "risk_factors",
        F.concat(
            F.when(
                F.col("elasticity_abs") >= 1.5,
                F.lit("HIGH: Elastic product -- volume loss risk. "),
            ).otherwise(F.lit("")),
            F.when(
                F.col("confidence_level") == "Low",
                F.lit("MEDIUM: Low-confidence elasticity estimate. "),
            ).otherwise(F.lit("")),
            F.when(
                F.col("suggested_price_increase_pct") >= 4.0,
                F.lit("MEDIUM: Large price increase may trigger competitive switching. "),
            ).otherwise(F.lit("")),
            F.when(
                F.col("current_annual_revenue") > 1000000,
                F.lit("INFO: High-revenue account -- executive approval recommended."),
            ).otherwise(F.lit("")),
        ),
    )

    return df


rationale_df = generate_rationale(cumulative_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 -- Final Schema Selection

# COMMAND ----------

def finalise_uplift_schema(df: DataFrame) -> DataFrame:
    """Select and order the final output columns for gold.uplift_simulation.

    Parameters
    ----------
    df : DataFrame
        Full uplift simulation DataFrame.

    Returns
    -------
    DataFrame
        Final schema.
    """
    df = df.withColumn("updated_at", F.current_timestamp())

    final_df = df.select(
        "rank",
        "product_id",
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
        F.round("current_avg_pocket_price", 2).alias("current_avg_pocket_price"),
        F.round("current_avg_list_price", 2).alias("current_avg_list_price"),
        F.round("current_discount_pct", 4).alias("current_discount_pct"),
        F.round("outlier_peer_avg_discount_pct", 4).alias("peer_avg_discount_pct"),
        F.round("outlier_discount_gap", 4).alias("discount_gap"),
        F.round("elasticity_coefficient", 6).alias("elasticity_coefficient"),
        "elasticity_classification",
        F.round("safe_increase_3pct_vol_loss", 4).alias("safe_increase_3pct_vol_loss"),
        "confidence_level",
        F.round("suggested_price_increase_pct", 4).alias("suggested_price_increase_pct"),
        F.round("suggested_new_pocket_price", 2).alias("suggested_new_pocket_price"),
        F.round("uplift_score", 6).alias("uplift_score"),
        F.round("expected_volume_change_pct", 4).alias("expected_volume_change_pct"),
        F.round("expected_volume_change_units", 0).alias("expected_volume_change_units"),
        F.round("expected_revenue_impact", 2).alias("expected_revenue_impact"),
        F.round("expected_margin_impact", 2).alias("expected_margin_impact"),
        F.round("current_annual_revenue", 2).alias("current_annual_revenue"),
        "current_annual_volume",
        "portfolio_weight_pct",
        "marginal_uplift_contribution_pct",
        "cumulative_portfolio_uplift_pct",
        "is_within_target_set",
        "target_uplift_pct",
        "rationale",
        "risk_factors",
        "updated_at",
    )

    return final_df


final_uplift_df = finalise_uplift_schema(rationale_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 -- Validation

# COMMAND ----------

def validate_uplift(df: DataFrame) -> None:
    """Run data quality checks on the final uplift simulation.

    Parameters
    ----------
    df : DataFrame
        Final uplift_simulation DataFrame.
    """
    row_count = df.count()
    print(f"Total uplift simulation rows: {row_count:,}")

    # Target set size
    target_set_count = df.filter(F.col("is_within_target_set") == True).count()
    print(f"Actions in target set (to reach {TARGET_UPLIFT_PCT}%): {target_set_count:,}")

    # Max cumulative uplift
    max_uplift = df.agg(
        F.max("cumulative_portfolio_uplift_pct")
    ).collect()[0][0]
    print(f"Maximum cumulative uplift: {max_uplift:.4f}%")

    # Score distribution
    print("\nUplift score distribution:")
    df.select("uplift_score").summary("min", "25%", "50%", "75%", "max").show()

    # Elasticity classification distribution within target set
    print("Elasticity classes in target set:")
    df.filter(F.col("is_within_target_set") == True).groupBy(
        "elasticity_classification"
    ).count().orderBy("elasticity_classification").show()

    # Total expected revenue impact of target set
    target_revenue_impact = (
        df.filter(F.col("is_within_target_set") == True)
        .agg(F.sum("expected_revenue_impact"))
        .collect()[0][0]
    )
    print(f"Total expected revenue impact (target set): ${target_revenue_impact:,.2f}")

    # Rank should be sequential starting at 1
    max_rank = df.agg(F.max("rank")).collect()[0][0]
    assert max_rank == row_count, f"Rank max ({max_rank}) != row count ({row_count})"
    print(f"\n[PASS] Ranks are sequential 1 to {row_count}")

    print("\nAll validation checks passed.")


validate_uplift(final_uplift_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 -- Write to Delta Lake

# COMMAND ----------

# Ensure gold schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}")

# Write as managed Delta table
(
    final_uplift_df
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
    "Uplift simulation scoring every (SKU, customer, rep) for price-increase potential. "
    "Uplift score = 0.30*inelasticity + 0.25*discount_gap + 0.20*revenue_weight "
    "+ 0.15*margin_headroom - 0.10*competitive_risk. "
    "Suggested increase = MIN(safe_increase_3pct, discount_gap*0.5, 5.0%). "
    "Rows sorted by score with cumulative portfolio uplift tracking toward target. "
    "Source: silver.ficm_pricing_master + gold.price_elasticity + gold.discount_outliers. "
    "Generated by notebook 15_gold_uplift_simulation."
)
spark.sql(f"COMMENT ON TABLE {TARGET_TABLE} IS '{_comment}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 -- Summary
# MAGIC
# MAGIC | Metric | Expected |
# MAGIC |--------|----------|
# MAGIC | Total rows | 2000-5000 |
# MAGIC | Target uplift | 1.0% (configurable via widget) |
# MAGIC | Actions to reach target | 80-200 |
# MAGIC | Score components | inelasticity, discount_gap, revenue, margin, competitive_risk |
# MAGIC | Max suggested increase | 5.0% |
# MAGIC | Output table | `hls_amer_catalog.gold.uplift_simulation` |

# COMMAND ----------

target_set_count = verify_df.filter(F.col("is_within_target_set") == True).count()

print("=" * 70)
print("  UPLIFT SIMULATION COMPLETE")
print("=" * 70)
print(f"  Source FICM      : {SOURCE_FICM}")
print(f"  Source Elasticity : {SOURCE_ELASTICITY}")
print(f"  Source Outliers   : {SOURCE_OUTLIERS}")
print(f"  Output           : {TARGET_TABLE}")
print(f"  Total rows       : {verify_count:,}")
print(f"  Target uplift    : {TARGET_UPLIFT_PCT}%")
print(f"  Target set size  : {target_set_count:,}")
print(f"  Score weights    : inelasticity={W_INELASTICITY}, gap={W_DISCOUNT_GAP}, "
      f"revenue={W_REVENUE}, margin={W_MARGIN}, risk={W_COMPETITIVE_RISK}")
print("=" * 70)
