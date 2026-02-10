# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 14 - Gold Layer: Price Elasticity Analysis
# MAGIC
# MAGIC **Purpose**: Compute price elasticity coefficients for every (SKU, customer_segment)
# MAGIC combination using log-log regression on monthly price-volume data. The elasticity
# MAGIC classification drives safe price-increase recommendations and revenue impact projections.
# MAGIC
# MAGIC **Algorithm**:
# MAGIC 1. Read from `hls_amer_catalog.silver.ficm_pricing_master`
# MAGIC 2. Aggregate to (sku, customer_segment, transaction_month) grain
# MAGIC 3. For groups with >= 6 monthly observations and >= 3 distinct price points,
# MAGIC    compute log-log regression: ln(volume) = alpha + beta * ln(price)
# MAGIC 4. Classify: |beta| < 0.5 = Highly Inelastic, 0.5-1.0 = Inelastic,
# MAGIC    1.0-1.5 = Unit Elastic, >= 1.5 = Elastic
# MAGIC 5. Compute safe price-increase ranges and revenue/margin impacts
# MAGIC
# MAGIC **Output**: `hls_amer_catalog.gold.price_elasticity`
# MAGIC
# MAGIC **Expected**: ~800-1400 rows; 30% Highly Inelastic, 35% Inelastic,
# MAGIC 20% Unit Elastic, 15% Elastic
# MAGIC
# MAGIC **Owner**: Pricing Intelligence Team

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0 -- Imports & Configuration

# COMMAND ----------

import numpy as np
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    DoubleType,
    StringType,
    IntegerType,
    StructType,
    StructField,
)
from datetime import datetime

# COMMAND ----------

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATALOG = "hls_amer_catalog"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"

SOURCE_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.ficm_pricing_master"
TARGET_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.price_elasticity"

# Minimum data requirements for reliable regression
MIN_MONTHLY_OBSERVATIONS = 6
MIN_DISTINCT_PRICE_POINTS = 3

# Elasticity classification thresholds (absolute value of beta)
HIGHLY_INELASTIC_THRESHOLD = 0.5
INELASTIC_THRESHOLD = 1.0
UNIT_ELASTIC_THRESHOLD = 1.5

spark = SparkSession.builder.getOrCreate()

print(f"Source table : {SOURCE_TABLE}")
print(f"Target table : {TARGET_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 -- Read Source Data

# COMMAND ----------

ficm_df = spark.read.table(SOURCE_TABLE)
print(f"Loaded {ficm_df.count():,} rows from {SOURCE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 -- Aggregate to Monthly Grain
# MAGIC
# MAGIC Group transactions by (sku, customer_segment, transaction_month) to compute
# MAGIC average price and total volume per period. This is the input for the
# MAGIC log-log regression.

# COMMAND ----------

def aggregate_monthly(df: DataFrame) -> DataFrame:
    """Aggregate FICM data to (sku, customer_segment, month) grain.

    Computes average pocket price, total volume, total revenue, and
    margin metrics for each time period. Also carries forward product
    dimension columns using first() aggregation.

    Parameters
    ----------
    df : DataFrame
        FICM pricing master.

    Returns
    -------
    DataFrame
        Monthly aggregation with avg_price, total_volume, and product metadata.
    """
    # Extract year-month from transaction date
    df = df.withColumn(
        "transaction_month",
        F.date_format(F.col("transaction_date"), "yyyy-MM"),
    )

    monthly = (
        df.groupBy(
            "sku", "customer_segment", "transaction_month",
        )
        .agg(
            # Price and volume
            F.avg("pocket_price").alias("avg_price"),
            F.sum("units_sold").alias("total_volume"),
            F.sum("total_revenue").alias("total_revenue"),
            F.avg("list_price").alias("avg_list_price"),
            F.avg("margin_pct").alias("avg_margin_pct"),
            # Product metadata (carry forward via first)
            F.first("product_id").alias("product_id"),
            F.first("product_name").alias("product_name"),
            F.first("product_family").alias("product_family"),
            F.first("business_unit").alias("business_unit"),
        )
        # Require positive price and volume for valid regression
        .filter((F.col("avg_price") > 0) & (F.col("total_volume") > 0))
    )

    print(f"Monthly aggregated rows: {monthly.count():,}")
    return monthly


monthly_df = aggregate_monthly(ficm_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 -- Identify Eligible Groups for Regression
# MAGIC
# MAGIC Filter to (sku, customer_segment) groups that have enough temporal
# MAGIC observations and price variation for a meaningful regression.

# COMMAND ----------

def identify_eligible_groups(df: DataFrame) -> DataFrame:
    """Identify (sku, customer_segment) groups eligible for elasticity regression.

    Eligibility criteria:
    - At least MIN_MONTHLY_OBSERVATIONS monthly data points
    - At least MIN_DISTINCT_PRICE_POINTS distinct price levels

    Parameters
    ----------
    df : DataFrame
        Monthly aggregated data.

    Returns
    -------
    DataFrame
        Eligibility metadata per (sku, customer_segment).
    """
    group_stats = (
        df.groupBy("sku", "customer_segment")
        .agg(
            F.count("*").alias("sample_months"),
            F.countDistinct(F.round("avg_price", 0)).alias("distinct_price_points"),
            F.avg("avg_price").alias("overall_avg_price"),
            F.avg("total_volume").alias("avg_volume_monthly"),
            F.sum("total_revenue").alias("total_revenue"),
            F.min("avg_price").alias("price_range_min"),
            F.max("avg_price").alias("price_range_max"),
            F.stddev("avg_price").alias("price_std"),
        )
        .filter(
            (F.col("sample_months") >= MIN_MONTHLY_OBSERVATIONS)
            & (F.col("distinct_price_points") >= MIN_DISTINCT_PRICE_POINTS)
        )
    )

    # Compute coefficient of variation for price
    group_stats = group_stats.withColumn(
        "price_cv",
        F.when(
            F.col("overall_avg_price") > 0,
            F.col("price_std") / F.col("overall_avg_price"),
        ).otherwise(F.lit(0.0)),
    )

    eligible_count = group_stats.count()
    print(f"Eligible (sku, segment) groups: {eligible_count:,}")
    return group_stats


eligible_groups_df = identify_eligible_groups(monthly_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 -- Log-Log Regression via Pandas UDF
# MAGIC
# MAGIC The core elasticity computation uses `scipy.stats.linregress` on
# MAGIC ln(price) vs ln(volume) within each (sku, customer_segment) group.
# MAGIC This is implemented as a grouped Pandas UDF for distributed execution.

# COMMAND ----------

# Define the output schema for the regression UDF
REGRESSION_SCHEMA = StructType([
    StructField("sku", StringType(), False),
    StructField("customer_segment", StringType(), False),
    StructField("elasticity_coefficient", DoubleType(), True),
    StructField("r_squared", DoubleType(), True),
    StructField("p_value", DoubleType(), True),
    StructField("intercept", DoubleType(), True),
    StructField("stderr", DoubleType(), True),
])


@F.pandas_udf(REGRESSION_SCHEMA, F.PandasUDFType.GROUPED_MAP)
def compute_log_log_regression(pdf):
    """Compute log-log regression for a single (sku, customer_segment) group.

    Fits the model: ln(volume) = alpha + beta * ln(price)

    The coefficient beta is the price elasticity of demand:
    - beta < 0 indicates normal demand (price up -> volume down)
    - |beta| < 1 indicates inelastic demand
    - |beta| > 1 indicates elastic demand

    Uses scipy.stats.linregress for OLS estimation with p-value and R-squared.

    Parameters
    ----------
    pdf : pandas.DataFrame
        Monthly data for one (sku, customer_segment) group with columns
        avg_price and total_volume.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame with regression results.
    """
    import pandas as pd
    from scipy import stats

    sku = pdf["sku"].iloc[0]
    segment = pdf["customer_segment"].iloc[0]

    try:
        # Compute log-log values
        ln_price = np.log(pdf["avg_price"].values)
        ln_volume = np.log(pdf["total_volume"].values)

        # Run OLS regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(ln_price, ln_volume)

        return pd.DataFrame([{
            "sku": sku,
            "customer_segment": segment,
            "elasticity_coefficient": float(slope),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "intercept": float(intercept),
            "stderr": float(std_err),
        }])
    except Exception:
        # Return null values if regression fails
        return pd.DataFrame([{
            "sku": sku,
            "customer_segment": segment,
            "elasticity_coefficient": None,
            "r_squared": None,
            "p_value": None,
            "intercept": None,
            "stderr": None,
        }])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 -- Run Regression Across All Eligible Groups

# COMMAND ----------

def run_elasticity_regression(
    monthly_df: DataFrame,
    eligible_df: DataFrame,
) -> DataFrame:
    """Run the log-log regression for all eligible (sku, customer_segment) groups.

    Filters monthly data to only eligible groups, then applies the grouped
    Pandas UDF to compute regression coefficients in parallel.

    Parameters
    ----------
    monthly_df : DataFrame
        Monthly aggregated price-volume data.
    eligible_df : DataFrame
        Eligible groups from identify_eligible_groups().

    Returns
    -------
    DataFrame
        Regression results for each eligible group.
    """
    # Inner join to filter monthly data to eligible groups only
    filtered = monthly_df.join(
        eligible_df.select("sku", "customer_segment"),
        on=["sku", "customer_segment"],
        how="inner",
    )

    # Apply the grouped regression UDF
    regression_results = (
        filtered
        .select("sku", "customer_segment", "avg_price", "total_volume")
        .groupBy("sku", "customer_segment")
        .apply(compute_log_log_regression)
    )

    # Drop rows where regression failed
    regression_results = regression_results.filter(
        F.col("elasticity_coefficient").isNotNull()
    )

    result_count = regression_results.count()
    print(f"Successful regressions: {result_count:,}")
    return regression_results


regression_df = run_elasticity_regression(monthly_df, eligible_groups_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 -- Classify Elasticity and Compute Safe Price Increases

# COMMAND ----------

def classify_and_compute_impacts(
    regression_df: DataFrame,
    eligible_df: DataFrame,
    monthly_df: DataFrame,
) -> DataFrame:
    """Classify elasticity and compute safe price-increase recommendations.

    Elasticity classification (by absolute value of beta):
    - |beta| < 0.5: Highly Inelastic -- strong pricing power
    - 0.5 <= |beta| < 1.0: Inelastic -- moderate pricing power
    - 1.0 <= |beta| < 1.5: Unit Elastic -- balanced sensitivity
    - |beta| >= 1.5: Elastic -- price-sensitive, caution required

    Safe price increase = target_volume_loss / |elasticity_coefficient|

    Revenue impact at X% volume loss:
    - new_price = price * (1 + increase_pct)
    - new_volume = volume * (1 - volume_loss_pct)
    - revenue_impact = new_price * new_volume - current_revenue

    Parameters
    ----------
    regression_df : DataFrame
        Regression results with elasticity_coefficient.
    eligible_df : DataFrame
        Group-level metadata (avg price, volume, revenue).
    monthly_df : DataFrame
        Monthly data for product metadata lookup.

    Returns
    -------
    DataFrame
        Fully enriched elasticity table.
    """
    # Join regression results with group metadata
    enriched = regression_df.join(
        eligible_df,
        on=["sku", "customer_segment"],
        how="inner",
    )

    # Get product metadata (first occurrence)
    product_meta = (
        monthly_df
        .select("sku", "customer_segment", "product_id", "product_name",
                "product_family", "business_unit")
        .dropDuplicates(["sku", "customer_segment"])
    )
    enriched = enriched.join(
        product_meta,
        on=["sku", "customer_segment"],
        how="left",
    )

    # Absolute elasticity
    enriched = enriched.withColumn(
        "elasticity_abs",
        F.abs(F.col("elasticity_coefficient")),
    )

    # Classification
    enriched = enriched.withColumn(
        "elasticity_classification",
        F.when(F.col("elasticity_abs") < HIGHLY_INELASTIC_THRESHOLD, F.lit("Highly Inelastic"))
        .when(F.col("elasticity_abs") < INELASTIC_THRESHOLD, F.lit("Inelastic"))
        .when(F.col("elasticity_abs") < UNIT_ELASTIC_THRESHOLD, F.lit("Unit Elastic"))
        .otherwise(F.lit("Elastic")),
    )

    # Safe price increases: increase_pct = target_vol_loss / |elasticity|
    # If elasticity_abs is near zero, cap the safe increase at 20%
    for vol_loss_pct in [1.0, 3.0, 5.0, 10.0]:
        col_name = f"safe_increase_{int(vol_loss_pct)}pct_vol_loss"
        enriched = enriched.withColumn(
            col_name,
            F.when(
                F.col("elasticity_abs") > 0.01,
                F.least(
                    F.round(F.lit(vol_loss_pct) / F.col("elasticity_abs"), 4),
                    F.lit(20.0),
                ),
            ).otherwise(F.lit(20.0)),
        )

    # Revenue-optimal price change: for log-log model, optimal = 1 / (|elasticity| - 1)
    # Only meaningful when |elasticity| > 1
    enriched = enriched.withColumn(
        "max_revenue_optimal_price_change_pct",
        F.when(
            F.col("elasticity_abs") > 1.01,
            F.round(F.lit(100.0) / (F.col("elasticity_abs") - 1.0), 2),
        ).otherwise(F.lit(None).cast(DoubleType())),
    )

    # Revenue and margin impact at 3% and 5% volume loss
    avg_price = F.col("overall_avg_price")
    avg_vol = F.col("avg_volume_monthly")

    for vol_loss in [3.0, 5.0]:
        safe_inc_col = f"safe_increase_{int(vol_loss)}pct_vol_loss"
        rev_col = f"revenue_impact_at_{int(vol_loss)}pct_vol_loss"
        margin_col = f"margin_impact_at_{int(vol_loss)}pct_vol_loss"

        # Revenue impact = (new_price * new_volume - old_price * old_volume) * 12
        enriched = enriched.withColumn(
            rev_col,
            F.round(
                (
                    (avg_price * (1 + F.col(safe_inc_col) / 100.0))
                    * (avg_vol * (1 - vol_loss / 100.0))
                    - avg_price * avg_vol
                ) * F.lit(12.0),
                2,
            ),
        )

        # Margin impact approximation: similar structure, assumes margin scales with price
        enriched = enriched.withColumn(
            margin_col,
            F.round(F.col(rev_col) * F.lit(0.65), 2),  # ~65% contribution margin
        )

    # Confidence level based on R-squared and sample size
    enriched = enriched.withColumn(
        "confidence_level",
        F.when(
            (F.col("r_squared") >= 0.5) & (F.col("sample_months") >= 12),
            F.lit("High"),
        )
        .when(
            (F.col("r_squared") >= 0.3) & (F.col("sample_months") >= 6),
            F.lit("Medium"),
        )
        .otherwise(F.lit("Low")),
    )

    return enriched


enriched_df = classify_and_compute_impacts(regression_df, eligible_groups_df, monthly_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 -- Generate Elasticity IDs and Final Schema

# COMMAND ----------

def finalise_elasticity_schema(df: DataFrame) -> DataFrame:
    """Generate unique elasticity IDs and select the final output columns.

    Parameters
    ----------
    df : DataFrame
        Enriched elasticity DataFrame.

    Returns
    -------
    DataFrame
        Final schema matching gold.price_elasticity specification.
    """
    # Generate deterministic elasticity_id
    df = df.withColumn(
        "elasticity_id",
        F.md5(F.concat_ws("|", F.col("sku"), F.col("customer_segment"))),
    )

    # Add updated_at timestamp
    df = df.withColumn("updated_at", F.current_timestamp())

    # Select final columns
    final_df = df.select(
        "elasticity_id",
        "product_id",
        "sku",
        "product_name",
        "product_family",
        "business_unit",
        "customer_segment",
        F.round("elasticity_coefficient", 6).alias("elasticity_coefficient"),
        F.round("elasticity_abs", 6).alias("elasticity_abs"),
        "elasticity_classification",
        F.round("r_squared", 6).alias("r_squared"),
        F.round("p_value", 6).alias("p_value"),
        "sample_months",
        "distinct_price_points",
        F.round("overall_avg_price", 2).alias("avg_price"),
        F.round("avg_volume_monthly", 2).alias("avg_volume_monthly"),
        F.round("total_revenue", 2).alias("total_revenue"),
        F.round("price_range_min", 2).alias("price_range_min"),
        F.round("price_range_max", 2).alias("price_range_max"),
        F.round("price_cv", 6).alias("price_cv"),
        F.round("safe_increase_1pct_vol_loss", 4).alias("safe_increase_1pct_vol_loss"),
        F.round("safe_increase_3pct_vol_loss", 4).alias("safe_increase_3pct_vol_loss"),
        F.round("safe_increase_5pct_vol_loss", 4).alias("safe_increase_5pct_vol_loss"),
        F.round("safe_increase_10pct_vol_loss", 4).alias("safe_increase_10pct_vol_loss"),
        "max_revenue_optimal_price_change_pct",
        "revenue_impact_at_3pct_vol_loss",
        "revenue_impact_at_5pct_vol_loss",
        "margin_impact_at_3pct_vol_loss",
        "margin_impact_at_5pct_vol_loss",
        "confidence_level",
        "updated_at",
    )

    return final_df


final_elasticity_df = finalise_elasticity_schema(enriched_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 -- Validation

# COMMAND ----------

def validate_elasticity(df: DataFrame) -> None:
    """Run data quality checks on the final elasticity DataFrame.

    Parameters
    ----------
    df : DataFrame
        Final price_elasticity DataFrame.
    """
    row_count = df.count()
    print(f"Total elasticity rows: {row_count:,}")

    # Classification distribution
    print("\nElasticity classification distribution:")
    df.groupBy("elasticity_classification").agg(
        F.count("*").alias("count"),
        F.round(F.avg("elasticity_abs"), 4).alias("avg_abs_elasticity"),
        F.round(F.avg("r_squared"), 4).alias("avg_r_squared"),
    ).orderBy("elasticity_classification").show(truncate=False)

    # Confidence level distribution
    print("Confidence level distribution:")
    df.groupBy("confidence_level").count().orderBy("confidence_level").show()

    # Elasticity coefficient range
    coef_stats = df.agg(
        F.min("elasticity_coefficient").alias("min_coef"),
        F.max("elasticity_coefficient").alias("max_coef"),
        F.avg("elasticity_coefficient").alias("avg_coef"),
        F.avg("r_squared").alias("avg_r2"),
    ).collect()[0]
    print(f"Elasticity coefficient range: {coef_stats['min_coef']:.4f} to {coef_stats['max_coef']:.4f}")
    print(f"Average elasticity coefficient: {coef_stats['avg_coef']:.4f}")
    print(f"Average R-squared: {coef_stats['avg_r2']:.4f}")

    # No null elasticity_ids
    null_ids = df.filter(F.col("elasticity_id").isNull()).count()
    assert null_ids == 0, f"Found {null_ids} null elasticity_ids"
    print("\n[PASS] No null elasticity_ids")

    # All safe increase values should be positive
    neg_safe = df.filter(F.col("safe_increase_3pct_vol_loss") < 0).count()
    assert neg_safe == 0, f"Found {neg_safe} negative safe increase values"
    print("[PASS] All safe increase values are non-negative")

    print("\nAll validation checks passed.")


validate_elasticity(final_elasticity_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 -- Write to Delta Lake

# COMMAND ----------

# Ensure gold schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}")

# Write as managed Delta table
(
    final_elasticity_df
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
    "Price elasticity coefficients computed via log-log regression (ln(volume) = a + b*ln(price)) "
    "for each (SKU, customer_segment) with >= 6 monthly observations and >= 3 distinct price points. "
    "Classifications: Highly Inelastic (|b|<0.5), Inelastic (0.5-1.0), Unit Elastic (1.0-1.5), "
    "Elastic (>=1.5). Includes safe price-increase ranges and revenue/margin impact projections. "
    "Source: silver.ficm_pricing_master. Generated by notebook 14_gold_price_elasticity."
)
spark.sql(f"COMMENT ON TABLE {TARGET_TABLE} IS '{_comment}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 -- Summary
# MAGIC
# MAGIC | Metric | Expected |
# MAGIC |--------|----------|
# MAGIC | Total elasticity rows | 800-1400 |
# MAGIC | Highly Inelastic (|beta| < 0.5) | ~30% |
# MAGIC | Inelastic (0.5-1.0) | ~35% |
# MAGIC | Unit Elastic (1.0-1.5) | ~20% |
# MAGIC | Elastic (>= 1.5) | ~15% |
# MAGIC | Min monthly observations | 6 |
# MAGIC | Min distinct price points | 3 |
# MAGIC | Output table | `hls_amer_catalog.gold.price_elasticity` |

# COMMAND ----------

print("=" * 70)
print("  PRICE ELASTICITY ANALYSIS COMPLETE")
print("=" * 70)
print(f"  Source         : {SOURCE_TABLE}")
print(f"  Output         : {TARGET_TABLE}")
print(f"  Elasticity rows: {verify_count:,}")
print(f"  Min months     : {MIN_MONTHLY_OBSERVATIONS}")
print(f"  Min price pts  : {MIN_DISTINCT_PRICE_POINTS}")
print("=" * 70)
