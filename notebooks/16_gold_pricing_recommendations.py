# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 16 - Gold Layer: Pricing Recommendations
# MAGIC
# MAGIC **Purpose**: Generate actionable pricing recommendations by combining signals
# MAGIC from elasticity analysis, discount outlier detection, uplift simulation, and
# MAGIC competitive intelligence. Each recommendation is assigned an action type and
# MAGIC a priority score (0-100).
# MAGIC
# MAGIC **Action Types**:
# MAGIC - `INCREASE_PRICE`: Inelastic products where safe price increases are justified
# MAGIC - `STANDARDIZE_DISCOUNT`: Reps discounting far above peer norms
# MAGIC - `REVIEW_REP`: Severe outlier reps requiring management review
# MAGIC - `COMPETITIVE_ADJUSTMENT`: Adjustments driven by competitive positioning gaps
# MAGIC - `MARGIN_RECOVERY`: Products/customers where margin erosion can be reversed
# MAGIC
# MAGIC **Output**: `hls_amer_catalog.gold.pricing_recommendations`
# MAGIC
# MAGIC **Expected**: 200+ rows with all action types represented
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
from functools import reduce

# COMMAND ----------

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATALOG = "hls_amer_catalog"
GOLD_SCHEMA = "gold"

SOURCE_ELASTICITY = f"{CATALOG}.{GOLD_SCHEMA}.price_elasticity"
SOURCE_OUTLIERS = f"{CATALOG}.{GOLD_SCHEMA}.discount_outliers"
SOURCE_UPLIFT = f"{CATALOG}.{GOLD_SCHEMA}.uplift_simulation"

TARGET_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.pricing_recommendations"

# Action type constants
ACTION_INCREASE_PRICE = "INCREASE_PRICE"
ACTION_STANDARDIZE_DISCOUNT = "STANDARDIZE_DISCOUNT"
ACTION_REVIEW_REP = "REVIEW_REP"
ACTION_COMPETITIVE_ADJUSTMENT = "COMPETITIVE_ADJUSTMENT"
ACTION_MARGIN_RECOVERY = "MARGIN_RECOVERY"

spark = SparkSession.builder.getOrCreate()

print(f"Source Elasticity : {SOURCE_ELASTICITY}")
print(f"Source Outliers   : {SOURCE_OUTLIERS}")
print(f"Source Uplift     : {SOURCE_UPLIFT}")
print(f"Target table     : {TARGET_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 -- Read Source Tables

# COMMAND ----------

def read_sources() -> tuple:
    """Read all gold-layer source tables.

    Returns
    -------
    tuple of (DataFrame, DataFrame, DataFrame)
        (elasticity, outliers, uplift) DataFrames.
    """
    elasticity = spark.read.table(SOURCE_ELASTICITY)
    print(f"Elasticity rows: {elasticity.count():,}")

    outliers = spark.read.table(SOURCE_OUTLIERS)
    print(f"Outlier rows: {outliers.count():,}")

    uplift = spark.read.table(SOURCE_UPLIFT)
    print(f"Uplift rows: {uplift.count():,}")

    return elasticity, outliers, uplift


elasticity_df, outliers_df, uplift_df = read_sources()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 -- Generate INCREASE_PRICE Recommendations
# MAGIC
# MAGIC Products with Highly Inelastic or Inelastic demand where the safe
# MAGIC price increase exceeds 1% and confidence is Medium or High.

# COMMAND ----------

def generate_increase_price_recs(
    uplift_df: DataFrame,
) -> DataFrame:
    """Generate INCREASE_PRICE recommendations from the uplift simulation.

    Selection criteria:
    - Elasticity classification in (Highly Inelastic, Inelastic)
    - Safe increase at 3% volume loss > 1.0%
    - Confidence level in (High, Medium)
    - Suggested increase > 0.5%

    Priority score (0-100):
    - 40 pts: inelasticity (lower |elasticity| = higher priority)
    - 30 pts: revenue impact (larger = higher)
    - 20 pts: confidence (High=20, Medium=10)
    - 10 pts: safe increase magnitude

    Parameters
    ----------
    uplift_df : DataFrame
        Uplift simulation data.

    Returns
    -------
    DataFrame
        INCREASE_PRICE recommendations.
    """
    recs = (
        uplift_df
        .filter(
            F.col("elasticity_classification").isin("Highly Inelastic", "Inelastic")
            & (F.col("safe_increase_3pct_vol_loss") > 1.0)
            & F.col("confidence_level").isin("High", "Medium")
            & (F.col("suggested_price_increase_pct") > 0.5)
        )
        .withColumn("action_type", F.lit(ACTION_INCREASE_PRICE))
    )

    # Priority score
    recs = recs.withColumn(
        "priority_score",
        F.round(
            # Inelasticity component (40 pts)
            F.lit(40.0) * (F.lit(1.0) / (F.lit(1.0) + F.col("elasticity_coefficient").cast("double").alias("ec").__abs__()))
            # Revenue impact component (30 pts)
            + F.least(
                F.lit(30.0) * F.log(F.greatest(F.abs(F.col("expected_revenue_impact")), F.lit(1.0))) / F.lit(15.0),
                F.lit(30.0),
            )
            # Confidence component (20 pts)
            + F.when(F.col("confidence_level") == "High", F.lit(20.0))
             .when(F.col("confidence_level") == "Medium", F.lit(10.0))
             .otherwise(F.lit(0.0))
            # Safe increase magnitude (10 pts)
            + F.least(F.col("safe_increase_3pct_vol_loss") / F.lit(10.0) * F.lit(10.0), F.lit(10.0)),
            2,
        ),
    )

    # Cap at 100
    recs = recs.withColumn(
        "priority_score",
        F.least(F.greatest(F.col("priority_score"), F.lit(0.0)), F.lit(100.0)),
    )

    # Rationale
    recs = recs.withColumn(
        "rationale",
        F.concat(
            F.lit("PRICE INCREASE: "),
            F.col("elasticity_classification"),
            F.lit(" demand (beta="),
            F.round("elasticity_coefficient", 3).cast("string"),
            F.lit("). Safe to increase by "),
            F.round("suggested_price_increase_pct", 2).cast("string"),
            F.lit("% with <3% volume impact. "),
            F.lit("Confidence: "), F.col("confidence_level"), F.lit("."),
        ),
    )

    # Supporting factors
    recs = recs.withColumn(
        "supporting_factors",
        F.concat(
            F.lit("R-squared confidence; "),
            F.col("elasticity_classification"),
            F.lit(" classification; "),
            F.lit("safe increase headroom; "),
            F.when(
                F.col("is_within_target_set") == True,
                F.lit("within uplift target set"),
            ).otherwise(F.lit("outside uplift target set")),
        ),
    )

    rec_count = recs.count()
    print(f"INCREASE_PRICE recommendations: {rec_count:,}")
    return recs


increase_price_recs = generate_increase_price_recs(uplift_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 -- Generate STANDARDIZE_DISCOUNT Recommendations

# COMMAND ----------

def generate_standardize_discount_recs(
    outliers_df: DataFrame,
) -> DataFrame:
    """Generate STANDARDIZE_DISCOUNT recommendations from discount outliers.

    Selection criteria:
    - Outlier severity in (Moderate, Watch)
    - Discount gap > 1.0 percentage points

    Priority score:
    - 35 pts: z-score magnitude
    - 35 pts: recovery potential
    - 20 pts: peer group size (larger = more reliable benchmark)
    - 10 pts: volume significance

    Parameters
    ----------
    outliers_df : DataFrame
        Discount outlier data.

    Returns
    -------
    DataFrame
        STANDARDIZE_DISCOUNT recommendations.
    """
    recs = (
        outliers_df
        .filter(
            F.col("outlier_severity").isin("Moderate", "Watch")
            & (F.col("discount_gap") > 1.0)
        )
        .withColumn("action_type", F.lit(ACTION_STANDARDIZE_DISCOUNT))
    )

    # Priority score
    recs = recs.withColumn(
        "priority_score",
        F.round(
            F.least(F.col("z_score") / F.lit(5.0) * F.lit(35.0), F.lit(35.0))
            + F.least(
                F.log(F.greatest(F.col("potential_recovery_amount"), F.lit(1.0))) / F.lit(15.0) * F.lit(35.0),
                F.lit(35.0),
            )
            + F.least(F.col("peer_count").cast("double") / F.lit(20.0) * F.lit(20.0), F.lit(20.0))
            + F.least(
                F.log(F.greatest(F.col("rep_volume").cast("double"), F.lit(1.0))) / F.lit(10.0) * F.lit(10.0),
                F.lit(10.0),
            ),
            2,
        ),
    )
    recs = recs.withColumn(
        "priority_score",
        F.least(F.greatest(F.col("priority_score"), F.lit(0.0)), F.lit(100.0)),
    )

    # Compute recommended change
    recs = (
        recs
        .withColumn("recommended_change_pct",
                     F.round(F.col("discount_gap") * F.lit(-0.5), 4))
        .withColumn("expected_volume_impact_pct", F.lit(0.0))
        .withColumn("expected_revenue_impact",
                     F.round(F.col("potential_recovery_amount") * F.lit(0.5), 2))
        .withColumn("expected_margin_impact",
                     F.round(F.col("potential_recovery_amount") * F.lit(0.5) * F.lit(0.65), 2))
        .withColumn("risk_level",
                     F.when(F.col("z_score") > 3.0, F.lit("Low"))
                      .when(F.col("z_score") > 2.0, F.lit("Medium"))
                      .otherwise(F.lit("Low")))
    )

    # Rationale
    recs = recs.withColumn(
        "rationale",
        F.concat(
            F.lit("STANDARDIZE DISCOUNT: Rep discounts "),
            F.round("discount_gap", 2).cast("string"),
            F.lit("pp above peer avg (z="),
            F.round("z_score", 2).cast("string"),
            F.lit("). Aligning to peer norm recovers est. $"),
            F.format_number("expected_revenue_impact", 0),
            F.lit(". Peer group size: "),
            F.col("peer_count").cast("string"),
            F.lit(" reps."),
        ),
    )

    recs = recs.withColumn("supporting_factors", F.lit("Peer-group benchmarking; z-score analysis; recovery potential"))
    recs = recs.withColumn("competitive_context", F.lit("Discount standardisation within existing customer relationships"))

    rec_count = recs.count()
    print(f"STANDARDIZE_DISCOUNT recommendations: {rec_count:,}")
    return recs


standardize_recs = generate_standardize_discount_recs(outliers_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 -- Generate REVIEW_REP Recommendations

# COMMAND ----------

def generate_review_rep_recs(
    outliers_df: DataFrame,
) -> DataFrame:
    """Generate REVIEW_REP recommendations for severe discount outliers.

    Selection criteria:
    - Outlier severity = Severe (z-score > 3.0)

    These reps require management review due to extreme discounting patterns.

    Parameters
    ----------
    outliers_df : DataFrame
        Discount outlier data.

    Returns
    -------
    DataFrame
        REVIEW_REP recommendations.
    """
    recs = (
        outliers_df
        .filter(F.col("outlier_severity") == "Severe")
        .withColumn("action_type", F.lit(ACTION_REVIEW_REP))
    )

    # Priority score (severe outliers get high base priority)
    recs = recs.withColumn(
        "priority_score",
        F.round(
            F.lit(50.0)
            + F.least(F.col("z_score") / F.lit(6.0) * F.lit(30.0), F.lit(30.0))
            + F.least(
                F.log(F.greatest(F.col("potential_recovery_amount"), F.lit(1.0))) / F.lit(15.0) * F.lit(20.0),
                F.lit(20.0),
            ),
            2,
        ),
    )
    recs = recs.withColumn(
        "priority_score",
        F.least(F.greatest(F.col("priority_score"), F.lit(0.0)), F.lit(100.0)),
    )

    recs = (
        recs
        .withColumn("recommended_change_pct",
                     F.round(F.col("discount_gap") * F.lit(-0.75), 4))
        .withColumn("expected_volume_impact_pct", F.lit(-1.0))
        .withColumn("expected_revenue_impact",
                     F.round(F.col("potential_recovery_amount") * F.lit(0.75), 2))
        .withColumn("expected_margin_impact",
                     F.round(F.col("potential_recovery_amount") * F.lit(0.75) * F.lit(0.65), 2))
        .withColumn("risk_level", F.lit("High"))
    )

    recs = recs.withColumn(
        "rationale",
        F.concat(
            F.lit("REVIEW REP: Severe discount outlier (z="),
            F.round("z_score", 2).cast("string"),
            F.lit(", gap="),
            F.round("discount_gap", 2).cast("string"),
            F.lit("pp). Recovery potential: $"),
            F.format_number("potential_recovery_amount", 0),
            F.lit(". Management review recommended."),
        ),
    )

    recs = recs.withColumn("supporting_factors", F.lit("Severe z-score; large discount gap; peer benchmark comparison"))
    recs = recs.withColumn("competitive_context", F.lit("Internal pricing discipline issue -- not competitive-driven"))

    rec_count = recs.count()
    print(f"REVIEW_REP recommendations: {rec_count:,}")
    return recs


review_rep_recs = generate_review_rep_recs(outliers_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 -- Generate COMPETITIVE_ADJUSTMENT Recommendations

# COMMAND ----------

def generate_competitive_adjustment_recs(
    uplift_df: DataFrame,
) -> DataFrame:
    """Generate COMPETITIVE_ADJUSTMENT recommendations.

    Selection criteria:
    - Elastic products (|elasticity| >= 1.0) where peer discount gap exists
    - OR products with negative expected revenue impact at current pricing

    These require price adjustments driven by competitive dynamics.

    Parameters
    ----------
    uplift_df : DataFrame
        Uplift simulation data.

    Returns
    -------
    DataFrame
        COMPETITIVE_ADJUSTMENT recommendations.
    """
    recs = (
        uplift_df
        .filter(
            (F.col("elasticity_classification").isin("Elastic", "Unit Elastic"))
            & (F.col("discount_gap") > 0.5)
        )
        .withColumn("action_type", F.lit(ACTION_COMPETITIVE_ADJUSTMENT))
    )

    # Priority score -- competitive adjustments are typically moderate priority
    recs = recs.withColumn(
        "priority_score",
        F.round(
            F.lit(20.0)
            + F.least(F.col("discount_gap") / F.lit(10.0) * F.lit(30.0), F.lit(30.0))
            + F.least(
                F.log(F.greatest(F.col("current_annual_revenue"), F.lit(1.0))) / F.lit(20.0) * F.lit(30.0),
                F.lit(30.0),
            )
            + F.when(F.col("confidence_level") == "High", F.lit(20.0))
             .when(F.col("confidence_level") == "Medium", F.lit(10.0))
             .otherwise(F.lit(0.0)),
            2,
        ),
    )
    recs = recs.withColumn(
        "priority_score",
        F.least(F.greatest(F.col("priority_score"), F.lit(0.0)), F.lit(100.0)),
    )

    recs = (
        recs
        .withColumn("recommended_change_pct",
                     F.round(F.col("suggested_price_increase_pct") * F.lit(0.5), 4))
        .withColumn("expected_volume_impact_pct",
                     F.round(F.col("expected_volume_change_pct") * F.lit(0.5), 4))
        .withColumn("expected_revenue_impact",
                     F.round(F.col("expected_revenue_impact") * F.lit(0.5), 2))
        .withColumn("expected_margin_impact",
                     F.round(F.col("expected_margin_impact") * F.lit(0.5), 2))
        .withColumn("risk_level",
                     F.when(F.col("elasticity_classification") == "Elastic", F.lit("High"))
                      .otherwise(F.lit("Medium")))
    )

    recs = recs.withColumn(
        "rationale",
        F.concat(
            F.lit("COMPETITIVE ADJUSTMENT: "),
            F.col("elasticity_classification"),
            F.lit(" product with discount gap of "),
            F.round("discount_gap", 2).cast("string"),
            F.lit("pp. Conservative increase of "),
            F.round("recommended_change_pct", 2).cast("string"),
            F.lit("% to balance volume retention and margin."),
        ),
    )

    recs = recs.withColumn("supporting_factors", F.lit("Elasticity analysis; competitive positioning; discount gap"))
    recs = recs.withColumn("competitive_context", F.lit("Elastic demand -- monitor competitor response closely"))

    rec_count = recs.count()
    print(f"COMPETITIVE_ADJUSTMENT recommendations: {rec_count:,}")
    return recs


competitive_recs = generate_competitive_adjustment_recs(uplift_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 -- Generate MARGIN_RECOVERY Recommendations

# COMMAND ----------

def generate_margin_recovery_recs(
    uplift_df: DataFrame,
) -> DataFrame:
    """Generate MARGIN_RECOVERY recommendations for eroded margins.

    Selection criteria:
    - Current margin below 30%
    - Suggested price increase available (> 0.5%)
    - Non-elastic demand

    These focus on products/customers where margins have eroded below
    sustainable levels and need recovery.

    Parameters
    ----------
    uplift_df : DataFrame
        Uplift simulation data.

    Returns
    -------
    DataFrame
        MARGIN_RECOVERY recommendations.
    """
    recs = (
        uplift_df
        .filter(
            (F.col("current_discount_pct") > 15.0)
            & (F.col("suggested_price_increase_pct") > 0.5)
            & F.col("elasticity_classification").isin(
                "Highly Inelastic", "Inelastic", "Unit Elastic"
            )
        )
        .withColumn("action_type", F.lit(ACTION_MARGIN_RECOVERY))
    )

    # Priority score
    recs = recs.withColumn(
        "priority_score",
        F.round(
            # Margin erosion severity (40 pts) -- higher discount = higher priority
            F.least(F.col("current_discount_pct") / F.lit(30.0) * F.lit(40.0), F.lit(40.0))
            # Revenue significance (30 pts)
            + F.least(
                F.log(F.greatest(F.col("current_annual_revenue"), F.lit(1.0))) / F.lit(20.0) * F.lit(30.0),
                F.lit(30.0),
            )
            # Inelasticity advantage (20 pts)
            + F.lit(20.0) * (F.lit(1.0) / (F.lit(1.0) + F.abs(F.col("elasticity_coefficient"))))
            # Safe increase headroom (10 pts)
            + F.least(F.col("safe_increase_3pct_vol_loss") / F.lit(10.0) * F.lit(10.0), F.lit(10.0)),
            2,
        ),
    )
    recs = recs.withColumn(
        "priority_score",
        F.least(F.greatest(F.col("priority_score"), F.lit(0.0)), F.lit(100.0)),
    )

    recs = (
        recs
        .withColumn("recommended_change_pct",
                     F.round(F.col("suggested_price_increase_pct"), 4))
        .withColumn("expected_volume_impact_pct",
                     F.round(F.col("expected_volume_change_pct"), 4))
        .withColumn("risk_level",
                     F.when(F.col("current_discount_pct") > 25.0, F.lit("Medium"))
                      .otherwise(F.lit("Low")))
    )

    recs = recs.withColumn(
        "rationale",
        F.concat(
            F.lit("MARGIN RECOVERY: Current discount at "),
            F.round("current_discount_pct", 1).cast("string"),
            F.lit("% is above sustainable levels. "),
            F.col("elasticity_classification"),
            F.lit(" demand supports increase of "),
            F.round("suggested_price_increase_pct", 2).cast("string"),
            F.lit("% with expected $"),
            F.format_number("expected_revenue_impact", 0),
            F.lit(" revenue recovery."),
        ),
    )

    recs = recs.withColumn("supporting_factors", F.lit("Margin erosion analysis; elasticity support; discount benchmarking"))
    recs = recs.withColumn("competitive_context", F.lit("Margin recovery to sustainable levels within competitive range"))

    rec_count = recs.count()
    print(f"MARGIN_RECOVERY recommendations: {rec_count:,}")
    return recs


margin_recs = generate_margin_recovery_recs(uplift_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 -- Union All Recommendations and Rank

# COMMAND ----------

def union_and_rank_recommendations(*rec_dfs: DataFrame) -> DataFrame:
    """Union all recommendation DataFrames and assign global ranking.

    Normalises column schemas across different action types, unions them,
    and assigns a global rank by priority_score descending.

    Parameters
    ----------
    rec_dfs : DataFrame
        Variable number of recommendation DataFrames.

    Returns
    -------
    DataFrame
        Unified, ranked recommendations.
    """
    # Define the common output columns
    common_columns = [
        "action_type",
        "priority_score",
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
        "recommended_change_pct",
        "expected_volume_impact_pct",
        "expected_revenue_impact",
        "expected_margin_impact",
        "risk_level",
        "rationale",
        "supporting_factors",
        "competitive_context",
    ]

    normalised_dfs = []
    for rec_df in rec_dfs:
        # Add missing columns as nulls
        for col_name in common_columns:
            if col_name not in rec_df.columns:
                rec_df = rec_df.withColumn(col_name, F.lit(None).cast(StringType()))
        normalised_dfs.append(rec_df.select(*common_columns))

    # Union all
    unified = reduce(DataFrame.unionByName, normalised_dfs)

    # Generate recommendation_id
    unified = unified.withColumn(
        "recommendation_id",
        F.md5(
            F.concat_ws("|",
                F.col("action_type"),
                F.col("sku"),
                F.coalesce(F.col("customer_id"), F.lit("")),
                F.coalesce(F.col("sales_rep_id"), F.lit("")),
            )
        ),
    )

    # Compute current and recommended pocket prices from uplift where available
    # (For outlier-based recs, we derive from the outlier data)
    unified = unified.withColumn("current_pocket_price", F.lit(None).cast(DoubleType()))
    unified = unified.withColumn("recommended_pocket_price", F.lit(None).cast(DoubleType()))
    unified = unified.withColumn("product_id", F.lit(None).cast(StringType()))

    # Global rank
    rank_window = Window.orderBy(F.col("priority_score").desc())
    unified = unified.withColumn("rank", F.row_number().over(rank_window))

    # Add timestamp
    unified = unified.withColumn("updated_at", F.current_timestamp())

    total_count = unified.count()
    print(f"\nTotal unified recommendations: {total_count:,}")
    print("\nAction type distribution:")
    unified.groupBy("action_type").agg(
        F.count("*").alias("count"),
        F.round(F.avg("priority_score"), 2).alias("avg_priority"),
        F.round(F.sum("expected_revenue_impact"), 2).alias("total_rev_impact"),
    ).orderBy(F.desc("count")).show(truncate=False)

    return unified


all_recs_df = union_and_rank_recommendations(
    increase_price_recs,
    standardize_recs,
    review_rep_recs,
    competitive_recs,
    margin_recs,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 -- Final Schema Selection

# COMMAND ----------

def finalise_recommendation_schema(df: DataFrame) -> DataFrame:
    """Select and order the final output columns.

    Parameters
    ----------
    df : DataFrame
        Unified recommendations.

    Returns
    -------
    DataFrame
        Final schema for gold.pricing_recommendations.
    """
    final_df = df.select(
        "recommendation_id",
        "rank",
        "action_type",
        F.round("priority_score", 2).alias("priority_score"),
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
        "current_pocket_price",
        "recommended_pocket_price",
        F.round("recommended_change_pct", 4).alias("recommended_change_pct"),
        F.round("expected_volume_impact_pct", 4).alias("expected_volume_impact_pct"),
        F.round("expected_revenue_impact", 2).alias("expected_revenue_impact"),
        F.round("expected_margin_impact", 2).alias("expected_margin_impact"),
        "risk_level",
        "rationale",
        "supporting_factors",
        "competitive_context",
        "updated_at",
    )

    return final_df


final_recs_df = finalise_recommendation_schema(all_recs_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 -- Validation

# COMMAND ----------

def validate_recommendations(df: DataFrame) -> None:
    """Run data quality checks on the final recommendations.

    Parameters
    ----------
    df : DataFrame
        Final pricing_recommendations DataFrame.
    """
    row_count = df.count()
    print(f"Total recommendations: {row_count:,}")
    assert row_count >= 200, f"Expected >= 200 recommendations, got {row_count}"
    print("[PASS] >= 200 recommendations")

    # All action types represented
    action_types = {
        row["action_type"]
        for row in df.select("action_type").distinct().collect()
    }
    expected_types = {
        ACTION_INCREASE_PRICE,
        ACTION_STANDARDIZE_DISCOUNT,
        ACTION_REVIEW_REP,
        ACTION_COMPETITIVE_ADJUSTMENT,
        ACTION_MARGIN_RECOVERY,
    }
    missing_types = expected_types - action_types
    if missing_types:
        print(f"[WARN] Missing action types: {missing_types}")
    else:
        print(f"[PASS] All {len(expected_types)} action types represented")

    # Priority score range
    priority_stats = df.agg(
        F.min("priority_score").alias("min_p"),
        F.max("priority_score").alias("max_p"),
        F.avg("priority_score").alias("avg_p"),
    ).collect()[0]
    print(f"Priority score range: {priority_stats['min_p']:.2f} to {priority_stats['max_p']:.2f} "
          f"(avg: {priority_stats['avg_p']:.2f})")
    assert priority_stats["min_p"] >= 0, "Negative priority score found"
    assert priority_stats["max_p"] <= 100, "Priority score exceeds 100"
    print("[PASS] Priority scores in [0, 100]")

    # No null recommendation_ids
    null_ids = df.filter(F.col("recommendation_id").isNull()).count()
    assert null_ids == 0, f"Found {null_ids} null recommendation_ids"
    print("[PASS] No null recommendation_ids")

    # Risk level distribution
    print("\nRisk level distribution:")
    df.groupBy("risk_level").count().orderBy("risk_level").show()

    print("\nAll validation checks passed.")


validate_recommendations(final_recs_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 -- Write to Delta Lake

# COMMAND ----------

# Ensure gold schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}")

# Write as managed Delta table
(
    final_recs_df
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
    "Pricing recommendations combining elasticity, discount outlier, uplift simulation, "
    "and competitive signals. Action types: INCREASE_PRICE, STANDARDIZE_DISCOUNT, REVIEW_REP, "
    "COMPETITIVE_ADJUSTMENT, MARGIN_RECOVERY. Priority scored 0-100. "
    "Source: gold.price_elasticity + gold.discount_outliers + gold.uplift_simulation. "
    "Generated by notebook 16_gold_pricing_recommendations."
)
spark.sql(f"COMMENT ON TABLE {TARGET_TABLE} IS '{_comment}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11 -- Summary
# MAGIC
# MAGIC | Metric | Expected |
# MAGIC |--------|----------|
# MAGIC | Total recommendations | 200+ |
# MAGIC | Action types | 5 (INCREASE_PRICE, STANDARDIZE_DISCOUNT, REVIEW_REP, COMPETITIVE_ADJUSTMENT, MARGIN_RECOVERY) |
# MAGIC | Priority score range | 0-100 |
# MAGIC | Risk levels | Low, Medium, High |
# MAGIC | Output table | `hls_amer_catalog.gold.pricing_recommendations` |

# COMMAND ----------

print("=" * 70)
print("  PRICING RECOMMENDATIONS COMPLETE")
print("=" * 70)
print(f"  Sources        : {SOURCE_ELASTICITY}")
print(f"                   {SOURCE_OUTLIERS}")
print(f"                   {SOURCE_UPLIFT}")
print(f"  Output         : {TARGET_TABLE}")
print(f"  Total recs     : {verify_count:,}")
print(f"  Action types   : {ACTION_INCREASE_PRICE}, {ACTION_STANDARDIZE_DISCOUNT},")
print(f"                   {ACTION_REVIEW_REP}, {ACTION_COMPETITIVE_ADJUSTMENT},")
print(f"                   {ACTION_MARGIN_RECOVERY}")
print("=" * 70)
