# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 17 - Gold Layer: Top 100 Recommended Price Changes
# MAGIC
# MAGIC **Purpose**: Curate the top 100 highest-priority pricing actions from the
# MAGIC full recommendations table. This is the "executive action list" -- a
# MAGIC filterable, dashboard-ready table enriched with all dimension columns
# MAGIC and a quick-action summary for each recommendation.
# MAGIC
# MAGIC **Algorithm**:
# MAGIC 1. Pull from `gold.pricing_recommendations`
# MAGIC 2. Filter to actionable types: INCREASE_PRICE, STANDARDIZE_DISCOUNT, MARGIN_RECOVERY
# MAGIC 3. Filter to High or Medium confidence where available
# MAGIC 4. Rank by priority_score DESC, take top 100
# MAGIC 5. Enrich with all filter dimensions from upstream tables
# MAGIC 6. Generate quick_action_summary column
# MAGIC
# MAGIC **Output**: `hls_amer_catalog.gold.top100_price_changes`
# MAGIC
# MAGIC **Expected**: Exactly 100 rows (or fewer if fewer qualifying recommendations exist)
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
GOLD_SCHEMA = "gold"

SOURCE_RECS = f"{CATALOG}.{GOLD_SCHEMA}.pricing_recommendations"
SOURCE_ELASTICITY = f"{CATALOG}.{GOLD_SCHEMA}.price_elasticity"
SOURCE_OUTLIERS = f"{CATALOG}.{GOLD_SCHEMA}.discount_outliers"
SOURCE_UPLIFT = f"{CATALOG}.{GOLD_SCHEMA}.uplift_simulation"

TARGET_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.top100_price_changes"

# Actionable recommendation types to include
ACTIONABLE_TYPES = [
    "INCREASE_PRICE",
    "STANDARDIZE_DISCOUNT",
    "MARGIN_RECOVERY",
]

TOP_N = 100

spark = SparkSession.builder.getOrCreate()

print(f"Source recs   : {SOURCE_RECS}")
print(f"Target table  : {TARGET_TABLE}")
print(f"Top N         : {TOP_N}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 -- Read Source Data

# COMMAND ----------

def read_recommendations(table_name: str) -> DataFrame:
    """Read the pricing recommendations table.

    Parameters
    ----------
    table_name : str
        Fully qualified table name.

    Returns
    -------
    DataFrame
        Pricing recommendations.
    """
    df = spark.read.table(table_name)
    row_count = df.count()
    print(f"Total recommendations: {row_count:,}")

    # Show action type distribution
    print("\nAction type distribution (before filter):")
    df.groupBy("action_type").count().orderBy(F.desc("count")).show()

    return df


recs_df = read_recommendations(SOURCE_RECS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 -- Filter to Actionable Recommendations

# COMMAND ----------

def filter_actionable_recs(df: DataFrame) -> DataFrame:
    """Filter recommendations to actionable types with sufficient confidence.

    Keeps only INCREASE_PRICE, STANDARDIZE_DISCOUNT, and MARGIN_RECOVERY
    action types. These represent direct pricing actions that can be
    implemented without extensive strategic review (unlike REVIEW_REP
    and COMPETITIVE_ADJUSTMENT which require broader analysis).

    Parameters
    ----------
    df : DataFrame
        Full recommendations DataFrame.

    Returns
    -------
    DataFrame
        Filtered to actionable recommendations.
    """
    filtered = df.filter(
        F.col("action_type").isin(*ACTIONABLE_TYPES)
    )

    filtered_count = filtered.count()
    print(f"Actionable recommendations: {filtered_count:,}")

    return filtered


actionable_df = filter_actionable_recs(recs_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 -- Enrich with Upstream Dimensions
# MAGIC
# MAGIC Join with elasticity, outlier, and uplift data to bring in all
# MAGIC filter dimensions needed for the dashboard.

# COMMAND ----------

def enrich_with_dimensions(
    recs_df: DataFrame,
) -> DataFrame:
    """Enrich recommendations with additional dimensions from upstream tables.

    Brings in:
    - From elasticity: elasticity_coefficient, classification, confidence
    - From outliers: discount_gap, peer_avg_discount
    - From uplift: current pricing, volume, margin details

    Parameters
    ----------
    recs_df : DataFrame
        Filtered recommendations.

    Returns
    -------
    DataFrame
        Enriched with all filter dimensions.
    """
    # Read upstream tables
    elasticity = spark.read.table(SOURCE_ELASTICITY).select(
        "sku", "customer_segment",
        F.col("elasticity_coefficient").alias("e_elasticity_coefficient"),
        F.col("elasticity_classification").alias("e_elasticity_class"),
        F.col("confidence_level").alias("e_confidence_level"),
    )

    outliers = spark.read.table(SOURCE_OUTLIERS).select(
        "sku", "customer_segment", "sales_rep_id",
        F.col("discount_gap").alias("o_discount_gap"),
        F.col("peer_avg_discount_pct").alias("o_peer_avg_discount"),
        F.col("sales_rep_territory").alias("o_sales_rep_territory"),
    )

    uplift = spark.read.table(SOURCE_UPLIFT).select(
        "sku", "customer_id", "sales_rep_id",
        F.col("current_avg_pocket_price").alias("u_current_pocket_price"),
        F.col("current_avg_list_price").alias("u_current_list_price"),
        F.col("current_discount_pct").alias("u_current_discount_pct"),
        F.col("current_annual_revenue").alias("u_current_annual_revenue"),
        F.col("current_annual_volume").alias("u_current_annual_volume"),
        F.col("suggested_price_increase_pct").alias("u_suggested_increase_pct"),
        F.col("suggested_new_pocket_price").alias("u_suggested_new_price"),
    )

    # Join elasticity (by sku + segment)
    enriched = recs_df.join(
        elasticity,
        on=["sku", "customer_segment"],
        how="left",
    )

    # Join outliers (by sku + segment + rep)
    enriched = enriched.join(
        outliers,
        on=["sku", "customer_segment", "sales_rep_id"],
        how="left",
    )

    # Join uplift (by sku + customer + rep)
    enriched = enriched.join(
        uplift,
        on=["sku", "customer_id", "sales_rep_id"],
        how="left",
    )

    # Coalesce enriched columns with recommendation columns
    enriched = (
        enriched
        .withColumn("current_pocket_price",
                     F.coalesce(F.col("current_pocket_price"), F.col("u_current_pocket_price")))
        .withColumn("current_discount_pct",
                     F.coalesce(F.col("u_current_discount_pct"), F.lit(0.0)))
        .withColumn("current_annual_revenue",
                     F.coalesce(F.col("u_current_annual_revenue"), F.lit(0.0)))
        .withColumn("current_annual_volume",
                     F.coalesce(F.col("u_current_annual_volume"), F.lit(0)))
        .withColumn("current_margin_pct", F.lit(None).cast(DoubleType()))
        .withColumn("recommended_new_price",
                     F.coalesce(F.col("recommended_pocket_price"), F.col("u_suggested_new_price")))
        .withColumn("recommended_new_discount_pct",
                     F.when(
                         F.col("u_current_list_price") > 0,
                         F.round(
                             (1.0 - F.coalesce(F.col("recommended_new_price"), F.col("u_current_pocket_price"))
                              / F.col("u_current_list_price")) * 100.0,
                             2,
                         ),
                     ).otherwise(F.lit(None).cast(DoubleType())))
        .withColumn("expected_volume_change_pct",
                     F.coalesce(F.col("expected_volume_impact_pct"), F.lit(0.0)))
        .withColumn("annualized_revenue_gain",
                     F.round(F.coalesce(F.col("expected_revenue_impact"), F.lit(0.0)), 2))
        .withColumn("elasticity_coefficient",
                     F.coalesce(F.col("e_elasticity_coefficient"), F.lit(None).cast(DoubleType())))
        .withColumn("elasticity_class",
                     F.coalesce(F.col("e_elasticity_class"), F.lit("Unknown")))
        .withColumn("discount_gap",
                     F.coalesce(F.col("o_discount_gap"), F.lit(0.0)))
        .withColumn("peer_avg_discount",
                     F.coalesce(F.col("o_peer_avg_discount"), F.lit(0.0)))
        .withColumn("confidence_level",
                     F.coalesce(F.col("e_confidence_level"), F.lit("Unknown")))
        .withColumn("sales_rep_territory",
                     F.coalesce(F.col("o_sales_rep_territory"), F.lit("Unknown")))
        # Placeholder dimension columns
        .withColumn("product_category", F.coalesce(F.col("product_family"), F.lit("Unknown")))
        .withColumn("customer_tier", F.lit("Standard"))
    )

    print(f"Enriched rows: {enriched.count():,}")
    return enriched


enriched_df = enrich_with_dimensions(actionable_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 -- Rank and Take Top 100

# COMMAND ----------

def rank_and_take_top_n(df: DataFrame, n: int = 100) -> DataFrame:
    """Rank by priority_score descending and take the top N rows.

    Re-assigns rank from 1 to N for the final output.

    Parameters
    ----------
    df : DataFrame
        Enriched, filtered recommendations.
    n : int
        Number of top recommendations to keep.

    Returns
    -------
    DataFrame
        Top N recommendations with rank 1 through N.
    """
    # Rank by priority_score descending, breaking ties by expected_revenue_impact
    rank_window = Window.orderBy(
        F.col("priority_score").desc(),
        F.abs(F.col("expected_revenue_impact")).desc(),
    )

    ranked = df.withColumn("rank", F.row_number().over(rank_window))

    # Take top N
    top_n = ranked.filter(F.col("rank") <= n)

    actual_count = top_n.count()
    print(f"Top {n} recommendations: {actual_count:,} rows")
    return top_n


top100_df = rank_and_take_top_n(enriched_df, TOP_N)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 -- Generate Quick Action Summary

# COMMAND ----------

def generate_quick_action_summary(df: DataFrame) -> DataFrame:
    """Generate a concise quick_action_summary for each recommendation.

    The summary is a single-line, dashboard-friendly text that tells
    the pricing analyst exactly what to do.

    Format: "[ACTION] SKU: {sku} | Customer: {name} | Change: {pct}% | Impact: ${amount}"

    Parameters
    ----------
    df : DataFrame
        Top 100 recommendations.

    Returns
    -------
    DataFrame
        With quick_action_summary column added.
    """
    df = df.withColumn(
        "quick_action_summary",
        F.concat(
            F.lit("["),
            F.col("action_type"),
            F.lit("] "),
            F.col("sku"),
            F.lit(" | "),
            F.col("customer_name"),
            F.lit(" | Change: "),
            F.when(
                F.col("recommended_change_pct").isNotNull(),
                F.concat(
                    F.when(F.col("recommended_change_pct") > 0, F.lit("+")).otherwise(F.lit("")),
                    F.round("recommended_change_pct", 2).cast("string"),
                    F.lit("%"),
                ),
            ).otherwise(F.lit("TBD")),
            F.lit(" | Impact: $"),
            F.format_number(F.coalesce("annualized_revenue_gain", F.lit(0.0)), 0),
            F.lit(" | Risk: "),
            F.coalesce(F.col("risk_level"), F.lit("Unknown")),
        ),
    )

    return df


top100_with_summary_df = generate_quick_action_summary(top100_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 -- Final Schema Selection

# COMMAND ----------

def finalise_top100_schema(df: DataFrame) -> DataFrame:
    """Select and order the final output columns for gold.top100_price_changes.

    Parameters
    ----------
    df : DataFrame
        Top 100 recommendations with all enrichments.

    Returns
    -------
    DataFrame
        Final schema.
    """
    df = df.withColumn("updated_at", F.current_timestamp())

    final_df = df.select(
        "rank",
        "recommendation_id",
        "action_type",
        "customer_country",
        "customer_region",
        "product_family",
        "product_category",
        "business_unit",
        "customer_segment",
        "customer_tier",
        "sales_rep_id",
        "sales_rep_name",
        "sales_rep_territory",
        "sku",
        "product_name",
        "customer_name",
        F.round("current_pocket_price", 2).alias("current_pocket_price"),
        F.round("current_discount_pct", 2).alias("current_discount_pct"),
        F.round("current_annual_revenue", 2).alias("current_annual_revenue"),
        "current_annual_volume",
        "current_margin_pct",
        F.round("recommended_new_price", 2).alias("recommended_new_price"),
        F.round("recommended_change_pct", 4).alias("recommended_change_pct"),
        F.round("recommended_new_discount_pct", 2).alias("recommended_new_discount_pct"),
        F.round("expected_volume_change_pct", 4).alias("expected_volume_change_pct"),
        F.round("expected_revenue_impact", 2).alias("expected_revenue_impact"),
        F.round("expected_margin_impact", 2).alias("expected_margin_impact"),
        F.round("annualized_revenue_gain", 2).alias("annualized_revenue_gain"),
        F.round("elasticity_coefficient", 6).alias("elasticity_coefficient"),
        "elasticity_class",
        F.round("discount_gap", 4).alias("discount_gap"),
        F.round("peer_avg_discount", 4).alias("peer_avg_discount"),
        "confidence_level",
        "risk_level",
        "quick_action_summary",
        "rationale",
        F.round("priority_score", 2).alias("priority_score"),
        "updated_at",
    )

    return final_df


final_top100_df = finalise_top100_schema(top100_with_summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 -- Validation

# COMMAND ----------

def validate_top100(df: DataFrame) -> None:
    """Run data quality checks on the top 100 recommendations.

    Parameters
    ----------
    df : DataFrame
        Final top100_price_changes DataFrame.
    """
    row_count = df.count()
    print(f"Total rows: {row_count:,}")
    assert row_count <= TOP_N, f"Expected <= {TOP_N} rows, got {row_count}"
    print(f"[PASS] Row count <= {TOP_N}")

    # Rank should be sequential 1 to N
    max_rank = df.agg(F.max("rank")).collect()[0][0]
    min_rank = df.agg(F.min("rank")).collect()[0][0]
    assert min_rank == 1, f"Expected min rank = 1, got {min_rank}"
    assert max_rank == row_count, f"Expected max rank = {row_count}, got {max_rank}"
    print(f"[PASS] Ranks sequential from 1 to {row_count}")

    # Priority scores should be descending
    ranks_ordered = df.select("rank", "priority_score").orderBy("rank").collect()
    for i in range(1, len(ranks_ordered)):
        assert ranks_ordered[i]["priority_score"] <= ranks_ordered[i - 1]["priority_score"], (
            f"Priority scores not descending at rank {ranks_ordered[i]['rank']}"
        )
    print("[PASS] Priority scores descending by rank")

    # Action type distribution
    print("\nAction type distribution:")
    df.groupBy("action_type").agg(
        F.count("*").alias("count"),
        F.round(F.avg("priority_score"), 2).alias("avg_priority"),
        F.round(F.sum("annualized_revenue_gain"), 2).alias("total_annual_gain"),
    ).orderBy(F.desc("count")).show(truncate=False)

    # Total annualized revenue gain
    total_gain = df.agg(
        F.sum("annualized_revenue_gain")
    ).collect()[0][0]
    print(f"Total annualized revenue gain: ${total_gain:,.2f}")

    # All rows should have quick_action_summary
    null_summary = df.filter(F.col("quick_action_summary").isNull()).count()
    assert null_summary == 0, f"Found {null_summary} null quick_action_summary values"
    print("[PASS] All rows have quick_action_summary")

    # No null recommendation_ids
    null_ids = df.filter(F.col("recommendation_id").isNull()).count()
    assert null_ids == 0, f"Found {null_ids} null recommendation_ids"
    print("[PASS] No null recommendation_ids")

    # Risk level distribution
    print("\nRisk level distribution:")
    df.groupBy("risk_level").count().orderBy("risk_level").show()

    # Confidence level distribution
    print("Confidence level distribution:")
    df.groupBy("confidence_level").count().orderBy("confidence_level").show()

    # Priority score range
    p_stats = df.agg(
        F.min("priority_score").alias("min_p"),
        F.max("priority_score").alias("max_p"),
    ).collect()[0]
    print(f"Priority score range: {p_stats['min_p']:.2f} to {p_stats['max_p']:.2f}")

    print("\nAll validation checks passed.")


validate_top100(final_top100_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 -- Write to Delta Lake

# COMMAND ----------

# Ensure gold schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}")

# Write as managed Delta table
(
    final_top100_df
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
    "Top 100 highest-priority pricing actions curated from gold.pricing_recommendations. "
    "Filtered to INCREASE_PRICE, STANDARDIZE_DISCOUNT, MARGIN_RECOVERY with High/Medium confidence. "
    "Ranked by priority_score (0-100). Enriched with all filter dimensions and quick_action_summary. "
    "Dashboard-ready executive action list. "
    "Source: gold.pricing_recommendations + gold.price_elasticity + gold.discount_outliers + gold.uplift_simulation. "
    "Generated by notebook 17_gold_top100_recommended_changes."
)
spark.sql(f"COMMENT ON TABLE {TARGET_TABLE} IS '{_comment}'")

# COMMAND ----------

# Show top 10 for quick review
print("\nTop 10 Recommended Price Changes:")
verify_df.select(
    "rank", "action_type", "sku", "customer_name",
    "recommended_change_pct", "annualized_revenue_gain",
    "priority_score", "risk_level",
).orderBy("rank").show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 -- Summary
# MAGIC
# MAGIC | Metric | Expected |
# MAGIC |--------|----------|
# MAGIC | Total rows | 100 (or fewer if insufficient qualifying recs) |
# MAGIC | Action types | INCREASE_PRICE, STANDARDIZE_DISCOUNT, MARGIN_RECOVERY |
# MAGIC | Priority score range | 0-100 (descending by rank) |
# MAGIC | Includes | All filter dimensions, quick_action_summary |
# MAGIC | Output table | `hls_amer_catalog.gold.top100_price_changes` |

# COMMAND ----------

total_gain = verify_df.agg(F.sum("annualized_revenue_gain")).collect()[0][0] or 0

print("=" * 70)
print("  TOP 100 RECOMMENDED PRICE CHANGES COMPLETE")
print("=" * 70)
print(f"  Source         : {SOURCE_RECS}")
print(f"  Output         : {TARGET_TABLE}")
print(f"  Total rows     : {verify_count:,}")
print(f"  Action types   : {', '.join(ACTIONABLE_TYPES)}")
print(f"  Total annual   : ${total_gain:,.2f}")
print(f"  Priority range : {verify_df.agg(F.min('priority_score')).collect()[0][0]:.2f} "
      f"to {verify_df.agg(F.max('priority_score')).collect()[0][0]:.2f}")
print("=" * 70)
