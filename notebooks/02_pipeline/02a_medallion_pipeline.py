# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stryker Pricing Intelligence - Medallion Pipeline
# MAGIC
# MAGIC **Spark Declarative Pipeline (formerly DLT) implementing the Bronze -> Silver -> Gold medallion
# MAGIC architecture for the Stryker Pricing Intelligence platform.**
# MAGIC
# MAGIC ## Pipeline Layers
# MAGIC | Layer | Table | Grain | Purpose |
# MAGIC |-------|-------|-------|---------|
# MAGIC | Silver | `silver_fact_sales` | transaction | Cleansed sales with derived pricing metrics and window features |
# MAGIC | Silver | `silver_external_enriched` | month | External market factors with lag features |
# MAGIC | Gold | `gold_pricing_features` | product-month | Final ML feature table joining internal + external signals |
# MAGIC
# MAGIC ## Data Quality
# MAGIC Expectations enforce row-level quality gates. Rows that violate `expect_or_drop` rules are
# MAGIC quarantined automatically by the pipeline runtime and surfaced in the event log for monitoring.
# MAGIC
# MAGIC ## Ownership
# MAGIC - **Team**: Pricing Analytics & Data Engineering
# MAGIC - **Catalog**: `hls_amer_catalog`
# MAGIC - **Update cadence**: Daily (triggered by upstream Bronze ingestion)

# COMMAND ----------

import dlt
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Constants

# COMMAND ----------

# ---------------------------------------------------------------------------
# Source catalog and schema references
# ---------------------------------------------------------------------------
CATALOG = "hls_amer_catalog"
BRONZE_SCHEMA = "bronze"

# Fully-qualified source table paths
TRANSACTIONS_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.stryker_transactions"
PRODUCTS_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.stryker_products"
MARKET_EXTERNAL_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.market_external"
COMPETITOR_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.competitor_pricing"

# Tariff impact weighting factors (sourced from supply-chain cost-structure analysis)
STEEL_TARIFF_WEIGHT = 0.45
TITANIUM_TARIFF_WEIGHT = 0.55

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Silver Layer: `silver_fact_sales`
# MAGIC
# MAGIC Cleansed and enriched sales transactions with derived pricing metrics and
# MAGIC trailing window features. This table serves as the canonical fact table for
# MAGIC all downstream pricing analytics.

# COMMAND ----------

@dlt.table(
    name="silver_fact_sales",
    comment="Cleansed sales transactions with derived pricing metrics, deduplication, and window features.",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true",
        "delta.autoOptimize.optimizeWrite": "true",
    },
)
@dlt.expect_or_drop("valid_price", "list_price > 0")
@dlt.expect_or_drop("valid_units", "units_sold > 0")
@dlt.expect("valid_date", "date IS NOT NULL")
def silver_fact_sales():
    """Build the Silver fact-sales table from Bronze transactions and products.

    Processing steps
    ----------------
    1. **Deduplication** -- Rows are deduplicated on ``transaction_id`` by keeping
       the record with the latest ``_ingested_at`` timestamp.  This guards against
       duplicate deliveries from upstream CDC or batch-load processes.

    2. **Product enrichment** -- A left join to the products dimension attaches
       ``product_name``, ``product_category``, ``cogs_pct``, ``innovation_tier``,
       and ``patent_years_remaining`` onto every transaction row.

    3. **Derived pricing columns**

       * ``margin_pct`` -- Unit-economics margin as
         ``(pocket_price - list_price * cogs_pct) / pocket_price``.
       * ``discount_depth`` -- Depth of the discount waterfall as
         ``(list_price - pocket_price) / list_price``.
       * ``price_realization_pct`` -- Fraction of the list price actually
         realized: ``pocket_price / list_price``.

    4. **Window features** (partitioned by ``product_id``, ordered by ``year_month``)

       * ``yoy_price_change`` -- Year-over-year percentage change in average
         pocket price for the same calendar month in the prior year.
       * ``rolling_3mo_volume`` -- Three-month rolling average of ``units_sold``.
       * ``seasonal_index`` -- Ratio of the month's average pocket price to the
         trailing-12-month average, capturing intra-year seasonality.

    Returns
    -------
    pyspark.sql.DataFrame
        Silver-quality sales fact table at transaction grain.
    """

    # ------------------------------------------------------------------
    # 1. Read Bronze sources
    # ------------------------------------------------------------------
    transactions_raw = spark.read.table(TRANSACTIONS_TABLE)
    products = spark.read.table(PRODUCTS_TABLE)

    # ------------------------------------------------------------------
    # 2. Deduplicate transactions on transaction_id (keep latest ingest)
    # ------------------------------------------------------------------
    dedup_window = Window.partitionBy("transaction_id").orderBy(F.col("_ingested_at").desc())

    transactions = (
        transactions_raw
        .withColumn("_row_num", F.row_number().over(dedup_window))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )

    # ------------------------------------------------------------------
    # 3. Enrich with product dimension
    # ------------------------------------------------------------------
    enriched = transactions.join(
        products.select(
            "product_id",
            "product_name",
            "product_category",
            "cogs_pct",
            "innovation_tier",
            "patent_years_remaining",
        ),
        on="product_id",
        how="left",
    )

    # ------------------------------------------------------------------
    # 4. Derive pricing metrics
    # ------------------------------------------------------------------
    enriched = (
        enriched
        .withColumn(
            "margin_pct",
            F.when(
                F.col("pocket_price") != 0,
                (F.col("pocket_price") - (F.col("list_price") * F.col("cogs_pct")))
                / F.col("pocket_price"),
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
        .withColumn(
            "discount_depth",
            F.when(
                F.col("list_price") != 0,
                (F.col("list_price") - F.col("pocket_price")) / F.col("list_price"),
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
        .withColumn(
            "price_realization_pct",
            F.when(
                F.col("list_price") != 0,
                F.col("pocket_price") / F.col("list_price"),
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
    )

    # Ensure year_month column exists for window operations
    enriched = enriched.withColumn(
        "year_month",
        F.coalesce(F.col("year_month"), F.date_format(F.col("date"), "yyyy-MM")),
    )

    # Numeric sort key for window ordering
    enriched = enriched.withColumn(
        "_ym_sort",
        (
            F.year(F.to_date(F.concat(F.col("year_month"), F.lit("-01")))) * 100
            + F.month(F.to_date(F.concat(F.col("year_month"), F.lit("-01"))))
        ),
    )

    # ------------------------------------------------------------------
    # 5. Window features
    # ------------------------------------------------------------------
    product_time_window = Window.partitionBy("product_id").orderBy("_ym_sort")

    # 5a. Year-over-year price change
    #     Compare current pocket_price to the value 12 months prior.
    enriched = enriched.withColumn(
        "_pocket_price_lag_12",
        F.lag("pocket_price", 12).over(product_time_window),
    ).withColumn(
        "yoy_price_change",
        F.when(
            F.col("_pocket_price_lag_12").isNotNull() & (F.col("_pocket_price_lag_12") != 0),
            (F.col("pocket_price") - F.col("_pocket_price_lag_12"))
            / F.col("_pocket_price_lag_12"),
        ).otherwise(F.lit(None).cast(DoubleType())),
    )

    # 5b. Rolling 3-month average volume
    rolling_3mo_window = (
        Window.partitionBy("product_id")
        .orderBy("_ym_sort")
        .rowsBetween(-2, Window.currentRow)
    )
    enriched = enriched.withColumn(
        "rolling_3mo_volume",
        F.avg("units_sold").over(rolling_3mo_window),
    )

    # 5c. Seasonal index: month avg / trailing 12-month avg
    rolling_12mo_window = (
        Window.partitionBy("product_id")
        .orderBy("_ym_sort")
        .rowsBetween(-11, Window.currentRow)
    )
    enriched = enriched.withColumn(
        "_annual_avg_price",
        F.avg("pocket_price").over(rolling_12mo_window),
    ).withColumn(
        "seasonal_index",
        F.when(
            F.col("_annual_avg_price").isNotNull() & (F.col("_annual_avg_price") != 0),
            F.col("pocket_price") / F.col("_annual_avg_price"),
        ).otherwise(F.lit(1.0)),
    )

    # ------------------------------------------------------------------
    # 6. Clean up internal helper columns
    # ------------------------------------------------------------------
    enriched = enriched.drop(
        "_ym_sort",
        "_pocket_price_lag_12",
        "_annual_avg_price",
    )

    return enriched

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Silver Layer: `silver_external_enriched`
# MAGIC
# MAGIC External market signals (tariffs, CPI, fuel costs, supply-chain stress)
# MAGIC enriched with lag features and composite indices used as macro-economic
# MAGIC controls in downstream pricing models.

# COMMAND ----------

@dlt.table(
    name="silver_external_enriched",
    comment="External market factors with lag features and composite macro indices.",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true",
        "delta.autoOptimize.optimizeWrite": "true",
    },
)
def silver_external_enriched():
    """Build Silver external-enrichment table from Bronze market data.

    Processing steps
    ----------------
    1. **Read** the ``market_external`` Bronze table which contains monthly
       snapshots of macro-economic and supply-chain indicators.

    2. **Lag features** -- For each key indicator (``cpi_medical``,
       ``supply_chain_pressure_index``, ``fuel_index``, ``steel_tariff_pct``,
       ``titanium_tariff_pct``), compute 1-month, 2-month, and 3-month lags
       to give the ML model access to recent trends without data leakage.

    3. **Tariff impact index** -- A weighted composite of steel and titanium
       tariff rates reflecting Stryker's raw-material cost exposure:

       ``tariff_impact_index = 0.45 * steel_tariff_pct + 0.55 * titanium_tariff_pct``

       Weights are derived from the corporate supply-chain cost-structure
       analysis and should be revisited annually.

    4. **Macro pressure score** -- A min-max normalised composite of CPI,
       supply-chain pressure, and fuel index that collapses multiple correlated
       signals into a single feature for regularisation-friendly modelling:

       ``macro_pressure_score = mean(norm(cpi), norm(pressure), norm(fuel))``

    Returns
    -------
    pyspark.sql.DataFrame
        Silver-quality external enrichment table at monthly grain.
    """

    # ------------------------------------------------------------------
    # 1. Read Bronze external data
    # ------------------------------------------------------------------
    external = spark.read.table(MARKET_EXTERNAL_TABLE)

    # Ensure a consistent sort key
    external = external.withColumn(
        "_ym_sort",
        (
            F.year(F.to_date(F.concat(F.col("year_month"), F.lit("-01")))) * 100
            + F.month(F.to_date(F.concat(F.col("year_month"), F.lit("-01"))))
        ),
    )

    time_window = Window.orderBy("_ym_sort")

    # ------------------------------------------------------------------
    # 2. Lag features for key indicators
    # ------------------------------------------------------------------
    lag_columns = [
        "cpi_medical",
        "supply_chain_pressure_index",
        "fuel_index",
        "steel_tariff_pct",
        "titanium_tariff_pct",
    ]

    for col_name in lag_columns:
        for lag_months in [1, 2, 3]:
            external = external.withColumn(
                f"{col_name}_lag{lag_months}",
                F.lag(col_name, lag_months).over(time_window),
            )

    # ------------------------------------------------------------------
    # 3. Tariff impact index (weighted composite)
    # ------------------------------------------------------------------
    external = external.withColumn(
        "tariff_impact_index",
        (
            F.lit(STEEL_TARIFF_WEIGHT) * F.col("steel_tariff_pct")
            + F.lit(TITANIUM_TARIFF_WEIGHT) * F.col("titanium_tariff_pct")
        ),
    )

    # ------------------------------------------------------------------
    # 4. Macro pressure score (min-max normalised composite)
    #    Normalisation uses the full-dataset min/max so the score is
    #    stable across incremental refreshes within a pipeline run.
    # ------------------------------------------------------------------
    norm_cols = ["cpi_medical", "supply_chain_pressure_index", "fuel_index"]
    full_window = Window.orderBy(F.lit(1)).rowsBetween(
        Window.unboundedPreceding, Window.unboundedFollowing
    )

    for col_name in norm_cols:
        col_min = F.min(col_name).over(full_window)
        col_max = F.max(col_name).over(full_window)
        external = external.withColumn(
            f"_norm_{col_name}",
            F.when(
                (col_max - col_min) != 0,
                (F.col(col_name) - col_min) / (col_max - col_min),
            ).otherwise(F.lit(0.5)),
        )

    external = external.withColumn(
        "macro_pressure_score",
        (
            F.col("_norm_cpi_medical")
            + F.col("_norm_supply_chain_pressure_index")
            + F.col("_norm_fuel_index")
        )
        / F.lit(3.0),
    )

    # ------------------------------------------------------------------
    # 5. Clean up internal helper columns
    # ------------------------------------------------------------------
    internal_cols = ["_ym_sort"] + [f"_norm_{c}" for c in norm_cols]
    external = external.drop(*internal_cols)

    return external

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Gold Layer: `gold_pricing_features`
# MAGIC
# MAGIC The final ML feature table at **product-month** grain. This table is the
# MAGIC single entry point for model training, scoring, and feature-store
# MAGIC registration. Every column is a documented, production-ready feature.

# COMMAND ----------

@dlt.table(
    name="gold_pricing_features",
    comment="Final ML feature table at product-month grain joining internal pricing signals with external market factors.",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true",
        "delta.autoOptimize.optimizeWrite": "true",
        "delta.autoOptimize.autoCompact": "true",
    },
)
def gold_pricing_features():
    """Build the Gold ML feature table at product-month grain.

    This table is the authoritative feature store input for the Stryker
    Pricing Intelligence ML models.  It combines:

    * **Internal pricing signals** aggregated from ``silver_fact_sales``
    * **External macro signals** from ``silver_external_enriched``
    * **Competitive intelligence** from ``competitor_pricing``

    Feature groups
    --------------
    **Pricing dynamics**
        ``avg_pocket_price``, ``avg_list_price``, ``price_delta_pct`` (MoM),
        ``discount_depth_avg``, ``price_realization_avg``, ``margin_pct_avg``

    **Volume signals**
        ``total_units_sold``, ``volume_delta_pct`` (MoM),
        ``rolling_3mo_volume_avg``, ``seasonal_index_avg``

    **Competitive positioning**
        ``competitor_asp_gap`` -- gap between Stryker pocket price and the
        competitor average selling price for the same product category and month.

    **Contract & channel mix**
        ``contract_mix_score`` -- proportion of volume under contract vs spot.
        ``gpo_concentration`` -- Herfindahl-Hirschman Index of GPO mix,
        measuring buyer-side concentration risk.
        ``customer_segment_mix`` -- share of volume from top customer segment.

    **Product attributes**
        ``innovation_tier``, ``patent_years_remaining``

    **Market share**
        ``market_share_pct`` -- Stryker units / (Stryker units + competitor units)
        for the product category in the given month.

    **Macro controls**
        ``tariff_impact_index``, ``macro_pressure_score`` (from Silver external).

    Returns
    -------
    pyspark.sql.DataFrame
        Gold-quality feature table at product_id x year_month grain.
    """

    # ------------------------------------------------------------------
    # 1. Read Silver tables
    # ------------------------------------------------------------------
    sales = dlt.read("silver_fact_sales")
    external = dlt.read("silver_external_enriched")

    # Competitor data comes straight from Bronze (simple lookup, no
    # separate Silver table needed).
    competitor = spark.read.table(COMPETITOR_TABLE)

    # ------------------------------------------------------------------
    # 2. Aggregate sales to product-month grain
    # ------------------------------------------------------------------
    sales_agg = (
        sales
        .groupBy("product_id", "product_name", "product_category", "year_month",
                 "innovation_tier", "patent_years_remaining")
        .agg(
            # Pricing
            F.avg("pocket_price").alias("avg_pocket_price"),
            F.avg("list_price").alias("avg_list_price"),
            F.avg("discount_depth").alias("discount_depth_avg"),
            F.avg("price_realization_pct").alias("price_realization_avg"),
            F.avg("margin_pct").alias("margin_pct_avg"),

            # Volume
            F.sum("units_sold").alias("total_units_sold"),
            F.avg("rolling_3mo_volume").alias("rolling_3mo_volume_avg"),
            F.avg("seasonal_index").alias("seasonal_index_avg"),

            # YoY pricing
            F.avg("yoy_price_change").alias("yoy_price_change_avg"),

            # Contract mix: fraction of rows flagged as contract
            F.avg(
                F.when(F.col("contract_flag") == True, F.lit(1.0)).otherwise(F.lit(0.0))
            ).alias("contract_mix_score"),

            # GPO concentration (HHI): sum of squared shares per GPO
            # We first collect GPO-level unit counts, then compute HHI.
            F.sum("units_sold").alias("_total_units_for_hhi"),
            F.collect_list(
                F.struct(
                    F.col("gpo_id").alias("gpo"),
                    F.col("units_sold").alias("units"),
                )
            ).alias("_gpo_units_list"),

            # Customer segment mix: share of dominant segment
            F.collect_list(
                F.struct(
                    F.col("customer_segment").alias("segment"),
                    F.col("units_sold").alias("units"),
                )
            ).alias("_segment_units_list"),
        )
    )

    # ------------------------------------------------------------------
    # 2a. Compute GPO concentration (HHI) via explode + re-aggregate
    # ------------------------------------------------------------------
    gpo_exploded = (
        sales_agg
        .select("product_id", "year_month", F.explode("_gpo_units_list").alias("gpo_rec"), "_total_units_for_hhi")
        .select(
            "product_id",
            "year_month",
            F.col("gpo_rec.gpo").alias("gpo_id"),
            F.col("gpo_rec.units").alias("gpo_units"),
            "_total_units_for_hhi",
        )
        .groupBy("product_id", "year_month", "gpo_id", "_total_units_for_hhi")
        .agg(F.sum("gpo_units").alias("gpo_total_units"))
        .withColumn(
            "gpo_share",
            F.col("gpo_total_units") / F.col("_total_units_for_hhi"),
        )
        .withColumn("gpo_share_sq", F.col("gpo_share") ** 2)
        .groupBy("product_id", "year_month")
        .agg(F.sum("gpo_share_sq").alias("gpo_concentration"))
    )

    sales_agg = sales_agg.join(gpo_exploded, on=["product_id", "year_month"], how="left")

    # ------------------------------------------------------------------
    # 2b. Compute customer segment mix (dominant segment share)
    # ------------------------------------------------------------------
    segment_exploded = (
        sales_agg
        .select("product_id", "year_month", F.explode("_segment_units_list").alias("seg_rec"), "_total_units_for_hhi")
        .select(
            "product_id",
            "year_month",
            F.col("seg_rec.segment").alias("segment"),
            F.col("seg_rec.units").alias("seg_units"),
            F.col("_total_units_for_hhi").alias("total_units"),
        )
        .groupBy("product_id", "year_month", "segment", "total_units")
        .agg(F.sum("seg_units").alias("segment_total_units"))
        .withColumn(
            "segment_share",
            F.col("segment_total_units") / F.col("total_units"),
        )
    )

    # Keep the maximum segment share as the concentration metric
    top_segment = (
        segment_exploded
        .groupBy("product_id", "year_month")
        .agg(F.max("segment_share").alias("customer_segment_mix"))
    )

    sales_agg = sales_agg.join(top_segment, on=["product_id", "year_month"], how="left")

    # Drop internal helper columns
    sales_agg = sales_agg.drop("_gpo_units_list", "_segment_units_list", "_total_units_for_hhi")

    # ------------------------------------------------------------------
    # 3. Month-over-month deltas (price and volume)
    # ------------------------------------------------------------------
    sales_agg = sales_agg.withColumn(
        "_ym_sort",
        (
            F.year(F.to_date(F.concat(F.col("year_month"), F.lit("-01")))) * 100
            + F.month(F.to_date(F.concat(F.col("year_month"), F.lit("-01"))))
        ),
    )

    product_month_window = Window.partitionBy("product_id").orderBy("_ym_sort")

    sales_agg = (
        sales_agg
        .withColumn("_prev_price", F.lag("avg_pocket_price", 1).over(product_month_window))
        .withColumn(
            "price_delta_pct",
            F.when(
                F.col("_prev_price").isNotNull() & (F.col("_prev_price") != 0),
                (F.col("avg_pocket_price") - F.col("_prev_price")) / F.col("_prev_price"),
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
        .withColumn("_prev_volume", F.lag("total_units_sold", 1).over(product_month_window))
        .withColumn(
            "volume_delta_pct",
            F.when(
                F.col("_prev_volume").isNotNull() & (F.col("_prev_volume") != 0),
                (F.col("total_units_sold") - F.col("_prev_volume")) / F.col("_prev_volume"),
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
        .drop("_prev_price", "_prev_volume", "_ym_sort")
    )

    # ------------------------------------------------------------------
    # 4. Competitor ASP gap
    #    Gap = (Stryker avg pocket price - competitor ASP) / competitor ASP
    # ------------------------------------------------------------------
    competitor_agg = (
        competitor
        .groupBy("product_category", "year_month")
        .agg(
            F.avg("competitor_asp").alias("competitor_avg_asp"),
            F.sum("competitor_units").alias("competitor_total_units"),
        )
    )

    sales_agg = sales_agg.join(
        competitor_agg,
        on=["product_category", "year_month"],
        how="left",
    )

    sales_agg = sales_agg.withColumn(
        "competitor_asp_gap",
        F.when(
            F.col("competitor_avg_asp").isNotNull() & (F.col("competitor_avg_asp") != 0),
            (F.col("avg_pocket_price") - F.col("competitor_avg_asp"))
            / F.col("competitor_avg_asp"),
        ).otherwise(F.lit(None).cast(DoubleType())),
    )

    # ------------------------------------------------------------------
    # 5. Market share
    #    Stryker units / (Stryker units + competitor units) at category-month
    # ------------------------------------------------------------------
    sales_agg = sales_agg.withColumn(
        "market_share_pct",
        F.when(
            (F.col("total_units_sold") + F.coalesce(F.col("competitor_total_units"), F.lit(0))) > 0,
            F.col("total_units_sold")
            / (F.col("total_units_sold") + F.coalesce(F.col("competitor_total_units"), F.lit(0))),
        ).otherwise(F.lit(None).cast(DoubleType())),
    )

    # Drop competitor helper columns
    sales_agg = sales_agg.drop("competitor_avg_asp", "competitor_total_units")

    # ------------------------------------------------------------------
    # 6. Join with external enrichment on year_month
    # ------------------------------------------------------------------
    features = sales_agg.join(
        external.select(
            "year_month",
            "tariff_impact_index",
            "macro_pressure_score",
            "cpi_medical",
            "supply_chain_pressure_index",
            "fuel_index",
            "steel_tariff_pct",
            "titanium_tariff_pct",
            # Lag features
            "cpi_medical_lag1",
            "cpi_medical_lag2",
            "cpi_medical_lag3",
            "supply_chain_pressure_index_lag1",
            "supply_chain_pressure_index_lag2",
            "supply_chain_pressure_index_lag3",
            "fuel_index_lag1",
            "fuel_index_lag2",
            "fuel_index_lag3",
        ),
        on="year_month",
        how="left",
    )

    # ------------------------------------------------------------------
    # 7. Final column selection and ordering
    # ------------------------------------------------------------------
    features = features.select(
        # Keys
        "product_id",
        "product_name",
        "product_category",
        "year_month",

        # Pricing dynamics
        "avg_pocket_price",
        "avg_list_price",
        "price_delta_pct",
        "discount_depth_avg",
        "price_realization_avg",
        "margin_pct_avg",

        # Volume signals
        "total_units_sold",
        "volume_delta_pct",
        "rolling_3mo_volume_avg",
        "seasonal_index_avg",
        "yoy_price_change_avg",

        # Competitive positioning
        "competitor_asp_gap",
        "market_share_pct",

        # Contract & channel mix
        "contract_mix_score",
        "gpo_concentration",
        "customer_segment_mix",

        # Product attributes
        "innovation_tier",
        "patent_years_remaining",

        # External / macro
        "tariff_impact_index",
        "macro_pressure_score",
        "cpi_medical",
        "supply_chain_pressure_index",
        "fuel_index",
        "steel_tariff_pct",
        "titanium_tariff_pct",
        "cpi_medical_lag1",
        "cpi_medical_lag2",
        "cpi_medical_lag3",
        "supply_chain_pressure_index_lag1",
        "supply_chain_pressure_index_lag2",
        "supply_chain_pressure_index_lag3",
        "fuel_index_lag1",
        "fuel_index_lag2",
        "fuel_index_lag3",
    )

    return features

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Pipeline Notes
# MAGIC
# MAGIC ### Data Quality Monitoring
# MAGIC - `silver_fact_sales` enforces `expect_or_drop` on `list_price > 0` and `units_sold > 0`.
# MAGIC   Dropped rows are logged in the DLT event log under `flow_progress.data_quality`.
# MAGIC - `valid_date` uses `expect` (warn-only) to surface NULL dates without discarding rows.
# MAGIC
# MAGIC ### Performance Considerations
# MAGIC - All Silver and Gold tables enable `autoOptimize` for write-time compaction.
# MAGIC - The Gold table additionally enables `autoCompact` to maintain optimal file sizes for
# MAGIC   downstream ML reads.
# MAGIC - Window functions in Silver use explicit partitioning by `product_id` to ensure
# MAGIC   parallelism across the product dimension.
# MAGIC
# MAGIC ### Refresh Strategy
# MAGIC - **Full refresh**: Required after schema evolution in Bronze source tables.
# MAGIC - **Incremental**: Default mode. The pipeline processes only new/changed rows
# MAGIC   when Bronze tables use Delta change-data-feed.
