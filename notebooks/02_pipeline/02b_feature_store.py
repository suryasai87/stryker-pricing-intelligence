# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02b - Feature Store Registration & Validation
# MAGIC
# MAGIC **Purpose**: Register gold-layer pricing features into the Databricks Feature Store,
# MAGIC define feature groups with documentation, create feature lookups for training data
# MAGIC assembly, and validate feature distributions for production readiness.
# MAGIC
# MAGIC **Pipeline Stage**: Medallion Pipeline - Gold Layer -> Feature Store
# MAGIC
# MAGIC **Source Table**: `hls_amer_catalog.gold.pricing_features`
# MAGIC
# MAGIC **Key Operations**:
# MAGIC 1. Initialize the Feature Engineering client
# MAGIC 2. Read and validate gold pricing features
# MAGIC 3. Register as a Feature Store table with composite primary keys and timestamp key
# MAGIC 4. Define and document six feature groups (Price, Volume, Competitive, Financial, External, Customer)
# MAGIC 5. Create feature lookups for point-in-time correct training data assembly
# MAGIC 6. Validate feature distributions (nulls, outliers, drift detection)
# MAGIC 7. Log feature metadata and statistics to MLflow
# MAGIC
# MAGIC **Authors**: Pricing Intelligence Team
# MAGIC **Last Updated**: 2026-02-09

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuration & Imports

# COMMAND ----------

# Standard library
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# PySpark
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Databricks Feature Engineering
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

# MLflow for metadata logging
import mlflow

# COMMAND ----------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("feature_store_registration")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuration Parameters

# COMMAND ----------

# -- Catalog and schema configuration --
CATALOG = "hls_amer_catalog"
GOLD_SCHEMA = "gold"
FEATURE_SCHEMA = "feature_store"

# -- Source and destination tables --
SOURCE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.pricing_features"
FEATURE_TABLE = f"{CATALOG}.{FEATURE_SCHEMA}.pricing_features_fs"

# -- Feature Store registration keys --
PRIMARY_KEYS = ["product_id", "month"]
TIMESTAMP_KEY = "month"

# -- Validation thresholds --
NULL_THRESHOLD_PCT = 5.0        # Max allowable null percentage per feature
OUTLIER_STD_THRESHOLD = 4.0     # Number of std deviations to flag as outlier
OUTLIER_MAX_PCT = 2.0           # Max allowable outlier percentage per feature
DRIFT_PSI_THRESHOLD = 0.2       # Population Stability Index threshold for drift

# -- MLflow experiment --
MLFLOW_EXPERIMENT = f"/Shared/stryker-pricing-intelligence/feature_store_validation"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Group Definitions
# MAGIC
# MAGIC Each feature group is documented with its constituent features, descriptions,
# MAGIC expected data types, and valid ranges for validation.

# COMMAND ----------

FEATURE_GROUPS: Dict[str, Dict[str, Any]] = {
    "price": {
        "description": (
            "Price-related features capturing current pricing levels, "
            "rate of change, and positioning relative to contractual bounds."
        ),
        "features": {
            "current_asp": {
                "description": "Current average selling price for the product in the period",
                "dtype": "double",
                "valid_range": (0.0, 1_000_000.0),
                "unit": "USD",
            },
            "price_delta_pct": {
                "description": "Period-over-period percentage change in ASP",
                "dtype": "double",
                "valid_range": (-100.0, 500.0),
                "unit": "percent",
            },
            "price_vs_floor": {
                "description": "Ratio of current ASP to contractual price floor (>1 means above floor)",
                "dtype": "double",
                "valid_range": (0.0, 10.0),
                "unit": "ratio",
            },
            "price_vs_ceiling": {
                "description": "Ratio of current ASP to contractual price ceiling (<1 means below ceiling)",
                "dtype": "double",
                "valid_range": (0.0, 10.0),
                "unit": "ratio",
            },
            "price_realization_pct": {
                "description": "Percentage of list price actually realized after discounts and rebates",
                "dtype": "double",
                "valid_range": (0.0, 150.0),
                "unit": "percent",
            },
        },
    },
    "volume": {
        "description": (
            "Volume and demand features capturing order quantities, trends, "
            "and seasonality patterns."
        ),
        "features": {
            "units_3mo_avg": {
                "description": "Rolling 3-month average unit volume",
                "dtype": "double",
                "valid_range": (0.0, 10_000_000.0),
                "unit": "units",
            },
            "units_yoy_change": {
                "description": "Year-over-year percentage change in unit volume",
                "dtype": "double",
                "valid_range": (-100.0, 1000.0),
                "unit": "percent",
            },
            "seasonal_index": {
                "description": "Seasonal adjustment index (1.0 = no seasonal effect)",
                "dtype": "double",
                "valid_range": (0.0, 5.0),
                "unit": "index",
            },
            "volume_trend_3mo": {
                "description": "3-month linear trend coefficient for volume (positive = growing)",
                "dtype": "double",
                "valid_range": (-1.0, 1.0),
                "unit": "coefficient",
            },
        },
    },
    "competitive": {
        "description": (
            "Competitive landscape features capturing relative positioning, "
            "market dynamics, and switching economics."
        ),
        "features": {
            "competitor_asp_gap": {
                "description": "Percentage gap between our ASP and the nearest competitor ASP",
                "dtype": "double",
                "valid_range": (-100.0, 200.0),
                "unit": "percent",
            },
            "market_share_trend": {
                "description": "Trailing 3-month trend in market share (positive = gaining share)",
                "dtype": "double",
                "valid_range": (-1.0, 1.0),
                "unit": "coefficient",
            },
            "innovation_gap": {
                "description": "Product innovation score relative to competitive set (0=lagging, 1=leading)",
                "dtype": "double",
                "valid_range": (0.0, 1.0),
                "unit": "score",
            },
            "switching_cost_index": {
                "description": "Estimated switching cost index for customers (higher = stickier)",
                "dtype": "double",
                "valid_range": (0.0, 10.0),
                "unit": "index",
            },
        },
    },
    "financial": {
        "description": (
            "Financial and margin features capturing cost structure, "
            "discount exposure, and profitability metrics."
        ),
        "features": {
            "cogs_pct": {
                "description": "Cost of goods sold as a percentage of revenue",
                "dtype": "double",
                "valid_range": (0.0, 100.0),
                "unit": "percent",
            },
            "gross_margin_pct": {
                "description": "Gross margin percentage (revenue minus COGS divided by revenue)",
                "dtype": "double",
                "valid_range": (-50.0, 100.0),
                "unit": "percent",
            },
            "discount_depth": {
                "description": "Average discount depth as percentage off list price",
                "dtype": "double",
                "valid_range": (0.0, 100.0),
                "unit": "percent",
            },
            "rebate_exposure": {
                "description": "Total rebate obligation as percentage of revenue",
                "dtype": "double",
                "valid_range": (0.0, 50.0),
                "unit": "percent",
            },
            "freight_pct": {
                "description": "Freight and logistics cost as a percentage of revenue",
                "dtype": "double",
                "valid_range": (0.0, 30.0),
                "unit": "percent",
            },
        },
    },
    "external": {
        "description": (
            "Macroeconomic and external environment features affecting "
            "pricing strategy and cost structure."
        ),
        "features": {
            "tariff_impact": {
                "description": "Estimated tariff impact as percentage of COGS",
                "dtype": "double",
                "valid_range": (0.0, 50.0),
                "unit": "percent",
            },
            "cpi_medical_trend": {
                "description": "Medical CPI trend (annualized rate of change)",
                "dtype": "double",
                "valid_range": (-5.0, 20.0),
                "unit": "percent",
            },
            "supply_chain_pressure": {
                "description": "Supply chain pressure index (0=no pressure, 1=severe disruption)",
                "dtype": "double",
                "valid_range": (0.0, 1.0),
                "unit": "index",
            },
            "fx_impact": {
                "description": "Foreign exchange impact on revenue as a percentage",
                "dtype": "double",
                "valid_range": (-20.0, 20.0),
                "unit": "percent",
            },
            "hospital_capex_trend": {
                "description": "Hospital capital expenditure trend index (>1 = expanding)",
                "dtype": "double",
                "valid_range": (0.5, 2.0),
                "unit": "index",
            },
        },
    },
    "customer": {
        "description": (
            "Customer concentration, channel mix, and contract structure features "
            "influencing pricing flexibility and risk."
        ),
        "features": {
            "gpo_concentration": {
                "description": "Revenue concentration through GPO contracts (0-1 scale)",
                "dtype": "double",
                "valid_range": (0.0, 1.0),
                "unit": "ratio",
            },
            "contract_tier_mix": {
                "description": "Weighted average contract tier (1=lowest tier, 5=highest tier)",
                "dtype": "double",
                "valid_range": (1.0, 5.0),
                "unit": "tier",
            },
            "customer_segment_distribution": {
                "description": "Herfindahl index of customer segment concentration (0=diverse, 1=concentrated)",
                "dtype": "double",
                "valid_range": (0.0, 1.0),
                "unit": "index",
            },
            "direct_channel_pct": {
                "description": "Percentage of revenue from direct sales channel vs. distribution",
                "dtype": "double",
                "valid_range": (0.0, 100.0),
                "unit": "percent",
            },
        },
    },
}

# Build flat list of all feature names for convenience
ALL_FEATURE_NAMES: List[str] = []
for group_cfg in FEATURE_GROUPS.values():
    ALL_FEATURE_NAMES.extend(group_cfg["features"].keys())

logger.info("Defined %d feature groups with %d total features.", len(FEATURE_GROUPS), len(ALL_FEATURE_NAMES))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Initialize Feature Engineering Client

# COMMAND ----------

def get_feature_engineering_client() -> FeatureEngineeringClient:
    """Initialize and return the Databricks Feature Engineering client.

    The client provides APIs for creating, reading, and writing feature tables
    in Unity Catalog. It handles serialization, versioning, and lineage tracking
    automatically.

    Returns:
        FeatureEngineeringClient: Initialized client instance.
    """
    fe_client = FeatureEngineeringClient()
    logger.info("Feature Engineering client initialized successfully.")
    return fe_client


fe = get_feature_engineering_client()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Read Gold Pricing Features

# COMMAND ----------

def read_gold_features(
    table_name: str,
    validate_schema: bool = True,
) -> DataFrame:
    """Read the gold-layer pricing features table and perform initial validation.

    Reads from the specified Unity Catalog table, validates that all expected
    feature columns are present, and logs summary statistics.

    Args:
        table_name: Fully qualified table name (catalog.schema.table).
        validate_schema: If True, verify all expected feature columns exist.

    Returns:
        DataFrame: The pricing features DataFrame.

    Raises:
        ValueError: If required columns are missing from the source table.
    """
    spark = SparkSession.builder.getOrCreate()

    logger.info("Reading gold features from: %s", table_name)
    df = spark.read.table(table_name)

    row_count = df.count()
    col_count = len(df.columns)
    logger.info("Loaded %d rows and %d columns from %s.", row_count, col_count, table_name)

    if validate_schema:
        source_columns = set(df.columns)
        required_columns = set(PRIMARY_KEYS + ALL_FEATURE_NAMES)
        missing = required_columns - source_columns
        if missing:
            raise ValueError(
                f"Source table is missing required columns: {sorted(missing)}. "
                f"Available columns: {sorted(source_columns)}"
            )
        logger.info("Schema validation passed. All %d required columns present.", len(required_columns))

    return df

# COMMAND ----------

gold_df = read_gold_features(SOURCE_TABLE)
display(gold_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Register Feature Store Table

# COMMAND ----------

def ensure_schema_exists(catalog: str, schema: str) -> None:
    """Create the feature store schema if it does not already exist.

    Args:
        catalog: Unity Catalog name.
        schema: Schema name to create.
    """
    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    logger.info("Ensured schema exists: %s.%s", catalog, schema)


def register_feature_table(
    fe_client: FeatureEngineeringClient,
    df: DataFrame,
    feature_table_name: str,
    primary_keys: List[str],
    timestamp_key: Optional[str] = None,
    description: Optional[str] = None,
) -> None:
    """Register a DataFrame as a Feature Store table in Unity Catalog.

    If the feature table already exists, this performs an upsert (merge) operation.
    If it does not exist, it creates a new feature table with the specified keys.

    Point-in-time correctness is enabled via the timestamp_key parameter, which
    ensures that feature lookups during training only use features available at
    the time of each training example.

    Args:
        fe_client: Initialized FeatureEngineeringClient.
        df: DataFrame containing feature data to register.
        feature_table_name: Fully qualified destination table name.
        primary_keys: List of column names forming the composite primary key.
        timestamp_key: Column name used for point-in-time lookups.
        description: Human-readable description of the feature table.
    """
    spark = SparkSession.builder.getOrCreate()

    table_exists = spark.catalog.tableExists(feature_table_name)

    if table_exists:
        logger.info("Feature table '%s' already exists. Performing upsert.", feature_table_name)
        fe_client.write_table(
            name=feature_table_name,
            df=df,
            mode="merge",
        )
        logger.info("Upserted %d rows into existing feature table.", df.count())
    else:
        logger.info("Creating new feature table: %s", feature_table_name)
        fe_client.create_table(
            name=feature_table_name,
            primary_keys=primary_keys,
            timestamp_keys=[timestamp_key] if timestamp_key else None,
            df=df,
            description=description or "Stryker Pricing Intelligence - Gold Pricing Features",
        )
        logger.info(
            "Created feature table '%s' with primary_keys=%s, timestamp_key=%s.",
            feature_table_name,
            primary_keys,
            timestamp_key,
        )

# COMMAND ----------

# Ensure the feature_store schema exists
ensure_schema_exists(CATALOG, FEATURE_SCHEMA)

# Register the feature table
register_feature_table(
    fe_client=fe,
    df=gold_df,
    feature_table_name=FEATURE_TABLE,
    primary_keys=PRIMARY_KEYS,
    timestamp_key=TIMESTAMP_KEY,
    description=(
        "Production feature table for Stryker pricing intelligence. Contains 27 features "
        "across 6 groups (price, volume, competitive, financial, external, customer). "
        "Primary keys: product_id + month. Timestamp key: month for point-in-time lookups."
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Document Feature Groups with Tags and Comments
# MAGIC
# MAGIC Apply column-level comments and table-level tags so that features are
# MAGIC discoverable and self-documenting inside Unity Catalog.

# COMMAND ----------

def apply_feature_documentation(
    catalog: str,
    schema: str,
    table: str,
    feature_groups: Dict[str, Dict[str, Any]],
) -> None:
    """Apply column comments and table tags to document feature groups in Unity Catalog.

    Sets SQL-level COMMENT on each feature column with its description, and
    applies ALTER TABLE SET TAGS to categorize features by group.

    Args:
        catalog: Unity Catalog name.
        schema: Schema name.
        table: Table name.
        feature_groups: Feature group definitions dictionary.
    """
    spark = SparkSession.builder.getOrCreate()
    fq_table = f"{catalog}.{schema}.{table}"

    # Apply column-level comments
    comment_count = 0
    for group_name, group_cfg in feature_groups.items():
        for feat_name, feat_meta in group_cfg["features"].items():
            comment_text = (
                f"[{group_name.upper()}] {feat_meta['description']} "
                f"(unit: {feat_meta['unit']}, range: {feat_meta['valid_range']})"
            )
            # Escape single quotes in the comment
            comment_text = comment_text.replace("'", "\\'")
            try:
                spark.sql(f"ALTER TABLE {fq_table} ALTER COLUMN {feat_name} COMMENT '{comment_text}'")
                comment_count += 1
            except Exception as e:
                logger.warning("Could not set comment on column '%s': %s", feat_name, str(e))

    logger.info("Applied comments to %d feature columns.", comment_count)

    # Apply table-level tags for feature groups
    group_names = list(feature_groups.keys())
    tags_json = ", ".join([f"'feature_group_{g}' = 'true'" for g in group_names])
    try:
        spark.sql(
            f"ALTER TABLE {fq_table} SET TAGS ("
            f"'pipeline' = 'stryker-pricing-intelligence', "
            f"'layer' = 'feature_store', "
            f"'feature_count' = '{len(ALL_FEATURE_NAMES)}', "
            f"{tags_json})"
        )
        logger.info("Applied table-level tags to %s.", fq_table)
    except Exception as e:
        logger.warning("Could not set table tags: %s", str(e))

# COMMAND ----------

# Parse table components from fully qualified name
_fs_parts = FEATURE_TABLE.split(".")
apply_feature_documentation(
    catalog=_fs_parts[0],
    schema=_fs_parts[1],
    table=_fs_parts[2],
    feature_groups=FEATURE_GROUPS,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create Feature Lookups for Training Data Assembly
# MAGIC
# MAGIC Feature lookups define how features are joined to training labels at
# MAGIC inference and training time. Point-in-time correctness is enforced
# MAGIC through the timestamp lookup key.

# COMMAND ----------

def build_feature_lookups(
    feature_table_name: str,
    feature_groups: Dict[str, Dict[str, Any]],
    lookup_key: List[str],
    timestamp_lookup_key: Optional[str] = None,
) -> List[FeatureLookup]:
    """Build FeatureLookup objects for each feature group.

    Creates one FeatureLookup per feature group, enabling modular feature
    selection during training. Each lookup uses the same primary key columns
    and optional timestamp key for point-in-time correctness.

    Args:
        feature_table_name: Fully qualified feature table name.
        feature_groups: Feature group definitions dictionary.
        lookup_key: Column names in the training DataFrame to join on.
        timestamp_lookup_key: Column in training DataFrame for point-in-time join.

    Returns:
        List[FeatureLookup]: List of configured FeatureLookup instances.
    """
    lookups = []

    for group_name, group_cfg in feature_groups.items():
        feature_names = list(group_cfg["features"].keys())

        lookup = FeatureLookup(
            table_name=feature_table_name,
            feature_names=feature_names,
            lookup_key=lookup_key,
            timestamp_lookup_key=timestamp_lookup_key,
        )
        lookups.append(lookup)
        logger.info(
            "Created FeatureLookup for group '%s' with %d features: %s",
            group_name,
            len(feature_names),
            feature_names,
        )

    logger.info("Total FeatureLookups created: %d", len(lookups))
    return lookups


def create_training_set(
    fe_client: FeatureEngineeringClient,
    labels_df: DataFrame,
    feature_lookups: List[FeatureLookup],
    label_column: str = "target_price",
    exclude_columns: Optional[List[str]] = None,
) -> Any:
    """Assemble a training set by joining labels with feature lookups.

    Uses the Feature Engineering client to perform point-in-time correct
    joins between the labels DataFrame and the registered feature tables.

    Args:
        fe_client: Initialized FeatureEngineeringClient.
        labels_df: DataFrame containing label data with join keys.
        feature_lookups: List of FeatureLookup objects.
        label_column: Name of the label/target column.
        exclude_columns: Columns to exclude from the final training set.

    Returns:
        TrainingSet: A Databricks TrainingSet object with features and labels.
    """
    training_set = fe_client.create_training_set(
        df=labels_df,
        feature_lookups=feature_lookups,
        label=label_column,
        exclude_columns=exclude_columns or [],
    )
    training_df = training_set.load_df()
    logger.info(
        "Training set assembled: %d rows, %d columns.",
        training_df.count(),
        len(training_df.columns),
    )
    return training_set

# COMMAND ----------

# Build feature lookups (these are reusable across model training notebooks)
feature_lookups = build_feature_lookups(
    feature_table_name=FEATURE_TABLE,
    feature_groups=FEATURE_GROUPS,
    lookup_key=["product_id", "month"],
    timestamp_lookup_key="month",
)

# Display the lookups for verification
for i, fl in enumerate(feature_lookups):
    print(f"Lookup {i+1}: table={fl.table_name}, features={fl.feature_names}, keys={fl.lookup_key}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validate Feature Distributions
# MAGIC
# MAGIC Production-quality validation checks for:
# MAGIC - **Null rates**: Flag features exceeding the null threshold
# MAGIC - **Outliers**: Detect values beyond configurable standard deviation bounds
# MAGIC - **Data drift**: Compute Population Stability Index (PSI) against a reference period

# COMMAND ----------

def validate_null_rates(
    df: DataFrame,
    feature_names: List[str],
    threshold_pct: float = 5.0,
) -> Dict[str, Dict[str, float]]:
    """Check null/missing value rates for each feature.

    Computes the percentage of null values for each feature column and
    flags those exceeding the threshold.

    Args:
        df: Feature DataFrame to validate.
        feature_names: List of feature column names to check.
        threshold_pct: Maximum allowable null percentage.

    Returns:
        Dictionary with per-feature null statistics and pass/fail status.
    """
    total_rows = df.count()
    if total_rows == 0:
        logger.warning("DataFrame is empty. Skipping null validation.")
        return {}

    results = {}
    failed_features = []

    # Compute null counts in a single pass for efficiency
    null_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in feature_names
        if c in df.columns
    ]
    null_counts_row = df.select(null_exprs).collect()[0]

    for feat_name in feature_names:
        if feat_name not in df.columns:
            logger.warning("Feature '%s' not found in DataFrame. Skipping.", feat_name)
            continue

        null_count = null_counts_row[feat_name]
        null_pct = (null_count / total_rows) * 100.0

        passed = null_pct <= threshold_pct
        results[feat_name] = {
            "null_count": int(null_count),
            "null_pct": round(null_pct, 4),
            "threshold_pct": threshold_pct,
            "passed": passed,
        }
        if not passed:
            failed_features.append(feat_name)

    if failed_features:
        logger.warning(
            "NULL VALIDATION FAILED for %d features: %s",
            len(failed_features),
            failed_features,
        )
    else:
        logger.info("Null validation PASSED for all %d features.", len(results))

    return results

# COMMAND ----------

def validate_outliers(
    df: DataFrame,
    feature_groups: Dict[str, Dict[str, Any]],
    std_threshold: float = 4.0,
    max_outlier_pct: float = 2.0,
) -> Dict[str, Dict[str, Any]]:
    """Detect outlier values for numeric features using range and statistical checks.

    Performs two complementary checks:
    1. **Range check**: Flags values outside the configured valid_range.
    2. **Statistical check**: Flags values beyond mean +/- std_threshold * stddev.

    Args:
        df: Feature DataFrame to validate.
        feature_groups: Feature group definitions with valid ranges.
        std_threshold: Number of standard deviations for statistical outlier detection.
        max_outlier_pct: Maximum allowable percentage of outlier values.

    Returns:
        Dictionary with per-feature outlier statistics and pass/fail status.
    """
    total_rows = df.count()
    if total_rows == 0:
        logger.warning("DataFrame is empty. Skipping outlier validation.")
        return {}

    results = {}
    failed_features = []

    for group_name, group_cfg in feature_groups.items():
        for feat_name, feat_meta in group_cfg["features"].items():
            if feat_name not in df.columns:
                continue

            low, high = feat_meta["valid_range"]

            # Compute statistics in a single pass
            stats_row = df.select(
                F.mean(F.col(feat_name)).alias("mean_val"),
                F.stddev(F.col(feat_name)).alias("std_val"),
                F.min(F.col(feat_name)).alias("min_val"),
                F.max(F.col(feat_name)).alias("max_val"),
                F.sum(
                    F.when(
                        (F.col(feat_name) < low) | (F.col(feat_name) > high), 1
                    ).otherwise(0)
                ).alias("range_outlier_count"),
            ).collect()[0]

            mean_val = stats_row["mean_val"] or 0.0
            std_val = stats_row["std_val"] or 0.0
            range_outlier_count = stats_row["range_outlier_count"]
            range_outlier_pct = (range_outlier_count / total_rows) * 100.0

            # Statistical outlier check
            stat_lower = mean_val - std_threshold * std_val
            stat_upper = mean_val + std_threshold * std_val
            stat_outlier_count = df.filter(
                (F.col(feat_name) < stat_lower) | (F.col(feat_name) > stat_upper)
            ).count()
            stat_outlier_pct = (stat_outlier_count / total_rows) * 100.0

            passed = (range_outlier_pct <= max_outlier_pct) and (stat_outlier_pct <= max_outlier_pct)
            results[feat_name] = {
                "group": group_name,
                "min": float(stats_row["min_val"]) if stats_row["min_val"] is not None else None,
                "max": float(stats_row["max_val"]) if stats_row["max_val"] is not None else None,
                "mean": round(float(mean_val), 4),
                "std": round(float(std_val), 4),
                "valid_range": (low, high),
                "range_outlier_count": int(range_outlier_count),
                "range_outlier_pct": round(range_outlier_pct, 4),
                "stat_outlier_count": int(stat_outlier_count),
                "stat_outlier_pct": round(stat_outlier_pct, 4),
                "passed": passed,
            }
            if not passed:
                failed_features.append(feat_name)

    if failed_features:
        logger.warning(
            "OUTLIER VALIDATION FAILED for %d features: %s",
            len(failed_features),
            failed_features,
        )
    else:
        logger.info("Outlier validation PASSED for all %d features.", len(results))

    return results

# COMMAND ----------

def compute_psi(
    reference: DataFrame,
    current: DataFrame,
    feature_name: str,
    num_bins: int = 10,
) -> float:
    """Compute the Population Stability Index (PSI) between two distributions.

    PSI measures how much a feature distribution has shifted between a reference
    period and the current period. Values above 0.1 indicate moderate drift;
    values above 0.2 indicate significant drift requiring investigation.

    PSI = SUM( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )

    Args:
        reference: Reference (baseline) DataFrame.
        current: Current (latest) DataFrame.
        feature_name: Column name to compute PSI for.
        num_bins: Number of equal-width bins for discretization.

    Returns:
        PSI value as a float. Lower is better (0 = identical distributions).
    """
    import numpy as np

    # Collect values (sample if very large)
    MAX_SAMPLE = 100_000
    ref_count = reference.count()
    cur_count = current.count()

    ref_vals = np.array(
        reference.select(feat_name)
        .filter(F.col(feature_name).isNotNull())
        .sample(fraction=min(1.0, MAX_SAMPLE / max(ref_count, 1)))
        .toPandas()[feature_name]
        .values,
        dtype=float,
    )
    cur_vals = np.array(
        current.select(feature_name)
        .filter(F.col(feature_name).isNotNull())
        .sample(fraction=min(1.0, MAX_SAMPLE / max(cur_count, 1)))
        .toPandas()[feature_name]
        .values,
        dtype=float,
    )

    if len(ref_vals) == 0 or len(cur_vals) == 0:
        return 0.0

    # Create bins from the reference distribution
    breakpoints = np.linspace(np.min(ref_vals), np.max(ref_vals), num_bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_hist = np.histogram(ref_vals, bins=breakpoints)[0] / len(ref_vals)
    cur_hist = np.histogram(cur_vals, bins=breakpoints)[0] / len(cur_vals)

    # Avoid division by zero with small epsilon
    eps = 1e-6
    ref_hist = np.clip(ref_hist, eps, None)
    cur_hist = np.clip(cur_hist, eps, None)

    psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
    return float(psi)


def validate_drift(
    df: DataFrame,
    feature_names: List[str],
    timestamp_col: str = "month",
    psi_threshold: float = 0.2,
    reference_months: int = 6,
) -> Dict[str, Dict[str, Any]]:
    """Detect data drift by comparing recent data against a historical reference window.

    Splits the data into a reference period (older) and a current period (recent),
    then computes the Population Stability Index for each feature.

    Args:
        df: Feature DataFrame with a timestamp column.
        feature_names: List of feature columns to check for drift.
        timestamp_col: Column name containing the temporal key.
        psi_threshold: PSI value above which drift is flagged.
        reference_months: Number of months to use as the reference window.

    Returns:
        Dictionary with per-feature PSI values and pass/fail status.
    """
    # Determine the split point
    max_date_row = df.agg(F.max(F.col(timestamp_col)).alias("max_date")).collect()[0]
    max_date = max_date_row["max_date"]

    if max_date is None:
        logger.warning("No valid dates found. Skipping drift validation.")
        return {}

    # Use the last `reference_months` months before the most recent data as reference
    split_date = max_date - timedelta(days=reference_months * 30)

    reference_df = df.filter(F.col(timestamp_col) < split_date)
    current_df = df.filter(F.col(timestamp_col) >= split_date)

    ref_count = reference_df.count()
    cur_count = current_df.count()

    if ref_count < 100 or cur_count < 100:
        logger.warning(
            "Insufficient data for drift detection (reference=%d, current=%d). "
            "Need at least 100 rows in each partition. Skipping.",
            ref_count,
            cur_count,
        )
        return {}

    logger.info(
        "Drift detection: reference period (%d rows) vs current period (%d rows), split at %s.",
        ref_count,
        cur_count,
        split_date,
    )

    results = {}
    drifted_features = []

    for feat_name in feature_names:
        if feat_name not in df.columns:
            continue
        try:
            psi_value = compute_psi(reference_df, current_df, feat_name)
            passed = psi_value < psi_threshold
            results[feat_name] = {
                "psi": round(psi_value, 6),
                "threshold": psi_threshold,
                "passed": passed,
                "drift_level": (
                    "none" if psi_value < 0.1 else "moderate" if psi_value < 0.2 else "significant"
                ),
            }
            if not passed:
                drifted_features.append(feat_name)
        except Exception as e:
            logger.warning("Could not compute PSI for '%s': %s", feat_name, str(e))
            results[feat_name] = {"psi": None, "passed": None, "error": str(e)}

    if drifted_features:
        logger.warning(
            "DRIFT DETECTED in %d features: %s",
            len(drifted_features),
            drifted_features,
        )
    else:
        logger.info("No significant drift detected across %d features.", len(results))

    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run All Validations

# COMMAND ----------

# 6a. Null rate validation
logger.info("=" * 60)
logger.info("RUNNING NULL RATE VALIDATION")
logger.info("=" * 60)
null_results = validate_null_rates(gold_df, ALL_FEATURE_NAMES, threshold_pct=NULL_THRESHOLD_PCT)

# Summarize
null_summary = {
    "total_features": len(null_results),
    "passed": sum(1 for r in null_results.values() if r["passed"]),
    "failed": sum(1 for r in null_results.values() if not r["passed"]),
}
print(f"Null Validation Summary: {json.dumps(null_summary, indent=2)}")

# COMMAND ----------

# 6b. Outlier validation
logger.info("=" * 60)
logger.info("RUNNING OUTLIER VALIDATION")
logger.info("=" * 60)
outlier_results = validate_outliers(
    gold_df, FEATURE_GROUPS, std_threshold=OUTLIER_STD_THRESHOLD, max_outlier_pct=OUTLIER_MAX_PCT,
)

# Summarize
outlier_summary = {
    "total_features": len(outlier_results),
    "passed": sum(1 for r in outlier_results.values() if r["passed"]),
    "failed": sum(1 for r in outlier_results.values() if not r["passed"]),
}
print(f"Outlier Validation Summary: {json.dumps(outlier_summary, indent=2)}")

# COMMAND ----------

# 6c. Data drift validation
logger.info("=" * 60)
logger.info("RUNNING DATA DRIFT VALIDATION (PSI)")
logger.info("=" * 60)
drift_results = validate_drift(
    gold_df, ALL_FEATURE_NAMES, timestamp_col=TIMESTAMP_KEY, psi_threshold=DRIFT_PSI_THRESHOLD,
)

# Summarize
drift_summary = {
    "total_features": len(drift_results),
    "passed": sum(1 for r in drift_results.values() if r.get("passed") is True),
    "failed": sum(1 for r in drift_results.values() if r.get("passed") is False),
    "skipped": sum(1 for r in drift_results.values() if r.get("passed") is None),
}
print(f"Drift Validation Summary: {json.dumps(drift_summary, indent=2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Log Feature Metadata and Statistics to MLflow

# COMMAND ----------

def log_feature_metadata_to_mlflow(
    experiment_name: str,
    feature_groups: Dict[str, Dict[str, Any]],
    null_results: Dict[str, Dict[str, float]],
    outlier_results: Dict[str, Dict[str, Any]],
    drift_results: Dict[str, Dict[str, Any]],
    feature_table_name: str,
    source_table_name: str,
    row_count: int,
) -> str:
    """Log comprehensive feature store metadata and validation results to MLflow.

    Creates an MLflow run that records:
    - Feature table metadata (source, destination, keys, counts)
    - Per-feature null rates, outlier statistics, and drift PSI values
    - Overall validation pass/fail status
    - Feature group composition as artifacts

    This provides an auditable record of every feature store refresh,
    supporting governance and debugging workflows.

    Args:
        experiment_name: MLflow experiment path.
        feature_groups: Feature group definitions.
        null_results: Output from validate_null_rates.
        outlier_results: Output from validate_outliers.
        drift_results: Output from validate_drift.
        feature_table_name: Destination feature table name.
        source_table_name: Source gold table name.
        row_count: Number of rows in the feature table.

    Returns:
        The MLflow run ID for reference.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"feature_store_refresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id

        # -- Table-level metadata --
        mlflow.log_param("source_table", source_table_name)
        mlflow.log_param("feature_table", feature_table_name)
        mlflow.log_param("primary_keys", json.dumps(PRIMARY_KEYS))
        mlflow.log_param("timestamp_key", TIMESTAMP_KEY)
        mlflow.log_param("total_features", len(ALL_FEATURE_NAMES))
        mlflow.log_param("total_feature_groups", len(feature_groups))
        mlflow.log_param("row_count", row_count)
        mlflow.log_param("refresh_timestamp", datetime.now().isoformat())

        # -- Feature group composition --
        for group_name, group_cfg in feature_groups.items():
            feature_list = list(group_cfg["features"].keys())
            mlflow.log_param(f"group_{group_name}_count", len(feature_list))
            mlflow.set_tag(f"feature_group.{group_name}", json.dumps(feature_list))

        # -- Null validation metrics --
        null_pass_count = 0
        for feat_name, stats in null_results.items():
            mlflow.log_metric(f"null_pct__{feat_name}", stats["null_pct"])
            if stats["passed"]:
                null_pass_count += 1
        mlflow.log_metric("validation_null_passed", null_pass_count)
        mlflow.log_metric("validation_null_failed", len(null_results) - null_pass_count)

        # -- Outlier validation metrics --
        outlier_pass_count = 0
        for feat_name, stats in outlier_results.items():
            mlflow.log_metric(f"outlier_range_pct__{feat_name}", stats["range_outlier_pct"])
            mlflow.log_metric(f"outlier_stat_pct__{feat_name}", stats["stat_outlier_pct"])
            mlflow.log_metric(f"mean__{feat_name}", stats["mean"])
            mlflow.log_metric(f"std__{feat_name}", stats["std"])
            if stats["passed"]:
                outlier_pass_count += 1
        mlflow.log_metric("validation_outlier_passed", outlier_pass_count)
        mlflow.log_metric("validation_outlier_failed", len(outlier_results) - outlier_pass_count)

        # -- Drift validation metrics --
        drift_pass_count = 0
        for feat_name, stats in drift_results.items():
            if stats.get("psi") is not None:
                mlflow.log_metric(f"psi__{feat_name}", stats["psi"])
            if stats.get("passed") is True:
                drift_pass_count += 1
        mlflow.log_metric("validation_drift_passed", drift_pass_count)
        mlflow.log_metric(
            "validation_drift_failed",
            sum(1 for s in drift_results.values() if s.get("passed") is False),
        )

        # -- Overall validation status --
        all_null_passed = all(r["passed"] for r in null_results.values()) if null_results else True
        all_outlier_passed = all(r["passed"] for r in outlier_results.values()) if outlier_results else True
        all_drift_passed = all(
            r.get("passed", True) for r in drift_results.values()
        ) if drift_results else True

        overall_passed = all_null_passed and all_outlier_passed and all_drift_passed
        mlflow.set_tag("validation.overall_status", "PASSED" if overall_passed else "FAILED")
        mlflow.set_tag("validation.null_status", "PASSED" if all_null_passed else "FAILED")
        mlflow.set_tag("validation.outlier_status", "PASSED" if all_outlier_passed else "FAILED")
        mlflow.set_tag("validation.drift_status", "PASSED" if all_drift_passed else "FAILED")

        # -- Log feature group definitions as a JSON artifact --
        feature_group_artifact = json.dumps(
            {
                group_name: {
                    "description": cfg["description"],
                    "features": {
                        fname: {
                            "description": fmeta["description"],
                            "unit": fmeta["unit"],
                            "valid_range": list(fmeta["valid_range"]),
                        }
                        for fname, fmeta in cfg["features"].items()
                    },
                }
                for group_name, cfg in feature_groups.items()
            },
            indent=2,
        )
        mlflow.log_text(feature_group_artifact, "feature_group_definitions.json")

        # -- Log full validation report as artifact --
        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "source_table": source_table_name,
            "feature_table": feature_table_name,
            "row_count": row_count,
            "overall_passed": overall_passed,
            "null_validation": null_results,
            "outlier_validation": {
                k: {kk: vv for kk, vv in v.items() if kk != "valid_range"}
                for k, v in outlier_results.items()
            },
            "drift_validation": drift_results,
        }
        mlflow.log_text(json.dumps(validation_report, indent=2, default=str), "validation_report.json")

        logger.info("MLflow run logged: %s (overall: %s)", run_id, "PASSED" if overall_passed else "FAILED")
        return run_id

# COMMAND ----------

# Log everything to MLflow
row_count = gold_df.count()

mlflow_run_id = log_feature_metadata_to_mlflow(
    experiment_name=MLFLOW_EXPERIMENT,
    feature_groups=FEATURE_GROUPS,
    null_results=null_results,
    outlier_results=outlier_results,
    drift_results=drift_results,
    feature_table_name=FEATURE_TABLE,
    source_table_name=SOURCE_TABLE,
    row_count=row_count,
)

print(f"MLflow Run ID: {mlflow_run_id}")
print(f"Experiment: {MLFLOW_EXPERIMENT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook completed the following:
# MAGIC
# MAGIC | Step | Description | Status |
# MAGIC |------|-------------|--------|
# MAGIC | 1 | Initialized Feature Engineering Client | Done |
# MAGIC | 2 | Read gold pricing features from `hls_amer_catalog.gold.pricing_features` | Done |
# MAGIC | 3 | Registered Feature Store table with PK=`(product_id, month)` and timestamp key=`month` | Done |
# MAGIC | 4 | Documented 6 feature groups (27 features) with column comments and table tags | Done |
# MAGIC | 5 | Created FeatureLookups for point-in-time correct training data assembly | Done |
# MAGIC | 6 | Validated feature distributions (nulls, outliers, drift) | Done |
# MAGIC | 7 | Logged all metadata and validation results to MLflow | Done |
# MAGIC
# MAGIC ### Next Steps
# MAGIC - Use the `feature_lookups` in notebook `03_ml_models` for model training
# MAGIC - Monitor drift metrics over time via the MLflow experiment dashboard
# MAGIC - Set up alerts for validation failures in production scheduling

# COMMAND ----------

# Final status output
print("=" * 60)
print("FEATURE STORE REGISTRATION COMPLETE")
print("=" * 60)
print(f"  Feature Table : {FEATURE_TABLE}")
print(f"  Primary Keys  : {PRIMARY_KEYS}")
print(f"  Timestamp Key : {TIMESTAMP_KEY}")
print(f"  Feature Groups: {len(FEATURE_GROUPS)}")
print(f"  Total Features: {len(ALL_FEATURE_NAMES)}")
print(f"  Total Rows    : {row_count}")
print(f"  MLflow Run    : {mlflow_run_id}")
print("=" * 60)
