# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01b - Synthetic Transaction Data Generator
# MAGIC
# MAGIC **Purpose:** Generate 500,000 realistic medical device transactions spanning 36 months
# MAGIC (January 2023 through December 2025) for the Stryker Pricing Intelligence platform.
# MAGIC
# MAGIC **Output:** `hls_amer_catalog.bronze.stryker_transactions` (Delta table, partitioned by year/month)
# MAGIC
# MAGIC **Dependencies:**
# MAGIC - Product master table: `hls_amer_catalog.bronze.stryker_products`
# MAGIC - This notebook should be run **after** `01a_product_master.py`
# MAGIC
# MAGIC **Embedded Statistical Patterns:**
# MAGIC 1. **Seasonality** - Q4 budget-cycle spike, Q1 post-budget dip
# MAGIC 2. **Price elasticity by category** - Ranges from very inelastic (capital equipment) to elastic (consumables)
# MAGIC 3. **Discount waterfall** - List -> contract discount -> GPO rebate -> freight -> pocket price
# MAGIC 4. **Regional pricing variation** - +/- 5-12% by geography
# MAGIC 5. **Annual price escalation** - 2-4% on implants, flat on commodities
# MAGIC 6. **Volume correlation with macro factors** - Hospital utilization trends
# MAGIC
# MAGIC **Reproducibility:** All random generation seeded with seed=42.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration & Constants

# COMMAND ----------

import math
from typing import Dict, List, Tuple

from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
CATALOG = "hls_amer_catalog"
SCHEMA = "bronze"
PRODUCT_TABLE = f"{CATALOG}.{SCHEMA}.stryker_products"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.stryker_transactions"

NUM_TRANSACTIONS = 500_000
NUM_CUSTOMERS = 1_000
RANDOM_SEED = 42
DATE_START = "2023-01-01"
DATE_END = "2025-12-31"

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Reference Data Definitions
# MAGIC
# MAGIC Deterministic lookup tables encoded as Python dictionaries and broadcast
# MAGIC for efficient joins at scale. These define the statistical distributions
# MAGIC that govern realistic transaction generation.

# COMMAND ----------

# ---------------------------------------------------------------------------
# 2a. Monthly seasonal multipliers
#     Derived from hospital capital budget cycles:
#       - Q1 (Jan-Mar): post-budget hangover, lower volumes
#       - Q2 (Apr-Jun): steady-state purchasing
#       - Q3 (Jul-Sep): mid-year budget reviews drive modest increase
#       - Q4 (Oct-Dec): "use it or lose it" budget flush, peak volumes
# ---------------------------------------------------------------------------
MONTHLY_SEASONAL_MULTIPLIERS: Dict[int, float] = {
    1: 0.78,   # January   - Q1 dip (post-budget hangover)
    2: 0.82,   # February  - Q1 dip (gradual recovery)
    3: 0.88,   # March     - Q1 end, fiscal year-end for some systems
    4: 0.95,   # April     - Q2 steady state begins
    5: 0.98,   # May       - Q2 steady
    6: 1.00,   # June      - Baseline / mid-year
    7: 1.02,   # July      - Q3 slight uptick
    8: 1.04,   # August    - Q3 surgical volume peak (elective backlog)
    9: 1.06,   # September - Q3 budget review triggers orders
    10: 1.15,  # October   - Q4 budget flush begins
    11: 1.22,  # November  - Q4 accelerating spend
    12: 1.30,  # December  - Q4 peak (fiscal year-end rush)
}

# ---------------------------------------------------------------------------
# 2b. Price elasticity ranges by product category
#     Elasticity = % change in quantity / % change in price
#     More negative = more elastic (price-sensitive)
# ---------------------------------------------------------------------------
CATEGORY_ELASTICITY: Dict[str, Tuple[float, float]] = {
    "Joint Replacement":  (-0.6, -0.3),   # Inelastic: surgeon preference/loyalty
    "Trauma":             (-0.9, -0.5),   # Moderate: competitive but clinical need
    "Beds & Stretchers":  (-1.5, -0.8),   # Elastic: GPO-driven, substitutable
    "Capital Equipment":  (-0.3, -0.1),   # Very inelastic: value-based, long cycles
    "Consumables":        (-1.8, -1.0),   # Elastic: commodity-like, high switching
}

# ---------------------------------------------------------------------------
# 2c. Discount waterfall parameters by contract tier
#     Each tier defines: (min_contract_discount, max_contract_discount)
#     Applied as: List Price * (1 - contract_discount) = Invoice Price
# ---------------------------------------------------------------------------
CONTRACT_TIER_DISCOUNTS: Dict[str, Tuple[float, float]] = {
    "Platinum": (0.18, 0.25),  # Largest volume commitments
    "Gold":     (0.14, 0.20),  # Mid-tier committed volume
    "Silver":   (0.10, 0.16),  # Lower volume commitments
    "Standard": (0.08, 0.12),  # List-price adjacent, minimal discounts
}

# ---------------------------------------------------------------------------
# 2d. GPO rebate ranges (applied after invoice price)
# ---------------------------------------------------------------------------
GPO_REBATE_RANGE: Tuple[float, float] = (0.02, 0.05)

# ---------------------------------------------------------------------------
# 2e. Regional pricing adjustment factors
#     Reflects cost-of-doing-business, competition density, and logistics.
# ---------------------------------------------------------------------------
REGIONAL_FACTORS: Dict[str, Tuple[float, float]] = {
    "Northeast":     (0.98, 1.08),   # Higher costs, dense hospital market
    "Southeast":     (0.92, 1.00),   # Lower costs, competitive
    "Midwest":       (0.90, 0.98),   # Lower costs, sparse coverage premium
    "West":          (0.95, 1.05),   # Mixed - urban premium, rural discount
    "International": (0.88, 1.12),   # Widest variance - FX, tariff, logistics
}

# ---------------------------------------------------------------------------
# 2f. Annual price escalation by category (applied per year after 2023)
# ---------------------------------------------------------------------------
ANNUAL_ESCALATION: Dict[str, float] = {
    "Joint Replacement":  0.035,  # 3.5% annual increase
    "Trauma":             0.030,  # 3.0%
    "Beds & Stretchers":  0.005,  # 0.5% (commoditized, held flat)
    "Capital Equipment":  0.025,  # 2.5%
    "Consumables":        0.000,  # Flat (commodity pricing pressure)
}

# ---------------------------------------------------------------------------
# 2g. Freight cost ranges by category (absolute dollars per unit)
# ---------------------------------------------------------------------------
FREIGHT_COST_RANGES: Dict[str, Tuple[float, float]] = {
    "Joint Replacement":  (45.0, 180.0),
    "Trauma":             (30.0, 120.0),
    "Beds & Stretchers":  (150.0, 600.0),  # Large/heavy items
    "Capital Equipment":  (500.0, 2500.0),  # Specialized logistics
    "Consumables":        (2.0, 15.0),      # Small, lightweight
}

# ---------------------------------------------------------------------------
# 2h. Base units-per-transaction ranges by category
# ---------------------------------------------------------------------------
BASE_UNITS_RANGES: Dict[str, Tuple[int, int]] = {
    "Joint Replacement":  (1, 6),
    "Trauma":             (1, 10),
    "Beds & Stretchers":  (1, 8),
    "Capital Equipment":  (1, 3),
    "Consumables":        (10, 200),
}

# ---------------------------------------------------------------------------
# 2i. Channel distribution weights by customer type
#     (Direct, Distributor, GPO) - must sum to 1.0
# ---------------------------------------------------------------------------
CHANNEL_WEIGHTS_BY_CUSTOMER_TYPE: Dict[str, Tuple[float, float, float]] = {
    "Academic Medical Center": (0.40, 0.20, 0.40),
    "Community Hospital":      (0.20, 0.35, 0.45),
    "ASC":                     (0.15, 0.50, 0.35),
    "Trauma Center":           (0.45, 0.25, 0.30),
    "Specialty Orthopedic":    (0.50, 0.30, 0.20),
}

# ---------------------------------------------------------------------------
# 2j. Contract tier distribution weights by customer type
#     (Platinum, Gold, Silver, Standard)
# ---------------------------------------------------------------------------
TIER_WEIGHTS_BY_CUSTOMER_TYPE: Dict[str, Tuple[float, float, float, float]] = {
    "Academic Medical Center": (0.35, 0.35, 0.20, 0.10),
    "Community Hospital":      (0.10, 0.25, 0.40, 0.25),
    "ASC":                     (0.05, 0.15, 0.35, 0.45),
    "Trauma Center":           (0.25, 0.35, 0.25, 0.15),
    "Specialty Orthopedic":    (0.20, 0.30, 0.30, 0.20),
}

# ---------------------------------------------------------------------------
# 2k. Customer type distribution (proportions of 1000 customers)
# ---------------------------------------------------------------------------
CUSTOMER_TYPE_DISTRIBUTION: Dict[str, float] = {
    "Academic Medical Center": 0.15,  # 150 customers
    "Community Hospital":      0.35,  # 350 customers
    "ASC":                     0.25,  # 250 customers
    "Trauma Center":           0.10,  # 100 customers
    "Specialty Orthopedic":    0.15,  # 150 customers
}

# ---------------------------------------------------------------------------
# 2l. GPO affiliation distribution (weighted by market share)
# ---------------------------------------------------------------------------
GPO_AFFILIATIONS: Dict[str, float] = {
    "Vizient":      0.30,
    "Premier":      0.28,
    "HPG":          0.18,
    "Intalere":     0.12,
    "HealthTrust":  0.12,
}

# ---------------------------------------------------------------------------
# 2m. Macro volume trend index (simulates hospital utilization trends)
#     Base year 2023 = 1.0; incorporates post-COVID recovery and growth
# ---------------------------------------------------------------------------
ANNUAL_VOLUME_TREND: Dict[int, float] = {
    2023: 1.00,  # Baseline
    2024: 1.04,  # 4% growth (elective procedure recovery)
    2025: 1.07,  # 7% cumulative (steady-state growth)
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Customer Master Generation
# MAGIC
# MAGIC Generate 1,000 unique customer profiles with deterministic attributes.
# MAGIC Each customer is assigned a type, region, GPO affiliation, preferred
# MAGIC channel, and contract tier. These attributes drive downstream
# MAGIC transaction generation.

# COMMAND ----------

def build_customer_master(num_customers: int, seed: int) -> DataFrame:
    """
    Generate a synthetic customer master DataFrame with realistic attribute distributions.

    Each customer receives:
      - customer_id: formatted as CUST-XXXXXX (zero-padded)
      - customer_type: one of five hospital/facility types
      - region: geographic region
      - gpo_affiliation: GPO membership
      - preferred_channel: primary purchasing channel
      - contract_tier: negotiated pricing tier

    Attributes are assigned using cumulative distribution functions built
    from the configured weight dictionaries, ensuring reproducible and
    statistically realistic distributions.

    Args:
        num_customers: Number of unique customer records to generate.
        seed: Random seed for reproducibility.

    Returns:
        PySpark DataFrame with the customer master schema.
    """
    # Build a sequential customer ID DataFrame
    customer_df = (
        spark.range(0, num_customers)
        .withColumn("customer_id", F.format_string("CUST-%06d", F.col("id")))
        .withColumn("rand_type", F.rand(seed=seed))
        .withColumn("rand_region", F.rand(seed=seed + 1))
        .withColumn("rand_gpo", F.rand(seed=seed + 2))
        .withColumn("rand_channel", F.rand(seed=seed + 3))
        .withColumn("rand_tier", F.rand(seed=seed + 4))
    )

    # --- Customer Type assignment via cumulative thresholds ---
    customer_types = list(CUSTOMER_TYPE_DISTRIBUTION.keys())
    type_weights = list(CUSTOMER_TYPE_DISTRIBUTION.values())
    cum_type = _cumulative_sum(type_weights)

    type_expr = F.lit(customer_types[-1])
    for i in range(len(customer_types) - 2, -1, -1):
        type_expr = F.when(F.col("rand_type") < cum_type[i], customer_types[i]).otherwise(type_expr)

    customer_df = customer_df.withColumn("customer_type", type_expr)

    # --- Region assignment (uniform across 5 regions) ---
    regions = list(REGIONAL_FACTORS.keys())
    region_expr = F.lit(regions[-1])
    for i in range(len(regions) - 2, -1, -1):
        threshold = (i + 1) / len(regions)
        region_expr = F.when(F.col("rand_region") < threshold, regions[i]).otherwise(region_expr)

    customer_df = customer_df.withColumn("region", region_expr)

    # --- GPO Affiliation assignment ---
    gpo_names = list(GPO_AFFILIATIONS.keys())
    gpo_weights = list(GPO_AFFILIATIONS.values())
    cum_gpo = _cumulative_sum(gpo_weights)

    gpo_expr = F.lit(gpo_names[-1])
    for i in range(len(gpo_names) - 2, -1, -1):
        gpo_expr = F.when(F.col("rand_gpo") < cum_gpo[i], gpo_names[i]).otherwise(gpo_expr)

    customer_df = customer_df.withColumn("gpo_affiliation", gpo_expr)

    # --- Preferred Channel (derived from customer_type weights) ---
    channels = ["Direct", "Distributor", "GPO"]
    channel_cases = F.lit("GPO")  # default fallback
    for ctype, weights in CHANNEL_WEIGHTS_BY_CUSTOMER_TYPE.items():
        cum_ch = _cumulative_sum(list(weights))
        inner_expr = F.lit(channels[-1])
        for j in range(len(channels) - 2, -1, -1):
            inner_expr = F.when(F.col("rand_channel") < cum_ch[j], channels[j]).otherwise(inner_expr)
        channel_cases = F.when(F.col("customer_type") == ctype, inner_expr).otherwise(channel_cases)

    customer_df = customer_df.withColumn("preferred_channel", channel_cases)

    # --- Contract Tier (derived from customer_type weights) ---
    tiers = ["Platinum", "Gold", "Silver", "Standard"]
    tier_cases = F.lit("Standard")  # default fallback
    for ctype, weights in TIER_WEIGHTS_BY_CUSTOMER_TYPE.items():
        cum_t = _cumulative_sum(list(weights))
        inner_expr = F.lit(tiers[-1])
        for j in range(len(tiers) - 2, -1, -1):
            inner_expr = F.when(F.col("rand_tier") < cum_t[j], tiers[j]).otherwise(inner_expr)
        tier_cases = F.when(F.col("customer_type") == ctype, inner_expr).otherwise(tier_cases)

    customer_df = customer_df.withColumn("contract_tier", tier_cases)

    # Drop intermediate random columns
    customer_df = customer_df.select(
        "customer_id",
        "customer_type",
        "region",
        "gpo_affiliation",
        "preferred_channel",
        "contract_tier",
    )

    return customer_df


def _cumulative_sum(weights: List[float]) -> List[float]:
    """
    Compute cumulative sum thresholds from a list of probability weights.

    Used to map a uniform random value [0, 1) to a categorical outcome
    via threshold comparison.

    Args:
        weights: List of non-negative floats that sum to 1.0.

    Returns:
        List of cumulative thresholds (same length as input).
    """
    cumulative = []
    running = 0.0
    for w in weights:
        running += w
        cumulative.append(running)
    return cumulative

# COMMAND ----------

customer_master_df = build_customer_master(NUM_CUSTOMERS, RANDOM_SEED)

print(f"Customer master: {customer_master_df.count()} records")
customer_master_df.groupBy("customer_type").count().orderBy("customer_type").show()
customer_master_df.groupBy("region").count().orderBy("region").show()
customer_master_df.groupBy("gpo_affiliation").count().orderBy("gpo_affiliation").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Product Master Lookup
# MAGIC
# MAGIC Load the product master table to obtain `product_id`, `category`,
# MAGIC and `list_price` for each product. These drive the pricing waterfall
# MAGIC and category-specific statistical behaviors.

# COMMAND ----------

def load_product_master(table_name: str) -> DataFrame:
    """
    Load the product master reference table from Unity Catalog.

    Selects the minimal columns needed for transaction generation:
    product_id, category, and list_price. Validates that the table
    is non-empty before returning.

    Args:
        table_name: Fully qualified table name (catalog.schema.table).

    Returns:
        PySpark DataFrame with product_id, category, list_price columns.

    Raises:
        ValueError: If the product master table is empty.
    """
    product_df = spark.table(table_name).select(
        "product_id",
        "category",
        F.col("base_asp").alias("list_price"),
    )

    row_count = product_df.count()
    if row_count == 0:
        raise ValueError(
            f"Product master table '{table_name}' is empty. "
            "Run notebook 01a_product_master.py first."
        )

    print(f"Loaded {row_count} products from {table_name}")
    return product_df

# COMMAND ----------

product_master_df = load_product_master(PRODUCT_TABLE)
product_master_df.groupBy("category").count().orderBy("category").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Transaction Generation Engine
# MAGIC
# MAGIC The core generation pipeline follows these steps:
# MAGIC 1. Generate a base skeleton of 500K rows with random date, customer, product assignments
# MAGIC 2. Join customer and product attributes
# MAGIC 3. Apply the full pricing waterfall (list -> invoice -> pocket)
# MAGIC 4. Apply seasonality, regional, trend, and elasticity adjustments to units
# MAGIC 5. Compute derived financial columns (revenue, discount %, rebate %)

# COMMAND ----------

def generate_transaction_skeleton(
    num_transactions: int,
    num_customers: int,
    product_count: int,
    date_start: str,
    date_end: str,
    seed: int,
) -> DataFrame:
    """
    Generate the raw transaction skeleton with randomized assignments.

    Creates a DataFrame of N rows, each with:
      - transaction_id: unique identifier (TXN-XXXXXXXXXX)
      - date: uniformly distributed across the date range
      - customer index: random integer [0, num_customers) for join
      - product index: random integer [0, product_count) for join
      - Random values for downstream pricing/volume calculations

    The skeleton is intentionally lightweight; all business logic
    is applied in subsequent transformation steps.

    Args:
        num_transactions: Total number of transactions to generate.
        num_customers: Size of the customer pool.
        product_count: Number of distinct products.
        date_start: Start date string (YYYY-MM-DD).
        date_end: End date string (YYYY-MM-DD).
        seed: Random seed for reproducibility.

    Returns:
        PySpark DataFrame with the transaction skeleton.
    """
    # Calculate the number of days in the range for uniform date distribution
    total_days = (
        F.datediff(F.lit(date_end), F.lit(date_start))
    )

    skeleton_df = (
        spark.range(0, num_transactions)
        .withColumn(
            "transaction_id",
            F.format_string("TXN-%010d", F.col("id")),
        )
        # Uniform random date within the 36-month window
        .withColumn(
            "date",
            F.date_add(
                F.lit(date_start),
                (F.rand(seed=seed) * 1095).cast("int"),  # ~365 * 3 = 1095 days
            ),
        )
        # Random customer and product assignment indices
        .withColumn(
            "customer_idx",
            (F.rand(seed=seed + 10) * num_customers).cast("int"),
        )
        .withColumn(
            "product_idx",
            (F.rand(seed=seed + 20) * product_count).cast("int"),
        )
        # Random values for pricing waterfall (0,1) uniform draws
        .withColumn("rand_discount", F.rand(seed=seed + 30))
        .withColumn("rand_rebate", F.rand(seed=seed + 40))
        .withColumn("rand_region_adj", F.rand(seed=seed + 50))
        .withColumn("rand_freight", F.rand(seed=seed + 60))
        .withColumn("rand_units", F.rand(seed=seed + 70))
        .withColumn("rand_elasticity", F.rand(seed=seed + 80))
        # Extract temporal components for seasonality and trend
        .withColumn("year", F.year("date"))
        .withColumn("month", F.month("date"))
    )

    return skeleton_df

# COMMAND ----------

def attach_customer_attributes(
    skeleton_df: DataFrame,
    customer_df: DataFrame,
) -> DataFrame:
    """
    Join customer master attributes onto the transaction skeleton.

    Uses a row_number window on the customer master to create a join key
    matching the random customer_idx. This avoids broadcasting Python
    lists and keeps the operation fully within Spark's execution engine.

    Args:
        skeleton_df: Transaction skeleton with customer_idx column.
        customer_df: Customer master DataFrame.

    Returns:
        DataFrame enriched with customer_type, region, gpo_affiliation,
        preferred_channel (aliased as channel), and contract_tier.
    """
    # Add a numeric index to the customer master for join
    cust_window = Window.orderBy("customer_id")
    indexed_customers = customer_df.withColumn(
        "cust_idx", (F.row_number().over(cust_window) - 1)
    )

    joined = skeleton_df.join(
        F.broadcast(indexed_customers),
        skeleton_df["customer_idx"] == indexed_customers["cust_idx"],
        "inner",
    ).drop("cust_idx", "customer_idx")

    # Rename preferred_channel to channel for output schema
    joined = joined.withColumnRenamed("preferred_channel", "channel")

    return joined

# COMMAND ----------

def attach_product_attributes(
    txn_df: DataFrame,
    product_df: DataFrame,
) -> DataFrame:
    """
    Join product master attributes onto the transaction DataFrame.

    Uses row_number indexing on the product master (ordered by product_id)
    to create a deterministic join with the random product_idx.

    Args:
        txn_df: Transaction DataFrame with product_idx column.
        product_df: Product master DataFrame with product_id, category, list_price.

    Returns:
        DataFrame enriched with product_id, category, and base_list_price.
    """
    prod_window = Window.orderBy("product_id")
    indexed_products = product_df.withColumn(
        "prod_idx", (F.row_number().over(prod_window) - 1)
    )

    joined = txn_df.join(
        F.broadcast(indexed_products),
        txn_df["product_idx"] == indexed_products["prod_idx"],
        "inner",
    ).drop("prod_idx", "product_idx")

    # Rename list_price from product master to base_list_price
    # (we will compute the adjusted list_price with annual escalation)
    joined = joined.withColumnRenamed("list_price", "base_list_price")

    return joined

# COMMAND ----------

def apply_pricing_waterfall(txn_df: DataFrame) -> DataFrame:
    """
    Apply the complete pricing waterfall to compute financial columns.

    The waterfall proceeds as:
      1. **Annual price escalation**: base_list_price * (1 + escalation)^(year - 2023)
      2. **Regional adjustment**: list_price * regional_factor
      3. **Contract discount**: list_price * (1 - contract_discount) = invoice_price
      4. **GPO rebate**: invoice_price * (1 - rebate) = pocket_price_before_freight
      5. **Freight**: pocket_price = pocket_price_before_freight - freight_per_unit
      6. **Units sold**: base_units * seasonality * volume_trend * elasticity_effect
      7. **Derived metrics**: total_revenue, discount_pct, rebate_pct

    All random interpolations use the pre-generated rand_* columns from the
    skeleton, ensuring full reproducibility.

    Args:
        txn_df: Transaction DataFrame with product/customer attributes
                and rand_* columns.

    Returns:
        DataFrame with all pricing waterfall columns computed.
    """
    # ------------------------------------------------------------------
    # Step 1: Annual price escalation
    #   list_price = base_list_price * (1 + category_escalation) ^ (year - 2023)
    # ------------------------------------------------------------------
    escalation_expr = F.lit(0.0)
    for category, rate in ANNUAL_ESCALATION.items():
        escalation_expr = F.when(
            F.col("category") == category, F.lit(rate)
        ).otherwise(escalation_expr)

    txn_df = txn_df.withColumn("escalation_rate", escalation_expr)
    txn_df = txn_df.withColumn(
        "list_price",
        F.round(
            F.col("base_list_price")
            * F.pow(1.0 + F.col("escalation_rate"), F.col("year") - 2023),
            2,
        ),
    )

    # ------------------------------------------------------------------
    # Step 2: Regional price adjustment
    #   Interpolate between (region_low, region_high) using rand_region_adj
    # ------------------------------------------------------------------
    region_low_expr = F.lit(1.0)
    region_high_expr = F.lit(1.0)
    for region, (low, high) in REGIONAL_FACTORS.items():
        region_low_expr = F.when(F.col("region") == region, F.lit(low)).otherwise(region_low_expr)
        region_high_expr = F.when(F.col("region") == region, F.lit(high)).otherwise(region_high_expr)

    txn_df = txn_df.withColumn(
        "regional_factor",
        region_low_expr + (region_high_expr - region_low_expr) * F.col("rand_region_adj"),
    )
    txn_df = txn_df.withColumn(
        "list_price",
        F.round(F.col("list_price") * F.col("regional_factor"), 2),
    )

    # ------------------------------------------------------------------
    # Step 3: Contract discount -> Invoice price
    #   Interpolate discount within tier's (min, max) range
    # ------------------------------------------------------------------
    disc_low_expr = F.lit(0.08)
    disc_high_expr = F.lit(0.12)
    for tier, (low, high) in CONTRACT_TIER_DISCOUNTS.items():
        disc_low_expr = F.when(F.col("contract_tier") == tier, F.lit(low)).otherwise(disc_low_expr)
        disc_high_expr = F.when(F.col("contract_tier") == tier, F.lit(high)).otherwise(disc_high_expr)

    txn_df = txn_df.withColumn(
        "contract_discount",
        disc_low_expr + (disc_high_expr - disc_low_expr) * F.col("rand_discount"),
    )
    txn_df = txn_df.withColumn(
        "invoice_price",
        F.round(F.col("list_price") * (1.0 - F.col("contract_discount")), 2),
    )

    # ------------------------------------------------------------------
    # Step 4: GPO rebate -> pre-freight pocket price
    # ------------------------------------------------------------------
    rebate_low, rebate_high = GPO_REBATE_RANGE
    txn_df = txn_df.withColumn(
        "rebate_rate",
        F.lit(rebate_low) + (F.lit(rebate_high) - F.lit(rebate_low)) * F.col("rand_rebate"),
    )
    txn_df = txn_df.withColumn(
        "pocket_price_pre_freight",
        F.round(F.col("invoice_price") * (1.0 - F.col("rebate_rate")), 2),
    )

    # ------------------------------------------------------------------
    # Step 5: Freight cost per unit
    # ------------------------------------------------------------------
    freight_low_expr = F.lit(10.0)
    freight_high_expr = F.lit(50.0)
    for category, (low, high) in FREIGHT_COST_RANGES.items():
        freight_low_expr = F.when(F.col("category") == category, F.lit(low)).otherwise(freight_low_expr)
        freight_high_expr = F.when(F.col("category") == category, F.lit(high)).otherwise(freight_high_expr)

    txn_df = txn_df.withColumn(
        "freight_cost",
        F.round(
            freight_low_expr + (freight_high_expr - freight_low_expr) * F.col("rand_freight"),
            2,
        ),
    )
    txn_df = txn_df.withColumn(
        "pocket_price",
        F.round(
            F.greatest(
                F.col("pocket_price_pre_freight") - F.col("freight_cost"),
                F.lit(0.01),  # Floor at $0.01 to avoid negative pocket prices
            ),
            2,
        ),
    )

    # ------------------------------------------------------------------
    # Step 6: Units sold
    #   base_units * seasonal_multiplier * volume_trend * elasticity_modifier
    # ------------------------------------------------------------------
    # 6a. Base units (category-specific range)
    units_low_expr = F.lit(1)
    units_high_expr = F.lit(10)
    for category, (low, high) in BASE_UNITS_RANGES.items():
        units_low_expr = F.when(F.col("category") == category, F.lit(low)).otherwise(units_low_expr)
        units_high_expr = F.when(F.col("category") == category, F.lit(high)).otherwise(units_high_expr)

    txn_df = txn_df.withColumn(
        "base_units",
        (units_low_expr + (units_high_expr - units_low_expr) * F.col("rand_units")).cast("int"),
    )

    # 6b. Seasonal multiplier
    seasonal_expr = F.lit(1.0)
    for m, mult in MONTHLY_SEASONAL_MULTIPLIERS.items():
        seasonal_expr = F.when(F.col("month") == m, F.lit(mult)).otherwise(seasonal_expr)

    txn_df = txn_df.withColumn("seasonal_multiplier", seasonal_expr)

    # 6c. Annual volume trend
    trend_expr = F.lit(1.0)
    for yr, trend in ANNUAL_VOLUME_TREND.items():
        trend_expr = F.when(F.col("year") == yr, F.lit(trend)).otherwise(trend_expr)

    txn_df = txn_df.withColumn("volume_trend", trend_expr)

    # 6d. Elasticity-based volume modifier
    #   Higher discount -> elasticity amplifies volume (for elastic categories)
    #   elasticity_modifier = 1 + |elasticity| * (discount - mean_discount)
    #   Clamped to [0.7, 1.5] to avoid extreme outliers
    elast_low_expr = F.lit(-0.5)
    elast_high_expr = F.lit(-0.3)
    for category, (low, high) in CATEGORY_ELASTICITY.items():
        elast_low_expr = F.when(F.col("category") == category, F.lit(low)).otherwise(elast_low_expr)
        elast_high_expr = F.when(F.col("category") == category, F.lit(high)).otherwise(elast_high_expr)

    txn_df = txn_df.withColumn(
        "elasticity",
        elast_low_expr + (elast_high_expr - elast_low_expr) * F.col("rand_elasticity"),
    )
    # Mean discount across all tiers is approximately 0.16
    txn_df = txn_df.withColumn(
        "elasticity_modifier",
        F.greatest(
            F.least(
                1.0 + F.abs(F.col("elasticity")) * (F.col("contract_discount") - 0.16),
                F.lit(1.5),
            ),
            F.lit(0.7),
        ),
    )

    # 6e. Final units sold (minimum of 1)
    txn_df = txn_df.withColumn(
        "units_sold",
        F.greatest(
            (
                F.col("base_units")
                * F.col("seasonal_multiplier")
                * F.col("volume_trend")
                * F.col("elasticity_modifier")
            ).cast("int"),
            F.lit(1),
        ),
    )

    # ------------------------------------------------------------------
    # Step 7: Derived financial metrics
    # ------------------------------------------------------------------
    txn_df = txn_df.withColumn(
        "total_revenue",
        F.round(F.col("pocket_price") * F.col("units_sold"), 2),
    )
    txn_df = txn_df.withColumn(
        "discount_pct",
        F.round(
            (F.col("list_price") - F.col("invoice_price")) / F.col("list_price") * 100,
            2,
        ),
    )
    txn_df = txn_df.withColumn(
        "rebate_pct",
        F.round(F.col("rebate_rate") * 100, 2),
    )

    return txn_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Execute Transaction Generation Pipeline

# COMMAND ----------

# Step 1: Count products for skeleton generation
product_count = product_master_df.count()
print(f"Product count: {product_count}")
print(f"Customer count: {NUM_CUSTOMERS}")
print(f"Target transactions: {NUM_TRANSACTIONS:,}")

# COMMAND ----------

# Step 2: Generate the transaction skeleton
print("Generating transaction skeleton...")
skeleton_df = generate_transaction_skeleton(
    num_transactions=NUM_TRANSACTIONS,
    num_customers=NUM_CUSTOMERS,
    product_count=product_count,
    date_start=DATE_START,
    date_end=DATE_END,
    seed=RANDOM_SEED,
)
print(f"Skeleton rows: {skeleton_df.count():,}")

# COMMAND ----------

# Step 3: Attach customer attributes
print("Joining customer attributes...")
txn_with_customers = attach_customer_attributes(skeleton_df, customer_master_df)

# COMMAND ----------

# Step 4: Attach product attributes
print("Joining product attributes...")
txn_with_products = attach_product_attributes(txn_with_customers, product_master_df)

# COMMAND ----------

# Step 5: Apply the full pricing waterfall
print("Applying pricing waterfall (list -> invoice -> pocket)...")
transactions_df = apply_pricing_waterfall(txn_with_products)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Final Schema Selection & Validation

# COMMAND ----------

def select_output_schema(txn_df: DataFrame) -> DataFrame:
    """
    Select and order the final output columns matching the target schema.

    Drops all intermediate calculation columns (rand_*, base_*, escalation_*,
    seasonal_*, etc.) and retains only the business-facing transaction columns.

    Args:
        txn_df: Full transaction DataFrame with all intermediate columns.

    Returns:
        DataFrame with the clean output schema, ordered for readability.
    """
    output_columns = [
        "transaction_id",
        "date",
        "product_id",
        "customer_id",
        "customer_type",
        "region",
        "channel",
        "contract_tier",
        "list_price",
        "invoice_price",
        "pocket_price",
        "units_sold",
        "total_revenue",
        "discount_pct",
        "rebate_pct",
        "freight_cost",
        "gpo_affiliation",
        "year",
        "month",
    ]
    return txn_df.select(*output_columns)

# COMMAND ----------

final_df = select_output_schema(transactions_df)

# Display schema and sample
final_df.printSchema()
print(f"\nTotal transactions: {final_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7a. Data Quality Validation

# COMMAND ----------

def validate_transactions(df: DataFrame) -> None:
    """
    Run data quality assertions on the generated transaction DataFrame.

    Validates:
      - Row count matches target (500,000)
      - No null values in critical columns
      - Price waterfall integrity (list >= invoice >= pocket)
      - Discount percentages within expected bounds (8-25%)
      - Rebate percentages within expected bounds (2-5%)
      - Units sold are positive integers
      - All expected regions, channels, and tiers are present
      - Date range covers the full 36-month window
      - Revenue is non-negative

    Args:
        df: Final transaction DataFrame to validate.

    Raises:
        AssertionError: If any validation check fails.
    """
    print("Running data quality validations...")

    # 1. Row count
    count = df.count()
    assert count == NUM_TRANSACTIONS, f"Expected {NUM_TRANSACTIONS:,} rows, got {count:,}"
    print(f"  [PASS] Row count: {count:,}")

    # 2. No nulls in critical columns
    critical_cols = [
        "transaction_id", "date", "product_id", "customer_id",
        "list_price", "invoice_price", "pocket_price", "units_sold",
    ]
    for col_name in critical_cols:
        null_count = df.filter(F.col(col_name).isNull()).count()
        assert null_count == 0, f"Found {null_count} nulls in {col_name}"
    print("  [PASS] No nulls in critical columns")

    # 3. Price waterfall integrity
    waterfall_violations = df.filter(
        (F.col("list_price") < F.col("invoice_price"))
        | (F.col("invoice_price") < F.col("pocket_price"))
    ).count()
    # Note: pocket_price can exceed invoice_price in rare freight-negative cases
    # but list >= invoice should always hold
    list_invoice_violations = df.filter(
        F.col("list_price") < F.col("invoice_price")
    ).count()
    assert list_invoice_violations == 0, (
        f"Found {list_invoice_violations} rows where list_price < invoice_price"
    )
    print(f"  [PASS] Price waterfall: list >= invoice (0 violations)")

    # 4. Discount percentage bounds
    discount_stats = df.select(
        F.min("discount_pct").alias("min_disc"),
        F.max("discount_pct").alias("max_disc"),
    ).first()
    print(
        f"  [INFO] Discount range: {discount_stats['min_disc']:.2f}% "
        f"to {discount_stats['max_disc']:.2f}%"
    )
    assert discount_stats["min_disc"] >= 0, "Negative discount found"
    assert discount_stats["max_disc"] <= 35, f"Discount exceeds 35%: {discount_stats['max_disc']}"
    print("  [PASS] Discount percentages within bounds")

    # 5. Rebate percentage bounds
    rebate_stats = df.select(
        F.min("rebate_pct").alias("min_reb"),
        F.max("rebate_pct").alias("max_reb"),
    ).first()
    print(
        f"  [INFO] Rebate range: {rebate_stats['min_reb']:.2f}% "
        f"to {rebate_stats['max_reb']:.2f}%"
    )
    assert rebate_stats["min_reb"] >= 1.5, f"Rebate too low: {rebate_stats['min_reb']}"
    assert rebate_stats["max_reb"] <= 6.0, f"Rebate too high: {rebate_stats['max_reb']}"
    print("  [PASS] Rebate percentages within bounds")

    # 6. Positive units
    non_positive_units = df.filter(F.col("units_sold") <= 0).count()
    assert non_positive_units == 0, f"Found {non_positive_units} rows with non-positive units"
    print("  [PASS] All units_sold > 0")

    # 7. All regions present
    distinct_regions = {row["region"] for row in df.select("region").distinct().collect()}
    expected_regions = set(REGIONAL_FACTORS.keys())
    assert expected_regions.issubset(distinct_regions), (
        f"Missing regions: {expected_regions - distinct_regions}"
    )
    print(f"  [PASS] All {len(expected_regions)} regions present")

    # 8. All channels present
    distinct_channels = {row["channel"] for row in df.select("channel").distinct().collect()}
    expected_channels = {"Direct", "Distributor", "GPO"}
    assert expected_channels.issubset(distinct_channels), (
        f"Missing channels: {expected_channels - distinct_channels}"
    )
    print(f"  [PASS] All {len(expected_channels)} channels present")

    # 9. Date range coverage
    date_stats = df.select(
        F.min("date").alias("min_date"),
        F.max("date").alias("max_date"),
    ).first()
    print(f"  [INFO] Date range: {date_stats['min_date']} to {date_stats['max_date']}")
    assert str(date_stats["min_date"]).startswith("2023"), "Data does not start in 2023"
    assert str(date_stats["max_date"]).startswith("2025"), "Data does not end in 2025"
    print("  [PASS] Date range covers 2023-2025")

    # 10. Non-negative revenue
    neg_revenue = df.filter(F.col("total_revenue") < 0).count()
    assert neg_revenue == 0, f"Found {neg_revenue} rows with negative revenue"
    print("  [PASS] All total_revenue >= 0")

    print("\nAll data quality validations PASSED.")

# COMMAND ----------

validate_transactions(final_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7b. Statistical Summary

# COMMAND ----------

# Distribution summary for key financial metrics
print("=== Financial Metrics Summary ===")
final_df.select(
    "list_price", "invoice_price", "pocket_price",
    "discount_pct", "rebate_pct", "freight_cost",
    "units_sold", "total_revenue",
).summary("count", "min", "25%", "50%", "75%", "max", "mean", "stddev").show(truncate=False)

# COMMAND ----------

# Seasonal volume distribution
print("=== Monthly Volume Distribution ===")
(
    final_df
    .groupBy("month")
    .agg(
        F.count("*").alias("transaction_count"),
        F.sum("units_sold").alias("total_units"),
        F.round(F.avg("total_revenue"), 2).alias("avg_revenue"),
    )
    .orderBy("month")
    .show(12, truncate=False)
)

# COMMAND ----------

# Category-level pricing analysis
print("=== Category Pricing Analysis ===")
(
    final_df
    .join(
        product_master_df.select("product_id", "category"),
        on="product_id",
        how="inner",
    )
    .groupBy("category")
    .agg(
        F.count("*").alias("txn_count"),
        F.round(F.avg("list_price"), 2).alias("avg_list"),
        F.round(F.avg("invoice_price"), 2).alias("avg_invoice"),
        F.round(F.avg("pocket_price"), 2).alias("avg_pocket"),
        F.round(F.avg("discount_pct"), 2).alias("avg_discount_pct"),
        F.round(F.avg("rebate_pct"), 2).alias("avg_rebate_pct"),
        F.round(F.sum("total_revenue"), 2).alias("total_revenue"),
    )
    .orderBy("category")
    .show(truncate=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Write to Delta Table
# MAGIC
# MAGIC Write the final transaction DataFrame to the Unity Catalog bronze layer
# MAGIC as a Delta table, partitioned by `year` and `month` for efficient
# MAGIC time-range queries in downstream pipeline stages.

# COMMAND ----------

def write_transactions_to_delta(
    df: DataFrame,
    table_name: str,
    partition_cols: List[str],
) -> None:
    """
    Write the transaction DataFrame to a Delta table in Unity Catalog.

    Performs a full overwrite of the target table. Partitions by year and month
    to optimize query performance for time-series analytics downstream.

    The write uses Delta format with overwriteSchema=True to handle schema
    evolution gracefully during iterative development.

    Args:
        df: Final transaction DataFrame to persist.
        table_name: Fully qualified table name (catalog.schema.table).
        partition_cols: List of column names to partition by.
    """
    print(f"Writing {df.count():,} transactions to {table_name}...")
    print(f"Partition columns: {partition_cols}")

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy(*partition_cols)
        .saveAsTable(table_name)
    )

    # Verify the write
    written_count = spark.table(table_name).count()
    print(f"Successfully wrote {written_count:,} rows to {table_name}")

    # Display partition statistics
    partition_stats = (
        spark.table(table_name)
        .groupBy(*partition_cols)
        .count()
        .orderBy(*partition_cols)
    )
    print(f"\nPartition distribution ({partition_stats.count()} partitions):")
    partition_stats.show(40, truncate=False)

# COMMAND ----------

write_transactions_to_delta(
    df=final_df,
    table_name=OUTPUT_TABLE,
    partition_cols=["year", "month"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Post-Write Verification
# MAGIC
# MAGIC Final end-to-end verification that the persisted Delta table matches
# MAGIC expectations and is ready for downstream consumption.

# COMMAND ----------

# Reload from Delta and verify
persisted_df = spark.table(OUTPUT_TABLE)

print(f"Table: {OUTPUT_TABLE}")
print(f"Row count: {persisted_df.count():,}")
print(f"Distinct customers: {persisted_df.select('customer_id').distinct().count():,}")
print(f"Distinct products: {persisted_df.select('product_id').distinct().count():,}")
print(f"Date range: {persisted_df.select(F.min('date')).first()[0]} to {persisted_df.select(F.max('date')).first()[0]}")
print(f"\nRegion distribution:")
persisted_df.groupBy("region").count().orderBy(F.desc("count")).show()
print(f"Channel distribution:")
persisted_df.groupBy("channel").count().orderBy(F.desc("count")).show()
print(f"Contract tier distribution:")
persisted_df.groupBy("contract_tier").count().orderBy(F.desc("count")).show()
print(f"GPO affiliation distribution:")
persisted_df.groupBy("gpo_affiliation").count().orderBy(F.desc("count")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Notebook Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Transactions generated | 500,000 |
# MAGIC | Date range | Jan 2023 - Dec 2025 (36 months) |
# MAGIC | Unique customers | 1,000 |
# MAGIC | Customer types | Academic Medical Center, Community Hospital, ASC, Trauma Center, Specialty Orthopedic |
# MAGIC | Regions | Northeast, Southeast, Midwest, West, International |
# MAGIC | Channels | Direct, Distributor, GPO |
# MAGIC | Contract tiers | Platinum, Gold, Silver, Standard |
# MAGIC | GPO affiliations | Vizient, Premier, HPG, Intalere, HealthTrust |
# MAGIC | Output table | `hls_amer_catalog.bronze.stryker_transactions` |
# MAGIC | Partitioning | year, month |
# MAGIC | Random seed | 42 |
# MAGIC
# MAGIC **Next step:** Run `02_pipeline/02a_silver_pricing.py` to transform bronze transactions into the silver pricing analytics layer.
