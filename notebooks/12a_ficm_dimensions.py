# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 12a - FICM Dimension Tables: Customers, Sales Reps, Products
# MAGIC
# MAGIC **Purpose**: Generate three deterministic dimension tables for the FICM (Full Invoice-to-Cash Margin)
# MAGIC pricing intelligence platform. These dimensions are the foundation for the 600K-row
# MAGIC `ficm_pricing_master` fact table created in notebook `12_ficm_pricing_master.py`.
# MAGIC
# MAGIC **Output Tables:**
# MAGIC | Table | Rows | Description |
# MAGIC |-------|------|-------------|
# MAGIC | `hls_amer_catalog.silver.dim_customers` | 500 | Hospital/facility customer master |
# MAGIC | `hls_amer_catalog.silver.dim_sales_reps` | 75 | Sales representative roster |
# MAGIC | `hls_amer_catalog.silver.dim_products` | 200 | Medical device product catalog |
# MAGIC
# MAGIC **Reproducibility**: All stochastic operations use `seed=42`.
# MAGIC
# MAGIC **Regional Distribution**: 60% US, 25% EMEA, 10% APAC, 5% LATAM

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration & Constants

# COMMAND ----------

import numpy as np
import uuid
import json
from datetime import date, timedelta
from typing import Dict, List, Tuple, Any

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    DateType,
    BooleanType,
)
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Deterministic seed -- every random draw in this notebook is anchored here.
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Unity Catalog target
# ---------------------------------------------------------------------------
TARGET_CATALOG: str = "hls_amer_catalog"
TARGET_SCHEMA: str = "silver"

CUSTOMERS_TABLE: str = f"{TARGET_CATALOG}.{TARGET_SCHEMA}.dim_customers"
SALES_REPS_TABLE: str = f"{TARGET_CATALOG}.{TARGET_SCHEMA}.dim_sales_reps"
PRODUCTS_TABLE: str = f"{TARGET_CATALOG}.{TARGET_SCHEMA}.dim_products"

# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------
NUM_CUSTOMERS: int = 500
NUM_SALES_REPS: int = 75
NUM_PRODUCTS: int = 200

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. dim_customers -- 500 Hospital & Facility Customers
# MAGIC
# MAGIC Schema:
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | customer_id | STRING (PK) | Unique customer identifier |
# MAGIC | customer_name | STRING | Realistic hospital/facility name |
# MAGIC | customer_segment | STRING | IDN / GPO / Direct / Distributor / Govt-VA / Academic |
# MAGIC | customer_tier | STRING | A / B / C / D |
# MAGIC | customer_country | STRING | ISO country code |
# MAGIC | customer_region | STRING | US / EMEA / APAC / LATAM |
# MAGIC | customer_state | STRING (nullable) | US state code or null for international |
# MAGIC | annual_spend | DOUBLE | Annual spend in USD |
# MAGIC | contract_count | INT | Number of active contracts |
# MAGIC | first_purchase_date | DATE | Date of first purchase |
# MAGIC | is_active | BOOLEAN | Whether customer is currently active |

# COMMAND ----------

# ---------------------------------------------------------------------------
# Customer name components for realistic hospital/facility naming
# ---------------------------------------------------------------------------
US_CITY_NAMES: List[str] = [
    "Boston", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio",
    "Dallas", "Austin", "Jacksonville", "San Jose", "Columbus", "Charlotte",
    "Indianapolis", "Denver", "Seattle", "Nashville", "Baltimore", "Louisville",
    "Portland", "Milwaukee", "Albuquerque", "Tucson", "Fresno", "Sacramento",
    "Mesa", "Atlanta", "Kansas City", "Omaha", "Raleigh", "Miami",
    "Cleveland", "Tampa", "New Orleans", "Pittsburgh", "St. Louis", "Cincinnati",
    "Orlando", "Minneapolis", "Richmond", "Hartford", "Salt Lake City", "Birmingham",
    "Buffalo", "Rochester", "Memphis", "Boise", "Des Moines", "Spokane",
    "Little Rock", "Lexington", "Anchorage", "Honolulu", "Providence", "Charleston",
    "Savannah", "Madison", "Knoxville", "Dayton", "Tulsa", "Springfield",
]

US_STATES: List[str] = [
    "MA", "IL", "TX", "AZ", "PA", "TX", "TX", "TX", "FL", "CA",
    "OH", "NC", "IN", "CO", "WA", "TN", "MD", "KY", "OR", "WI",
    "NM", "AZ", "CA", "CA", "AZ", "GA", "MO", "NE", "NC", "FL",
    "OH", "FL", "LA", "PA", "MO", "OH", "FL", "MN", "VA", "CT",
    "UT", "AL", "NY", "NY", "TN", "ID", "IA", "WA", "AR", "KY",
    "AK", "HI", "RI", "SC", "GA", "WI", "TN", "OH", "OK", "IL",
]

EMEA_NAMES: List[str] = [
    "Berlin", "Munich", "Hamburg", "Frankfurt", "London", "Manchester",
    "Birmingham", "Leeds", "Paris", "Lyon", "Marseille", "Toulouse",
    "Rome", "Milan", "Naples", "Turin", "Amsterdam", "Rotterdam",
    "Brussels", "Zurich", "Vienna", "Stockholm", "Copenhagen", "Oslo",
    "Madrid", "Barcelona", "Lisbon", "Dublin", "Edinburgh", "Warsaw",
]

EMEA_COUNTRIES: List[str] = [
    "DE", "DE", "DE", "DE", "GB", "GB", "GB", "GB", "FR", "FR",
    "FR", "FR", "IT", "IT", "IT", "IT", "NL", "NL", "BE", "CH",
    "AT", "SE", "DK", "NO", "ES", "ES", "PT", "IE", "GB", "PL",
]

APAC_NAMES: List[str] = [
    "Tokyo", "Osaka", "Yokohama", "Nagoya", "Sydney", "Melbourne",
    "Brisbane", "Perth", "Seoul", "Busan", "Singapore", "Hong Kong",
    "Shanghai", "Beijing", "Taipei", "Bangkok", "Mumbai", "Delhi",
]

APAC_COUNTRIES: List[str] = [
    "JP", "JP", "JP", "JP", "AU", "AU", "AU", "AU",
    "KR", "KR", "SG", "HK", "CN", "CN", "TW", "TH", "IN", "IN",
]

LATAM_NAMES: List[str] = [
    "Mexico City", "Guadalajara", "Monterrey", "Sao Paulo", "Rio de Janeiro",
    "Brasilia", "Buenos Aires", "Santiago", "Lima", "Bogota",
    "Medellin", "Cancun",
]

LATAM_COUNTRIES: List[str] = [
    "MX", "MX", "MX", "BR", "BR", "BR", "AR", "CL", "PE", "CO",
    "CO", "MX",
]

HOSPITAL_SUFFIXES: List[str] = [
    "General Hospital", "Medical Center", "University Hospital",
    "Regional Medical Center", "Community Hospital", "Memorial Hospital",
    "Health System", "Healthcare", "Surgical Center", "Clinic",
    "Teaching Hospital", "Veterans Medical Center", "Childrens Hospital",
    "Orthopedic Institute", "Specialty Hospital",
]

CUSTOMER_SEGMENTS: List[str] = ["IDN", "GPO", "Direct", "Distributor", "Govt-VA", "Academic"]
SEGMENT_WEIGHTS: List[float] = [0.25, 0.25, 0.20, 0.15, 0.08, 0.07]

CUSTOMER_TIERS: List[str] = ["A", "B", "C", "D"]
TIER_WEIGHTS: List[float] = [0.15, 0.30, 0.35, 0.20]

# Annual spend ranges by tier (USD)
TIER_SPEND_RANGES: Dict[str, Tuple[float, float]] = {
    "A": (5_000_000.0, 50_000_000.0),
    "B": (1_000_000.0, 8_000_000.0),
    "C": (250_000.0, 2_000_000.0),
    "D": (50_000.0, 500_000.0),
}

# Contract count ranges by tier
TIER_CONTRACT_RANGES: Dict[str, Tuple[int, int]] = {
    "A": (8, 25),
    "B": (4, 12),
    "C": (1, 6),
    "D": (1, 3),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a. Customer Generation Logic

# COMMAND ----------

def _generate_customer_rows(seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    """Generate 500 customer master rows deterministically.

    Regional distribution: 60% US (300), 25% EMEA (125), 10% APAC (50), 5% LATAM (25).
    Uses cumulative probability thresholds for segment and tier assignment.

    Returns
    -------
    List[Dict[str, Any]]
        Each dict maps column name to its scalar value.
    """
    rng = np.random.RandomState(seed)
    _uuid_ns = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

    # Regional allocation
    n_us = 300
    n_emea = 125
    n_apac = 50
    n_latam = 25
    assert n_us + n_emea + n_apac + n_latam == NUM_CUSTOMERS

    rows: List[Dict[str, Any]] = []

    def _pick_weighted(choices: List[str], weights: List[float]) -> str:
        """Pick from choices using cumulative weight thresholds."""
        r = rng.random()
        cumulative = 0.0
        for choice, w in zip(choices, weights):
            cumulative += w
            if r < cumulative:
                return choice
        return choices[-1]

    def _make_customer(
        idx: int,
        city: str,
        country: str,
        region: str,
        state: str = None,
    ) -> Dict[str, Any]:
        customer_id = f"CUST-{idx:05d}"
        suffix = HOSPITAL_SUFFIXES[rng.randint(0, len(HOSPITAL_SUFFIXES))]
        customer_name = f"{city} {suffix}"

        segment = _pick_weighted(CUSTOMER_SEGMENTS, SEGMENT_WEIGHTS)
        tier = _pick_weighted(CUSTOMER_TIERS, TIER_WEIGHTS)

        spend_lo, spend_hi = TIER_SPEND_RANGES[tier]
        annual_spend = round(float(np.exp(rng.uniform(np.log(spend_lo), np.log(spend_hi)))), 2)

        contract_lo, contract_hi = TIER_CONTRACT_RANGES[tier]
        contract_count = int(rng.randint(contract_lo, contract_hi + 1))

        # First purchase date: 2010-01-01 to 2023-12-31
        days_offset = int(rng.randint(0, 5110))  # ~14 years
        first_purchase_date = date(2010, 1, 1) + timedelta(days=days_offset)

        # 90% active, 10% inactive
        is_active = bool(rng.random() < 0.90)

        return {
            "customer_id": customer_id,
            "customer_name": customer_name,
            "customer_segment": segment,
            "customer_tier": tier,
            "customer_country": country,
            "customer_region": region,
            "customer_state": state,
            "annual_spend": annual_spend,
            "contract_count": contract_count,
            "first_purchase_date": first_purchase_date,
            "is_active": is_active,
        }

    idx = 0

    # --- US customers (300) ---
    for i in range(n_us):
        city_idx = rng.randint(0, len(US_CITY_NAMES))
        city = US_CITY_NAMES[city_idx]
        state = US_STATES[city_idx]
        rows.append(_make_customer(idx, city, "US", "US", state))
        idx += 1

    # --- EMEA customers (125) ---
    for i in range(n_emea):
        city_idx = rng.randint(0, len(EMEA_NAMES))
        city = EMEA_NAMES[city_idx]
        country = EMEA_COUNTRIES[city_idx]
        rows.append(_make_customer(idx, city, country, "EMEA", None))
        idx += 1

    # --- APAC customers (50) ---
    for i in range(n_apac):
        city_idx = rng.randint(0, len(APAC_NAMES))
        city = APAC_NAMES[city_idx]
        country = APAC_COUNTRIES[city_idx]
        rows.append(_make_customer(idx, city, country, "APAC", None))
        idx += 1

    # --- LATAM customers (25) ---
    for i in range(n_latam):
        city_idx = rng.randint(0, len(LATAM_NAMES))
        city = LATAM_NAMES[city_idx]
        country = LATAM_COUNTRIES[city_idx]
        rows.append(_make_customer(idx, city, country, "LATAM", None))
        idx += 1

    return rows


customer_rows = _generate_customer_rows()
print(f"Generated {len(customer_rows)} customer rows")
print(f"Sample: {json.dumps({k: str(v) for k, v in customer_rows[0].items()}, indent=2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b. Build Customer DataFrame & Write to Delta

# COMMAND ----------

CUSTOMER_SCHEMA = StructType([
    StructField("customer_id", StringType(), nullable=False),
    StructField("customer_name", StringType(), nullable=False),
    StructField("customer_segment", StringType(), nullable=False),
    StructField("customer_tier", StringType(), nullable=False),
    StructField("customer_country", StringType(), nullable=False),
    StructField("customer_region", StringType(), nullable=False),
    StructField("customer_state", StringType(), nullable=True),
    StructField("annual_spend", DoubleType(), nullable=False),
    StructField("contract_count", IntegerType(), nullable=False),
    StructField("first_purchase_date", DateType(), nullable=False),
    StructField("is_active", BooleanType(), nullable=False),
])

df_customers = spark.createDataFrame(customer_rows, schema=CUSTOMER_SCHEMA)

print(f"Customer DataFrame: {df_customers.count()} rows, {len(df_customers.columns)} columns")
df_customers.printSchema()

# COMMAND ----------

# --- Customer data quality checks ---
assert df_customers.count() == NUM_CUSTOMERS
assert df_customers.select("customer_id").distinct().count() == NUM_CUSTOMERS

# Regional distribution
print("Regional distribution:")
df_customers.groupBy("customer_region").count().orderBy("customer_region").show()

print("Segment distribution:")
df_customers.groupBy("customer_segment").count().orderBy("customer_segment").show()

print("Tier distribution:")
df_customers.groupBy("customer_tier").count().orderBy("customer_tier").show()

# US customers should have non-null state
us_no_state = df_customers.filter(
    (F.col("customer_region") == "US") & F.col("customer_state").isNull()
).count()
assert us_no_state == 0, f"Found {us_no_state} US customers with null state"
print("[CHECK] All US customers have a state assigned")

# International customers should have null state
intl_with_state = df_customers.filter(
    (F.col("customer_region") != "US") & F.col("customer_state").isNotNull()
).count()
assert intl_with_state == 0, f"Found {intl_with_state} international customers with state"
print("[CHECK] All international customers have null state")

print("\n=== Customer data quality checks passed ===")

# COMMAND ----------

# Write dim_customers to Delta
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {TARGET_CATALOG}.{TARGET_SCHEMA}")

(
    df_customers
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CUSTOMERS_TABLE)
)

print(f"Delta table written: {CUSTOMERS_TABLE} ({df_customers.count()} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. dim_sales_reps -- 75 Sales Representatives
# MAGIC
# MAGIC Schema:
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | sales_rep_id | STRING (PK) | Unique rep identifier |
# MAGIC | sales_rep_name | STRING | Full name |
# MAGIC | sales_rep_territory | STRING | Sales territory name |
# MAGIC | sales_rep_region | STRING | US / EMEA / APAC / LATAM |
# MAGIC | sales_rep_country | STRING | ISO country code |
# MAGIC | hire_date | DATE | Date of hire |
# MAGIC | is_active | BOOLEAN | Currently active flag |
# MAGIC | annual_quota | DOUBLE | Annual revenue quota (USD) |
# MAGIC | ytd_attainment_pct | DOUBLE | Year-to-date quota attainment % |
# MAGIC | is_high_discounter | BOOLEAN | Flagged as systematic over-discounter |

# COMMAND ----------

# ---------------------------------------------------------------------------
# Sales rep name components
# ---------------------------------------------------------------------------
FIRST_NAMES: List[str] = [
    "James", "Michael", "Robert", "David", "William", "John", "Richard",
    "Joseph", "Thomas", "Christopher", "Daniel", "Matthew", "Anthony",
    "Mark", "Steven", "Andrew", "Paul", "Kevin", "Brian", "Jason",
    "Jennifer", "Maria", "Sarah", "Jessica", "Emily", "Amanda", "Ashley",
    "Stephanie", "Nicole", "Melissa", "Rebecca", "Katherine", "Laura",
    "Rachel", "Heather", "Christine", "Lisa", "Samantha", "Angela", "Karen",
    "Alejandro", "Carlos", "Hiroshi", "Takeshi", "Hans", "Stefan",
    "Pierre", "Jean-Luc", "Marco", "Alessandro",
]

LAST_NAMES: List[str] = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "White", "Harris", "Clark", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Hill", "Scott", "Green",
    "Adams", "Baker", "Nelson", "Campbell", "Mitchell", "Roberts",
    "Mueller", "Schmidt", "Weber", "Fischer", "Nakamura", "Tanaka",
    "Dubois", "Moreau", "Rossi", "Bianchi",
]

TERRITORY_NAMES_US: List[str] = [
    "Northeast Corridor", "Mid-Atlantic", "Southeast", "Gulf Coast",
    "Great Lakes", "Upper Midwest", "Central Plains", "Mountain West",
    "Pacific Northwest", "California North", "California South",
    "Texas Metro", "Florida", "Carolinas", "New England",
    "Ohio Valley", "Tennessee Valley", "Southwest", "Rocky Mountain",
    "Alaska-Hawaii",
]

TERRITORY_NAMES_EMEA: List[str] = [
    "DACH", "UK & Ireland", "France", "Iberia", "Italy",
    "Benelux", "Nordics", "Central Europe", "Eastern Europe",
]

TERRITORY_NAMES_APAC: List[str] = [
    "Japan", "ANZ", "Greater China", "Southeast Asia", "Korea",
]

TERRITORY_NAMES_LATAM: List[str] = [
    "Mexico", "Brazil", "Southern Cone", "Andean",
]

# Indices of reps who will be systematic high discounters (0-indexed within the 75)
# These 7 reps will give 35-50% discounts consistently
HIGH_DISCOUNTER_INDICES: List[int] = [3, 14, 27, 38, 45, 58, 67]

# COMMAND ----------

def _generate_sales_rep_rows(seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    """Generate 75 sales rep rows deterministically.

    Regional allocation: ~45 US (60%), ~19 EMEA (25%), ~8 APAC (11%), ~3 LATAM (4%).
    7 reps are flagged as systematic high discounters.

    Returns
    -------
    List[Dict[str, Any]]
        Each dict maps column name to its scalar value.
    """
    rng = np.random.RandomState(seed + 100)  # Offset seed to avoid correlation with customers

    n_us = 45
    n_emea = 19
    n_apac = 8
    n_latam = 3
    assert n_us + n_emea + n_apac + n_latam == NUM_SALES_REPS

    rows: List[Dict[str, Any]] = []
    idx = 0

    def _make_rep(
        rep_idx: int,
        territory: str,
        region: str,
        country: str,
    ) -> Dict[str, Any]:
        sales_rep_id = f"REP-{rep_idx:04d}"

        first = FIRST_NAMES[rng.randint(0, len(FIRST_NAMES))]
        last = LAST_NAMES[rng.randint(0, len(LAST_NAMES))]
        sales_rep_name = f"{first} {last}"

        # Hire date: 2012-01-01 to 2024-06-30
        days_offset = int(rng.randint(0, 4565))
        hire_date = date(2012, 1, 1) + timedelta(days=days_offset)

        is_active = bool(rng.random() < 0.93)

        # Annual quota: $1.5M - $12M based on region and seniority
        tenure_years = (date(2024, 12, 31) - hire_date).days / 365.25
        base_quota = float(rng.uniform(1_500_000, 8_000_000))
        seniority_multiplier = 1.0 + min(tenure_years, 10) * 0.05  # up to 50% boost
        annual_quota = round(base_quota * seniority_multiplier, 2)

        # YTD attainment: mean ~85%, std ~20%
        ytd_attainment_pct = round(float(np.clip(rng.normal(85, 20), 30, 150)), 1)

        is_high_discounter = rep_idx in HIGH_DISCOUNTER_INDICES

        return {
            "sales_rep_id": sales_rep_id,
            "sales_rep_name": sales_rep_name,
            "sales_rep_territory": territory,
            "sales_rep_region": region,
            "sales_rep_country": country,
            "hire_date": hire_date,
            "is_active": is_active,
            "annual_quota": annual_quota,
            "ytd_attainment_pct": ytd_attainment_pct,
            "is_high_discounter": is_high_discounter,
        }

    # --- US reps (45) ---
    for i in range(n_us):
        territory = TERRITORY_NAMES_US[rng.randint(0, len(TERRITORY_NAMES_US))]
        rows.append(_make_rep(idx, territory, "US", "US"))
        idx += 1

    # --- EMEA reps (19) ---
    emea_country_map = {
        "DACH": "DE", "UK & Ireland": "GB", "France": "FR",
        "Iberia": "ES", "Italy": "IT", "Benelux": "NL",
        "Nordics": "SE", "Central Europe": "AT", "Eastern Europe": "PL",
    }
    for i in range(n_emea):
        territory = TERRITORY_NAMES_EMEA[rng.randint(0, len(TERRITORY_NAMES_EMEA))]
        country = emea_country_map.get(territory, "DE")
        rows.append(_make_rep(idx, territory, "EMEA", country))
        idx += 1

    # --- APAC reps (8) ---
    apac_country_map = {
        "Japan": "JP", "ANZ": "AU", "Greater China": "CN",
        "Southeast Asia": "SG", "Korea": "KR",
    }
    for i in range(n_apac):
        territory = TERRITORY_NAMES_APAC[rng.randint(0, len(TERRITORY_NAMES_APAC))]
        country = apac_country_map.get(territory, "JP")
        rows.append(_make_rep(idx, territory, "APAC", country))
        idx += 1

    # --- LATAM reps (3) ---
    latam_country_map = {
        "Mexico": "MX", "Brazil": "BR", "Southern Cone": "AR", "Andean": "CO",
    }
    for i in range(n_latam):
        territory = TERRITORY_NAMES_LATAM[rng.randint(0, len(TERRITORY_NAMES_LATAM))]
        country = latam_country_map.get(territory, "MX")
        rows.append(_make_rep(idx, territory, "LATAM", country))
        idx += 1

    return rows


sales_rep_rows = _generate_sales_rep_rows()
print(f"Generated {len(sales_rep_rows)} sales rep rows")
high_disc_count = sum(1 for r in sales_rep_rows if r["is_high_discounter"])
print(f"High discounter reps: {high_disc_count}")
print(f"High discounter IDs: {[r['sales_rep_id'] for r in sales_rep_rows if r['is_high_discounter']]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Build Sales Rep DataFrame & Write to Delta

# COMMAND ----------

SALES_REP_SCHEMA = StructType([
    StructField("sales_rep_id", StringType(), nullable=False),
    StructField("sales_rep_name", StringType(), nullable=False),
    StructField("sales_rep_territory", StringType(), nullable=False),
    StructField("sales_rep_region", StringType(), nullable=False),
    StructField("sales_rep_country", StringType(), nullable=False),
    StructField("hire_date", DateType(), nullable=False),
    StructField("is_active", BooleanType(), nullable=False),
    StructField("annual_quota", DoubleType(), nullable=False),
    StructField("ytd_attainment_pct", DoubleType(), nullable=False),
    StructField("is_high_discounter", BooleanType(), nullable=False),
])

df_sales_reps = spark.createDataFrame(sales_rep_rows, schema=SALES_REP_SCHEMA)

print(f"Sales Rep DataFrame: {df_sales_reps.count()} rows, {len(df_sales_reps.columns)} columns")
df_sales_reps.printSchema()

# COMMAND ----------

# --- Sales rep data quality checks ---
assert df_sales_reps.count() == NUM_SALES_REPS
assert df_sales_reps.select("sales_rep_id").distinct().count() == NUM_SALES_REPS

high_disc = df_sales_reps.filter(F.col("is_high_discounter") == True).count()
assert high_disc >= 5 and high_disc <= 8, f"Expected 5-8 high discounters, got {high_disc}"
print(f"[CHECK] High discounter count: {high_disc} (expected 5-8)")

print("Regional distribution:")
df_sales_reps.groupBy("sales_rep_region").count().orderBy("sales_rep_region").show()

print("\n=== Sales rep data quality checks passed ===")

# COMMAND ----------

(
    df_sales_reps
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SALES_REPS_TABLE)
)

print(f"Delta table written: {SALES_REPS_TABLE} ({df_sales_reps.count()} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. dim_products -- 200 Medical Device Products
# MAGIC
# MAGIC Schema:
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | product_id | STRING (PK) | Unique product identifier |
# MAGIC | sku | STRING | Stock keeping unit code |
# MAGIC | product_name | STRING | Realistic Stryker-branded product name |
# MAGIC | product_family | STRING | Endoscopy / Joint Replacement / Trauma & Extremities / Spine / Instruments / Neurovascular / Sports Medicine |
# MAGIC | product_category | STRING | Sub-category within family |
# MAGIC | business_unit | STRING | MedSurg / Ortho / Neuro |
# MAGIC | list_price | DOUBLE | Catalog list price (USD) |
# MAGIC | cogs_per_unit | DOUBLE | Cost of goods sold per unit (USD) |
# MAGIC | target_margin_pct | DOUBLE | Target gross margin percentage |
# MAGIC | is_sole_source | BOOLEAN | Whether product is sole-sourced |
# MAGIC | competitor_count | INT | Number of direct competitors |
# MAGIC | launch_date | DATE | Product launch date |
# MAGIC | lifecycle_stage | STRING | Launch / Growth / Mature / Decline |

# COMMAND ----------

# ---------------------------------------------------------------------------
# Product family configuration with categories, pricing, and unit economics
# ---------------------------------------------------------------------------
PRODUCT_FAMILY_CONFIG: Dict[str, Dict[str, Any]] = {
    "Endoscopy": {
        "business_unit": "MedSurg",
        "count": 30,
        "categories": [
            "4K Camera Systems", "Light Sources", "Arthroscopes",
            "Laparoscopes", "Insufflators", "Video Towers",
        ],
        "brands": ["1688", "1788", "ProCare", "CrossFire", "AIM"],
        "models": ["4K", "Elite", "HD", "Fluorescence", "Ultra", "i-Suite"],
        "descriptors": [
            "Camera Head", "Camera System", "Light Source", "Scope Assembly",
            "Insufflator Unit", "Image Management System", "Video Column",
        ],
        "list_price_range": (8_000.0, 350_000.0),
        "cogs_pct_range": (0.35, 0.55),
        "target_margin_range": (0.45, 0.65),
        "competitor_range": (3, 6),
        "sole_source_prob": 0.10,
    },
    "Joint Replacement": {
        "business_unit": "Ortho",
        "count": 40,
        "categories": [
            "Hip Systems", "Knee Systems", "Shoulder Systems",
            "Revision Systems", "Bearing Surfaces", "Cement & Accessories",
        ],
        "brands": ["Triathlon", "Accolade", "Trident", "Mako", "ReUnion", "Scorpio"],
        "models": ["II", "III", "Plus", "CR", "PS", "TS", "X3", "Tritanium"],
        "descriptors": [
            "Total Hip System", "Cementless Hip Stem", "Acetabular Shell",
            "Total Knee System", "Cruciate Retaining Knee", "Tibial Baseplate",
            "Femoral Component", "Reverse Shoulder", "Humeral Stem",
            "Revision System", "Bearing Insert",
        ],
        "list_price_range": (2_500.0, 12_000.0),
        "cogs_pct_range": (0.20, 0.35),
        "target_margin_range": (0.55, 0.75),
        "competitor_range": (3, 5),
        "sole_source_prob": 0.15,
    },
    "Trauma & Extremities": {
        "business_unit": "Ortho",
        "count": 35,
        "categories": [
            "Locking Plates", "IM Nails", "Screws & Fixation",
            "External Fixation", "Foot & Ankle", "Hand & Wrist",
        ],
        "brands": ["VariAx", "T2", "Gamma", "AxSOS", "Peri-Loc", "SolidLoc"],
        "models": ["2", "3", "Distal", "Proximal", "Lateral", "LCP", "Mini"],
        "descriptors": [
            "Distal Radius Plate", "Proximal Humerus Plate", "Femoral Nail",
            "Tibial Nail", "Clavicle Plate", "Locking Screw Set",
            "Cannulated Screw Set", "External Fixator", "Foot Plating System",
            "Wrist Fusion Plate",
        ],
        "list_price_range": (200.0, 6_000.0),
        "cogs_pct_range": (0.15, 0.30),
        "target_margin_range": (0.60, 0.80),
        "competitor_range": (4, 7),
        "sole_source_prob": 0.05,
    },
    "Spine": {
        "business_unit": "Neuro",
        "count": 25,
        "categories": [
            "Interbody Devices", "Pedicle Screw Systems", "Cervical Plates",
            "Spinal Cord Stimulators", "Biologics",
        ],
        "brands": ["Tritanium", "Serrato", "Xia", "ES2", "Mantis"],
        "models": ["PL", "C", "TL", "Advanced", "Mini", "Lateral"],
        "descriptors": [
            "Posterior Lumbar Interbody", "Transforaminal Interbody",
            "Cervical Plate System", "Pedicle Screw System",
            "Lateral Interbody Cage", "Anterior Cervical Plate",
            "Spinal Cord Stimulator", "Bone Graft Substitute",
        ],
        "list_price_range": (1_500.0, 25_000.0),
        "cogs_pct_range": (0.18, 0.32),
        "target_margin_range": (0.55, 0.75),
        "competitor_range": (3, 6),
        "sole_source_prob": 0.12,
    },
    "Instruments": {
        "business_unit": "MedSurg",
        "count": 30,
        "categories": [
            "Power Tools", "Blades & Burs", "Reaming Systems",
            "Surgical Lighting", "Waste Management",
        ],
        "brands": ["System", "Stryker", "Neptune", "Sabo", "Core"],
        "models": ["8", "9", "7+", "TPS", "RemB", "Plus", "Ultra"],
        "descriptors": [
            "Sagittal Saw", "Oscillating Saw", "Reciprocating Saw",
            "Battery-Powered Drill", "Reamer System", "Sternum Saw",
            "Craniotomy Perforator", "Surgical Headlight", "Waste Cartridge",
            "Blade Assembly",
        ],
        "list_price_range": (500.0, 45_000.0),
        "cogs_pct_range": (0.30, 0.50),
        "target_margin_range": (0.45, 0.65),
        "competitor_range": (3, 5),
        "sole_source_prob": 0.08,
    },
    "Neurovascular": {
        "business_unit": "Neuro",
        "count": 20,
        "categories": [
            "Detachable Coils", "Flow Diverters", "Stent Retrievers",
            "Microcatheters", "Aspiration Systems",
        ],
        "brands": ["Target", "Excelsior", "Surpass", "Neuroform", "AXS"],
        "models": ["360", "Ultra", "SL-10", "Evolve", "Atlas", "Nano", "Helical"],
        "descriptors": [
            "Detachable Coil", "Microcatheter", "Flow Diverter Stent",
            "Stent Retriever", "Intracranial Stent", "Aspiration Catheter",
            "Guiding Catheter",
        ],
        "list_price_range": (500.0, 8_000.0),
        "cogs_pct_range": (0.18, 0.30),
        "target_margin_range": (0.60, 0.78),
        "competitor_range": (2, 5),
        "sole_source_prob": 0.20,
    },
    "Sports Medicine": {
        "business_unit": "Ortho",
        "count": 20,
        "categories": [
            "Anchors & Sutures", "ACL Reconstruction", "Rotator Cuff Repair",
            "Arthroscopic Instruments",
        ],
        "brands": ["CrossFit", "ReelX", "Iconix", "SwiveLock", "Tiger"],
        "models": ["STT", "SP", "FT", "Mini", "XL", "Bio"],
        "descriptors": [
            "Suture Anchor", "ACL Graft System", "Rotator Cuff Anchor",
            "Meniscal Repair Device", "Labral Repair Anchor",
            "Arthroscopic Shaver", "Interference Screw",
        ],
        "list_price_range": (150.0, 3_000.0),
        "cogs_pct_range": (0.15, 0.28),
        "target_margin_range": (0.60, 0.80),
        "competitor_range": (4, 8),
        "sole_source_prob": 0.05,
    },
}

# Validate total product count
_total_products = sum(cfg["count"] for cfg in PRODUCT_FAMILY_CONFIG.values())
assert _total_products == NUM_PRODUCTS, f"Expected {NUM_PRODUCTS}, got {_total_products}"
print(f"Product config validated: {_total_products} products across {len(PRODUCT_FAMILY_CONFIG)} families")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4a. Product Generation Logic

# COMMAND ----------

LIFECYCLE_STAGES: List[str] = ["Launch", "Growth", "Mature", "Decline"]

def _generate_product_rows(seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    """Generate 200 product rows deterministically.

    Products are distributed across 7 families with realistic naming,
    pricing, and lifecycle attributes. Uses log-uniform distribution
    for list prices to handle wide ASP ranges.

    Returns
    -------
    List[Dict[str, Any]]
        Each dict maps column name to its scalar value.
    """
    rng = np.random.RandomState(seed + 200)  # Offset seed
    _uuid_ns = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

    rows: List[Dict[str, Any]] = []
    global_idx: int = 0

    for family in sorted(PRODUCT_FAMILY_CONFIG.keys()):
        cfg = PRODUCT_FAMILY_CONFIG[family]
        count = cfg["count"]
        bu = cfg["business_unit"]
        categories = cfg["categories"]
        brands = cfg["brands"]
        models = cfg["models"]
        descriptors = cfg["descriptors"]
        lp_lo, lp_hi = cfg["list_price_range"]
        cogs_lo, cogs_hi = cfg["cogs_pct_range"]
        margin_lo, margin_hi = cfg["target_margin_range"]
        comp_lo, comp_hi = cfg["competitor_range"]
        sole_source_prob = cfg["sole_source_prob"]

        for i in range(count):
            global_idx += 1

            product_id = str(uuid.uuid5(_uuid_ns, f"ficm-product-{global_idx:04d}"))
            sku = f"SKU-{family[:3].upper()}-{global_idx:04d}"

            brand = brands[rng.randint(0, len(brands))]
            model = models[rng.randint(0, len(models))]
            descriptor = descriptors[rng.randint(0, len(descriptors))]
            product_name = f"Stryker {brand} {model} {descriptor}"

            product_category = categories[rng.randint(0, len(categories))]

            # List price (log-uniform for wide ranges, uniform for narrow)
            if lp_hi / max(lp_lo, 1) > 10:
                list_price = float(np.exp(rng.uniform(np.log(lp_lo), np.log(lp_hi))))
            else:
                list_price = float(rng.uniform(lp_lo, lp_hi))
            list_price = round(list_price, 2)

            # COGS per unit
            cogs_pct = float(rng.uniform(cogs_lo, cogs_hi))
            cogs_per_unit = round(list_price * cogs_pct, 2)

            # Target margin
            target_margin_pct = round(float(rng.uniform(margin_lo, margin_hi)) * 100, 1)

            # Sole source
            is_sole_source = bool(rng.random() < sole_source_prob)

            # Competitor count (0 if sole source)
            if is_sole_source:
                competitor_count = 0
            else:
                competitor_count = int(rng.randint(comp_lo, comp_hi + 1))

            # Launch date (2005-2024)
            launch_year = int(np.clip(rng.normal(2017, 5), 2005, 2024))
            launch_month = int(rng.randint(1, 13))
            launch_day = int(rng.randint(1, 29))  # Avoid month-end edge cases
            launch_date = date(launch_year, launch_month, launch_day)

            # Lifecycle stage based on launch year
            years_since_launch = 2024 - launch_year
            if years_since_launch <= 2:
                lifecycle_stage = "Launch"
            elif years_since_launch <= 6:
                lifecycle_stage = "Growth"
            elif years_since_launch <= 14:
                lifecycle_stage = "Mature"
            else:
                lifecycle_stage = "Decline"

            rows.append({
                "product_id": product_id,
                "sku": sku,
                "product_name": product_name,
                "product_family": family,
                "product_category": product_category,
                "business_unit": bu,
                "list_price": list_price,
                "cogs_per_unit": cogs_per_unit,
                "target_margin_pct": target_margin_pct,
                "is_sole_source": is_sole_source,
                "competitor_count": competitor_count,
                "launch_date": launch_date,
                "lifecycle_stage": lifecycle_stage,
            })

    return rows


product_rows = _generate_product_rows()
print(f"Generated {len(product_rows)} product rows")
print(f"Sample: {json.dumps({k: str(v) for k, v in product_rows[0].items()}, indent=2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b. Build Product DataFrame & Write to Delta

# COMMAND ----------

PRODUCT_SCHEMA = StructType([
    StructField("product_id", StringType(), nullable=False),
    StructField("sku", StringType(), nullable=False),
    StructField("product_name", StringType(), nullable=False),
    StructField("product_family", StringType(), nullable=False),
    StructField("product_category", StringType(), nullable=False),
    StructField("business_unit", StringType(), nullable=False),
    StructField("list_price", DoubleType(), nullable=False),
    StructField("cogs_per_unit", DoubleType(), nullable=False),
    StructField("target_margin_pct", DoubleType(), nullable=False),
    StructField("is_sole_source", BooleanType(), nullable=False),
    StructField("competitor_count", IntegerType(), nullable=False),
    StructField("launch_date", DateType(), nullable=False),
    StructField("lifecycle_stage", StringType(), nullable=False),
])

df_products = spark.createDataFrame(product_rows, schema=PRODUCT_SCHEMA)

print(f"Product DataFrame: {df_products.count()} rows, {len(df_products.columns)} columns")
df_products.printSchema()

# COMMAND ----------

# --- Product data quality checks ---
assert df_products.count() == NUM_PRODUCTS
assert df_products.select("product_id").distinct().count() == NUM_PRODUCTS
assert df_products.select("sku").distinct().count() == NUM_PRODUCTS

print("Product family distribution:")
df_products.groupBy("product_family").count().orderBy("product_family").show()

print("Business unit distribution:")
df_products.groupBy("business_unit").count().orderBy("business_unit").show()

print("Lifecycle stage distribution:")
df_products.groupBy("lifecycle_stage").count().orderBy("lifecycle_stage").show()

# Price range sanity
price_stats = df_products.select(
    F.min("list_price").alias("min_price"),
    F.max("list_price").alias("max_price"),
    F.mean("list_price").alias("avg_price"),
).collect()[0]
print(f"[CHECK] Price range: ${price_stats['min_price']:,.2f} - ${price_stats['max_price']:,.2f} (avg: ${price_stats['avg_price']:,.2f})")

# COGS < list_price
invalid_cogs = df_products.filter(F.col("cogs_per_unit") >= F.col("list_price")).count()
assert invalid_cogs == 0, f"Found {invalid_cogs} products where COGS >= list_price"
print("[CHECK] COGS < list_price for all products")

# Sole source should have 0 competitors
sole_source_with_competitors = df_products.filter(
    (F.col("is_sole_source") == True) & (F.col("competitor_count") > 0)
).count()
assert sole_source_with_competitors == 0, f"Found {sole_source_with_competitors} sole-source products with competitors"
print("[CHECK] Sole source products have 0 competitors")

print("\n=== Product data quality checks passed ===")

# COMMAND ----------

(
    df_products
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(PRODUCTS_TABLE)
)

print(f"Delta table written: {PRODUCTS_TABLE} ({df_products.count()} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Post-Write Verification & Table Metadata

# COMMAND ----------

# Add table comments
_customer_comment = f"FICM dimension: {NUM_CUSTOMERS} synthetic hospital/facility customers across 4 regions (60% US, 25% EMEA, 10% APAC, 5% LATAM). Seed=42."
spark.sql(f"COMMENT ON TABLE {CUSTOMERS_TABLE} IS '{_customer_comment}'")

_rep_comment = f"FICM dimension: {NUM_SALES_REPS} sales representatives with {len(HIGH_DISCOUNTER_INDICES)} flagged high-discounters. Seed=42."
spark.sql(f"COMMENT ON TABLE {SALES_REPS_TABLE} IS '{_rep_comment}'")

_product_comment = f"FICM dimension: {NUM_PRODUCTS} medical device products across 7 families and 3 business units. Seed=42."
spark.sql(f"COMMENT ON TABLE {PRODUCTS_TABLE} IS '{_product_comment}'")

print("Table comments applied to all three dimension tables")

# COMMAND ----------

# Final verification: read back and confirm counts
for table_name, expected_count in [
    (CUSTOMERS_TABLE, NUM_CUSTOMERS),
    (SALES_REPS_TABLE, NUM_SALES_REPS),
    (PRODUCTS_TABLE, NUM_PRODUCTS),
]:
    verify_count = spark.table(table_name).count()
    assert verify_count == expected_count, f"{table_name}: expected {expected_count}, got {verify_count}"
    print(f"[VERIFIED] {table_name}: {verify_count} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary
# MAGIC
# MAGIC | Table | Rows | Key Attributes |
# MAGIC |-------|------|----------------|
# MAGIC | `hls_amer_catalog.silver.dim_customers` | 500 | 6 segments, 4 tiers, 60/25/10/5 regional split |
# MAGIC | `hls_amer_catalog.silver.dim_sales_reps` | 75 | 7 flagged high-discounters, 4 regional territories |
# MAGIC | `hls_amer_catalog.silver.dim_products` | 200 | 7 product families, 3 business units, lifecycle stages |
# MAGIC
# MAGIC **Next notebook**: `12_ficm_pricing_master.py` -- generates 600K FICM transactions referencing these dimensions.
