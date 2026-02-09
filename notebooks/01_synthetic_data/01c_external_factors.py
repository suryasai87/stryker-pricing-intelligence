# Databricks notebook source

# MAGIC %md
# MAGIC # 01c - External Market Factors (Synthetic Data Generation)
# MAGIC
# MAGIC **Purpose:** Generate 36 months of monthly macroeconomic and market indicators
# MAGIC (Jan 2023 - Dec 2025) for use in pricing intelligence models.
# MAGIC
# MAGIC **Output Table:** `hls_amer_catalog.bronze.market_external`
# MAGIC
# MAGIC **Schema:**
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | month | date | First day of each month |
# MAGIC | cpi_medical | double | Medical CPI year-over-year % |
# MAGIC | cpi_general | double | General CPI year-over-year % |
# MAGIC | tariff_rate_steel | double | Steel tariff rate % |
# MAGIC | tariff_rate_titanium | double | Titanium tariff rate % (Section 301) |
# MAGIC | fuel_index | double | Fuel price index (100 = Jan 2023) |
# MAGIC | container_freight_index | double | Container freight cost index |
# MAGIC | fed_funds_rate | double | Federal funds effective rate % |
# MAGIC | usd_eur | double | USD to EUR exchange rate |
# MAGIC | usd_jpy | double | USD to JPY exchange rate |
# MAGIC | supply_chain_pressure_index | double | Global supply chain pressure (GSCPI-inspired) |
# MAGIC | hospital_capex_index | double | Hospital capital expenditure index |
# MAGIC | cms_payment_update_pct | double | CMS payment update percentage |
# MAGIC | resin_price_index | double | Medical-grade resin price index |
# MAGIC | cobalt_chrome_price_index | double | Cobalt-chrome alloy price index |

# COMMAND ----------

import numpy as np
from datetime import date, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    DateType,
    DoubleType,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "hls_amer_catalog"
SCHEMA = "bronze"
TABLE_NAME = "market_external"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

SEED = 42
N_MONTHS = 36  # Jan 2023 - Dec 2025
START_YEAR = 2023
START_MONTH = 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def generate_months(start_year: int, start_month: int, n: int) -> list:
    """Generate a list of first-of-month dates.

    Args:
        start_year: Starting year.
        start_month: Starting month (1-12).
        n: Number of months to generate.

    Returns:
        List of ``datetime.date`` objects, one per month.
    """
    months = []
    year, month = start_year, start_month
    for _ in range(n):
        months.append(date(year, month, 1))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def ar1_series(
    rng: np.random.Generator,
    n: int,
    start: float,
    phi: float,
    sigma: float,
    trend_per_step: float = 0.0,
    lower: float = -np.inf,
    upper: float = np.inf,
) -> np.ndarray:
    """Generate an AR(1) time series with optional linear trend and bounds.

    The process follows:
        x[t] = phi * x[t-1] + (1 - phi) * mu[t] + eps[t]
    where mu[t] drifts linearly and eps ~ N(0, sigma^2).

    Args:
        rng: NumPy random generator instance.
        n: Length of the series.
        start: Initial value of the series.
        phi: Autoregressive coefficient (0 < phi < 1 for stationarity).
        sigma: Standard deviation of the innovation noise.
        trend_per_step: Linear drift added to the mean each step.
        lower: Hard lower bound (values are clipped).
        upper: Hard upper bound (values are clipped).

    Returns:
        1-D numpy array of length *n*.
    """
    series = np.empty(n)
    series[0] = start
    mu = start
    for t in range(1, n):
        mu += trend_per_step
        innovation = rng.normal(0, sigma)
        series[t] = phi * series[t - 1] + (1 - phi) * mu + innovation
        series[t] = np.clip(series[t], lower, upper)
    return series


def seasonal_component(n: int, amplitude: float, phase_months: int = 0) -> np.ndarray:
    """Generate a sinusoidal seasonal component with a 12-month period.

    Args:
        n: Number of data points (months).
        amplitude: Peak-to-trough half-range.
        phase_months: Phase offset in months (0 = peak at month 0).

    Returns:
        1-D numpy array of length *n*.
    """
    t = np.arange(n)
    return amplitude * np.sin(2 * np.pi * (t - phase_months) / 12.0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Time Series Data

# COMMAND ----------

rng = np.random.default_rng(SEED)
months = generate_months(START_YEAR, START_MONTH, N_MONTHS)

# ---------------------------------------------------------------------------
# CPI-Medical: 3.5 - 6.2 %, starting high ~5.8%, declining to ~3.8%
# Trend: gradual disinflation over 36 months
# ---------------------------------------------------------------------------
cpi_medical = ar1_series(
    rng, N_MONTHS,
    start=5.8, phi=0.92, sigma=0.12,
    trend_per_step=-0.055,
    lower=3.5, upper=6.2,
)

# ---------------------------------------------------------------------------
# CPI-General: 3.0 - 5.5 %, starting ~5.2%, declining faster than medical
# ---------------------------------------------------------------------------
cpi_general = ar1_series(
    rng, N_MONTHS,
    start=5.2, phi=0.90, sigma=0.15,
    trend_per_step=-0.06,
    lower=3.0, upper=5.5,
)

# ---------------------------------------------------------------------------
# Steel tariff rate: 25% base with +/- 3% fluctuation (Section 232)
# ---------------------------------------------------------------------------
tariff_rate_steel = ar1_series(
    rng, N_MONTHS,
    start=25.0, phi=0.85, sigma=0.5,
    trend_per_step=0.0,
    lower=22.0, upper=28.0,
)

# ---------------------------------------------------------------------------
# Titanium tariff rate: 0 - 15 %, Section 301 scenarios
# Step function: 0% for months 0-11, jump to ~7.5% months 12-23, ~12% months 24-35
# Plus small AR noise around each plateau
# ---------------------------------------------------------------------------
titanium_base = np.zeros(N_MONTHS)
titanium_base[:12] = 0.0    # 2023: no tariff
titanium_base[12:24] = 7.5  # 2024: Section 301 partial
titanium_base[24:] = 12.0   # 2025: escalation
tariff_rate_titanium = titanium_base + ar1_series(
    rng, N_MONTHS,
    start=0.0, phi=0.7, sigma=0.4,
    trend_per_step=0.0,
    lower=-2.0, upper=3.0,
)
tariff_rate_titanium = np.clip(tariff_rate_titanium, 0.0, 15.0)

# ---------------------------------------------------------------------------
# Fuel index: 80 - 140, base 100 = Jan 2023, seasonal pattern
# Summer peaks (~July), winter dips; mild upward trend in 2023, flat after
# ---------------------------------------------------------------------------
fuel_trend = ar1_series(
    rng, N_MONTHS,
    start=100.0, phi=0.93, sigma=2.0,
    trend_per_step=0.15,
    lower=80.0, upper=140.0,
)
fuel_seasonal = seasonal_component(N_MONTHS, amplitude=8.0, phase_months=6)  # peak ~ July
fuel_index = np.clip(fuel_trend + fuel_seasonal, 80.0, 140.0)

# ---------------------------------------------------------------------------
# Container freight index: 70 - 180
# Spike mid-2023 (~month 5-8), then normalize toward 90-100
# ---------------------------------------------------------------------------
freight_base = ar1_series(
    rng, N_MONTHS,
    start=130.0, phi=0.88, sigma=4.0,
    trend_per_step=-1.2,
    lower=70.0, upper=180.0,
)
# Add a spike centered around month 6 (July 2023)
freight_spike = np.zeros(N_MONTHS)
for i in range(N_MONTHS):
    freight_spike[i] = 45.0 * np.exp(-0.5 * ((i - 6) / 2.5) ** 2)
container_freight_index = np.clip(freight_base + freight_spike, 70.0, 180.0)

# ---------------------------------------------------------------------------
# Fed funds rate: 4.25 - 5.50 %
# Hiking cycle: 4.50 -> 5.25 by mid-2023, hold, then cuts starting late 2024
# ---------------------------------------------------------------------------
fed_target = np.zeros(N_MONTHS)
# 2023 hiking: 4.50 -> 5.25 by month 6, hold at 5.33
fed_target[:7] = np.linspace(4.50, 5.33, 7)
fed_target[7:18] = 5.33  # Hold through mid-2024
# 2024 H2: first cut
fed_target[18:24] = np.linspace(5.33, 5.00, 6)
# 2025: continued easing
fed_target[24:30] = np.linspace(5.00, 4.50, 6)
fed_target[30:] = np.linspace(4.50, 4.25, N_MONTHS - 30)
fed_funds_rate = fed_target + rng.normal(0, 0.02, N_MONTHS)
fed_funds_rate = np.clip(fed_funds_rate, 4.25, 5.50)
# Round to nearest 0.01 (basis-point precision)
fed_funds_rate = np.round(fed_funds_rate, 2)

# ---------------------------------------------------------------------------
# USD/EUR: 0.88 - 0.95, mean ~0.91, mild appreciation of EUR over time
# ---------------------------------------------------------------------------
usd_eur = ar1_series(
    rng, N_MONTHS,
    start=0.92, phi=0.94, sigma=0.005,
    trend_per_step=-0.0005,
    lower=0.88, upper=0.95,
)

# ---------------------------------------------------------------------------
# USD/JPY: 130 - 155, trend up (yen weakness) then partial reversal
# ---------------------------------------------------------------------------
usd_jpy_target = np.concatenate([
    np.linspace(132, 150, 18),   # 2023-mid2024: yen weakening
    np.linspace(150, 145, 10),   # mid2024-early2025: partial reversal
    np.linspace(145, 140, N_MONTHS - 28),  # 2025: further normalization
])
usd_jpy = usd_jpy_target + ar1_series(
    rng, N_MONTHS,
    start=0.0, phi=0.8, sigma=1.2,
    trend_per_step=0.0,
    lower=-8.0, upper=8.0,
)
usd_jpy = np.clip(usd_jpy, 130.0, 155.0)

# ---------------------------------------------------------------------------
# Supply chain pressure index: -1.5 to +3.5 (GSCPI-inspired)
# Declining from elevated levels; lingering spikes possible
# ---------------------------------------------------------------------------
supply_chain_pressure_index = ar1_series(
    rng, N_MONTHS,
    start=2.8, phi=0.90, sigma=0.25,
    trend_per_step=-0.1,
    lower=-1.5, upper=3.5,
)

# ---------------------------------------------------------------------------
# Hospital CapEx index: 85 - 115, post-COVID recovery
# Gradual recovery from ~90 toward ~108
# ---------------------------------------------------------------------------
hospital_capex_index = ar1_series(
    rng, N_MONTHS,
    start=90.0, phi=0.91, sigma=1.5,
    trend_per_step=0.45,
    lower=85.0, upper=115.0,
)

# ---------------------------------------------------------------------------
# CMS payment update %: 2.0 - 3.5 %
# Stepped annually (set once per fiscal year, Oct-Sep), with minor revisions
# FY2023: ~2.8%, FY2024: ~3.1%, FY2025: ~3.3%
# ---------------------------------------------------------------------------
cms_base = np.zeros(N_MONTHS)
# FY2023 (Oct 2022 - Sep 2023): months 0-8 (Jan-Sep 2023)
cms_base[:9] = 2.8
# FY2024 (Oct 2023 - Sep 2024): months 9-20 (Oct 2023 - Sep 2024)
cms_base[9:21] = 3.1
# FY2025 (Oct 2024 - Dec 2025): months 21-35
cms_base[21:] = 3.3
cms_payment_update_pct = cms_base + rng.normal(0, 0.05, N_MONTHS)
cms_payment_update_pct = np.clip(cms_payment_update_pct, 2.0, 3.5)
cms_payment_update_pct = np.round(cms_payment_update_pct, 2)

# ---------------------------------------------------------------------------
# Resin price index: 90 - 130
# Elevated in 2023, gradual normalization, mild seasonal variation
# ---------------------------------------------------------------------------
resin_trend = ar1_series(
    rng, N_MONTHS,
    start=120.0, phi=0.92, sigma=2.0,
    trend_per_step=-0.6,
    lower=90.0, upper=130.0,
)
resin_seasonal = seasonal_component(N_MONTHS, amplitude=4.0, phase_months=3)
resin_price_index = np.clip(resin_trend + resin_seasonal, 90.0, 130.0)

# ---------------------------------------------------------------------------
# Cobalt chrome price index: 95 - 145
# Volatile commodity; upward pressure from EV battery demand spillover
# ---------------------------------------------------------------------------
cobalt_chrome_price_index = ar1_series(
    rng, N_MONTHS,
    start=110.0, phi=0.88, sigma=3.5,
    trend_per_step=0.3,
    lower=95.0, upper=145.0,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assemble and Validate Data

# COMMAND ----------

# Round all continuous series to 2 decimal places for cleanliness
records = []
for i in range(N_MONTHS):
    records.append((
        months[i],
        round(float(cpi_medical[i]), 2),
        round(float(cpi_general[i]), 2),
        round(float(tariff_rate_steel[i]), 2),
        round(float(tariff_rate_titanium[i]), 2),
        round(float(fuel_index[i]), 2),
        round(float(container_freight_index[i]), 2),
        round(float(fed_funds_rate[i]), 2),
        round(float(usd_eur[i]), 4),         # FX to 4 decimals
        round(float(usd_jpy[i]), 2),
        round(float(supply_chain_pressure_index[i]), 2),
        round(float(hospital_capex_index[i]), 2),
        round(float(cms_payment_update_pct[i]), 2),
        round(float(resin_price_index[i]), 2),
        round(float(cobalt_chrome_price_index[i]), 2),
    ))

schema = StructType([
    StructField("month", DateType(), nullable=False),
    StructField("cpi_medical", DoubleType(), nullable=False),
    StructField("cpi_general", DoubleType(), nullable=False),
    StructField("tariff_rate_steel", DoubleType(), nullable=False),
    StructField("tariff_rate_titanium", DoubleType(), nullable=False),
    StructField("fuel_index", DoubleType(), nullable=False),
    StructField("container_freight_index", DoubleType(), nullable=False),
    StructField("fed_funds_rate", DoubleType(), nullable=False),
    StructField("usd_eur", DoubleType(), nullable=False),
    StructField("usd_jpy", DoubleType(), nullable=False),
    StructField("supply_chain_pressure_index", DoubleType(), nullable=False),
    StructField("hospital_capex_index", DoubleType(), nullable=False),
    StructField("cms_payment_update_pct", DoubleType(), nullable=False),
    StructField("resin_price_index", DoubleType(), nullable=False),
    StructField("cobalt_chrome_price_index", DoubleType(), nullable=False),
])

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame(records, schema=schema)

# COMMAND ----------

# Validation: confirm row count and date range
assert df.count() == N_MONTHS, f"Expected {N_MONTHS} rows, got {df.count()}"

date_bounds = df.selectExpr("min(month) as min_dt", "max(month) as max_dt").first()
assert str(date_bounds["min_dt"]) == "2023-01-01", f"Unexpected min date: {date_bounds['min_dt']}"
assert str(date_bounds["max_dt"]) == "2025-12-01", f"Unexpected max date: {date_bounds['max_dt']}"

print(f"Validation passed: {N_MONTHS} rows, {date_bounds['min_dt']} to {date_bounds['max_dt']}")

# COMMAND ----------

# Validation: confirm all values fall within specified ranges
from pyspark.sql import functions as F

bounds = {
    "cpi_medical": (3.5, 6.2),
    "cpi_general": (3.0, 5.5),
    "tariff_rate_steel": (22.0, 28.0),
    "tariff_rate_titanium": (0.0, 15.0),
    "fuel_index": (80.0, 140.0),
    "container_freight_index": (70.0, 180.0),
    "fed_funds_rate": (4.25, 5.50),
    "usd_eur": (0.88, 0.95),
    "usd_jpy": (130.0, 155.0),
    "supply_chain_pressure_index": (-1.5, 3.5),
    "hospital_capex_index": (85.0, 115.0),
    "cms_payment_update_pct": (2.0, 3.5),
    "resin_price_index": (90.0, 130.0),
    "cobalt_chrome_price_index": (95.0, 145.0),
}

for col_name, (lo, hi) in bounds.items():
    stats = df.select(
        F.min(col_name).alias("col_min"),
        F.max(col_name).alias("col_max"),
    ).first()
    col_min, col_max = stats["col_min"], stats["col_max"]
    assert col_min >= lo, f"{col_name} min {col_min} below lower bound {lo}"
    assert col_max <= hi, f"{col_name} max {col_max} above upper bound {hi}"
    print(f"  {col_name:35s}  [{col_min:>8.2f}, {col_max:>8.2f}]  OK")

print("\nAll range validations passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview Sample Data

# COMMAND ----------

display(df.orderBy("month"))

# COMMAND ----------

# Summary statistics for documentation and review
df.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Delta Table

# COMMAND ----------

# Ensure the target schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FULL_TABLE_NAME)
)

print(f"Successfully wrote {df.count()} rows to {FULL_TABLE_NAME}")

# COMMAND ----------

# Add table comment for discoverability
spark.sql(f"""
    ALTER TABLE {FULL_TABLE_NAME}
    SET TBLPROPERTIES (
        'quality' = 'bronze',
        'source' = 'synthetic',
        'generator' = '01c_external_factors',
        'seed' = '42',
        'date_range' = '2023-01-01 to 2025-12-01',
        'description' = 'Monthly macroeconomic and market indicators for pricing intelligence models. Synthetic data with realistic autocorrelation, trends, and seasonal patterns.'
    )
""")

# COMMAND ----------

# Final verification: read back from table and confirm
verification_df = spark.table(FULL_TABLE_NAME)
assert verification_df.count() == N_MONTHS, "Row count mismatch after write"
print(f"Verification complete: {FULL_TABLE_NAME} contains {verification_df.count()} rows")
verification_df.printSchema()
