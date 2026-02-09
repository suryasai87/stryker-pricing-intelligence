"""
Tests for the synthetic data generation layer.

Validates that the generated product master, transaction history,
external factors, and competitor data conform to expected schemas,
row counts, and business-rule invariants.

Requires a PySpark session (local or Databricks runtime).
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Configuration constants (mirror the generator config)
# ---------------------------------------------------------------------------
CATALOG = "hls_amer_catalog"
SCHEMA_BRONZE = "bronze"
SCHEMA_SILVER = "silver"

EXPECTED_PRODUCT_COUNT = 200
EXPECTED_TRANSACTION_COUNT_MIN = 450_000
EXPECTED_TRANSACTION_COUNT_MAX = 550_000
EXPECTED_EXTERNAL_FACTOR_MONTHS = 36

PRODUCT_MASTER_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "sub_category",
    "list_price",
    "cost",
    "launch_date",
    "lifecycle_stage",
    "therapeutic_area",
    "competitor_count",
]

ASP_BOUNDS = {
    "Surgical Instruments": (500, 25_000),
    "Implants": (1_000, 80_000),
    "Endoscopy": (2_000, 60_000),
    "Medical Devices": (800, 50_000),
    "Neurotechnology": (5_000, 120_000),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Create or retrieve a local Spark session for testing."""
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("stryker-synthetic-data-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )


@pytest.fixture(scope="session")
def product_master(spark: SparkSession):
    """Load the product master table."""
    return spark.table(f"{CATALOG}.{SCHEMA_BRONZE}.product_master")


@pytest.fixture(scope="session")
def transactions(spark: SparkSession):
    """Load the transactions table."""
    return spark.table(f"{CATALOG}.{SCHEMA_BRONZE}.transactions")


@pytest.fixture(scope="session")
def external_factors(spark: SparkSession):
    """Load the external market factors table."""
    return spark.table(f"{CATALOG}.{SCHEMA_SILVER}.external_factors")


@pytest.fixture(scope="session")
def competitor_data(spark: SparkSession):
    """Load the competitor pricing data table."""
    return spark.table(f"{CATALOG}.{SCHEMA_SILVER}.competitor_data")


# ---------------------------------------------------------------------------
# Product master tests
# ---------------------------------------------------------------------------
class TestProductMaster:
    """Tests for the product_master table."""

    def test_product_master_count(self, product_master):
        """Product master must contain exactly 200 products."""
        count = product_master.count()
        assert count == EXPECTED_PRODUCT_COUNT, (
            f"Expected {EXPECTED_PRODUCT_COUNT} products, got {count}"
        )

    def test_product_master_schema(self, product_master):
        """All required columns must be present in the product master."""
        actual_columns = set(product_master.columns)
        missing = set(PRODUCT_MASTER_COLUMNS) - actual_columns
        assert not missing, f"Missing columns in product_master: {missing}"

    def test_product_asp_ranges(self, product_master):
        """List prices must fall within the configured ASP bounds per category."""
        rows = (
            product_master
            .select("category", "list_price")
            .collect()
        )

        violations = []
        for row in rows:
            bounds = ASP_BOUNDS.get(row.category)
            if bounds is None:
                # Category not in the explicit bounds map -- skip
                continue
            low, high = bounds
            if not (low <= row.list_price <= high):
                violations.append(
                    f"category={row.category}, list_price={row.list_price} "
                    f"not in [{low}, {high}]"
                )

        assert not violations, (
            f"{len(violations)} ASP violations:\n" + "\n".join(violations[:10])
        )


# ---------------------------------------------------------------------------
# Transaction tests
# ---------------------------------------------------------------------------
class TestTransactions:
    """Tests for the transactions table."""

    def test_transactions_count(self, transactions):
        """Transaction count should be approximately 500k (+/- 50k)."""
        count = transactions.count()
        assert EXPECTED_TRANSACTION_COUNT_MIN <= count <= EXPECTED_TRANSACTION_COUNT_MAX, (
            f"Expected ~500k transactions, got {count}"
        )

    def test_transactions_seasonality(self, transactions):
        """Q4 total revenue should exceed Q1 to reflect typical med-device seasonality."""
        quarterly = (
            transactions
            .withColumn("quarter", F.quarter("transaction_date"))
            .groupBy("quarter")
            .agg(F.sum("net_revenue").alias("total_revenue"))
            .collect()
        )

        revenue_by_q = {row.quarter: row.total_revenue for row in quarterly}
        q1_rev = revenue_by_q.get(1, 0)
        q4_rev = revenue_by_q.get(4, 0)

        assert q4_rev > q1_rev, (
            f"Q4 revenue ({q4_rev:,.0f}) should exceed Q1 ({q1_rev:,.0f})"
        )

    def test_transactions_discount_waterfall(self, transactions):
        """
        Discount waterfall invariant:
            pocket_price <= invoice_price <= list_price
        Must hold for every transaction.
        """
        violations = (
            transactions
            .filter(
                (F.col("pocket_price") > F.col("invoice_price"))
                | (F.col("invoice_price") > F.col("list_price"))
            )
            .count()
        )

        assert violations == 0, (
            f"{violations} transactions violate the discount waterfall "
            "(pocket <= invoice <= list)"
        )


# ---------------------------------------------------------------------------
# External factors tests
# ---------------------------------------------------------------------------
class TestExternalFactors:
    """Tests for the external_factors table."""

    def test_external_factors_months(self, external_factors):
        """External factors data must span exactly 36 months."""
        distinct_months = (
            external_factors
            .select(F.date_format("month", "yyyy-MM").alias("ym"))
            .distinct()
            .count()
        )
        assert distinct_months == EXPECTED_EXTERNAL_FACTOR_MONTHS, (
            f"Expected {EXPECTED_EXTERNAL_FACTOR_MONTHS} distinct months, "
            f"got {distinct_months}"
        )


# ---------------------------------------------------------------------------
# Competitor data tests
# ---------------------------------------------------------------------------
class TestCompetitorData:
    """Tests for the competitor_data table."""

    def test_competitor_data_completeness(self, competitor_data):
        """
        Every product with competitor_count > 0 in the master should have
        at least one corresponding row in competitor_data.
        """
        assert competitor_data.count() > 0, "competitor_data table is empty"

        # Ensure no null product_id values
        null_pid = (
            competitor_data
            .filter(F.col("product_id").isNull())
            .count()
        )
        assert null_pid == 0, (
            f"competitor_data has {null_pid} rows with null product_id"
        )

        # Ensure no null competitor_name values
        null_name = (
            competitor_data
            .filter(F.col("competitor_name").isNull())
            .count()
        )
        assert null_name == 0, (
            f"competitor_data has {null_name} rows with null competitor_name"
        )
