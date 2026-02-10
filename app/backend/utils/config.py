"""
Configuration module for Stryker Pricing Intelligence backend.

All settings are configurable via environment variables with sensible defaults
for Databricks Apps deployment.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Unity Catalog
# ---------------------------------------------------------------------------
CATALOG_NAME: str = os.getenv("CATALOG_NAME", "hls_amer_catalog")
SCHEMA_BRONZE: str = os.getenv("SCHEMA_BRONZE", "bronze")
SCHEMA_SILVER: str = os.getenv("SCHEMA_SILVER", "silver")
SCHEMA_GOLD: str = os.getenv("SCHEMA_GOLD", "gold")

# v2 Advanced Analytics schemas (isolated from v1)
SCHEMA_GOLD_V2: str = os.getenv("SCHEMA_GOLD_V2", "gold_v2")
SCHEMA_SILVER_V2: str = os.getenv("SCHEMA_SILVER_V2", "silver_v2")

# Fully-qualified table helpers
def _fqn(schema: str, table: str) -> str:
    """Return a fully-qualified three-level Unity Catalog table name."""
    return f"{CATALOG_NAME}.{schema}.{table}"


# Gold tables
TABLE_PRODUCTS: str = _fqn(SCHEMA_GOLD, "stryker_products")
TABLE_PRICE_WATERFALL: str = _fqn(SCHEMA_GOLD, "stryker_price_waterfall")
TABLE_COMPETITORS: str = _fqn(SCHEMA_GOLD, "stryker_competitors")
TABLE_EXTERNAL_FACTORS: str = _fqn(SCHEMA_GOLD, "stryker_external_factors")
TABLE_REVENUE_SUMMARY: str = _fqn(SCHEMA_GOLD, "stryker_revenue_summary")
TABLE_PRODUCT_FEATURES: str = _fqn(SCHEMA_GOLD, "stryker_product_features")

# ---------------------------------------------------------------------------
# SQL Warehouse
# ---------------------------------------------------------------------------
WAREHOUSE_ID: str = os.getenv("DATABRICKS_WAREHOUSE_ID", "your-warehouse-id")

# ---------------------------------------------------------------------------
# Model Serving
# ---------------------------------------------------------------------------
SERVING_ENDPOINT: str = os.getenv("SERVING_ENDPOINT", "stryker-pricing-models")

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
CACHE_TTL: int = int(os.getenv("CACHE_TTL", "300"))  # seconds

# ---------------------------------------------------------------------------
# Databricks connection (local dev fallback)
# ---------------------------------------------------------------------------
DATABRICKS_HOST: str = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN: str = os.getenv("DATABRICKS_TOKEN", "")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
APP_TITLE: str = "Stryker Pricing Intelligence"
APP_VERSION: str = "1.0.0"
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
STATIC_FILES_DIR: str = os.getenv("STATIC_FILES_DIR", "static")
