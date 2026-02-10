"""
FICM (Field Inventory & Contract Management) pricing master router.

Provides summary statistics and schema introspection for the
``silver.ficm_pricing_master`` table.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.utils.config import CATALOG_NAME, SCHEMA_SILVER
from backend.utils.databricks_client import execute_sql

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/ficm", tags=["ficm"])

_TABLE = f"{CATALOG_NAME}.{SCHEMA_SILVER}.ficm_pricing_master"


# ---------------------------------------------------------------------------
# GET /summary
# ---------------------------------------------------------------------------
@router.get(
    "/summary",
    summary="FICM pricing master summary statistics",
)
async def get_ficm_summary() -> dict[str, Any]:
    """Return row count, date range, and breakdowns by country, segment,
    and product family from the FICM pricing master table.
    """
    try:
        # Overall stats
        overview_query = f"""
            SELECT
                COUNT(*)                       AS row_count,
                MIN(pricing_date)              AS min_date,
                MAX(pricing_date)              AS max_date
            FROM {_TABLE}
        """
        overview_rows = execute_sql(
            overview_query,
            cache_key="ficm:summary:overview",
            schema=SCHEMA_SILVER,
        )
        overview = overview_rows[0] if overview_rows else {}

        # Country breakdown
        country_query = f"""
            SELECT customer_country AS country, COUNT(*) AS count
            FROM {_TABLE}
            GROUP BY customer_country
            ORDER BY count DESC
        """
        country_rows = execute_sql(
            country_query,
            cache_key="ficm:summary:country",
            schema=SCHEMA_SILVER,
        )

        # Segment breakdown
        segment_query = f"""
            SELECT customer_segment AS segment, COUNT(*) AS count
            FROM {_TABLE}
            GROUP BY customer_segment
            ORDER BY count DESC
        """
        segment_rows = execute_sql(
            segment_query,
            cache_key="ficm:summary:segment",
            schema=SCHEMA_SILVER,
        )

        # Product family breakdown
        product_family_query = f"""
            SELECT product_family, COUNT(*) AS count
            FROM {_TABLE}
            GROUP BY product_family
            ORDER BY count DESC
        """
        product_family_rows = execute_sql(
            product_family_query,
            cache_key="ficm:summary:product_family",
            schema=SCHEMA_SILVER,
        )

        return {
            "row_count": int(overview.get("row_count", 0)),
            "date_range": {
                "min": overview.get("min_date"),
                "max": overview.get("max_date"),
            },
            "country_breakdown": country_rows,
            "segment_breakdown": segment_rows,
            "product_family_breakdown": product_family_rows,
        }
    except Exception as exc:
        logger.exception("Failed to fetch FICM summary")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /schema
# ---------------------------------------------------------------------------
@router.get(
    "/schema",
    summary="FICM pricing master column schema",
)
async def get_ficm_schema() -> dict[str, Any]:
    """Return the full column list with data types for the FICM pricing
    master table.
    """
    try:
        schema_query = f"DESCRIBE TABLE {_TABLE}"
        rows = execute_sql(
            schema_query,
            cache_key="ficm:schema",
            schema=SCHEMA_SILVER,
        )

        columns = [
            {
                "column_name": str(r.get("col_name", "")),
                "data_type": str(r.get("data_type", "")),
                "comment": str(r.get("comment", "")),
            }
            for r in rows
        ]

        return {
            "table": _TABLE,
            "column_count": len(columns),
            "columns": columns,
        }
    except Exception as exc:
        logger.exception("Failed to fetch FICM schema")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
