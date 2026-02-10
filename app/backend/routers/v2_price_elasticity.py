"""
Price elasticity analysis router.

Provides access to precomputed price elasticity coefficients, safe pricing
ranges, and distribution data for histogram visualisation.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from backend.utils.config import CATALOG_NAME, SCHEMA_GOLD_V2
from backend.utils.databricks_client import execute_sql

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/price-elasticity", tags=["price-elasticity"])

_TABLE = f"{CATALOG_NAME}.{SCHEMA_GOLD_V2}.price_elasticity"


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
@router.get(
    "/",
    summary="List price elasticity records with optional filters",
)
async def get_price_elasticity(
    business_unit: str | None = Query(None, description="Filter by business unit"),
    customer_segment: str | None = Query(None, description="Filter by customer segment"),
    elasticity_class: str | None = Query(None, description="Filter by elasticity class"),
    product_family: str | None = Query(None, description="Filter by product family"),
    confidence: str | None = Query(None, description="Filter by confidence level"),
) -> dict[str, Any]:
    """Return price elasticity records matching the specified filters."""
    try:
        where_clauses: list[str] = []

        if business_unit:
            where_clauses.append(f"business_unit = '{business_unit}'")
        if customer_segment:
            where_clauses.append(f"customer_segment = '{customer_segment}'")
        if elasticity_class:
            where_clauses.append(f"elasticity_class = '{elasticity_class}'")
        if product_family:
            where_clauses.append(f"product_family = '{product_family}'")
        if confidence:
            where_clauses.append(f"confidence = '{confidence}'")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query = f"""
            SELECT *
            FROM {_TABLE}
            {where_sql}
            ORDER BY elasticity_coefficient ASC
        """

        cache_key = (
            f"price_elasticity:list:{business_unit}:{customer_segment}"
            f":{elasticity_class}:{product_family}:{confidence}"
        )
        rows = execute_sql(query, cache_key=cache_key)

        return {
            "count": len(rows),
            "filters": {
                "business_unit": business_unit,
                "customer_segment": customer_segment,
                "elasticity_class": elasticity_class,
                "product_family": product_family,
                "confidence": confidence,
            },
            "data": rows,
        }
    except Exception as exc:
        logger.exception("Failed to fetch price elasticity data")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /safe-ranges
# ---------------------------------------------------------------------------
@router.get(
    "/safe-ranges",
    summary="Safe pricing ranges for a SKU and segment",
)
async def get_safe_ranges(
    sku: str = Query(..., description="SKU identifier"),
    customer_segment: str | None = Query(None, description="Customer segment filter"),
) -> dict[str, Any]:
    """Return detailed safe pricing ranges for a specific SKU, optionally
    filtered by customer segment.
    """
    try:
        where_clauses: list[str] = [f"sku = '{sku}'"]
        if customer_segment:
            where_clauses.append(f"customer_segment = '{customer_segment}'")

        where_sql = f"WHERE {' AND '.join(where_clauses)}"

        query = f"""
            SELECT *
            FROM {_TABLE}
            {where_sql}
            ORDER BY customer_segment
        """

        cache_key = f"price_elasticity:safe_ranges:{sku}:{customer_segment}"
        rows = execute_sql(query, cache_key=cache_key)

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No elasticity data found for SKU '{sku}'",
            )

        return {
            "sku": sku,
            "customer_segment": customer_segment,
            "count": len(rows),
            "data": rows,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch safe ranges for SKU %s", sku)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /distribution
# ---------------------------------------------------------------------------
@router.get(
    "/distribution",
    summary="Histogram data for elasticity coefficient distribution",
)
async def get_elasticity_distribution() -> dict[str, Any]:
    """Return histogram bucket data of elasticity coefficients for
    visualisation.
    """
    try:
        query = f"""
            SELECT
                FLOOR(elasticity_coefficient * 10) / 10 AS bucket,
                COUNT(*)                                AS count
            FROM {_TABLE}
            GROUP BY FLOOR(elasticity_coefficient * 10) / 10
            ORDER BY bucket ASC
        """
        rows = execute_sql(query, cache_key="price_elasticity:distribution")

        return {
            "bucket_count": len(rows),
            "data": rows,
        }
    except Exception as exc:
        logger.exception("Failed to fetch elasticity distribution")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
