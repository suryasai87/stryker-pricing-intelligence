"""
Discount outlier detection router.

Surfaces discount outliers identified by Z-score analysis, with filtering by
business unit, customer segment, country, severity, and Z-score threshold.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from backend.utils.config import CATALOG_NAME, SCHEMA_GOLD_V2
from backend.utils.databricks_client import execute_sql

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/discount-outliers", tags=["discount-outliers"])

_TABLE = f"{CATALOG_NAME}.{SCHEMA_GOLD_V2}.discount_outliers"


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
@router.get(
    "/",
    summary="List discount outliers with optional filters",
)
async def get_discount_outliers(
    business_unit: str | None = Query(None, description="Filter by business unit"),
    customer_segment: str | None = Query(None, description="Filter by customer segment"),
    customer_country: str | None = Query(None, description="Filter by customer country"),
    min_z_score: float = Query(2.0, ge=0.0, description="Minimum Z-score threshold"),
    severity: str | None = Query(None, description="Filter by severity level"),
    limit: int = Query(100, ge=1, le=5000, description="Max rows returned"),
) -> dict[str, Any]:
    """Return discount outliers matching the specified filter criteria."""
    try:
        where_clauses: list[str] = [f"z_score >= {min_z_score}"]

        if business_unit:
            where_clauses.append(f"business_unit = '{business_unit}'")
        if customer_segment:
            where_clauses.append(f"customer_segment = '{customer_segment}'")
        if customer_country:
            where_clauses.append(f"customer_country = '{customer_country}'")
        if severity:
            where_clauses.append(f"severity = '{severity}'")

        where_sql = f"WHERE {' AND '.join(where_clauses)}"

        query = f"""
            SELECT *
            FROM {_TABLE}
            {where_sql}
            ORDER BY z_score DESC
            LIMIT {limit}
        """

        cache_key = (
            f"discount_outliers:list:{business_unit}:{customer_segment}"
            f":{customer_country}:{min_z_score}:{severity}:{limit}"
        )
        rows = execute_sql(query, cache_key=cache_key)

        return {
            "count": len(rows),
            "filters": {
                "business_unit": business_unit,
                "customer_segment": customer_segment,
                "customer_country": customer_country,
                "min_z_score": min_z_score,
                "severity": severity,
            },
            "data": rows,
        }
    except Exception as exc:
        logger.exception("Failed to fetch discount outliers")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /summary
# ---------------------------------------------------------------------------
@router.get(
    "/summary",
    summary="Aggregate discount outlier statistics",
)
async def get_discount_outlier_summary() -> dict[str, Any]:
    """Return aggregate stats: total outliers, total recovery opportunity,
    and breakdowns by business unit, severity, and country.
    """
    try:
        # Overall totals
        totals_query = f"""
            SELECT
                COUNT(*)                       AS total_outliers,
                COALESCE(SUM(recovery_amount), 0) AS total_recovery
            FROM {_TABLE}
        """
        totals_rows = execute_sql(totals_query, cache_key="discount_outliers:summary:totals")
        totals = totals_rows[0] if totals_rows else {}

        # By business unit
        bu_query = f"""
            SELECT business_unit, COUNT(*) AS count,
                   COALESCE(SUM(recovery_amount), 0) AS recovery
            FROM {_TABLE}
            GROUP BY business_unit
            ORDER BY count DESC
        """
        bu_rows = execute_sql(bu_query, cache_key="discount_outliers:summary:bu")

        # By severity
        severity_query = f"""
            SELECT severity, COUNT(*) AS count,
                   COALESCE(SUM(recovery_amount), 0) AS recovery
            FROM {_TABLE}
            GROUP BY severity
            ORDER BY count DESC
        """
        severity_rows = execute_sql(
            severity_query, cache_key="discount_outliers:summary:severity"
        )

        # By country
        country_query = f"""
            SELECT customer_country AS country, COUNT(*) AS count,
                   COALESCE(SUM(recovery_amount), 0) AS recovery
            FROM {_TABLE}
            GROUP BY customer_country
            ORDER BY count DESC
        """
        country_rows = execute_sql(
            country_query, cache_key="discount_outliers:summary:country"
        )

        return {
            "total_outliers": int(totals.get("total_outliers", 0)),
            "total_recovery": float(totals.get("total_recovery", 0)),
            "by_business_unit": bu_rows,
            "by_severity": severity_rows,
            "by_country": country_rows,
        }
    except Exception as exc:
        logger.exception("Failed to fetch discount outlier summary")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /by-rep
# ---------------------------------------------------------------------------
@router.get(
    "/by-rep",
    summary="Discount outliers for a specific sales rep",
)
async def get_outliers_by_rep(
    sales_rep_id: str = Query(..., description="Sales rep identifier"),
) -> dict[str, Any]:
    """Return all discount outliers associated with a specific sales rep."""
    try:
        query = f"""
            SELECT *
            FROM {_TABLE}
            WHERE sales_rep_id = '{sales_rep_id}'
            ORDER BY z_score DESC
        """
        rows = execute_sql(query, cache_key=f"discount_outliers:rep:{sales_rep_id}")

        return {
            "sales_rep_id": sales_rep_id,
            "count": len(rows),
            "data": rows,
        }
    except Exception as exc:
        logger.exception(
            "Failed to fetch discount outliers for rep %s", sales_rep_id
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
