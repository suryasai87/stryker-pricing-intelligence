"""
Pricing recommendations router.

Surfaces ML-generated pricing recommendations with filtering by action type,
risk level, business unit, and product family.  Includes summary KPIs
aggregated by action type.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from backend.utils.config import CATALOG_NAME, SCHEMA_GOLD_V2
from backend.utils.databricks_client import execute_sql

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2/pricing-recommendations", tags=["recommendations"]
)

_TABLE = f"{CATALOG_NAME}.{SCHEMA_GOLD_V2}.pricing_recommendations"

# Allowed sort columns
_ALLOWED_SORT_COLUMNS = {
    "priority_score",
    "revenue_impact",
    "action_type",
    "risk_level",
    "product_family",
    "business_unit",
}


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
@router.get(
    "/",
    summary="List pricing recommendations with optional filters",
)
async def get_pricing_recommendations(
    action_type: str | None = Query(None, description="Filter by action type"),
    risk_level: str | None = Query(None, description="Filter by risk level"),
    business_unit: str | None = Query(None, description="Filter by business unit"),
    product_family: str | None = Query(None, description="Filter by product family"),
    limit: int = Query(100, ge=1, le=5000, description="Max rows returned"),
    sort_by: str = Query("priority_score", description="Column to sort by"),
) -> dict[str, Any]:
    """Return pricing recommendations matching the specified filters."""
    try:
        where_clauses: list[str] = []

        if action_type:
            where_clauses.append(f"action_type = '{action_type}'")
        if risk_level:
            where_clauses.append(f"risk_level = '{risk_level}'")
        if business_unit:
            where_clauses.append(f"business_unit = '{business_unit}'")
        if product_family:
            where_clauses.append(f"product_family = '{product_family}'")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        safe_sort = sort_by if sort_by in _ALLOWED_SORT_COLUMNS else "priority_score"

        query = f"""
            SELECT *
            FROM {_TABLE}
            {where_sql}
            ORDER BY {safe_sort} DESC
            LIMIT {limit}
        """

        cache_key = (
            f"recommendations:list:{action_type}:{risk_level}"
            f":{business_unit}:{product_family}:{limit}:{safe_sort}"
        )
        rows = execute_sql(query, cache_key=cache_key)

        return {
            "count": len(rows),
            "filters": {
                "action_type": action_type,
                "risk_level": risk_level,
                "business_unit": business_unit,
                "product_family": product_family,
            },
            "sort_by": safe_sort,
            "data": rows,
        }
    except Exception as exc:
        logger.exception("Failed to fetch pricing recommendations")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /summary
# ---------------------------------------------------------------------------
@router.get(
    "/summary",
    summary="Summary KPIs by action type",
)
async def get_recommendations_summary() -> dict[str, Any]:
    """Return summary KPIs grouped by action type, including counts,
    average priority score, and total revenue impact.
    """
    try:
        query = f"""
            SELECT
                action_type,
                COUNT(*)                                    AS count,
                COALESCE(AVG(priority_score), 0)            AS avg_priority_score,
                COALESCE(SUM(revenue_impact), 0)            AS total_revenue_impact,
                COALESCE(AVG(revenue_impact), 0)            AS avg_revenue_impact
            FROM {_TABLE}
            GROUP BY action_type
            ORDER BY total_revenue_impact DESC
        """
        rows = execute_sql(query, cache_key="recommendations:summary")

        return {
            "by_action_type": [
                {
                    "action_type": str(r.get("action_type", "")),
                    "count": int(r.get("count", 0)),
                    "avg_priority_score": float(r.get("avg_priority_score", 0)),
                    "total_revenue_impact": float(r.get("total_revenue_impact", 0)),
                    "avg_revenue_impact": float(r.get("avg_revenue_impact", 0)),
                }
                for r in rows
            ],
        }
    except Exception as exc:
        logger.exception("Failed to fetch recommendations summary")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
