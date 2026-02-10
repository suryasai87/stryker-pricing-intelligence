"""
Uplift simulation router.

Supports both on-the-fly uplift simulations (POST) and retrieval of
precomputed simulation results from the gold layer.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.utils.config import CATALOG_NAME, SCHEMA_GOLD_V2
from backend.utils.databricks_client import execute_sql

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/uplift-simulation", tags=["uplift-simulation"])

_TABLE = f"{CATALOG_NAME}.{SCHEMA_GOLD_V2}.uplift_simulation"


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class UpliftSimulationRequest(BaseModel):
    """Payload for an on-the-fly uplift simulation."""

    target_uplift_pct: float = Field(
        ..., description="Target revenue uplift percentage"
    )
    excluded_skus: list[str] = Field(
        default_factory=list, description="SKUs to exclude from the simulation"
    )
    excluded_segments: list[str] = Field(
        default_factory=list, description="Segments to exclude"
    )
    excluded_countries: list[str] = Field(
        default_factory=list, description="Countries to exclude"
    )
    max_per_sku_increase: float = Field(
        5.0, description="Maximum price increase per SKU (%)"
    )


# ---------------------------------------------------------------------------
# POST /
# ---------------------------------------------------------------------------
@router.post(
    "/",
    summary="Run an on-the-fly uplift simulation",
)
async def run_uplift_simulation(
    request: UpliftSimulationRequest,
) -> dict[str, Any]:
    """Run an uplift simulation with the given constraints.

    Queries the precomputed uplift data and applies the requested exclusions
    and constraints to produce a filtered action plan.
    """
    try:
        where_clauses: list[str] = []

        if request.excluded_skus:
            sku_list = ", ".join(f"'{s}'" for s in request.excluded_skus)
            where_clauses.append(f"sku NOT IN ({sku_list})")
        if request.excluded_segments:
            seg_list = ", ".join(f"'{s}'" for s in request.excluded_segments)
            where_clauses.append(f"customer_segment NOT IN ({seg_list})")
        if request.excluded_countries:
            country_list = ", ".join(f"'{c}'" for c in request.excluded_countries)
            where_clauses.append(f"customer_country NOT IN ({country_list})")

        where_clauses.append(
            f"proposed_increase_pct <= {request.max_per_sku_increase}"
        )

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query = f"""
            SELECT *
            FROM {_TABLE}
            {where_sql}
            ORDER BY revenue_impact DESC
        """
        rows = execute_sql(query)

        # Calculate achieved uplift from the filtered actions
        total_revenue_impact = sum(float(r.get("revenue_impact", 0)) for r in rows)
        total_volume_impact = sum(float(r.get("volume_impact", 0)) for r in rows)

        return {
            "target_uplift_pct": request.target_uplift_pct,
            "max_per_sku_increase": request.max_per_sku_increase,
            "exclusions": {
                "skus": request.excluded_skus,
                "segments": request.excluded_segments,
                "countries": request.excluded_countries,
            },
            "actions_count": len(rows),
            "total_revenue_impact": round(total_revenue_impact, 2),
            "total_volume_impact": round(total_volume_impact, 2),
            "actions": rows,
        }
    except Exception as exc:
        logger.exception("Failed to run uplift simulation")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /precomputed
# ---------------------------------------------------------------------------
@router.get(
    "/precomputed",
    summary="Retrieve precomputed uplift simulation results",
)
async def get_precomputed_uplift(
    target: float = Query(1.0, description="Target uplift percentage filter"),
    limit: int = Query(100, ge=1, le=5000, description="Max rows returned"),
) -> dict[str, Any]:
    """Return precomputed uplift simulation rows from the gold table."""
    try:
        query = f"""
            SELECT *
            FROM {_TABLE}
            WHERE target_uplift_pct = {target}
            ORDER BY revenue_impact DESC
            LIMIT {limit}
        """
        cache_key = f"uplift:precomputed:{target}:{limit}"
        rows = execute_sql(query, cache_key=cache_key)

        return {
            "target": target,
            "count": len(rows),
            "data": rows,
        }
    except Exception as exc:
        logger.exception("Failed to fetch precomputed uplift data")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /summary
# ---------------------------------------------------------------------------
@router.get(
    "/summary",
    summary="Uplift simulation summary KPIs",
)
async def get_uplift_summary() -> dict[str, Any]:
    """Return high-level summary: target, achieved uplift, actions needed,
    revenue and volume impacts.
    """
    try:
        query = f"""
            SELECT
                target_uplift_pct                           AS target,
                COUNT(*)                                    AS actions_needed,
                COALESCE(SUM(revenue_impact), 0)            AS total_revenue_impact,
                COALESCE(SUM(volume_impact), 0)             AS total_volume_impact,
                COALESCE(AVG(proposed_increase_pct), 0)     AS avg_increase_pct
            FROM {_TABLE}
            GROUP BY target_uplift_pct
            ORDER BY target_uplift_pct ASC
        """
        rows = execute_sql(query, cache_key="uplift:summary")

        return {
            "scenarios": [
                {
                    "target": float(r.get("target", 0)),
                    "actions_needed": int(r.get("actions_needed", 0)),
                    "total_revenue_impact": float(r.get("total_revenue_impact", 0)),
                    "total_volume_impact": float(r.get("total_volume_impact", 0)),
                    "avg_increase_pct": float(r.get("avg_increase_pct", 0)),
                }
                for r in rows
            ],
        }
    except Exception as exc:
        logger.exception("Failed to fetch uplift summary")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
