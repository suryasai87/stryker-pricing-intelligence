"""
Top 100 price changes router.

Fully filterable, sortable, and paginated endpoint for the top price changes.
Includes filter option discovery and CSV export.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.utils.config import CATALOG_NAME, SCHEMA_GOLD_V2
from backend.utils.databricks_client import execute_sql

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/top100-price-changes", tags=["top100"])

_TABLE = f"{CATALOG_NAME}.{SCHEMA_GOLD_V2}.top100_price_changes"

# Allowed sort columns to prevent SQL injection via sort_by
_ALLOWED_SORT_COLUMNS = {
    "revenue_impact",
    "price_change_pct",
    "product_name",
    "customer_country",
    "product_family",
    "customer_segment",
    "risk_level",
    "business_unit",
    "sales_rep",
}


def _build_where_clause(
    country: list[str] | None,
    product_family: list[str] | None,
    segment: list[str] | None,
    rep: str | None,
    business_unit: str | None,
    risk_level: str | None,
    min_revenue_impact: float | None,
) -> str:
    """Build a SQL WHERE clause from the provided filters."""
    where_clauses: list[str] = []

    if country:
        vals = ", ".join(f"'{c}'" for c in country)
        where_clauses.append(f"customer_country IN ({vals})")
    if product_family:
        vals = ", ".join(f"'{f}'" for f in product_family)
        where_clauses.append(f"product_family IN ({vals})")
    if segment:
        vals = ", ".join(f"'{s}'" for s in segment)
        where_clauses.append(f"customer_segment IN ({vals})")
    if rep:
        where_clauses.append(f"sales_rep LIKE '%{rep}%'")
    if business_unit:
        where_clauses.append(f"business_unit = '{business_unit}'")
    if risk_level:
        where_clauses.append(f"risk_level = '{risk_level}'")
    if min_revenue_impact is not None:
        where_clauses.append(f"ABS(revenue_impact) >= {min_revenue_impact}")

    return f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
@router.get(
    "/",
    summary="List top price changes with full filtering, sorting, and pagination",
)
async def get_top100_price_changes(
    country: list[str] | None = Query(None, description="Filter by country (multi)"),
    product_family: list[str] | None = Query(
        None, description="Filter by product family (multi)"
    ),
    segment: list[str] | None = Query(None, description="Filter by segment (multi)"),
    rep: str | None = Query(None, description="Search sales rep name"),
    business_unit: str | None = Query(None, description="Filter by business unit"),
    risk_level: str | None = Query(None, description="Filter by risk level"),
    min_revenue_impact: float | None = Query(
        None, description="Minimum absolute revenue impact"
    ),
    sort_by: str = Query("revenue_impact", description="Column to sort by"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=1, le=500, description="Rows per page"),
) -> dict[str, Any]:
    """Return paginated, filtered, and sorted top price changes."""
    try:
        where_sql = _build_where_clause(
            country, product_family, segment, rep, business_unit,
            risk_level, min_revenue_impact,
        )

        # Validate sort column
        safe_sort = sort_by if sort_by in _ALLOWED_SORT_COLUMNS else "revenue_impact"
        safe_order = "ASC" if sort_order.lower() == "asc" else "DESC"
        offset = (page - 1) * page_size

        # Count query
        count_query = f"SELECT COUNT(*) AS total FROM {_TABLE} {where_sql}"
        count_rows = execute_sql(count_query)
        total = int(count_rows[0].get("total", 0)) if count_rows else 0

        # Data query
        data_query = f"""
            SELECT *
            FROM {_TABLE}
            {where_sql}
            ORDER BY {safe_sort} {safe_order}
            LIMIT {page_size}
            OFFSET {offset}
        """
        rows = execute_sql(data_query)

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if page_size else 0,
            "sort_by": safe_sort,
            "sort_order": safe_order.lower(),
            "data": rows,
        }
    except Exception as exc:
        logger.exception("Failed to fetch top 100 price changes")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /filter-options
# ---------------------------------------------------------------------------
@router.get(
    "/filter-options",
    summary="Distinct filter values for each dimension",
)
async def get_filter_options() -> dict[str, Any]:
    """Return distinct values for each filterable dimension."""
    try:
        dimensions = {
            "country": "customer_country",
            "product_family": "product_family",
            "segment": "customer_segment",
            "business_unit": "business_unit",
            "risk_level": "risk_level",
        }

        result: dict[str, list[str]] = {}
        for key, column in dimensions.items():
            query = f"""
                SELECT DISTINCT {column} AS val
                FROM {_TABLE}
                WHERE {column} IS NOT NULL
                ORDER BY val ASC
            """
            rows = execute_sql(
                query, cache_key=f"top100:filter_options:{key}"
            )
            result[key] = [str(r.get("val", "")) for r in rows]

        return result
    except Exception as exc:
        logger.exception("Failed to fetch filter options")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /export
# ---------------------------------------------------------------------------
@router.get(
    "/export",
    summary="Export filtered price changes as CSV",
)
async def export_top100_csv(
    country: list[str] | None = Query(None),
    product_family: list[str] | None = Query(None),
    segment: list[str] | None = Query(None),
    rep: str | None = Query(None),
    business_unit: str | None = Query(None),
    risk_level: str | None = Query(None),
    min_revenue_impact: float | None = Query(None),
    sort_by: str = Query("revenue_impact"),
    sort_order: str = Query("desc"),
) -> StreamingResponse:
    """Stream a CSV download of the filtered top price changes."""
    try:
        where_sql = _build_where_clause(
            country, product_family, segment, rep, business_unit,
            risk_level, min_revenue_impact,
        )

        safe_sort = sort_by if sort_by in _ALLOWED_SORT_COLUMNS else "revenue_impact"
        safe_order = "ASC" if sort_order.lower() == "asc" else "DESC"

        query = f"""
            SELECT *
            FROM {_TABLE}
            {where_sql}
            ORDER BY {safe_sort} {safe_order}
        """
        rows = execute_sql(query)

        # Build CSV in memory
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        else:
            output.write("No data matching filters\n")

        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=top100_price_changes.csv"
            },
        )
    except Exception as exc:
        logger.exception("Failed to export top 100 price changes")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
