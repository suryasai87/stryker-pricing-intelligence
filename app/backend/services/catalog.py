"""
Unity Catalog query service.

Encapsulates all SQL queries that read from gold-layer tables in Unity Catalog.
Results are cached through the ``execute_sql`` helper in the Databricks client
module.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.models import (
    CompetitorData,
    ExternalFactors,
    PortfolioKPIs,
    Product,
    SegmentMetric,
    WaterfallStep,
)
from backend.utils.config import (
    TABLE_COMPETITORS,
    TABLE_EXTERNAL_FACTORS,
    TABLE_PRICE_WATERFALL,
    TABLE_PRODUCTS,
    TABLE_REVENUE_SUMMARY,
)
from backend.utils.databricks_client import execute_sql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------
def get_products(
    category: str | None = None,
    segment: str | None = None,
    limit: int = 500,
) -> list[Product]:
    """Return products from the gold product catalog.

    Parameters
    ----------
    category:
        Optional filter on product category.
    segment:
        Optional filter on business segment.
    limit:
        Maximum number of rows returned.
    """
    where_clauses: list[str] = []
    if category:
        where_clauses.append(f"category = '{category}'")
    if segment:
        where_clauses.append(f"segment = '{segment}'")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    query = f"""
        SELECT product_id, product_name, category, sub_category, segment,
               base_asp, cogs_pct, innovation_tier, market_share_pct
        FROM {TABLE_PRODUCTS}
        {where_sql}
        ORDER BY product_name
        LIMIT {limit}
    """

    cache_key = f"products:{category}:{segment}:{limit}"
    rows = execute_sql(query, cache_key=cache_key)

    return [
        Product(
            product_id=str(r["product_id"]),
            product_name=str(r["product_name"]),
            category=str(r["category"]),
            sub_category=str(r["sub_category"]),
            segment=str(r["segment"]),
            base_asp=float(r["base_asp"]),
            cogs_pct=float(r["cogs_pct"]),
            innovation_tier=str(r["innovation_tier"]),
            market_share_pct=float(r["market_share_pct"]),
        )
        for r in rows
    ]


def get_product_by_id(product_id: str) -> Product | None:
    """Look up a single product by its identifier."""
    query = f"""
        SELECT product_id, product_name, category, sub_category, segment,
               base_asp, cogs_pct, innovation_tier, market_share_pct
        FROM {TABLE_PRODUCTS}
        WHERE product_id = '{product_id}'
        LIMIT 1
    """
    rows = execute_sql(query, cache_key=f"product:{product_id}")
    if not rows:
        return None

    r = rows[0]
    return Product(
        product_id=str(r["product_id"]),
        product_name=str(r["product_name"]),
        category=str(r["category"]),
        sub_category=str(r["sub_category"]),
        segment=str(r["segment"]),
        base_asp=float(r["base_asp"]),
        cogs_pct=float(r["cogs_pct"]),
        innovation_tier=str(r["innovation_tier"]),
        market_share_pct=float(r["market_share_pct"]),
    )


# ---------------------------------------------------------------------------
# Portfolio KPIs
# ---------------------------------------------------------------------------
def get_portfolio_kpis() -> PortfolioKPIs:
    """Aggregate key portfolio metrics from gold tables."""

    # Overall KPIs
    kpi_query = f"""
        SELECT
            SUM(revenue)                         AS total_revenue,
            AVG(margin_pct)                      AS avg_margin_pct,
            AVG(yoy_growth_pct)                  AS yoy_growth_pct,
            COUNT(DISTINCT product_id)           AS total_products
        FROM {TABLE_REVENUE_SUMMARY}
    """
    kpi_rows = execute_sql(kpi_query, cache_key="portfolio_kpis:overall")
    kpi = kpi_rows[0] if kpi_rows else {}

    # Revenue by segment
    rev_seg_query = f"""
        SELECT segment, SUM(revenue) AS value
        FROM {TABLE_REVENUE_SUMMARY}
        GROUP BY segment
        ORDER BY value DESC
    """
    rev_seg_rows = execute_sql(rev_seg_query, cache_key="portfolio_kpis:rev_seg")

    # Margin by segment
    margin_seg_query = f"""
        SELECT segment, AVG(margin_pct) AS value
        FROM {TABLE_REVENUE_SUMMARY}
        GROUP BY segment
        ORDER BY value DESC
    """
    margin_seg_rows = execute_sql(margin_seg_query, cache_key="portfolio_kpis:margin_seg")

    return PortfolioKPIs(
        total_revenue=float(kpi.get("total_revenue", 0)),
        avg_margin_pct=float(kpi.get("avg_margin_pct", 0)),
        yoy_growth_pct=float(kpi.get("yoy_growth_pct", 0)),
        total_products=int(kpi.get("total_products", 0)),
        revenue_by_segment=[
            SegmentMetric(segment=str(r["segment"]), value=float(r["value"]))
            for r in rev_seg_rows
        ],
        margin_by_segment=[
            SegmentMetric(segment=str(r["segment"]), value=float(r["value"]))
            for r in margin_seg_rows
        ],
    )


# ---------------------------------------------------------------------------
# Price waterfall
# ---------------------------------------------------------------------------
def get_price_waterfall(product_id: str) -> list[WaterfallStep]:
    """Return the list-to-pocket waterfall components for a product."""
    query = f"""
        SELECT step_name, step_value, cumulative_value
        FROM {TABLE_PRICE_WATERFALL}
        WHERE product_id = '{product_id}'
        ORDER BY step_order ASC
    """
    rows = execute_sql(query, cache_key=f"waterfall:{product_id}")

    return [
        WaterfallStep(
            name=str(r["step_name"]),
            value=float(r["step_value"]),
            cumulative=float(r["cumulative_value"]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Competitive landscape
# ---------------------------------------------------------------------------
def get_competitive_landscape(category: str) -> list[CompetitorData]:
    """Return competitor ASPs and market share for a product category."""
    query = f"""
        SELECT competitor, avg_asp, market_share, innovation_score, asp_trend_pct
        FROM {TABLE_COMPETITORS}
        WHERE category = '{category}'
        ORDER BY market_share DESC
    """
    rows = execute_sql(query, cache_key=f"competitors:{category}")

    return [
        CompetitorData(
            competitor=str(r["competitor"]),
            avg_asp=float(r["avg_asp"]),
            market_share=float(r["market_share"]),
            innovation_score=float(r["innovation_score"]),
            asp_trend_pct=float(r["asp_trend_pct"]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# External / macro factors
# ---------------------------------------------------------------------------
def get_external_factors() -> ExternalFactors | None:
    """Return the most recent row of macro-economic indicators."""
    query = f"""
        SELECT month, cpi_medical, tariff_rate_steel, supply_chain_pressure,
               hospital_capex_index, reimbursement_trend, raw_material_index
        FROM {TABLE_EXTERNAL_FACTORS}
        ORDER BY month DESC
        LIMIT 1
    """
    rows = execute_sql(query, cache_key="external_factors:latest")
    if not rows:
        return None

    r = rows[0]
    return ExternalFactors(
        month=str(r["month"]),
        cpi_medical=float(r["cpi_medical"]),
        tariff_rate_steel=float(r["tariff_rate_steel"]),
        supply_chain_pressure=float(r["supply_chain_pressure"]),
        hospital_capex_index=float(r["hospital_capex_index"]),
        reimbursement_trend=float(r["reimbursement_trend"]),
        raw_material_index=float(r["raw_material_index"]),
    )
