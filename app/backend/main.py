"""
Stryker Pricing Intelligence -- FastAPI application.

Provides REST endpoints for price-change simulations, product catalog queries,
portfolio KPIs, price waterfall analysis, competitive landscape data, external
macro-factor retrieval, and batch scenario scoring.

The application is designed to run inside a Databricks App with SDK
auto-authentication.  For local development, set DATABRICKS_HOST and
DATABRICKS_TOKEN environment variables.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.models import (
    BatchScenarioRequest,
    BatchScenarioResponse,
    CompetitorData,
    ExternalFactors,
    PortfolioKPIs,
    PriceChangeRequest,
    PriceChangeResponse,
    Product,
    WaterfallStep,
)
from backend.services.catalog import (
    get_competitive_landscape,
    get_external_factors,
    get_portfolio_kpis,
    get_price_waterfall,
    get_products,
)
from backend.services.prediction import simulate_price_change
from backend.utils.config import APP_TITLE, APP_VERSION, LOG_LEVEL, STATIC_FILES_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown hooks."""
    logger.info("Starting %s v%s", APP_TITLE, APP_VERSION)
    yield
    logger.info("Shutting down %s", APP_TITLE)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    lifespan=lifespan,
)

# CORS -- allow all origins for Databricks App iframe embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Return a simple health-check response."""
    return {"status": "healthy", "version": APP_VERSION}


# ---------------------------------------------------------------------------
# Price simulation
# ---------------------------------------------------------------------------
@app.post(
    "/api/v1/simulate-price-change",
    response_model=PriceChangeResponse,
    tags=["simulation"],
    summary="Simulate a single-product price change",
)
async def api_simulate_price_change(
    request: PriceChangeRequest,
) -> PriceChangeResponse:
    """Run a what-if price-change simulation.

    Calls the volume-elasticity, revenue-impact, and margin-impact ML models
    via the Databricks Model Serving endpoint and returns a comprehensive
    response including confidence intervals, sensitivity factors, and a
    competitive risk score.
    """
    try:
        result = simulate_price_change(
            product_id=request.product_id,
            price_change_pct=request.price_change_pct,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Simulation failed for product %s", request.product_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Product catalog
# ---------------------------------------------------------------------------
@app.get(
    "/api/v1/products",
    response_model=list[Product],
    tags=["catalog"],
    summary="Query the product catalog",
)
async def api_get_products(
    category: str | None = Query(None, description="Filter by category"),
    segment: str | None = Query(None, description="Filter by segment"),
    limit: int = Query(500, ge=1, le=5000, description="Max rows returned"),
) -> list[Product]:
    """Return products from the Unity Catalog gold product table."""
    try:
        return get_products(category=category, segment=segment, limit=limit)
    except Exception as exc:
        logger.exception("Failed to fetch products")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Portfolio KPIs
# ---------------------------------------------------------------------------
@app.get(
    "/api/v1/portfolio-kpis",
    response_model=PortfolioKPIs,
    tags=["analytics"],
    summary="Aggregate portfolio KPIs",
)
async def api_get_portfolio_kpis() -> PortfolioKPIs:
    """Return high-level portfolio metrics from gold tables.

    Includes total revenue, average margin, year-over-year growth, product
    count, and per-segment breakdowns.
    """
    try:
        return get_portfolio_kpis()
    except Exception as exc:
        logger.exception("Failed to fetch portfolio KPIs")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Price waterfall
# ---------------------------------------------------------------------------
@app.get(
    "/api/v1/price-waterfall/{product_id}",
    response_model=list[WaterfallStep],
    tags=["analytics"],
    summary="Price-to-pocket waterfall",
)
async def api_get_price_waterfall(product_id: str) -> list[WaterfallStep]:
    """Return the list-to-pocket price waterfall for a product.

    Steps: list price -> contract adjustment -> GPO rebate -> freight ->
    pocket price.
    """
    try:
        steps = get_price_waterfall(product_id)
        if not steps:
            raise HTTPException(
                status_code=404,
                detail=f"No waterfall data for product {product_id}",
            )
        return steps
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch waterfall for %s", product_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Competitive landscape
# ---------------------------------------------------------------------------
@app.get(
    "/api/v1/competitive-landscape/{category}",
    response_model=list[CompetitorData],
    tags=["analytics"],
    summary="Competitor ASPs and market share",
)
async def api_get_competitive_landscape(category: str) -> list[CompetitorData]:
    """Return competitor pricing and market-share data for a category."""
    try:
        data = get_competitive_landscape(category)
        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"No competitive data for category '{category}'",
            )
        return data
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch competitive data for %s", category)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# External factors
# ---------------------------------------------------------------------------
@app.get(
    "/api/v1/external-factors",
    response_model=ExternalFactors,
    tags=["analytics"],
    summary="Latest macro-economic indicators",
)
async def api_get_external_factors() -> ExternalFactors:
    """Return the most recent macro and supply-chain indicators."""
    try:
        factors = get_external_factors()
        if factors is None:
            raise HTTPException(
                status_code=404,
                detail="No external factor data available",
            )
        return factors
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch external factors")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Batch scenario
# ---------------------------------------------------------------------------
@app.post(
    "/api/v1/batch-scenario",
    response_model=BatchScenarioResponse,
    tags=["simulation"],
    summary="Score multiple products in batch",
)
async def api_batch_scenario(
    request: BatchScenarioRequest,
) -> BatchScenarioResponse:
    """Run price-change simulations for a list of products.

    Each scenario in the request is scored independently.  Individual failures
    are logged but do not abort the entire batch -- those products are simply
    omitted from the response.
    """
    results: list[PriceChangeResponse] = []
    for scenario in request.scenarios:
        try:
            result = simulate_price_change(
                product_id=scenario.product_id,
                price_change_pct=scenario.price_change_pct,
            )
            results.append(result)
        except Exception as exc:
            logger.warning(
                "Batch scenario failed for product %s: %s",
                scenario.product_id,
                exc,
            )
    return BatchScenarioResponse(results=results)


# ---------------------------------------------------------------------------
# Static files (frontend) -- must be last so it doesn't shadow API routes
# ---------------------------------------------------------------------------
_static_dir = os.path.join(os.path.dirname(__file__), "..", STATIC_FILES_DIR)
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
    logger.info("Mounted static files from %s", _static_dir)
else:
    logger.warning(
        "Static directory %s not found; frontend will not be served", _static_dir
    )
