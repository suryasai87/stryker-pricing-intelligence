"""
Pydantic data models for the Stryker Pricing Intelligence API.

All request / response schemas are defined here so they can be shared across
routers, services, and tests.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Simulation request / response
# ---------------------------------------------------------------------------
class PriceChangeRequest(BaseModel):
    """Payload sent by the UI when a user runs a what-if simulation."""

    product_id: str = Field(..., description="Unique product identifier")
    price_change_pct: float = Field(
        ...,
        description="Proposed price change as a percentage (e.g. 5.0 for +5%)",
    )
    scenario_name: str = Field(
        "default",
        description="Optional label for the scenario",
    )


class SensitivityFactor(BaseModel):
    """A single SHAP-style driver behind the prediction."""

    feature: str
    impact: float = Field(..., description="Signed impact on the prediction")


class ConfidenceInterval(BaseModel):
    """Lower / upper bounds of a prediction."""

    lower: float
    upper: float


class PriceChangeResponse(BaseModel):
    """Full simulation result returned to the caller."""

    product_id: str
    product_name: str
    current_asp: float
    proposed_asp: float
    predicted_volume_change_pct: float
    predicted_revenue_impact: float
    predicted_margin_impact: float
    confidence_interval: ConfidenceInterval
    top_sensitivity_factors: list[SensitivityFactor]
    competitive_risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="0 = low risk, 1 = high risk"
    )


# ---------------------------------------------------------------------------
# Product catalog
# ---------------------------------------------------------------------------
class Product(BaseModel):
    """A single product from the Unity Catalog product table."""

    product_id: str
    product_name: str
    category: str
    sub_category: str
    segment: str
    base_asp: float
    cogs_pct: float
    innovation_tier: str
    market_share_pct: float


# ---------------------------------------------------------------------------
# Portfolio KPIs
# ---------------------------------------------------------------------------
class SegmentMetric(BaseModel):
    """Revenue or margin aggregated by segment."""

    segment: str
    value: float


class PortfolioKPIs(BaseModel):
    """High-level portfolio metrics surfaced on the dashboard."""

    total_revenue: float
    avg_margin_pct: float
    yoy_growth_pct: float
    total_products: int
    revenue_by_segment: list[SegmentMetric]
    margin_by_segment: list[SegmentMetric]


# ---------------------------------------------------------------------------
# Price waterfall
# ---------------------------------------------------------------------------
class WaterfallStep(BaseModel):
    """One leg of the price-to-pocket waterfall."""

    name: str
    value: float
    cumulative: float


# ---------------------------------------------------------------------------
# Competitive landscape
# ---------------------------------------------------------------------------
class CompetitorData(BaseModel):
    """Competitor pricing and positioning for a given product category."""

    competitor: str
    avg_asp: float
    market_share: float
    innovation_score: float
    asp_trend_pct: float


# ---------------------------------------------------------------------------
# External / macro factors
# ---------------------------------------------------------------------------
class ExternalFactors(BaseModel):
    """Latest macro-economic and supply-chain indicators."""

    month: str
    cpi_medical: float
    tariff_rate_steel: float
    supply_chain_pressure: float
    hospital_capex_index: float
    reimbursement_trend: float
    raw_material_index: float


# ---------------------------------------------------------------------------
# Batch scenario
# ---------------------------------------------------------------------------
class BatchScenarioRequest(BaseModel):
    """Run multiple what-if simulations in one call."""

    scenarios: list[PriceChangeRequest]


class BatchScenarioResponse(BaseModel):
    """Aggregated results for a batch simulation."""

    results: list[PriceChangeResponse]
