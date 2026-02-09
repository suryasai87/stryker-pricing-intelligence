"""
Model-serving prediction client.

Calls the ``stryker-pricing-models`` Databricks Model Serving endpoint which
hosts three models behind a single inference table:

1. **Volume elasticity model** -- predicts unit-volume change given a price
   change and product features.
2. **Revenue impact model** -- predicts net revenue delta accounting for
   cross-product cannibalisation and substitution effects.
3. **Margin impact model** -- predicts margin change incorporating COGS
   pressure, mix shifts, and contract terms.

Each model returns a point estimate and a standard deviation.  This module
computes 95 % confidence intervals (point +/- 1.96 * std), SHAP-style
sensitivity factors, and a composite competitive risk score.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from backend.models import (
    ConfidenceInterval,
    PriceChangeResponse,
    SensitivityFactor,
)
from backend.services.features import compute_scenario_features, get_product_features
from backend.services.catalog import get_product_by_id
from backend.utils.config import SERVING_ENDPOINT
from backend.utils.databricks_client import call_serving_endpoint

logger = logging.getLogger(__name__)

# Z-score for 95 % confidence interval
_Z95 = 1.96

# Default sensitivity features surfaced to the user
_SENSITIVITY_FEATURES = [
    "price_change_pct",
    "market_share_pct",
    "cogs_pct",
    "innovation_tier_score",
    "competitive_density",
    "hospital_capex_index",
    "supply_chain_pressure",
    "reimbursement_trend",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _call_model(
    model_name: str,
    features: dict[str, Any],
) -> dict[str, float]:
    """Invoke a single sub-model within the serving endpoint.

    The serving endpoint uses a *model_name* field in the payload to route to
    the appropriate model.  Expected response shape::

        {
          "predictions": [
            {"value": 3.2, "std": 0.8, "shap_values": {...}}
          ]
        }

    Returns a flat dict with ``value``, ``std``, and ``shap_values``.
    """
    payload = {
        "model_name": model_name,
        "dataframe_records": [features],
    }

    try:
        response = call_serving_endpoint(SERVING_ENDPOINT, payload)
    except Exception:
        logger.exception("Model %s call failed; returning fallback zeros", model_name)
        return {"value": 0.0, "std": 0.0, "shap_values": {}}

    predictions = response.get("predictions", [])
    if not predictions:
        logger.warning("Empty predictions from model %s", model_name)
        return {"value": 0.0, "std": 0.0, "shap_values": {}}

    pred = predictions[0]
    return {
        "value": float(pred.get("value", 0.0)),
        "std": float(pred.get("std", 0.0)),
        "shap_values": pred.get("shap_values", {}),
    }


def _compute_sensitivity_factors(
    shap_values: dict[str, float],
    top_n: int = 5,
) -> list[SensitivityFactor]:
    """Return the top-N SHAP features sorted by absolute impact."""
    filtered = {
        k: float(v) for k, v in shap_values.items() if k in _SENSITIVITY_FEATURES
    }
    sorted_features = sorted(filtered.items(), key=lambda x: abs(x[1]), reverse=True)
    return [
        SensitivityFactor(feature=feat, impact=round(impact, 4))
        for feat, impact in sorted_features[:top_n]
    ]


def _compute_competitive_risk(
    price_change_pct: float,
    market_share_pct: float,
    competitive_density: float,
) -> float:
    """Heuristic competitive risk score in [0, 1].

    Larger price increases on products with low market share in highly
    competitive categories yield higher risk.
    """
    price_factor = min(abs(price_change_pct) / 20.0, 1.0)
    share_factor = max(0.0, 1.0 - (market_share_pct / 50.0))
    density_factor = min(competitive_density / 10.0, 1.0)
    risk = 0.4 * price_factor + 0.3 * share_factor + 0.3 * density_factor
    return round(min(max(risk, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def simulate_price_change(
    product_id: str,
    price_change_pct: float,
) -> PriceChangeResponse:
    """Run a full price-change simulation for one product.

    Steps:
    1. Look up product metadata and current features.
    2. Compute scenario features incorporating the proposed change.
    3. Call all three ML models via the serving endpoint.
    4. Assemble confidence intervals, sensitivity factors, and risk score.
    """
    # 1. Product metadata
    product = get_product_by_id(product_id)
    if product is None:
        raise ValueError(f"Product {product_id} not found in catalog")

    # 2. Features
    base_features = get_product_features(product_id)
    scenario_features = compute_scenario_features(
        product_id, price_change_pct, base_features=base_features
    )

    # 3. Model calls
    volume_result = _call_model("volume_elasticity", scenario_features)
    revenue_result = _call_model("revenue_impact", scenario_features)
    margin_result = _call_model("margin_impact", scenario_features)

    # 4. Derived values
    current_asp = product.base_asp
    proposed_asp = round(current_asp * (1 + price_change_pct / 100.0), 2)

    # Confidence interval on revenue impact (widest uncertainty driver)
    revenue_std = revenue_result["std"] or abs(revenue_result["value"]) * 0.1
    ci = ConfidenceInterval(
        lower=round(revenue_result["value"] - _Z95 * revenue_std, 2),
        upper=round(revenue_result["value"] + _Z95 * revenue_std, 2),
    )

    # Merge SHAP values from all models (average where overlapping)
    merged_shap: dict[str, list[float]] = {}
    for result in (volume_result, revenue_result, margin_result):
        for feat, val in result.get("shap_values", {}).items():
            merged_shap.setdefault(feat, []).append(float(val))
    avg_shap = {k: sum(v) / len(v) for k, v in merged_shap.items()}

    sensitivity = _compute_sensitivity_factors(avg_shap)

    competitive_density = float(scenario_features.get("competitive_density", 5))
    risk_score = _compute_competitive_risk(
        price_change_pct, product.market_share_pct, competitive_density
    )

    return PriceChangeResponse(
        product_id=product.product_id,
        product_name=product.product_name,
        current_asp=current_asp,
        proposed_asp=proposed_asp,
        predicted_volume_change_pct=round(volume_result["value"], 4),
        predicted_revenue_impact=round(revenue_result["value"], 2),
        predicted_margin_impact=round(margin_result["value"], 4),
        confidence_interval=ci,
        top_sensitivity_factors=sensitivity,
        competitive_risk_score=risk_score,
    )
