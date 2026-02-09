"""
Feature Store lookups and scenario feature computation.

This module retrieves precomputed product-level features from the gold feature
table in Unity Catalog and builds the what-if feature vectors sent to the ML
serving endpoint during price-change simulations.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.utils.config import TABLE_PRODUCT_FEATURES
from backend.utils.databricks_client import execute_sql

logger = logging.getLogger(__name__)

# Columns expected in the feature table
_FEATURE_COLUMNS = [
    "product_id",
    "price_change_pct",
    "base_asp",
    "cogs_pct",
    "innovation_tier_score",
    "market_share_pct",
    "competitive_density",
    "hospital_capex_index",
    "supply_chain_pressure",
    "reimbursement_trend",
    "cpi_medical",
    "tariff_rate_steel",
    "raw_material_index",
    "contract_mix_ratio",
    "gpo_penetration",
    "volume_trend_6m",
    "asp_trend_6m",
]


# ---------------------------------------------------------------------------
# Single-product feature lookup
# ---------------------------------------------------------------------------
def get_product_features(product_id: str) -> dict[str, Any]:
    """Fetch the latest feature vector for a single product.

    Parameters
    ----------
    product_id:
        The unique product identifier.

    Returns
    -------
    dict
        Feature name -> value mapping.  Returns an empty dict if the product
        is not found in the feature table.
    """
    cols = ", ".join(_FEATURE_COLUMNS)
    query = f"""
        SELECT {cols}
        FROM {TABLE_PRODUCT_FEATURES}
        WHERE product_id = '{product_id}'
        ORDER BY updated_at DESC
        LIMIT 1
    """
    rows = execute_sql(query, cache_key=f"features:{product_id}")

    if not rows:
        logger.warning("No features found for product %s", product_id)
        return {}

    row = rows[0]
    return {col: _safe_cast(row.get(col)) for col in _FEATURE_COLUMNS}


# ---------------------------------------------------------------------------
# Batch feature lookup
# ---------------------------------------------------------------------------
def get_batch_features(product_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch feature vectors for multiple products in one query.

    Parameters
    ----------
    product_ids:
        List of product identifiers.

    Returns
    -------
    dict
        Mapping of product_id -> feature dict.
    """
    if not product_ids:
        return {}

    id_list = ", ".join(f"'{pid}'" for pid in product_ids)
    cols = ", ".join(_FEATURE_COLUMNS)
    query = f"""
        SELECT {cols}
        FROM {TABLE_PRODUCT_FEATURES}
        WHERE product_id IN ({id_list})
        QUALIFY ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY updated_at DESC) = 1
    """
    rows = execute_sql(query, cache_key=f"batch_features:{','.join(sorted(product_ids))}")

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = str(row.get("product_id", ""))
        result[pid] = {col: _safe_cast(row.get(col)) for col in _FEATURE_COLUMNS}
    return result


# ---------------------------------------------------------------------------
# Scenario (what-if) feature computation
# ---------------------------------------------------------------------------
def compute_scenario_features(
    product_id: str,
    price_change_pct: float,
    *,
    base_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the feature vector for a what-if price-change scenario.

    Starts from the product's current features (fetched if not provided) and
    overlays the proposed price change along with any derived adjustments.

    Parameters
    ----------
    product_id:
        The unique product identifier.
    price_change_pct:
        Proposed price change as a percentage.
    base_features:
        Pre-fetched features.  When ``None``, they are loaded from the feature
        table.

    Returns
    -------
    dict
        Adjusted feature vector ready for model scoring.
    """
    if base_features is None or not base_features:
        base_features = get_product_features(product_id)

    if not base_features:
        logger.error(
            "Cannot compute scenario features: no base features for %s", product_id
        )
        return {"product_id": product_id, "price_change_pct": price_change_pct}

    features = dict(base_features)

    # Override the price change
    features["price_change_pct"] = price_change_pct

    # Adjust ASP to reflect the proposed change
    current_asp = float(features.get("base_asp", 0))
    features["proposed_asp"] = round(current_asp * (1.0 + price_change_pct / 100.0), 2)

    # Estimate volume trend dampening -- larger price hikes dampen volume
    volume_trend = float(features.get("volume_trend_6m", 0))
    dampening = 1.0 - min(abs(price_change_pct) / 50.0, 0.5)
    features["volume_trend_6m_adj"] = round(volume_trend * dampening, 4)

    # ASP trend adjustment
    asp_trend = float(features.get("asp_trend_6m", 0))
    features["asp_trend_6m_adj"] = round(asp_trend + price_change_pct, 4)

    return features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_cast(value: Any) -> Any:
    """Attempt to cast stringified numbers from SQL results to float."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return value
