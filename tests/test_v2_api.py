"""
Tests for all v2 API endpoints in the Stryker Pricing Intelligence Platform.

Covers all 23 v2 API routes using FastAPI TestClient with a mocked Databricks
SDK ``execute_sql`` function so tests run without a live workspace connection.

Groups:
  1. FICM (2 endpoints)
  2. Discount Outliers (3 endpoints)
  3. Price Elasticity (3 endpoints)
  4. Uplift Simulation (3 endpoints)
  5. Top 100 Price Changes (3 endpoints)
  6. Pricing Recommendations (2 endpoints)
  7. External Data (3 endpoints)
  8. Pricing Scenarios (4 endpoints)
"""

from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# Ensure the app package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------
def _mock_ficm_overview_row() -> dict:
    return {"row_count": "600000", "min_date": "2022-01-01", "max_date": "2024-12-31"}


def _mock_ficm_country_rows() -> list[dict]:
    return [
        {"country": "US", "count": "300000"},
        {"country": "DE", "count": "150000"},
    ]


def _mock_ficm_segment_rows() -> list[dict]:
    return [
        {"segment": "IDN", "count": "250000"},
        {"segment": "GPO", "count": "200000"},
    ]


def _mock_ficm_product_family_rows() -> list[dict]:
    return [
        {"product_family": "Endoscopy", "count": "180000"},
        {"product_family": "MedSurg", "count": "120000"},
    ]


def _mock_ficm_schema_rows() -> list[dict]:
    return [
        {"col_name": "product_id", "data_type": "string", "comment": "Primary key"},
        {"col_name": "pricing_date", "data_type": "date", "comment": "Pricing date"},
        {"col_name": "list_price", "data_type": "double", "comment": "List price"},
    ]


def _mock_outlier_row() -> dict:
    return {
        "outlier_id": "OUT-001",
        "sku": "SKU-100",
        "business_unit": "Orthopaedics",
        "customer_segment": "IDN",
        "customer_country": "US",
        "z_score": "4.2",
        "severity": "high",
        "recovery_amount": "12500.00",
        "sales_rep_id": "REP-001",
    }


def _mock_outlier_summary_totals() -> dict:
    return {"total_outliers": "450", "total_recovery": "5600000.00"}


def _mock_outlier_bu_rows() -> list[dict]:
    return [
        {"business_unit": "Orthopaedics", "count": "200", "recovery": "2500000.00"},
        {"business_unit": "MedSurg", "count": "150", "recovery": "1800000.00"},
    ]


def _mock_outlier_severity_rows() -> list[dict]:
    return [
        {"severity": "high", "count": "120", "recovery": "3000000.00"},
        {"severity": "medium", "count": "200", "recovery": "2000000.00"},
    ]


def _mock_outlier_country_rows() -> list[dict]:
    return [
        {"country": "US", "count": "300", "recovery": "4000000.00"},
        {"country": "DE", "count": "100", "recovery": "1000000.00"},
    ]


def _mock_elasticity_row() -> dict:
    return {
        "sku": "SKU-001",
        "business_unit": "Orthopaedics",
        "customer_segment": "IDN",
        "product_family": "Joint Replacement",
        "elasticity_coefficient": "-1.35",
        "elasticity_class": "elastic",
        "confidence": "high",
        "safe_low_pct": "-3.0",
        "safe_high_pct": "2.0",
    }


def _mock_elasticity_distribution_rows() -> list[dict]:
    return [
        {"bucket": "-2.0", "count": "15"},
        {"bucket": "-1.5", "count": "45"},
        {"bucket": "-1.0", "count": "80"},
        {"bucket": "-0.5", "count": "60"},
    ]


def _mock_uplift_row() -> dict:
    return {
        "sku": "SKU-050",
        "product_name": "Surgical Drill X",
        "customer_segment": "IDN",
        "customer_country": "US",
        "proposed_increase_pct": "2.5",
        "revenue_impact": "50000.00",
        "volume_impact": "-200.0",
        "target_uplift_pct": "1.0",
    }


def _mock_uplift_summary_rows() -> list[dict]:
    return [
        {
            "target": "1.0",
            "actions_needed": "25",
            "total_revenue_impact": "500000.00",
            "total_volume_impact": "-2000.00",
            "avg_increase_pct": "2.1",
        },
        {
            "target": "2.0",
            "actions_needed": "50",
            "total_revenue_impact": "1100000.00",
            "total_volume_impact": "-5000.00",
            "avg_increase_pct": "3.4",
        },
    ]


def _mock_top100_row() -> dict:
    return {
        "product_name": "Hip Implant Premium",
        "product_family": "Endoscopy",
        "customer_country": "US",
        "customer_segment": "IDN",
        "business_unit": "Orthopaedics",
        "price_change_pct": "5.2",
        "revenue_impact": "125000.00",
        "risk_level": "medium",
        "sales_rep": "John Doe",
    }


def _mock_top100_filter_option_rows(column: str) -> list[dict]:
    options_map = {
        "customer_country": [{"val": "DE"}, {"val": "US"}],
        "product_family": [{"val": "Endoscopy"}, {"val": "MedSurg"}],
        "customer_segment": [{"val": "GPO"}, {"val": "IDN"}],
        "business_unit": [{"val": "MedSurg"}, {"val": "Orthopaedics"}],
        "risk_level": [{"val": "high"}, {"val": "low"}, {"val": "medium"}],
    }
    return options_map.get(column, [])


def _mock_recommendation_row() -> dict:
    return {
        "recommendation_id": "REC-001",
        "sku": "SKU-200",
        "product_family": "Joint Replacement",
        "business_unit": "Orthopaedics",
        "action_type": "increase",
        "risk_level": "low",
        "priority_score": "0.92",
        "revenue_impact": "75000.00",
    }


def _mock_recommendation_summary_rows() -> list[dict]:
    return [
        {
            "action_type": "increase",
            "count": "80",
            "avg_priority_score": "0.85",
            "total_revenue_impact": "2000000.00",
            "avg_revenue_impact": "25000.00",
        },
        {
            "action_type": "decrease",
            "count": "30",
            "avg_priority_score": "0.72",
            "total_revenue_impact": "-500000.00",
            "avg_revenue_impact": "-16666.67",
        },
    ]


def _mock_external_data_row() -> dict:
    return {
        "source_name": "market_report_q4.csv",
        "category": "market_research",
        "file_path": "/Volumes/hls_amer_catalog/gold/external_uploads/market_report_q4.csv",
        "uploaded_at": "2024-12-15T10:00:00Z",
        "row_count": "500",
        "column_count": "12",
        "indicator": "CPI",
        "value": "3.2",
    }


def _mock_external_sources_rows() -> list[dict]:
    return [
        {
            "source_name": "market_report_q4.csv",
            "category": "market_research",
            "file_path": "/Volumes/hls_amer_catalog/gold/external_uploads/market_report_q4.csv",
            "uploaded_at": "2024-12-15T10:00:00Z",
            "row_count": "500",
            "column_count": "12",
        },
    ]


def _mock_scenario_row() -> dict:
    return {
        "scenario_id": "abc12345",
        "name": "Q1 Price Adjustment",
        "description": "Quarterly review adjustments",
        "parameters": '{"target_margin": 0.35}',
        "status": "draft",
        "created_by": "anonymous@stryker.com",
        "created_at": "2024-12-01T10:00:00Z",
        "updated_at": "2024-12-01T10:00:00Z",
    }


# ---------------------------------------------------------------------------
# SQL side-effect router -- dispatches mock data based on query content
# ---------------------------------------------------------------------------
def _sql_side_effect(query: str, **kwargs) -> list[dict]:
    """Return mock data based on the SQL query content."""
    q = query.lower().strip()
    cache_key = kwargs.get("cache_key", "")

    # ---- FICM ----
    if "ficm_pricing_master" in q:
        if "count(*)" in q and "group by" not in q:
            return [_mock_ficm_overview_row()]
        if "customer_country" in q and "group by" in q:
            return _mock_ficm_country_rows()
        if "customer_segment" in q and "group by" in q:
            return _mock_ficm_segment_rows()
        if "product_family" in q and "group by" in q:
            return _mock_ficm_product_family_rows()
        if "describe table" in q:
            return _mock_ficm_schema_rows()

    # ---- Discount Outliers ----
    if "discount_outliers" in q:
        # Summary totals
        if "count(*)" in q and "sum(recovery_amount)" in q and "group by" not in q:
            return [_mock_outlier_summary_totals()]
        # By business unit
        if "business_unit" in q and "group by" in q and "severity" not in q and "customer_country" not in q:
            return _mock_outlier_bu_rows()
        # By severity
        if "severity" in q and "group by" in q:
            return _mock_outlier_severity_rows()
        # By country (summary)
        if "customer_country as country" in q and "group by" in q:
            return _mock_outlier_country_rows()
        # By rep or list
        return [_mock_outlier_row()]

    # ---- Price Elasticity ----
    if "price_elasticity" in q:
        if "floor(" in q and "group by" in q:
            return _mock_elasticity_distribution_rows()
        return [_mock_elasticity_row()]

    # ---- Uplift Simulation ----
    if "uplift_simulation" in q:
        if "group by" in q and "target_uplift_pct" in q:
            return _mock_uplift_summary_rows()
        return [_mock_uplift_row()]

    # ---- Top 100 Price Changes ----
    if "top100_price_changes" in q:
        if "count(*) as total" in q:
            return [{"total": "5"}]
        if "distinct" in q:
            # Filter options -- identify column from query
            for col in ["customer_country", "product_family", "customer_segment", "business_unit", "risk_level"]:
                if f"distinct {col}" in q:
                    return _mock_top100_filter_option_rows(col)
            return []
        return [_mock_top100_row()]

    # ---- Pricing Recommendations ----
    if "pricing_recommendations" in q:
        if "group by" in q and "action_type" in q:
            return _mock_recommendation_summary_rows()
        return [_mock_recommendation_row()]

    # ---- External Data ----
    if "external_data" in q:
        if "group by" in q:
            return _mock_external_sources_rows()
        return [_mock_external_data_row()]

    # ---- Pricing Scenarios ----
    if "pricing_scenarios" in q:
        if "count(*) as total" in q:
            return [{"total": "3"}]
        if "insert into" in q:
            return []  # INSERT returns no rows
        return [_mock_scenario_row()]

    return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """Create a TestClient with the Databricks SDK fully mocked out."""
    with patch("backend.utils.databricks_client.get_workspace_client") as mock_ws, \
         patch("backend.utils.databricks_client.execute_sql", side_effect=_sql_side_effect) as mock_sql:

        # Also patch at the router module level since they import execute_sql directly
        with patch("backend.routers.v2_ficm.execute_sql", side_effect=_sql_side_effect), \
             patch("backend.routers.v2_discount_outliers.execute_sql", side_effect=_sql_side_effect), \
             patch("backend.routers.v2_price_elasticity.execute_sql", side_effect=_sql_side_effect), \
             patch("backend.routers.v2_uplift_simulation.execute_sql", side_effect=_sql_side_effect), \
             patch("backend.routers.v2_top100_changes.execute_sql", side_effect=_sql_side_effect), \
             patch("backend.routers.v2_pricing_recommendations.execute_sql", side_effect=_sql_side_effect), \
             patch("backend.routers.v2_external_data.execute_sql", side_effect=_sql_side_effect), \
             patch("backend.routers.v2_external_data.get_workspace_client", return_value=MagicMock()), \
             patch("backend.routers.v2_pricing_scenarios.execute_sql", side_effect=_sql_side_effect):

            from backend.main import app
            from fastapi.testclient import TestClient

            yield TestClient(app)


# ===========================================================================
# 1. FICM Endpoints
# ===========================================================================
class TestFICM:
    """Tests for /api/v2/ficm/* endpoints."""

    def test_ficm_summary(self, client):
        """GET /api/v2/ficm/summary returns 200 with row count and breakdowns."""
        response = client.get("/api/v2/ficm/summary")
        assert response.status_code == 200

        data = response.json()
        assert "row_count" in data
        assert data["row_count"] == 600000
        assert "date_range" in data
        assert data["date_range"]["min"] == "2022-01-01"
        assert data["date_range"]["max"] == "2024-12-31"
        assert "country_breakdown" in data
        assert "segment_breakdown" in data
        assert "product_family_breakdown" in data
        assert len(data["country_breakdown"]) > 0
        assert len(data["segment_breakdown"]) > 0

    def test_ficm_schema(self, client):
        """GET /api/v2/ficm/schema returns column list with data types."""
        response = client.get("/api/v2/ficm/schema")
        assert response.status_code == 200

        data = response.json()
        assert "table" in data
        assert "column_count" in data
        assert data["column_count"] == 3
        assert "columns" in data
        assert isinstance(data["columns"], list)

        first_col = data["columns"][0]
        assert "column_name" in first_col
        assert "data_type" in first_col
        assert "comment" in first_col


# ===========================================================================
# 2. Discount Outliers
# ===========================================================================
class TestDiscountOutliers:
    """Tests for /api/v2/discount-outliers/* endpoints."""

    def test_discount_outliers_list(self, client):
        """GET /api/v2/discount-outliers/ returns outlier rows."""
        response = client.get("/api/v2/discount-outliers/")
        assert response.status_code == 200

        data = response.json()
        assert "count" in data
        assert "filters" in data
        assert "data" in data
        assert isinstance(data["data"], list)
        assert data["count"] >= 1

        row = data["data"][0]
        assert "sku" in row
        assert "z_score" in row
        assert "severity" in row

    def test_discount_outliers_with_filters(self, client):
        """GET with business_unit, severity, min_z_score filters returns filtered results."""
        response = client.get(
            "/api/v2/discount-outliers/",
            params={
                "business_unit": "Orthopaedics",
                "severity": "high",
                "min_z_score": 3.0,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "filters" in data
        assert data["filters"]["business_unit"] == "Orthopaedics"
        assert data["filters"]["severity"] == "high"
        assert data["filters"]["min_z_score"] == 3.0
        assert "data" in data

    def test_discount_outliers_summary(self, client):
        """GET /api/v2/discount-outliers/summary returns aggregate stats."""
        response = client.get("/api/v2/discount-outliers/summary")
        assert response.status_code == 200

        data = response.json()
        assert "total_outliers" in data
        assert data["total_outliers"] == 450
        assert "total_recovery" in data
        assert data["total_recovery"] == 5600000.00
        assert "by_business_unit" in data
        assert "by_severity" in data
        assert "by_country" in data
        assert isinstance(data["by_business_unit"], list)
        assert len(data["by_business_unit"]) > 0

    def test_discount_outliers_by_rep(self, client):
        """GET /api/v2/discount-outliers/by-rep?sales_rep_id=REP-001 returns rep-specific outliers."""
        response = client.get(
            "/api/v2/discount-outliers/by-rep",
            params={"sales_rep_id": "REP-001"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["sales_rep_id"] == "REP-001"
        assert "count" in data
        assert "data" in data
        assert isinstance(data["data"], list)


# ===========================================================================
# 3. Price Elasticity
# ===========================================================================
class TestPriceElasticity:
    """Tests for /api/v2/price-elasticity/* endpoints."""

    def test_price_elasticity_list(self, client):
        """GET /api/v2/price-elasticity/ returns elasticity records."""
        response = client.get("/api/v2/price-elasticity/")
        assert response.status_code == 200

        data = response.json()
        assert "count" in data
        assert "filters" in data
        assert "data" in data
        assert isinstance(data["data"], list)
        assert data["count"] >= 1

        row = data["data"][0]
        assert "sku" in row
        assert "elasticity_coefficient" in row
        assert "elasticity_class" in row

    def test_price_elasticity_safe_ranges(self, client):
        """GET /api/v2/price-elasticity/safe-ranges?sku=SKU-001 returns safe pricing data."""
        response = client.get(
            "/api/v2/price-elasticity/safe-ranges",
            params={"sku": "SKU-001"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["sku"] == "SKU-001"
        assert "count" in data
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1

    def test_price_elasticity_distribution(self, client):
        """GET /api/v2/price-elasticity/distribution returns histogram buckets."""
        response = client.get("/api/v2/price-elasticity/distribution")
        assert response.status_code == 200

        data = response.json()
        assert "bucket_count" in data
        assert "data" in data
        assert isinstance(data["data"], list)
        assert data["bucket_count"] == len(data["data"])
        assert data["bucket_count"] > 0

        bucket = data["data"][0]
        assert "bucket" in bucket
        assert "count" in bucket


# ===========================================================================
# 4. Uplift Simulation
# ===========================================================================
class TestUpliftSimulation:
    """Tests for /api/v2/uplift-simulation/* endpoints."""

    def test_uplift_precomputed(self, client):
        """GET /api/v2/uplift-simulation/precomputed returns precomputed simulation results."""
        response = client.get(
            "/api/v2/uplift-simulation/precomputed",
            params={"target": 1.0, "limit": 50},
        )
        assert response.status_code == 200

        data = response.json()
        assert "target" in data
        assert data["target"] == 1.0
        assert "count" in data
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_uplift_simulation_post(self, client):
        """POST /api/v2/uplift-simulation/ runs an on-the-fly uplift simulation."""
        payload = {
            "target_uplift_pct": 2.0,
            "excluded_skus": ["SKU-999"],
            "excluded_segments": [],
            "excluded_countries": [],
            "max_per_sku_increase": 5.0,
        }
        response = client.post("/api/v2/uplift-simulation/", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["target_uplift_pct"] == 2.0
        assert data["max_per_sku_increase"] == 5.0
        assert "exclusions" in data
        assert data["exclusions"]["skus"] == ["SKU-999"]
        assert "actions_count" in data
        assert "total_revenue_impact" in data
        assert "total_volume_impact" in data
        assert "actions" in data
        assert isinstance(data["actions"], list)

    def test_uplift_summary(self, client):
        """GET /api/v2/uplift-simulation/summary returns KPI scenarios."""
        response = client.get("/api/v2/uplift-simulation/summary")
        assert response.status_code == 200

        data = response.json()
        assert "scenarios" in data
        assert isinstance(data["scenarios"], list)
        assert len(data["scenarios"]) > 0

        scenario = data["scenarios"][0]
        assert "target" in scenario
        assert "actions_needed" in scenario
        assert "total_revenue_impact" in scenario
        assert "total_volume_impact" in scenario
        assert "avg_increase_pct" in scenario


# ===========================================================================
# 5. Top 100 Price Changes
# ===========================================================================
class TestTop100PriceChanges:
    """Tests for /api/v2/top100-price-changes/* endpoints."""

    def test_top100_list(self, client):
        """GET /api/v2/top100-price-changes/ returns paginated price changes."""
        response = client.get("/api/v2/top100-price-changes/")
        assert response.status_code == 200

        data = response.json()
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data
        assert "sort_by" in data
        assert "sort_order" in data
        assert "data" in data
        assert isinstance(data["data"], list)

        row = data["data"][0]
        assert "product_name" in row
        assert "revenue_impact" in row

    def test_top100_with_country_filter(self, client):
        """GET with ?country=US returns filtered results."""
        response = client.get(
            "/api/v2/top100-price-changes/",
            params={"country": "US"},
        )
        assert response.status_code == 200

        data = response.json()
        assert "total" in data
        assert "data" in data

    def test_top100_with_multiple_filters(self, client):
        """GET with country, segment, and product_family filters."""
        response = client.get(
            "/api/v2/top100-price-changes/",
            params={
                "country": "US",
                "segment": "IDN",
                "product_family": "Endoscopy",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "total" in data
        assert "data" in data
        assert data["page"] == 1

    def test_top100_filter_options(self, client):
        """GET /api/v2/top100-price-changes/filter-options returns distinct values."""
        response = client.get("/api/v2/top100-price-changes/filter-options")
        assert response.status_code == 200

        data = response.json()
        assert "country" in data
        assert "product_family" in data
        assert "segment" in data
        assert "business_unit" in data
        assert "risk_level" in data

        assert isinstance(data["country"], list)
        assert len(data["country"]) > 0

    def test_top100_export_csv(self, client):
        """GET /api/v2/top100-price-changes/export returns a CSV file."""
        response = client.get("/api/v2/top100-price-changes/export")
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")
        assert "attachment" in response.headers.get("content-disposition", "")

        body = response.text
        assert len(body) > 0
        # CSV should contain header row with known field names
        assert "product_name" in body
        assert "revenue_impact" in body


# ===========================================================================
# 6. Pricing Recommendations
# ===========================================================================
class TestPricingRecommendations:
    """Tests for /api/v2/pricing-recommendations/* endpoints."""

    def test_recommendations_list(self, client):
        """GET /api/v2/pricing-recommendations/ returns recommendation rows."""
        response = client.get("/api/v2/pricing-recommendations/")
        assert response.status_code == 200

        data = response.json()
        assert "count" in data
        assert "filters" in data
        assert "sort_by" in data
        assert "data" in data
        assert isinstance(data["data"], list)
        assert data["count"] >= 1

        row = data["data"][0]
        assert "action_type" in row
        assert "priority_score" in row
        assert "revenue_impact" in row

    def test_recommendations_summary(self, client):
        """GET /api/v2/pricing-recommendations/summary returns KPIs by action type."""
        response = client.get("/api/v2/pricing-recommendations/summary")
        assert response.status_code == 200

        data = response.json()
        assert "by_action_type" in data
        assert isinstance(data["by_action_type"], list)
        assert len(data["by_action_type"]) > 0

        entry = data["by_action_type"][0]
        assert "action_type" in entry
        assert "count" in entry
        assert "avg_priority_score" in entry
        assert "total_revenue_impact" in entry
        assert "avg_revenue_impact" in entry


# ===========================================================================
# 7. External Data
# ===========================================================================
class TestExternalData:
    """Tests for /api/v2/external-data/* endpoints."""

    def test_external_data_list(self, client):
        """GET /api/v2/external-data/ returns external data records."""
        response = client.get("/api/v2/external-data/")
        assert response.status_code == 200

        data = response.json()
        assert "count" in data
        assert "category" in data
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_external_data_sources(self, client):
        """GET /api/v2/external-data/sources returns uploaded data sources."""
        response = client.get("/api/v2/external-data/sources")
        assert response.status_code == 200

        data = response.json()
        assert "count" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_external_data_list_with_category_filter(self, client):
        """GET /api/v2/external-data/?category=market_research filters by category."""
        response = client.get(
            "/api/v2/external-data/",
            params={"category": "market_research"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["category"] == "market_research"
        assert "data" in data


# ===========================================================================
# 8. Pricing Scenarios
# ===========================================================================
class TestPricingScenarios:
    """Tests for /api/v2/pricing-scenarios/* endpoints."""

    def test_scenarios_user_info(self, client):
        """GET /api/v2/pricing-scenarios/user-info returns user identity."""
        response = client.get("/api/v2/pricing-scenarios/user-info")
        assert response.status_code == 200

        data = response.json()
        assert "user_id" in data
        assert "user_email" in data
        assert "is_admin" in data
        # Default headers yield anonymous user
        assert data["user_email"] == "anonymous@stryker.com"
        assert isinstance(data["is_admin"], bool)

    def test_scenarios_list(self, client):
        """GET /api/v2/pricing-scenarios/ returns paginated scenarios."""
        response = client.get("/api/v2/pricing-scenarios/")
        assert response.status_code == 200

        data = response.json()
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data
        assert "user" in data
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_scenarios_create(self, client):
        """POST /api/v2/pricing-scenarios/ creates a new scenario."""
        payload = {
            "name": "Q1 2025 Review",
            "description": "Quarterly pricing adjustments",
            "parameters": {"target_margin": 0.35, "max_increase": 5.0},
            "status": "draft",
        }
        response = client.post("/api/v2/pricing-scenarios/", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "created"
        assert "scenario_id" in data
        assert data["name"] == "Q1 2025 Review"
        assert "created_by" in data
        assert "created_at" in data

    def test_scenarios_get_by_id(self, client):
        """GET /api/v2/pricing-scenarios/{scenario_id} returns a single scenario."""
        response = client.get("/api/v2/pricing-scenarios/abc12345")
        assert response.status_code == 200

        data = response.json()
        assert "scenario" in data
        assert "user" in data
        assert data["scenario"]["scenario_id"] == "abc12345"
