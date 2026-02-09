"""
Tests for the FastAPI backend endpoints.

Uses fastapi.testclient.TestClient with a mocked Databricks SDK so that
tests can run without a live workspace connection.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mock_workspace_client():
    """Create a mocked Databricks WorkspaceClient."""
    client = MagicMock()

    # Mock statement execution for product listing
    mock_response = MagicMock()
    mock_response.status.state.value = "SUCCEEDED"
    mock_response.manifest.schema.columns = [
        MagicMock(name="product_id"),
        MagicMock(name="product_name"),
        MagicMock(name="category"),
        MagicMock(name="list_price"),
        MagicMock(name="cost"),
    ]
    mock_response.result.data_array = [
        ["PROD-001", "Hip Implant System A", "Implants", "15000.00", "6000.00"],
        ["PROD-002", "Knee Replacement Kit B", "Implants", "22000.00", "9000.00"],
        ["PROD-003", "Arthroscopy Camera C", "Endoscopy", "8500.00", "3500.00"],
    ]
    client.statement_execution.execute_statement.return_value = mock_response

    # Mock serving endpoint invocation for predictions
    client.api_client.do.return_value = {
        "predictions": [
            {
                "predicted_volume_change_pct": -6.2,
                "predicted_revenue_impact": -93_000.0,
                "predicted_margin_impact": -31_000.0,
                "confidence_interval": {"lower": -9.0, "upper": -3.5},
                "top_sensitivity_factors": [
                    {"factor": "competitor_price_ratio", "weight": 0.32},
                    {"factor": "contract_tier", "weight": 0.22},
                ],
                "competitive_risk_score": 0.65,
            }
        ]
    }

    return client


@pytest.fixture(scope="module")
def client(mock_workspace_client):
    """
    Create a TestClient with the Databricks SDK patched out.

    The patch replaces the WorkspaceClient at import time so the FastAPI
    app initialises with the mock instead of attempting a real connection.
    """
    with patch(
        "backend.main.WorkspaceClient", return_value=mock_workspace_client
    ), patch(
        "backend.main.Config"
    ):
        # Import after patching so the module-level client uses the mock
        from backend.main import app

        yield TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
class TestHealthCheck:
    """Tests for the /health endpoint."""

    def test_health_check(self, client):
        """GET /health should return 200 with a status field."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------
class TestProducts:
    """Tests for the /api/v1/products endpoint."""

    def test_list_products(self, client):
        """GET /api/v1/products should return a list of product objects."""
        response = client.get("/api/v1/products")
        assert response.status_code == 200

        data = response.json()
        products = data if isinstance(data, list) else data.get("products", [])
        assert len(products) > 0, "Expected at least one product"

        first = products[0]
        assert "product_id" in first
        assert "product_name" in first or "name" in first


# ---------------------------------------------------------------------------
# Price simulation
# ---------------------------------------------------------------------------
class TestPriceSimulation:
    """Tests for the /api/v1/simulate-price-change endpoint."""

    def test_simulate_price_change(self, client):
        """POST with a valid request should return prediction fields."""
        payload = {
            "product_id": "PROD-001",
            "price_change_pct": 5.0,
        }
        response = client.post("/api/v1/simulate-price-change", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "predicted_volume_change_pct" in data
        assert "predicted_revenue_impact" in data
        assert "predicted_margin_impact" in data
        assert "confidence_interval" in data
        assert "competitive_risk_score" in data

    def test_simulate_price_change_invalid(self, client):
        """POST with a missing or invalid product_id should return 4xx."""
        payload = {
            "product_id": "",
            "price_change_pct": 5.0,
        }
        response = client.post("/api/v1/simulate-price-change", json=payload)
        assert response.status_code in (400, 404, 422), (
            f"Expected 4xx for invalid product_id, got {response.status_code}"
        )


# ---------------------------------------------------------------------------
# Portfolio KPIs
# ---------------------------------------------------------------------------
class TestPortfolioKPIs:
    """Tests for the /api/v1/portfolio-kpis endpoint."""

    def test_portfolio_kpis(self, client):
        """GET should return a KPI structure with expected fields."""
        response = client.get("/api/v1/portfolio-kpis")
        assert response.status_code == 200

        data = response.json()
        # The KPI response should contain top-level metrics
        expected_keys = {
            "total_revenue",
            "avg_margin_pct",
            "total_products",
            "avg_discount_pct",
        }
        actual_keys = set(data.keys())
        missing = expected_keys - actual_keys
        assert not missing, f"Missing KPI fields: {missing}"


# ---------------------------------------------------------------------------
# Price waterfall
# ---------------------------------------------------------------------------
class TestPriceWaterfall:
    """Tests for the /api/v1/price-waterfall endpoint."""

    def test_price_waterfall(self, client):
        """GET should return waterfall steps for a product."""
        response = client.get(
            "/api/v1/price-waterfall", params={"product_id": "PROD-001"}
        )
        assert response.status_code == 200

        data = response.json()
        steps = data if isinstance(data, list) else data.get("steps", [])
        assert len(steps) > 0, "Expected at least one waterfall step"

        first_step = steps[0]
        assert "name" in first_step or "label" in first_step
        assert "value" in first_step or "amount" in first_step


# ---------------------------------------------------------------------------
# Competitive landscape
# ---------------------------------------------------------------------------
class TestCompetitiveLandscape:
    """Tests for the /api/v1/competitive-landscape endpoint."""

    def test_competitive_landscape(self, client):
        """GET should return competitor data."""
        response = client.get(
            "/api/v1/competitive-landscape",
            params={"product_id": "PROD-001"},
        )
        assert response.status_code == 200

        data = response.json()
        competitors = (
            data if isinstance(data, list) else data.get("competitors", [])
        )
        assert isinstance(competitors, list)


# ---------------------------------------------------------------------------
# Batch scenario
# ---------------------------------------------------------------------------
class TestBatchScenario:
    """Tests for the /api/v1/batch-scenario endpoint."""

    def test_batch_scenario(self, client):
        """POST with multiple product scenarios should return results for each."""
        payload = {
            "scenarios": [
                {"product_id": "PROD-001", "price_change_pct": 3.0},
                {"product_id": "PROD-002", "price_change_pct": -2.0},
                {"product_id": "PROD-003", "price_change_pct": 7.5},
            ]
        }
        response = client.post("/api/v1/batch-scenario", json=payload)
        assert response.status_code == 200

        data = response.json()
        results = data if isinstance(data, list) else data.get("results", [])
        assert len(results) == 3, (
            f"Expected 3 scenario results, got {len(results)}"
        )

        for result in results:
            assert "product_id" in result
            assert "predicted_volume_change_pct" in result or "error" in result
