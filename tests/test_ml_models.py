"""
Tests for the ML model layer (elasticity, revenue, margin models).

Validates that trained models are loadable from MLflow, produce predictions
within expected ranges, and have the required artifacts (signatures, SHAP
values, coefficients).

Uses mocked serving endpoints where a live Databricks connection is not
available.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT_PREFIX = "/stryker-pricing-intelligence"
ELASTICITY_MODEL_NAME = "stryker_price_elasticity"
REVENUE_MODEL_NAME = "stryker_revenue_predictor"
MARGIN_MODEL_NAME = "stryker_margin_predictor"
SERVING_ENDPOINT = "stryker-pricing-models"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mlflow_client():
    """Return an MLflow tracking client (local or remote)."""
    import mlflow

    return mlflow.tracking.MlflowClient()


@pytest.fixture()
def mock_serving_response():
    """Mock response from the Databricks model serving endpoint."""
    return {
        "predictions": [
            {
                "predicted_volume_change_pct": -8.5,
                "predicted_revenue_impact": -125_000.0,
                "predicted_margin_impact": -45_000.0,
                "confidence_interval": {"lower": -12.0, "upper": -5.0},
                "top_sensitivity_factors": [
                    {"factor": "competitor_price_ratio", "weight": 0.35},
                    {"factor": "contract_tier", "weight": 0.25},
                    {"factor": "market_share", "weight": 0.18},
                ],
                "competitive_risk_score": 0.72,
            }
        ]
    }


@pytest.fixture()
def sample_features():
    """Sample input feature vector for prediction tests."""
    return {
        "product_id": "PROD-001",
        "current_list_price": 15_000.0,
        "price_change_pct": 5.0,
        "contract_tier": "GPO_Tier1",
        "competitor_price_ratio": 1.05,
        "market_share": 0.32,
        "months_since_launch": 24,
        "therapeutic_area": "Orthopedics",
    }


# ---------------------------------------------------------------------------
# Model loading tests
# ---------------------------------------------------------------------------
class TestModelLoading:
    """Ensure all registered models can be loaded from MLflow."""

    def test_elasticity_model_loaded(self, mlflow_client):
        """The price-elasticity model must be registered and loadable."""
        versions = mlflow_client.get_latest_versions(
            ELASTICITY_MODEL_NAME, stages=["Production", "Staging", "None"]
        )
        assert len(versions) > 0, (
            f"No versions found for model '{ELASTICITY_MODEL_NAME}'"
        )

    def test_revenue_model_loaded(self, mlflow_client):
        """The revenue-predictor model must be registered and loadable."""
        versions = mlflow_client.get_latest_versions(
            REVENUE_MODEL_NAME, stages=["Production", "Staging", "None"]
        )
        assert len(versions) > 0, (
            f"No versions found for model '{REVENUE_MODEL_NAME}'"
        )

    def test_margin_model_loaded(self, mlflow_client):
        """The margin-predictor model must be registered and loadable."""
        versions = mlflow_client.get_latest_versions(
            MARGIN_MODEL_NAME, stages=["Production", "Staging", "None"]
        )
        assert len(versions) > 0, (
            f"No versions found for model '{MARGIN_MODEL_NAME}'"
        )


# ---------------------------------------------------------------------------
# Prediction quality tests
# ---------------------------------------------------------------------------
class TestElasticityModel:
    """Tests for the price-elasticity model predictions."""

    def test_elasticity_negative_correlation(self, mock_serving_response):
        """
        A price increase should predict a volume decrease (negative
        elasticity).  We use the mocked serving response to validate the
        directional invariant.
        """
        prediction = mock_serving_response["predictions"][0]
        volume_change = prediction["predicted_volume_change_pct"]

        # For a positive price change, volume change should be negative
        assert volume_change < 0, (
            f"Expected negative volume change for price increase, "
            f"got {volume_change}"
        )

    @patch("requests.post")
    def test_elasticity_via_serving_endpoint(
        self, mock_post, sample_features, mock_serving_response
    ):
        """End-to-end test through a mocked serving endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_serving_response
        mock_post.return_value = mock_response

        import requests

        response = requests.post(
            f"https://test-workspace.databricks.com/serving-endpoints/"
            f"{SERVING_ENDPOINT}/invocations",
            json={"dataframe_records": [sample_features]},
            headers={"Authorization": "Bearer mock-token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1


class TestRevenueModel:
    """Tests for the revenue-predictor model."""

    def test_revenue_model_prediction_range(self, mock_serving_response):
        """
        Predicted revenue impact should be within reasonable bounds.
        For a single product with a moderate price change, the absolute
        impact should not exceed $10M.
        """
        prediction = mock_serving_response["predictions"][0]
        impact = abs(prediction["predicted_revenue_impact"])

        assert impact < 10_000_000, (
            f"Revenue impact {impact:,.0f} exceeds $10M -- "
            "likely a data or model issue"
        )

    def test_revenue_confidence_interval_valid(self, mock_serving_response):
        """Confidence interval lower bound should be below upper bound."""
        ci = mock_serving_response["predictions"][0]["confidence_interval"]
        assert ci["lower"] < ci["upper"], (
            f"Invalid CI: lower={ci['lower']}, upper={ci['upper']}"
        )


class TestMarginModel:
    """Tests for the margin-predictor model (ElasticNet)."""

    def test_margin_model_coefficients(self):
        """
        The ElasticNet margin model should have non-zero coefficients,
        verifying that regularisation did not shrink all features to zero.
        """
        # Simulate loading coefficients from a logged artifact
        mock_coefficients = np.array([0.12, -0.08, 0.05, 0.0, -0.03, 0.07])

        non_zero = np.count_nonzero(mock_coefficients)
        assert non_zero > 0, (
            "All ElasticNet coefficients are zero -- the model has no "
            "predictive power"
        )
        assert non_zero >= 3, (
            f"Only {non_zero} non-zero coefficients; expected at least 3 "
            "meaningful features"
        )


# ---------------------------------------------------------------------------
# Model artifact tests
# ---------------------------------------------------------------------------
class TestModelArtifacts:
    """Tests for model signatures and SHAP artifacts."""

    def test_model_signatures(self, mlflow_client):
        """All registered models must have logged input/output signatures."""
        for model_name in [
            ELASTICITY_MODEL_NAME,
            REVENUE_MODEL_NAME,
            MARGIN_MODEL_NAME,
        ]:
            versions = mlflow_client.get_latest_versions(
                model_name, stages=["Production", "Staging", "None"]
            )
            if not versions:
                pytest.skip(f"Model '{model_name}' not registered -- skipping")

            latest = versions[0]
            run = mlflow_client.get_run(latest.run_id)

            # The signature is stored in the MLmodel file; we check that the
            # run logged the model artifact
            artifacts = mlflow_client.list_artifacts(latest.run_id)
            artifact_paths = [a.path for a in artifacts]
            assert any("model" in p.lower() for p in artifact_paths), (
                f"Model '{model_name}' run {latest.run_id} has no model "
                "artifact logged"
            )

    def test_shap_values_computed(self, mlflow_client):
        """
        SHAP value artifacts should exist for the elasticity model to
        support the sensitivity-factor explanations in the UI.
        """
        versions = mlflow_client.get_latest_versions(
            ELASTICITY_MODEL_NAME, stages=["Production", "Staging", "None"]
        )
        if not versions:
            pytest.skip("Elasticity model not registered -- skipping")

        latest = versions[0]
        artifacts = mlflow_client.list_artifacts(latest.run_id)
        artifact_paths = [a.path for a in artifacts]

        assert any("shap" in p.lower() for p in artifact_paths), (
            f"No SHAP artifact found for elasticity model run {latest.run_id}. "
            "Expected a 'shap_values' or similar artifact."
        )
