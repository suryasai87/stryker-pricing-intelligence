# Stryker Pricing Intelligence Platform

An end-to-end pricing analytics and simulation platform built on Databricks. The system generates synthetic medical-device transaction data, processes it through a medallion architecture, trains ML models for price-elasticity and margin prediction, and serves interactive what-if scenarios through a React + FastAPI application deployed as a Databricks App.

---

## Architecture

```
 +---------------------+      +------------------------------+      +---------------------+
 |  Synthetic Data     |      |  Medallion Pipeline          |      |  ML Models          |
 |  Generation         | ---> |  Bronze -> Silver -> Gold    | ---> |  (MLflow Registry)  |
 |  (Notebooks Grp 1)  |      |  (Notebooks Grp 2)          |      |  (Notebooks Grp 3)  |
 +---------------------+      +------------------------------+      +---------------------+
                                                                            |
                                                                            v
                                                                    +---------------------+
                                                                    |  Model Serving      |
                                                                    |  Endpoint           |
                                                                    +---------------------+
                                                                            |
                                                                            v
 +---------------------+      +------------------------------+      +---------------------+
 |  React UI           | <--- |  FastAPI Backend             | <--- |  Databricks SDK     |
 |  (Material-UI)      |      |  /api/v1/*                   |      |  + SQL Warehouse    |
 +---------------------+      +------------------------------+      +---------------------+
                                        |
                                        v
                               +---------------------+
                               |  Databricks App     |
                               |  (Deployed via CLI) |
                               +---------------------+
```

**Data flow**: Synthetic Data --> Bronze (raw) --> Silver (cleaned/enriched) --> Gold (aggregated) --> Feature Store --> ML Models --> Serving Endpoint --> FastAPI --> React UI

---

## Tech Stack

| Layer            | Technology                                              |
|------------------|---------------------------------------------------------|
| Data Platform    | Databricks, Unity Catalog, Delta Lake                   |
| Pipeline         | PySpark, Medallion Architecture (Bronze/Silver/Gold)    |
| ML               | scikit-learn, XGBoost, ElasticNet, MLflow, SHAP         |
| Model Serving    | Databricks Model Serving Endpoints                      |
| Backend          | Python 3.10+, FastAPI, Uvicorn, Databricks SDK          |
| Frontend         | React 18, TypeScript, Material-UI v5, Vite, Recharts   |
| Deployment       | Databricks Apps, Service Principal Auth                 |

---

## Prerequisites

- **Databricks Workspace** with Unity Catalog enabled
- **SQL Warehouse** (Serverless or Pro)
- **ML Runtime** cluster (13.3 LTS+ recommended)
- **Databricks CLI** v0.200+ configured with a profile
- **Node.js** 18+ and **npm** 9+ (for frontend development)
- **Python** 3.10+ (for backend and tests)

---

## Setup Instructions

### Step 1: Generate Synthetic Data (Group 1 Notebooks)

Run the following notebooks in order on a cluster with ML Runtime:

1. `notebooks/01_synthetic_product_master.py` -- Creates 200 medical-device products across Stryker business units
2. `notebooks/02_synthetic_transactions.py` -- Generates ~500k transaction records with discount waterfall (list -> invoice -> pocket)
3. `notebooks/03_synthetic_external_factors.py` -- Produces 36 months of market/competitor/regulatory factors
4. `notebooks/04_synthetic_competitor_data.py` -- Creates competitor pricing benchmarks

All tables are written to `hls_amer_catalog.bronze.*`.

### Step 2: Run Medallion Pipeline (Group 2 Notebooks)

5. `notebooks/05_bronze_to_silver.py` -- Cleans, deduplicates, and enriches raw data
6. `notebooks/06_silver_to_gold.py` -- Aggregates into analytics-ready tables
7. `notebooks/07_feature_engineering.py` -- Builds the feature store for ML

Tables progress from `bronze` to `silver` to `gold` schemas.

### Step 3: Train ML Models (Group 3 Notebooks)

8. `notebooks/08_train_elasticity_model.py` -- Price-elasticity model (XGBoost) with SHAP explanations
9. `notebooks/09_train_revenue_model.py` -- Revenue impact predictor
10. `notebooks/10_train_margin_model.py` -- Margin predictor (ElasticNet)
11. `notebooks/11_register_and_serve.py` -- Registers models in MLflow, deploys serving endpoint

Models are registered under `hls_amer_catalog` in Unity Catalog.

### Step 4: Deploy the Databricks App (Group 4)

```bash
# Build frontend and deploy
cd app
python build.py
python deploy_to_databricks.py --app-name stryker-pricing-intel
```

---

## Service Principal Permissions

After deployment, the app's service principal needs access to data and models. Retrieve the service principal client ID:

```bash
databricks apps get stryker-pricing-intel --output json | jq '.service_principal_client_id'
```

### SQL Warehouse Access

```bash
databricks permissions update sql/warehouses <WAREHOUSE_ID> --json '{
  "access_control_list": [{
    "service_principal_name": "<CLIENT_ID>",
    "permission_level": "CAN_USE"
  }]
}'
```

### Unity Catalog Grants

```sql
GRANT USE CATALOG ON CATALOG hls_amer_catalog TO `<CLIENT_ID>`;
GRANT USE SCHEMA ON SCHEMA hls_amer_catalog.gold TO `<CLIENT_ID>`;
GRANT USE SCHEMA ON SCHEMA hls_amer_catalog.silver TO `<CLIENT_ID>`;
GRANT SELECT ON SCHEMA hls_amer_catalog.gold TO `<CLIENT_ID>`;
GRANT SELECT ON SCHEMA hls_amer_catalog.silver TO `<CLIENT_ID>`;
```

### Model Serving Endpoint

```bash
databricks permissions update serving-endpoints/stryker-pricing-models --json '{
  "access_control_list": [{
    "service_principal_name": "apps/stryker-pricing-intel",
    "permission_level": "CAN_QUERY"
  }]
}'
```

---

## API Documentation

All endpoints are served under the FastAPI backend at port 8000.

| Method | Endpoint                          | Description                                        |
|--------|-----------------------------------|----------------------------------------------------|
| GET    | `/health`                         | Health check; returns `{"status": "healthy"}`      |
| GET    | `/api/v1/products`                | List all products in the catalog                   |
| POST   | `/api/v1/simulate-price-change`   | Simulate a price change and get ML predictions     |
| GET    | `/api/v1/portfolio-kpis`          | Aggregate portfolio-level KPIs                     |
| GET    | `/api/v1/price-waterfall`         | Discount waterfall for a product (`?product_id=`)  |
| GET    | `/api/v1/competitive-landscape`   | Competitor pricing data (`?product_id=`)           |
| POST   | `/api/v1/batch-scenario`          | Run multiple price-change scenarios at once        |

### Example: Simulate Price Change

```bash
curl -X POST http://localhost:8000/api/v1/simulate-price-change \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "PROD-001",
    "price_change_pct": 5.0
  }'
```

Response:

```json
{
  "predicted_volume_change_pct": -6.2,
  "predicted_revenue_impact": -93000.0,
  "predicted_margin_impact": -31000.0,
  "confidence_interval": { "lower": -9.0, "upper": -3.5 },
  "top_sensitivity_factors": [
    { "factor": "competitor_price_ratio", "weight": 0.32 },
    { "factor": "contract_tier", "weight": 0.22 }
  ],
  "competitive_risk_score": 0.65
}
```

---

## Frontend Development

```bash
cd app/frontend

# Install dependencies
npm install

# Start dev server (proxied to backend at :8000)
npm run dev

# Build for production
npm run build
```

The frontend uses Vite for development and builds to `app/frontend/dist/`. The `build.py` script copies the production build into `app/backend/static/` for serving by FastAPI.

### Key React Hooks

| Hook                 | Purpose                                         |
|----------------------|-------------------------------------------------|
| `useModelPrediction` | Debounced ML prediction fetching                |
| `useProducts`        | Product catalog with local search               |
| `usePortfolioData`   | Dashboard KPIs with 5-minute auto-refresh       |

---

## Project Structure

```
stryker-pricing-intelligence/
|-- README.md
|-- notebooks/
|   |-- 01_synthetic_product_master.py
|   |-- 02_synthetic_transactions.py
|   |-- 03_synthetic_external_factors.py
|   |-- 04_synthetic_competitor_data.py
|   |-- 05_bronze_to_silver.py
|   |-- 06_silver_to_gold.py
|   |-- 07_feature_engineering.py
|   |-- 08_train_elasticity_model.py
|   |-- 09_train_revenue_model.py
|   |-- 10_train_margin_model.py
|   |-- 11_register_and_serve.py
|-- app/
|   |-- app.yaml                          # Databricks App manifest
|   |-- build.py                          # Frontend build script
|   |-- deploy_to_databricks.py           # Deployment automation
|   |-- frontend/
|   |   |-- package.json
|   |   |-- vite.config.ts
|   |   |-- src/
|   |       |-- hooks/
|   |       |   |-- useModelPrediction.js
|   |       |   |-- useProducts.js
|   |       |   |-- usePortfolioData.js
|   |       |-- components/
|   |       |-- pages/
|   |-- backend/
|       |-- main.py                       # FastAPI application
|       |-- requirements.txt
|       |-- static/                       # Production frontend build (generated)
|-- tests/
|   |-- __init__.py
|   |-- test_synthetic_data.py
|   |-- test_ml_models.py
|   |-- test_api.py
```

---

## Configuration

Environment variables used by the application:

| Variable                        | Description                           | Default                  |
|---------------------------------|---------------------------------------|--------------------------|
| `DATABRICKS_SERVING_ENDPOINT`   | Model serving endpoint name           | `stryker-pricing-models` |
| `CATALOG_NAME`                  | Unity Catalog name                    | `hls_amer_catalog`       |
| `SCHEMA_NAME`                   | Gold schema name                      | `gold`                   |
| `DATABRICKS_WAREHOUSE_ID`       | SQL Warehouse ID for queries          | (required)               |
| `DATABRICKS_HOST`               | Workspace URL (local dev only)        | (auto in Databricks App) |
| `DATABRICKS_TOKEN`              | PAT for local development             | (auto in Databricks App) |

---

## Running Tests

```bash
# API tests (no Databricks connection needed)
pytest tests/test_api.py -v

# ML model tests (requires MLflow tracking server)
pytest tests/test_ml_models.py -v

# Synthetic data tests (requires Spark + Unity Catalog)
pytest tests/test_synthetic_data.py -v

# All tests
pytest tests/ -v
```

---

## License

Internal use only. Proprietary to the organization.
