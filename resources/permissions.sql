-- =============================================================================
-- Stryker Pricing Intelligence Platform - Service Principal Permissions
-- =============================================================================
--
-- Run these statements in a Databricks SQL editor AFTER deploying the app.
-- Replace ${SP_CLIENT_ID} with the actual Service Principal Application/Client
-- ID (UUID) returned by:
--
--   databricks apps get stryker-pricing-intel-<target> --output json \
--     | jq -r '.service_principal_client_id'
--
-- The client ID must be enclosed in BACKTICKS in all GRANT statements.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- 1. Catalog and Schema access
-- ---------------------------------------------------------------------------
GRANT USE CATALOG ON CATALOG hls_amer_catalog TO `${SP_CLIENT_ID}`;

GRANT USE SCHEMA ON SCHEMA hls_amer_catalog.silver TO `${SP_CLIENT_ID}`;
GRANT USE SCHEMA ON SCHEMA hls_amer_catalog.gold   TO `${SP_CLIENT_ID}`;


-- ---------------------------------------------------------------------------
-- 2. Silver layer tables (read-only)
-- ---------------------------------------------------------------------------
GRANT SELECT ON TABLE hls_amer_catalog.silver.dim_customers       TO `${SP_CLIENT_ID}`;
GRANT SELECT ON TABLE hls_amer_catalog.silver.dim_sales_reps      TO `${SP_CLIENT_ID}`;
GRANT SELECT ON TABLE hls_amer_catalog.silver.dim_products        TO `${SP_CLIENT_ID}`;
GRANT SELECT ON TABLE hls_amer_catalog.silver.ficm_pricing_master TO `${SP_CLIENT_ID}`;


-- ---------------------------------------------------------------------------
-- 3. Gold layer tables (read-only)
-- ---------------------------------------------------------------------------
GRANT SELECT ON TABLE hls_amer_catalog.gold.stryker_products           TO `${SP_CLIENT_ID}`;
GRANT SELECT ON TABLE hls_amer_catalog.gold.stryker_price_waterfall    TO `${SP_CLIENT_ID}`;
GRANT SELECT ON TABLE hls_amer_catalog.gold.stryker_competitors        TO `${SP_CLIENT_ID}`;
GRANT SELECT ON TABLE hls_amer_catalog.gold.stryker_external_factors   TO `${SP_CLIENT_ID}`;
GRANT SELECT ON TABLE hls_amer_catalog.gold.stryker_revenue_summary    TO `${SP_CLIENT_ID}`;
GRANT SELECT ON TABLE hls_amer_catalog.gold.stryker_product_features   TO `${SP_CLIENT_ID}`;
GRANT SELECT ON TABLE hls_amer_catalog.gold.discount_outliers          TO `${SP_CLIENT_ID}`;
GRANT SELECT ON TABLE hls_amer_catalog.gold.price_elasticity           TO `${SP_CLIENT_ID}`;
GRANT SELECT ON TABLE hls_amer_catalog.gold.uplift_simulation          TO `${SP_CLIENT_ID}`;
GRANT SELECT ON TABLE hls_amer_catalog.gold.pricing_recommendations    TO `${SP_CLIENT_ID}`;
GRANT SELECT ON TABLE hls_amer_catalog.gold.top100_price_changes       TO `${SP_CLIENT_ID}`;
GRANT SELECT ON TABLE hls_amer_catalog.gold.external_market_data       TO `${SP_CLIENT_ID}`;


-- ---------------------------------------------------------------------------
-- 4. Custom pricing scenarios table (read + write for user scenarios)
-- ---------------------------------------------------------------------------
GRANT ALL PRIVILEGES ON TABLE hls_amer_catalog.gold.custom_pricing_scenarios TO `${SP_CLIENT_ID}`;


-- ---------------------------------------------------------------------------
-- 5. External uploads volume (for CSV file ingestion)
-- ---------------------------------------------------------------------------
-- Create the volume if it does not already exist
CREATE VOLUME IF NOT EXISTS hls_amer_catalog.gold.external_uploads;

GRANT READ VOLUME  ON VOLUME hls_amer_catalog.gold.external_uploads TO `${SP_CLIENT_ID}`;
GRANT WRITE VOLUME ON VOLUME hls_amer_catalog.gold.external_uploads TO `${SP_CLIENT_ID}`;
