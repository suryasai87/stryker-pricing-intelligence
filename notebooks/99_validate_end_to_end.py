# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 99 - End-to-End Validation
# MAGIC
# MAGIC **Purpose**: Comprehensive validation of ALL tables in the Stryker Pricing Intelligence
# MAGIC platform.  This notebook verifies table existence, row counts, schema completeness,
# MAGIC null percentages, and data quality for every Silver and Gold table.  It produces a
# MAGIC formatted validation report and summarises API endpoint readiness.
# MAGIC
# MAGIC **Run this notebook** after executing the full pipeline (notebooks 01-21) to confirm
# MAGIC the platform is ready for production use.
# MAGIC
# MAGIC ## Validation Targets
# MAGIC | Table | Layer | Min Rows |
# MAGIC |-------|-------|----------|
# MAGIC | `silver.ficm_pricing_master` | Silver | 500,000 |
# MAGIC | `silver.dim_customers` | Silver | 400 |
# MAGIC | `silver.dim_sales_reps` | Silver | 60 |
# MAGIC | `gold.discount_outliers` | Gold | 100 |
# MAGIC | `gold.price_elasticity` | Gold | 500 |
# MAGIC | `gold.uplift_simulation` | Gold | 1,000 |
# MAGIC | `gold.pricing_recommendations` | Gold | 200 |
# MAGIC | `gold.top100_price_changes` | Gold | 100 (exact) |
# MAGIC | `gold.custom_pricing_scenarios` | Gold | 15 |
# MAGIC | `gold.external_market_data` | Gold | 30 |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

from datetime import datetime
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Unity Catalog
# ---------------------------------------------------------------------------
CATALOG: str = "hls_amer_catalog"

# ---------------------------------------------------------------------------
# Validation specifications: (schema, table, min_rows, exact_rows, required_columns)
#
# If exact_rows is not None, the row count must match exactly.
# required_columns is a list of column names that MUST exist in the table.
# ---------------------------------------------------------------------------
VALIDATION_SPECS = [
    {
        "schema": "silver",
        "table": "ficm_pricing_master",
        "min_rows": 500_000,
        "exact_rows": None,
        "required_columns": [
            "product_id", "list_price", "pocket_price", "units_sold",
            "year_month", "customer_segment",
        ],
    },
    {
        "schema": "silver",
        "table": "dim_customers",
        "min_rows": 400,
        "exact_rows": None,
        "required_columns": [
            "customer_id", "customer_name",
        ],
    },
    {
        "schema": "silver",
        "table": "dim_sales_reps",
        "min_rows": 60,
        "exact_rows": None,
        "required_columns": [
            "sales_rep_id", "rep_name",
        ],
    },
    {
        "schema": "gold",
        "table": "discount_outliers",
        "min_rows": 100,
        "exact_rows": None,
        "required_columns": [
            "product_id", "discount_pct",
        ],
    },
    {
        "schema": "gold",
        "table": "price_elasticity",
        "min_rows": 500,
        "exact_rows": None,
        "required_columns": [
            "product_id", "elasticity",
        ],
    },
    {
        "schema": "gold",
        "table": "uplift_simulation",
        "min_rows": 1_000,
        "exact_rows": None,
        "required_columns": [
            "product_id", "simulated_revenue",
        ],
    },
    {
        "schema": "gold",
        "table": "pricing_recommendations",
        "min_rows": 200,
        "exact_rows": None,
        "required_columns": [
            "product_id", "recommended_price",
        ],
    },
    {
        "schema": "gold",
        "table": "top100_price_changes",
        "min_rows": 100,
        "exact_rows": 100,
        "required_columns": [
            "product_id", "price_change_pct",
        ],
    },
    {
        "schema": "gold",
        "table": "custom_pricing_scenarios",
        "min_rows": 15,
        "exact_rows": None,
        "required_columns": [
            "scenario_id", "user_id", "user_email", "scenario_name",
            "assumptions", "target_uplift_pct", "selected_skus",
            "selected_segments", "simulation_results", "status",
            "created_at", "updated_at", "is_deleted",
        ],
    },
    {
        "schema": "gold",
        "table": "external_market_data",
        "min_rows": 30,
        "exact_rows": None,
        "required_columns": [
            "data_source", "upload_timestamp", "category", "item_key",
            "item_description", "value", "unit", "effective_date",
        ],
    },
]

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
total_checks = 0
passed_checks = 0
failed_checks = 0
warnings = 0
results = []

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Validation Functions

# COMMAND ----------

def validate_table(spec: dict) -> dict:
    """Validate a single table against its specification.

    Parameters
    ----------
    spec : dict
        Validation specification containing schema, table, min_rows,
        exact_rows, and required_columns.

    Returns
    -------
    dict
        Validation result with status, details, and diagnostics.
    """
    fqn = f"{CATALOG}.{spec['schema']}.{spec['table']}"
    result = {
        "table": fqn,
        "exists": False,
        "row_count": 0,
        "row_count_pass": False,
        "columns_present": [],
        "columns_missing": [],
        "columns_pass": False,
        "null_percentages": {},
        "overall_pass": False,
        "errors": [],
        "warnings": [],
    }

    # ----- Check 1: Table existence -----
    try:
        df = spark.table(fqn)
        result["exists"] = True
    except Exception as e:
        result["errors"].append(f"Table does not exist: {e}")
        return result

    # ----- Check 2: Row count -----
    try:
        row_count = df.count()
        result["row_count"] = row_count

        if spec["exact_rows"] is not None:
            result["row_count_pass"] = row_count == spec["exact_rows"]
            if not result["row_count_pass"]:
                result["errors"].append(
                    f"Row count {row_count:,} != expected {spec['exact_rows']:,}"
                )
        else:
            result["row_count_pass"] = row_count >= spec["min_rows"]
            if not result["row_count_pass"]:
                result["errors"].append(
                    f"Row count {row_count:,} < minimum {spec['min_rows']:,}"
                )
    except Exception as e:
        result["errors"].append(f"Could not count rows: {e}")

    # ----- Check 3: Required columns -----
    try:
        actual_columns = set(df.columns)
        required = set(spec["required_columns"])
        result["columns_present"] = sorted(required & actual_columns)
        result["columns_missing"] = sorted(required - actual_columns)
        result["columns_pass"] = len(result["columns_missing"]) == 0

        if result["columns_missing"]:
            result["errors"].append(
                f"Missing columns: {result['columns_missing']}"
            )
    except Exception as e:
        result["errors"].append(f"Column check failed: {e}")

    # ----- Check 4: Null percentages -----
    try:
        if row_count > 0:
            null_exprs = [
                (
                    F.round(
                        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)) / F.count("*") * 100,
                        2,
                    ).alias(c)
                )
                for c in df.columns
            ]
            null_row = df.select(*null_exprs).collect()[0]
            result["null_percentages"] = {
                col: float(null_row[col]) for col in df.columns
            }

            # Warn on columns with > 50% nulls
            high_null_cols = [
                col for col, pct in result["null_percentages"].items()
                if pct > 50.0
            ]
            if high_null_cols:
                result["warnings"].append(
                    f"High null columns (>50%): {high_null_cols}"
                )
    except Exception as e:
        result["warnings"].append(f"Null check failed: {e}")

    # ----- Overall pass -----
    result["overall_pass"] = (
        result["exists"]
        and result["row_count_pass"]
        and result["columns_pass"]
    )

    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run Validations

# COMMAND ----------

print(f"{'=' * 100}")
print(f"  STRYKER PRICING INTELLIGENCE - END-TO-END VALIDATION")
print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Catalog: {CATALOG}")
print(f"{'=' * 100}")
print()

for spec in VALIDATION_SPECS:
    fqn = f"{CATALOG}.{spec['schema']}.{spec['table']}"
    print(f"Validating {fqn} ...", end=" ")

    result = validate_table(spec)
    results.append(result)

    if result["overall_pass"]:
        passed_checks += 1
        status_icon = "PASS"
    else:
        failed_checks += 1
        status_icon = "FAIL"

    if result["warnings"]:
        warnings += len(result["warnings"])

    total_checks += 1
    print(f"[{status_icon}]")

print(f"\nValidation sweep complete: {total_checks} tables checked")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Detailed Validation Report

# COMMAND ----------

print(f"\n{'=' * 100}")
print(f"  DETAILED VALIDATION REPORT")
print(f"{'=' * 100}")

for i, result in enumerate(results):
    spec = VALIDATION_SPECS[i]
    print(f"\n{'─' * 80}")
    status = "PASS" if result["overall_pass"] else "FAIL"
    print(f"  [{status}] {result['table']}")
    print(f"{'─' * 80}")

    # Existence
    print(f"  Exists:           {'Yes' if result['exists'] else 'NO'}")

    if not result["exists"]:
        for err in result["errors"]:
            print(f"    ERROR: {err}")
        continue

    # Row count
    if spec["exact_rows"] is not None:
        count_label = f"{result['row_count']:,} (expected exactly {spec['exact_rows']:,})"
    else:
        count_label = f"{result['row_count']:,} (min {spec['min_rows']:,})"
    row_status = "PASS" if result["row_count_pass"] else "FAIL"
    print(f"  Row count:        {count_label} [{row_status}]")

    # Columns
    col_status = "PASS" if result["columns_pass"] else "FAIL"
    n_required = len(spec["required_columns"])
    n_present = len(result["columns_present"])
    print(f"  Required columns: {n_present}/{n_required} present [{col_status}]")
    if result["columns_missing"]:
        print(f"    Missing: {result['columns_missing']}")

    # Null percentages (show top 5 by null %)
    if result["null_percentages"]:
        sorted_nulls = sorted(
            result["null_percentages"].items(),
            key=lambda x: -x[1],
        )
        non_zero_nulls = [(c, p) for c, p in sorted_nulls if p > 0]
        if non_zero_nulls:
            print(f"  Null percentages (top 5):")
            for col, pct in non_zero_nulls[:5]:
                print(f"    {col}: {pct:.1f}%")
        else:
            print(f"  Null percentages: all columns 0.0%")

    # Errors
    for err in result["errors"]:
        print(f"  ERROR: {err}")

    # Warnings
    for warn in result["warnings"]:
        print(f"  WARNING: {warn}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Summary Dashboard

# COMMAND ----------

print(f"\n{'=' * 100}")
print(f"  VALIDATION SUMMARY")
print(f"{'=' * 100}")
print()

# Table-level summary
print(f"  {'Table':<55} {'Rows':>12} {'Status':>8}")
print(f"  {'─' * 55} {'─' * 12} {'─' * 8}")
for result in results:
    table_short = result["table"].replace(f"{CATALOG}.", "")
    row_str = f"{result['row_count']:,}" if result["exists"] else "N/A"
    status = "PASS" if result["overall_pass"] else "FAIL"
    print(f"  {table_short:<55} {row_str:>12} {status:>8}")

print(f"\n  {'─' * 77}")
print(f"  Total tables:   {total_checks}")
print(f"  Passed:         {passed_checks}")
print(f"  Failed:         {failed_checks}")
print(f"  Warnings:       {warnings}")
print(f"  Pass rate:      {passed_checks / max(total_checks, 1):.0%}")
print(f"  {'─' * 77}")

# Overall verdict
if failed_checks == 0:
    print(f"\n  VERDICT: ALL VALIDATIONS PASSED")
else:
    print(f"\n  VERDICT: {failed_checks} VALIDATION(S) FAILED")
    print(f"  Review the detailed report above for error details.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. API Endpoint Smoke Test Expectations
# MAGIC
# MAGIC The following table maps each validated Gold table to its corresponding API
# MAGIC endpoint(s) in the Stryker Pricing Intelligence application.  Use this as a
# MAGIC reference for manual or automated API smoke testing after deployment.

# COMMAND ----------

API_ENDPOINT_MAP = [
    {
        "table": f"{CATALOG}.silver.ficm_pricing_master",
        "endpoints": [
            {"method": "GET", "path": "/api/v1/waterfall/{product_id}", "description": "Price waterfall breakdown per SKU"},
            {"method": "GET", "path": "/api/v1/products", "description": "Product catalog with pricing metadata"},
        ],
    },
    {
        "table": f"{CATALOG}.silver.dim_customers",
        "endpoints": [
            {"method": "GET", "path": "/api/v1/customers", "description": "Customer dimension lookup"},
        ],
    },
    {
        "table": f"{CATALOG}.silver.dim_sales_reps",
        "endpoints": [
            {"method": "GET", "path": "/api/v1/sales-reps", "description": "Sales rep directory"},
        ],
    },
    {
        "table": f"{CATALOG}.gold.discount_outliers",
        "endpoints": [
            {"method": "GET", "path": "/api/v1/discount-outliers", "description": "Anomalous discount transactions"},
            {"method": "GET", "path": "/api/v1/kpis", "description": "Portfolio KPIs including outlier count"},
        ],
    },
    {
        "table": f"{CATALOG}.gold.price_elasticity",
        "endpoints": [
            {"method": "GET", "path": "/api/v1/elasticity/{product_id}", "description": "Price elasticity per SKU"},
            {"method": "POST", "path": "/api/v1/simulate", "description": "Price change simulation with elasticity"},
        ],
    },
    {
        "table": f"{CATALOG}.gold.uplift_simulation",
        "endpoints": [
            {"method": "POST", "path": "/api/v1/simulate", "description": "Revenue uplift simulation"},
            {"method": "POST", "path": "/api/v1/batch-scenario", "description": "Batch scenario scoring"},
        ],
    },
    {
        "table": f"{CATALOG}.gold.pricing_recommendations",
        "endpoints": [
            {"method": "GET", "path": "/api/v1/recommendations", "description": "AI-generated pricing recommendations"},
        ],
    },
    {
        "table": f"{CATALOG}.gold.top100_price_changes",
        "endpoints": [
            {"method": "GET", "path": "/api/v1/top-changes", "description": "Top 100 price change opportunities"},
        ],
    },
    {
        "table": f"{CATALOG}.gold.custom_pricing_scenarios",
        "endpoints": [
            {"method": "GET", "path": "/api/v1/scenarios", "description": "List user pricing scenarios"},
            {"method": "POST", "path": "/api/v1/scenarios", "description": "Create new pricing scenario"},
            {"method": "PUT", "path": "/api/v1/scenarios/{scenario_id}", "description": "Update scenario status"},
            {"method": "DELETE", "path": "/api/v1/scenarios/{scenario_id}", "description": "Soft-delete scenario"},
        ],
    },
    {
        "table": f"{CATALOG}.gold.external_market_data",
        "endpoints": [
            {"method": "GET", "path": "/api/v1/external-data", "description": "External market data feed"},
            {"method": "GET", "path": "/api/v1/external-data/{category}", "description": "Market data by category"},
        ],
    },
]

print(f"\n{'=' * 100}")
print(f"  API ENDPOINT SMOKE TEST REFERENCE")
print(f"{'=' * 100}")
print()

total_endpoints = 0
for mapping in API_ENDPOINT_MAP:
    # Find the validation result for this table
    table_result = next(
        (r for r in results if r["table"] == mapping["table"]),
        None,
    )
    data_status = "READY" if (table_result and table_result["overall_pass"]) else "NOT READY"

    print(f"  Table: {mapping['table']}")
    print(f"  Data status: [{data_status}]")
    print(f"  Endpoints:")
    for ep in mapping["endpoints"]:
        total_endpoints += 1
        print(f"    {ep['method']:>6} {ep['path']:<45} -- {ep['description']}")
    print()

print(f"  {'─' * 77}")
print(f"  Total API endpoints mapped: {total_endpoints}")
print(f"  Tables with data ready: {passed_checks}/{total_checks}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Data Freshness Check

# COMMAND ----------

print(f"\n{'=' * 100}")
print(f"  DATA FRESHNESS CHECK")
print(f"{'=' * 100}")
print()

freshness_cols = {
    f"{CATALOG}.silver.ficm_pricing_master": "year_month",
    f"{CATALOG}.gold.custom_pricing_scenarios": "updated_at",
    f"{CATALOG}.gold.external_market_data": "effective_date",
}

for fqn, date_col in freshness_cols.items():
    try:
        df = spark.table(fqn)
        if date_col in df.columns:
            max_date = df.agg(F.max(date_col)).collect()[0][0]
            min_date = df.agg(F.min(date_col)).collect()[0][0]
            print(f"  {fqn}")
            print(f"    Column:  {date_col}")
            print(f"    Range:   {min_date} to {max_date}")
            print()
    except Exception as e:
        print(f"  {fqn}: Could not check freshness ({e})")
        print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Final Verdict

# COMMAND ----------

print(f"\n{'=' * 100}")
if failed_checks == 0:
    print(f"  STRYKER PRICING INTELLIGENCE PLATFORM -- VALIDATION COMPLETE")
    print(f"  ALL {total_checks} TABLES VALIDATED SUCCESSFULLY")
    print(f"  The platform is ready for application deployment and API testing.")
else:
    print(f"  STRYKER PRICING INTELLIGENCE PLATFORM -- VALIDATION INCOMPLETE")
    print(f"  {failed_checks} of {total_checks} tables failed validation.")
    print(f"  Review the detailed report and re-run the corresponding pipeline notebooks.")
print(f"{'=' * 100}")

# Assert for CI/CD pipeline integration
assert failed_checks == 0, (
    f"End-to-end validation failed: {failed_checks} table(s) did not pass. "
    f"See detailed report above."
)
