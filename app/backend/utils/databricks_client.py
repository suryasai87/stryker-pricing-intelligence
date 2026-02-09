"""
Databricks client singleton.

Provides a single WorkspaceClient instance with SDK auto-auth for Databricks
Apps deployment and token fallback for local development.  Also exposes
helper methods for SQL execution (with in-memory caching) and model-serving
endpoint invocations.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.config import Config
from databricks.sdk.service.sql import StatementState

from backend.utils.config import (
    CACHE_TTL,
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
    WAREHOUSE_ID,
    CATALOG_NAME,
    SCHEMA_GOLD,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory query cache
# ---------------------------------------------------------------------------
_cache: dict[str, Any] = {}
_cache_time: dict[str, float] = {}


def _cache_get(key: str) -> Any | None:
    """Return cached value if still within TTL, else None."""
    if key in _cache and (time.time() - _cache_time.get(key, 0)) < CACHE_TTL:
        return _cache[key]
    return None


def _cache_set(key: str, value: Any) -> None:
    _cache[key] = value
    _cache_time[key] = time.time()


def invalidate_cache(prefix: str | None = None) -> None:
    """Clear all cached entries, or only those whose key starts with *prefix*."""
    if prefix is None:
        _cache.clear()
        _cache_time.clear()
    else:
        keys = [k for k in _cache if k.startswith(prefix)]
        for k in keys:
            _cache.pop(k, None)
            _cache_time.pop(k, None)


# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------
_client: WorkspaceClient | None = None


def get_workspace_client() -> WorkspaceClient:
    """Return a cached WorkspaceClient (created on first call).

    In Databricks Apps the SDK auto-authenticates via the service principal
    bound to the app.  For local development, set DATABRICKS_HOST and
    DATABRICKS_TOKEN environment variables.
    """
    global _client
    if _client is not None:
        return _client

    if DATABRICKS_TOKEN:
        logger.info("Initializing WorkspaceClient with token (local dev mode)")
        _client = WorkspaceClient(
            host=DATABRICKS_HOST,
            token=DATABRICKS_TOKEN,
            config=Config(http_timeout_seconds=120),
        )
    else:
        logger.info("Initializing WorkspaceClient with SDK auto-auth")
        config = Config(http_timeout_seconds=120)
        _client = WorkspaceClient(config=config)

    return _client


# ---------------------------------------------------------------------------
# SQL helper
# ---------------------------------------------------------------------------
def execute_sql(
    query: str,
    *,
    cache_key: str | None = None,
    catalog: str | None = None,
    schema: str | None = None,
) -> list[dict[str, Any]]:
    """Execute a SQL statement via the Databricks SQL Statement Execution API.

    Parameters
    ----------
    query:
        The SQL query string.
    cache_key:
        If provided the result is cached under this key for ``CACHE_TTL``
        seconds.  Subsequent calls with the same key skip execution.
    catalog / schema:
        Override the default catalog / schema for this execution.

    Returns
    -------
    list[dict]
        Each dict maps column name -> value for one row.
    """
    if cache_key:
        cached = _cache_get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for %s", cache_key)
            return cached

    w = get_workspace_client()
    response = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=query,
        wait_timeout="30s",
        catalog=catalog or CATALOG_NAME,
        schema=schema or SCHEMA_GOLD,
    )

    if response.status.state != StatementState.SUCCEEDED:
        error_msg = getattr(response.status, "error", None)
        raise RuntimeError(
            f"SQL execution failed ({response.status.state}): {error_msg}"
        )

    columns = [col.name for col in response.manifest.schema.columns]
    rows: list[dict[str, Any]] = []
    if response.result and response.result.data_array:
        for row in response.result.data_array:
            rows.append(dict(zip(columns, row)))

    if cache_key:
        _cache_set(cache_key, rows)
    return rows


# ---------------------------------------------------------------------------
# Serving endpoint helper
# ---------------------------------------------------------------------------
def call_serving_endpoint(
    endpoint_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Invoke a Databricks Model Serving endpoint and return the response.

    Parameters
    ----------
    endpoint_name:
        Name of the serving endpoint (e.g. ``stryker-pricing-models``).
    payload:
        JSON-serialisable request body.

    Returns
    -------
    dict
        The parsed JSON response from the endpoint.
    """
    w = get_workspace_client()
    api_url = f"/serving-endpoints/{endpoint_name}/invocations"

    try:
        response = w.api_client.do("POST", api_url, body=payload)
    except Exception as exc:
        logger.error("Serving endpoint call failed: %s", exc)
        raise

    if isinstance(response, dict):
        return response
    # Some SDK versions return raw bytes
    import json

    return json.loads(response)
