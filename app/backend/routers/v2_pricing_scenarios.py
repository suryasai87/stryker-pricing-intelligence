"""
Pricing scenarios router.

User-specific pricing scenarios with OBO (On-Behalf-Of) authentication.
Regular users see only their own scenarios; admin users can see all.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.services.obo_auth import get_user_identity
from backend.utils.config import CATALOG_NAME, SCHEMA_GOLD
from backend.utils.databricks_client import execute_sql

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/pricing-scenarios", tags=["scenarios"])

_TABLE = f"{CATALOG_NAME}.{SCHEMA_GOLD}.pricing_scenarios"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class CreateScenarioRequest(BaseModel):
    """Payload for creating a new pricing scenario."""

    name: str = Field(..., description="Scenario name")
    description: str = Field("", description="Scenario description")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Scenario parameters (JSON)"
    )
    status: str = Field("draft", description="Initial status")


# ---------------------------------------------------------------------------
# GET /user-info
# ---------------------------------------------------------------------------
@router.get(
    "/user-info",
    summary="Return the current user's identity from OBO headers",
)
async def get_user_info(request: Request) -> dict[str, Any]:
    """Return the identity of the currently authenticated user as extracted
    from the OBO forwarded headers.
    """
    identity = get_user_identity(request)
    return identity


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
@router.get(
    "/",
    summary="List pricing scenarios (user-scoped or admin-all)",
)
async def list_scenarios(
    request: Request,
    status: str | None = Query(None, description="Filter by scenario status"),
    search: str | None = Query(None, description="Search in scenario name"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=1, le=500, description="Rows per page"),
) -> dict[str, Any]:
    """Return pricing scenarios.  Regular users see only their own scenarios;
    admin users see all scenarios.
    """
    try:
        identity = get_user_identity(request)
        where_clauses: list[str] = []

        # Non-admin users can only see their own scenarios
        if not identity["is_admin"]:
            where_clauses.append(f"created_by = '{identity['user_email']}'")

        if status:
            where_clauses.append(f"status = '{status}'")
        if search:
            where_clauses.append(f"name LIKE '%{search}%'")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        offset = (page - 1) * page_size

        # Count
        count_query = f"SELECT COUNT(*) AS total FROM {_TABLE} {where_sql}"
        count_rows = execute_sql(count_query)
        total = int(count_rows[0].get("total", 0)) if count_rows else 0

        # Data
        data_query = f"""
            SELECT *
            FROM {_TABLE}
            {where_sql}
            ORDER BY created_at DESC
            LIMIT {page_size}
            OFFSET {offset}
        """
        rows = execute_sql(data_query)

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if page_size else 0,
            "user": identity,
            "data": rows,
        }
    except Exception as exc:
        logger.exception("Failed to list pricing scenarios")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /
# ---------------------------------------------------------------------------
@router.post(
    "/",
    summary="Create a new pricing scenario",
)
async def create_scenario(
    request: Request,
    body: CreateScenarioRequest,
) -> dict[str, Any]:
    """Create a new pricing scenario owned by the authenticated user."""
    try:
        import json

        identity = get_user_identity(request)
        scenario_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()
        params_json = json.dumps(body.parameters).replace("'", "''")

        query = f"""
            INSERT INTO {_TABLE}
                (scenario_id, name, description, parameters, status,
                 created_by, created_at, updated_at)
            VALUES
                ('{scenario_id}', '{body.name}', '{body.description}',
                 '{params_json}', '{body.status}',
                 '{identity["user_email"]}', '{now}', '{now}')
        """
        execute_sql(query)

        return {
            "status": "created",
            "scenario_id": scenario_id,
            "name": body.name,
            "created_by": identity["user_email"],
            "created_at": now,
        }
    except Exception as exc:
        logger.exception("Failed to create pricing scenario")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /{scenario_id}
# ---------------------------------------------------------------------------
@router.get(
    "/{scenario_id}",
    summary="Retrieve a single pricing scenario by ID",
)
async def get_scenario(
    request: Request,
    scenario_id: str,
) -> dict[str, Any]:
    """Return a single pricing scenario.  Non-admin users can only access
    their own scenarios.
    """
    try:
        identity = get_user_identity(request)

        where_clauses: list[str] = [f"scenario_id = '{scenario_id}'"]
        if not identity["is_admin"]:
            where_clauses.append(f"created_by = '{identity['user_email']}'")

        where_sql = f"WHERE {' AND '.join(where_clauses)}"

        query = f"""
            SELECT *
            FROM {_TABLE}
            {where_sql}
            LIMIT 1
        """
        rows = execute_sql(query)

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"Scenario '{scenario_id}' not found",
            )

        return {
            "scenario": rows[0],
            "user": identity,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch scenario %s", scenario_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
