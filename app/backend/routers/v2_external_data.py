"""
External data upload and retrieval router.

Allows users to upload Excel/CSV files that are ingested and stored into a
Unity Catalog Volume.  Provides endpoints to list uploaded sources and
retrieve current external data.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.services.file_ingestion import ingest_file
from backend.utils.config import CATALOG_NAME, SCHEMA_GOLD_V2
from backend.utils.databricks_client import execute_sql, get_workspace_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/external-data", tags=["external-data"])

_TABLE = f"{CATALOG_NAME}.{SCHEMA_GOLD_V2}.external_data"
_VOLUME_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_GOLD_V2}/external_uploads"


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------
@router.post(
    "/upload",
    summary="Upload an external data file (Excel or CSV)",
)
async def upload_external_data(
    file: UploadFile = File(..., description="Excel (.xlsx/.xls) or CSV file"),
    category: str = Query("general", description="Data category label"),
) -> dict[str, Any]:
    """Accept a file upload, parse its contents, and write the raw file to
    the Unity Catalog Volume for downstream processing.
    """
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in (".xlsx", ".xls", ".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted: .xlsx, .xls, .csv",
        )

    try:
        # Parse the file contents
        result = await ingest_file(file, category)

        # Write raw file to the Volume
        await file.seek(0)
        raw_bytes = await file.read()
        volume_file_path = f"{_VOLUME_PATH}/{filename}"

        try:
            w = get_workspace_client()
            w.files.upload(volume_file_path, raw_bytes, overwrite=True)
            logger.info("Uploaded file to Volume: %s", volume_file_path)
        except Exception as upload_exc:
            logger.warning(
                "Could not upload to Volume %s: %s", volume_file_path, upload_exc
            )
            # Non-fatal -- the parsed data is still returned

        return {
            "status": "success",
            "filename": result["filename"],
            "category": category,
            "row_count": result["row_count"],
            "columns": result["columns"],
            "volume_path": volume_file_path,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "preview": result["rows"][:10],  # Return first 10 rows as preview
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to process uploaded file %s", filename)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
@router.get(
    "/",
    summary="Retrieve current external data with optional category filter",
)
async def get_external_data(
    category: str | None = Query(None, description="Filter by data category"),
) -> dict[str, Any]:
    """Return external data records, optionally filtered by category."""
    try:
        where_clauses: list[str] = []
        if category:
            where_clauses.append(f"category = '{category}'")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query = f"""
            SELECT *
            FROM {_TABLE}
            {where_sql}
            ORDER BY uploaded_at DESC
        """
        cache_key = f"external_data:list:{category}"
        rows = execute_sql(query, cache_key=cache_key)

        return {
            "count": len(rows),
            "category": category,
            "data": rows,
        }
    except Exception as exc:
        logger.exception("Failed to fetch external data")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /sources
# ---------------------------------------------------------------------------
@router.get(
    "/sources",
    summary="List uploaded external data sources with metadata",
)
async def get_external_sources() -> dict[str, Any]:
    """Return a list of uploaded data sources with file metadata."""
    try:
        query = f"""
            SELECT
                source_name,
                category,
                file_path,
                uploaded_at,
                row_count,
                column_count
            FROM {_TABLE}
            GROUP BY source_name, category, file_path, uploaded_at,
                     row_count, column_count
            ORDER BY uploaded_at DESC
        """
        rows = execute_sql(query, cache_key="external_data:sources")

        return {
            "count": len(rows),
            "sources": rows,
        }
    except Exception as exc:
        logger.exception("Failed to fetch external data sources")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
