"""
File ingestion helper for external data uploads.

Accepts Excel (.xlsx, .xls) and CSV files, parses their contents into a list
of dictionaries, and returns the parsed rows along with inferred schema
metadata.  Files are temporarily saved to disk for processing.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import tempfile
from typing import Any

from fastapi import UploadFile

logger = logging.getLogger(__name__)


async def ingest_file(
    file: UploadFile,
    category: str,
) -> dict[str, Any]:
    """Read an uploaded file and return parsed rows with schema metadata.

    Parameters
    ----------
    file:
        The uploaded file (must be .xlsx, .xls, or .csv).
    category:
        A user-supplied label categorising the data (e.g. ``"competitor"``,
        ``"market_research"``).

    Returns
    -------
    dict
        Keys: ``filename``, ``category``, ``row_count``, ``columns``, ``rows``.
        ``rows`` is a ``list[dict]`` where each dict maps column name to value.

    Raises
    ------
    ValueError
        If the file extension is unsupported or the file cannot be parsed.
    """
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in (".xlsx", ".xls", ".csv"):
        raise ValueError(
            f"Unsupported file type '{ext}'. Accepted: .xlsx, .xls, .csv"
        )

    contents = await file.read()

    if ext in (".xlsx", ".xls"):
        rows, columns = _parse_excel(contents, filename)
    else:
        rows, columns = _parse_csv(contents, filename)

    logger.info(
        "Ingested file %s (%s): %d rows, %d columns",
        filename,
        category,
        len(rows),
        len(columns),
    )

    return {
        "filename": filename,
        "category": category,
        "row_count": len(rows),
        "columns": columns,
        "rows": rows,
    }


def _parse_excel(
    contents: bytes,
    filename: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Parse Excel bytes into rows and column metadata.

    Uses pandas for reading; falls back with a clear error if pandas or
    openpyxl are not installed.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ValueError(
            "pandas is required for Excel file processing. "
            "Install it with: pip install pandas openpyxl"
        ) from exc

    # Write to a temporary file so pandas can read it
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        df = pd.read_excel(tmp_path)
        columns = [
            {"name": str(col), "type": str(df[col].dtype)}
            for col in df.columns
        ]
        rows = df.where(df.notnull(), None).to_dict(orient="records")
        return rows, columns
    finally:
        os.unlink(tmp_path)


def _parse_csv(
    contents: bytes,
    filename: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Parse CSV bytes into rows and column metadata."""
    text = contents.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, Any]] = []

    for row in reader:
        parsed_row: dict[str, Any] = {}
        for key, value in row.items():
            parsed_row[key or "unnamed"] = _infer_value(value)
        rows.append(parsed_row)

    if not rows:
        return rows, []

    # Infer column types from first row
    first_row = rows[0]
    columns = [
        {"name": str(col), "type": _infer_type(first_row.get(col))}
        for col in first_row
    ]
    return rows, columns


def _infer_value(value: str | None) -> Any:
    """Attempt to cast a CSV string value to its most likely Python type."""
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _infer_type(value: Any) -> str:
    """Return a human-readable type string for a parsed value."""
    if value is None:
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    return "string"
