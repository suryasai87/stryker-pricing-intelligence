"""
On-Behalf-Of (OBO) authentication helper.

Extracts user identity from HTTP headers injected by the Databricks Apps
reverse proxy.  In production, ``x-forwarded-email`` and ``x-forwarded-user``
are set automatically by the SSO layer.  For local development these headers
will be absent and safe defaults are returned.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request


# Users with full admin access (can view all scenarios, manage all data)
_ADMIN_USERS: list[str] = [
    "admin@stryker.com",
]


def get_user_identity(request: Request) -> dict[str, Any]:
    """Extract user identity from the forwarded request headers.

    Parameters
    ----------
    request:
        The incoming FastAPI ``Request`` object.

    Returns
    -------
    dict
        Keys: ``user_id``, ``user_email``, ``is_admin``.
    """
    headers = dict(request.headers)
    user_email: str = headers.get("x-forwarded-email", "anonymous@stryker.com")
    user_id: str = headers.get("x-forwarded-user", "anonymous")
    is_admin: bool = user_email in _ADMIN_USERS

    return {
        "user_id": user_id,
        "user_email": user_email,
        "is_admin": is_admin,
    }
