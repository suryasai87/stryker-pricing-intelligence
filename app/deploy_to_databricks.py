#!/usr/bin/env python3
"""
Deployment script for the Stryker Pricing Intelligence Databricks App.

Automates the full deployment lifecycle:
  1. Build the frontend (npm install + npm run build, copy to backend/static/)
  2. Upload the application source to the Databricks workspace
  3. Create or re-deploy the Databricks App
  4. Print service-principal details for permission setup

Usage:
    python deploy_to_databricks.py --app-name stryker-pricing-intel
    python deploy_to_databricks.py --app-name stryker-pricing-intel --hard-redeploy
    python deploy_to_databricks.py --app-name stryker-pricing-intel --profile STAGING
    python deploy_to_databricks.py --app-name stryker-pricing-intel --skip-build
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = SCRIPT_DIR / "build.py"
WORKSPACE_UPLOAD_BASE = "/Workspace/Apps"

# Files/dirs to exclude from workspace upload
UPLOAD_EXCLUDES = [
    "frontend/node_modules",
    "frontend/.vite",
    "frontend/dist",
    "__pycache__",
    ".git",
    ".env",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_cli(
    args: list[str],
    capture: bool = False,
    profile: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """
    Run a Databricks CLI command.

    Args:
        args: CLI arguments (without the 'databricks' prefix).
        capture: If True, capture stdout for parsing.
        profile: Optional Databricks CLI profile name.

    Returns:
        CompletedProcess instance.
    """
    cmd = ["databricks"] + args
    if profile:
        cmd += ["--profile", profile]

    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr if capture else ""
        print(f"ERROR: CLI command failed (rc={result.returncode})")
        if stderr:
            print(stderr)
        sys.exit(result.returncode)

    return result


def _cli_json(
    args: list[str],
    profile: Optional[str] = None,
) -> dict:
    """Run a CLI command and parse its JSON output."""
    result = _run_cli(args + ["--output", "json"], capture=True, profile=profile)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse JSON output:\n{result.stdout[:500]}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Build step
# ---------------------------------------------------------------------------
def build_frontend() -> None:
    """Run the frontend build script."""
    print("\n[1/4] Building frontend ...")
    result = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--clean"],
        cwd=str(SCRIPT_DIR),
    )
    if result.returncode != 0:
        print("ERROR: Frontend build failed.")
        sys.exit(result.returncode)
    print("Frontend build complete.")


# ---------------------------------------------------------------------------
# Upload step
# ---------------------------------------------------------------------------
def upload_source(app_name: str, profile: Optional[str] = None) -> str:
    """
    Upload the application source to the Databricks workspace.

    Returns:
        The workspace path where the source was uploaded.
    """
    workspace_path = f"{WORKSPACE_UPLOAD_BASE}/{app_name}"
    print(f"\n[2/4] Uploading source to {workspace_path} ...")

    # Create the workspace directory
    _run_cli(
        ["workspace", "mkdirs", workspace_path],
        profile=profile,
    )

    # Upload the app directory contents
    _run_cli(
        [
            "workspace",
            "import-dir",
            str(SCRIPT_DIR),
            workspace_path,
            "--overwrite",
        ],
        profile=profile,
    )

    print(f"Source uploaded to {workspace_path}")
    return workspace_path


# ---------------------------------------------------------------------------
# App management
# ---------------------------------------------------------------------------
def delete_app(app_name: str, profile: Optional[str] = None) -> None:
    """Delete an existing Databricks App (for hard redeploy)."""
    print(f"\n  Deleting existing app '{app_name}' ...")
    try:
        _run_cli(["apps", "delete", app_name], profile=profile)
        print(f"  App '{app_name}' deleted. Waiting 10s for cleanup ...")
        time.sleep(10)
    except SystemExit:
        print(f"  App '{app_name}' does not exist or could not be deleted. Continuing ...")


def create_app(app_name: str, profile: Optional[str] = None) -> dict:
    """Create the Databricks App if it does not already exist."""
    print(f"\n[3/4] Creating app '{app_name}' ...")
    try:
        info = _cli_json(["apps", "get", app_name], profile=profile)
        print(f"  App '{app_name}' already exists.")
        return info
    except SystemExit:
        pass

    info = _cli_json(["apps", "create", app_name], profile=profile)
    print(f"  App '{app_name}' created.")
    return info


def deploy_app(
    app_name: str,
    workspace_path: str,
    profile: Optional[str] = None,
) -> dict:
    """Deploy the app from the uploaded workspace source."""
    print(f"\n  Deploying app '{app_name}' from {workspace_path} ...")
    info = _cli_json(
        [
            "apps",
            "deploy",
            app_name,
            "--source-code-path",
            workspace_path,
        ],
        profile=profile,
    )
    print(f"  Deployment initiated.")
    return info


# ---------------------------------------------------------------------------
# Permissions info
# ---------------------------------------------------------------------------
def print_permissions_info(app_name: str, profile: Optional[str] = None) -> None:
    """
    Retrieve and print the service-principal information so the user can
    set up the required permissions.
    """
    print("\n[4/4] Service Principal Permissions Setup")
    print("=" * 60)

    try:
        info = _cli_json(["apps", "get", app_name], profile=profile)
    except SystemExit:
        print("  WARNING: Could not retrieve app info. Set permissions manually.")
        return

    sp_client_id = info.get("service_principal_client_id", "N/A")
    sp_name = info.get("service_principal_name", f"apps/{app_name}")
    app_url = info.get("url", "N/A")

    print(f"  App Name:                {app_name}")
    print(f"  App URL:                 {app_url}")
    print(f"  Service Principal ID:    {sp_client_id}")
    print(f"  Service Principal Name:  {sp_name}")
    print()
    print("Grant the following permissions to the service principal:")
    print()
    print("1. SQL Warehouse access:")
    print(f'   databricks permissions update sql/warehouses <WAREHOUSE_ID> --json \'{{')
    print(f'     "access_control_list": [{{')
    print(f'       "service_principal_name": "{sp_client_id}",')
    print(f'       "permission_level": "CAN_USE"')
    print(f"     }}]")
    print(f"   }}'")
    print()
    print("2. Unity Catalog grants (run in SQL editor):")
    print(f"   GRANT USE CATALOG ON CATALOG hls_amer_catalog TO `{sp_client_id}`;")
    print(f"   GRANT USE SCHEMA ON SCHEMA hls_amer_catalog.gold TO `{sp_client_id}`;")
    print(f"   GRANT USE SCHEMA ON SCHEMA hls_amer_catalog.silver TO `{sp_client_id}`;")
    print(f"   GRANT SELECT ON SCHEMA hls_amer_catalog.gold TO `{sp_client_id}`;")
    print(f"   GRANT SELECT ON SCHEMA hls_amer_catalog.silver TO `{sp_client_id}`;")
    print()
    print("3. Model serving endpoint:")
    print(f'   databricks permissions update serving-endpoints/stryker-pricing-models --json \'{{')
    print(f'     "access_control_list": [{{')
    print(f'       "service_principal_name": "{sp_name}",')
    print(f'       "permission_level": "CAN_QUERY"')
    print(f"     }}]")
    print(f"   }}'")
    print()
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Parse arguments and execute the deployment pipeline."""
    parser = argparse.ArgumentParser(
        description="Deploy the Stryker Pricing Intelligence App to Databricks"
    )
    parser.add_argument(
        "--app-name",
        required=True,
        help="Name for the Databricks App (e.g. stryker-pricing-intel)",
    )
    parser.add_argument(
        "--hard-redeploy",
        action="store_true",
        help="Delete the existing app before re-creating it",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Databricks CLI profile to use (default: DEFAULT)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the frontend build step (use existing static files)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Stryker Pricing Intelligence Platform - Deployment")
    print("=" * 60)

    # Step 1: Build
    if not args.skip_build:
        build_frontend()
    else:
        print("\n[1/4] Skipping frontend build (--skip-build)")

    # Step 2: Upload
    workspace_path = upload_source(args.app_name, profile=args.profile)

    # Step 3: Create / Deploy
    if args.hard_redeploy:
        delete_app(args.app_name, profile=args.profile)

    create_app(args.app_name, profile=args.profile)
    deploy_app(args.app_name, workspace_path, profile=args.profile)

    # Step 4: Permissions info
    print_permissions_info(args.app_name, profile=args.profile)

    print("\nDeployment complete.")


if __name__ == "__main__":
    main()
