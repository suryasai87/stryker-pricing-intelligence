#!/usr/bin/env python3
"""
Automated deployment script for the Stryker Pricing Intelligence Platform
using Databricks Asset Bundles.

Performs the full deployment lifecycle:
  1. Build the React frontend (npm install + npm run build)
  2. Copy build artifacts to backend static directory (via build.py)
  3. Deploy via ``databricks bundle deploy -t <target>``
  4. Print deployment URL and status

Usage:
    python deploy.py dev          # Deploy to development (default)
    python deploy.py staging      # Deploy to staging
    python deploy.py prod         # Deploy to production
    python deploy.py dev --skip-build   # Skip frontend build
    python deploy.py dev --validate     # Validate bundle only (no deploy)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "app"
BUILD_SCRIPT = APP_DIR / "build.py"
FRONTEND_DIR = APP_DIR / "frontend"

VALID_TARGETS = ("dev", "staging", "prod")

# ---------------------------------------------------------------------------
# ANSI colour helpers (gracefully degrades in non-TTY environments)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Wrap *text* in an ANSI escape sequence when outputting to a terminal."""
    if _USE_COLOR:
        return f"\033[{code}m{text}\033[0m"
    return text


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def red(t: str) -> str:
    return _c("31", t)


def cyan(t: str) -> str:
    return _c("36", t)


def bold(t: str) -> str:
    return _c("1", t)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def banner(msg: str) -> None:
    """Print a section banner."""
    width = 64
    print()
    print(cyan("=" * width))
    print(cyan(f"  {msg}"))
    print(cyan("=" * width))


def step(num: int, total: int, msg: str) -> None:
    """Print a numbered step header."""
    print(f"\n{bold(f'[{num}/{total}]')} {msg}")


def run(cmd: list[str], cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command, streaming output unless *capture* is set."""
    print(f"  {yellow('>')} {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=capture,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr if capture else ""
        print(red(f"  ERROR: Command exited with code {result.returncode}"))
        if stderr:
            print(red(f"  {stderr.strip()}"))
        sys.exit(result.returncode)
    return result


# ---------------------------------------------------------------------------
# Step 1: Build frontend
# ---------------------------------------------------------------------------
def build_frontend() -> None:
    """Run npm install, npm run build, then copy output to static/."""
    step(1, 4, "Building frontend ...")

    if not FRONTEND_DIR.is_dir():
        print(red(f"  Frontend directory not found: {FRONTEND_DIR}"))
        sys.exit(1)

    run(["npm", "install"], cwd=FRONTEND_DIR)
    run(["npm", "run", "build"], cwd=FRONTEND_DIR)

    # Use the existing build.py to copy dist -> static
    if BUILD_SCRIPT.is_file():
        run([sys.executable, str(BUILD_SCRIPT), "--clean"], cwd=APP_DIR)
    else:
        print(yellow("  WARNING: build.py not found; skipping static copy"))

    print(green("  Frontend build complete."))


# ---------------------------------------------------------------------------
# Step 2: Validate bundle
# ---------------------------------------------------------------------------
def validate_bundle(target: str) -> None:
    """Run ``databricks bundle validate`` and report any issues."""
    step(2, 4, f"Validating bundle for target '{target}' ...")
    run(
        ["databricks", "bundle", "validate", "-t", target],
        cwd=PROJECT_ROOT,
    )
    print(green("  Bundle validation passed."))


# ---------------------------------------------------------------------------
# Step 3: Deploy bundle
# ---------------------------------------------------------------------------
def deploy_bundle(target: str) -> None:
    """Run ``databricks bundle deploy`` for the given target."""
    step(3, 4, f"Deploying bundle to target '{target}' ...")
    start = time.time()
    run(
        ["databricks", "bundle", "deploy", "-t", target],
        cwd=PROJECT_ROOT,
    )
    elapsed = time.time() - start
    print(green(f"  Bundle deployed in {elapsed:.1f}s."))


# ---------------------------------------------------------------------------
# Step 4: Print summary
# ---------------------------------------------------------------------------
def print_summary(target: str) -> None:
    """Print deployment URLs and next steps."""
    step(4, 4, "Deployment summary")

    # Attempt to fetch app info from the bundle output
    app_name = f"stryker-pricing-intel-{target}"
    print()
    print(f"  {bold('Target:')}     {target}")
    print(f"  {bold('App name:')}   {app_name}")
    print(f"  {bold('Bundle:')}     stryker-pricing-intel")
    print()

    # Try to get the app URL via Databricks CLI
    try:
        result = subprocess.run(
            ["databricks", "apps", "get", app_name, "--output", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            info = json.loads(result.stdout)
            app_url = info.get("url", "N/A")
            sp_id = info.get("service_principal_client_id", "N/A")
            status = info.get("status", {}).get("state", "UNKNOWN")

            print(f"  {bold('App URL:')}    {green(app_url)}")
            print(f"  {bold('Status:')}     {status}")
            print(f"  {bold('SP Client:')} {sp_id}")
            print()
            print(f"  {bold('Permissions SQL:')}")
            print(f"    See resources/permissions.sql (replace ${{SP_CLIENT_ID}} with {sp_id})")
        else:
            print(yellow(f"  Could not fetch app info for '{app_name}'."))
            print(yellow("  The app may still be deploying. Check with:"))
            print(f"    databricks apps get {app_name}")
    except Exception:
        print(yellow("  Could not query app status. Check manually:"))
        print(f"    databricks apps get {app_name}")

    print()
    print(green("  Deployment complete."))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Parse arguments and run the deployment pipeline."""
    parser = argparse.ArgumentParser(
        description="Deploy the Stryker Pricing Intelligence Platform via Databricks Asset Bundles",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="dev",
        choices=VALID_TARGETS,
        help="Deployment target (default: dev)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the frontend build step (use existing static files)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate the bundle; do not deploy",
    )
    args = parser.parse_args()

    banner(f"Stryker Pricing Intelligence - Deploy to {args.target.upper()}")

    # Step 1: Build
    if args.skip_build:
        step(1, 4, "Skipping frontend build (--skip-build)")
    else:
        build_frontend()

    # Step 2: Validate
    validate_bundle(args.target)

    if args.validate:
        print()
        print(green("Validation-only mode: skipping deploy."))
        return

    # Step 3: Deploy
    deploy_bundle(args.target)

    # Step 4: Summary
    print_summary(args.target)


if __name__ == "__main__":
    main()
