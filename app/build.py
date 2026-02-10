#!/usr/bin/env python3
"""
Build script for the Stryker Pricing Intelligence Platform.

Performs three steps:
  1. Runs ``npm install`` in the frontend/ directory.
  2. Runs ``npm run build`` to produce a production bundle.
  3. Copies the build output from ``frontend/dist/`` into ``backend/static/``
     so that FastAPI can serve the SPA directly.

Usage:
    python build.py          # Standard build
    python build.py --clean  # Remove previous static output before building
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = SCRIPT_DIR / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
BACKEND_STATIC_DIR = SCRIPT_DIR / "static"

# Files that must exist in the build output for the app to work
REQUIRED_BUILD_FILES = [
    "index.html",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run(cmd: List[str], cwd: Path) -> None:
    """Run a subprocess command, streaming output to stdout/stderr."""
    print(f"\n>>> {' '.join(cmd)}  (cwd={cwd})")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"ERROR: command exited with code {result.returncode}")
        sys.exit(result.returncode)


def _validate_directory(path: Path, label: str) -> None:
    """Abort if the given directory does not exist."""
    if not path.is_dir():
        print(f"ERROR: {label} directory not found at {path}")
        sys.exit(1)


def _validate_build_output() -> None:
    """Ensure all required files are present in the backend static folder."""
    missing: List[str] = []
    for filename in REQUIRED_BUILD_FILES:
        if not (BACKEND_STATIC_DIR / filename).exists():
            missing.append(filename)

    if missing:
        print(f"ERROR: Missing required build files in {BACKEND_STATIC_DIR}:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    # Count total files copied
    total_files = sum(1 for _ in BACKEND_STATIC_DIR.rglob("*") if _.is_file())
    print(f"Validation passed: {total_files} files in {BACKEND_STATIC_DIR}")


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------
def step_npm_install() -> None:
    """Install frontend npm dependencies."""
    _validate_directory(FRONTEND_DIR, "Frontend")
    _run(["npm", "install"], cwd=FRONTEND_DIR)


def step_npm_build() -> None:
    """Run the Vite production build."""
    _run(["npm", "run", "build"], cwd=FRONTEND_DIR)

    if not FRONTEND_DIST_DIR.is_dir():
        print(f"ERROR: Build did not produce output at {FRONTEND_DIST_DIR}")
        sys.exit(1)


def step_copy_to_static(clean: bool = False) -> None:
    """Copy frontend build output into the backend static directory."""
    if clean and BACKEND_STATIC_DIR.exists():
        print(f"Cleaning {BACKEND_STATIC_DIR} ...")
        shutil.rmtree(BACKEND_STATIC_DIR)

    BACKEND_STATIC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Copying {FRONTEND_DIST_DIR} -> {BACKEND_STATIC_DIR} ...")
    # Copy contents (not the dist/ folder itself)
    for item in FRONTEND_DIST_DIR.iterdir():
        dest = BACKEND_STATIC_DIR / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full build pipeline."""
    parser = argparse.ArgumentParser(
        description="Build the Stryker Pricing Intelligence frontend"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous static build before copying",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Stryker Pricing Intelligence Platform - Frontend Build")
    print("=" * 60)

    step_npm_install()
    step_npm_build()
    step_copy_to_static(clean=args.clean)
    _validate_build_output()

    print("\n" + "=" * 60)
    print("Build completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
