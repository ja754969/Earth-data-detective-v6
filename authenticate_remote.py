"""Authenticate protected remote data services from a Windows terminal.

Run this file with ``cmd.exe`` in the ``OC`` environment before starting
Streamlit. Secrets are never written to ``data_config.json``.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
COMMON_FUNCTION_ROOT = Path(
    os.environ.get(
        "EARTH_DATA_COMMON_FUNCTION_ROOT",
        APP_DIR.parent / "common-files" / "function",
    )
)
if str(COMMON_FUNCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_FUNCTION_ROOT))

from shared_helpers.earth_data_loader import authenticate_earth_engine, save_earthdata_credentials


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate Google Earth Engine and/or NASA Earthdata in the terminal."
    )
    parser.add_argument("--gee", action="store_true", help="Run the interactive Google Earth Engine OAuth flow.")
    parser.add_argument(
        "--earthdata",
        action="store_true",
        help="Prompt for NASA Earthdata credentials and save them locally.",
    )
    parser.add_argument("--project", help="Google Cloud project ID used by Earth Engine.")
    parser.add_argument(
        "--credentials-path",
        type=Path,
        default=APP_DIR / "config" / "earthdata_credentials.json",
        help="Private JSON path for Earthdata credentials.",
    )
    arguments = parser.parse_args()
    if not arguments.gee and not arguments.earthdata:
        parser.error("Choose at least one of --gee or --earthdata.")
    if arguments.gee and not (arguments.project or os.environ.get("GEE_PROJECT")):
        parser.error("--gee requires --project or the GEE_PROJECT environment variable.")
    return arguments


def main() -> int:
    arguments = _parse_args()
    if arguments.gee:
        project = (arguments.project or os.environ["GEE_PROJECT"]).strip()
        print(f"Starting Earth Engine authentication for project {project!r}.")
        authenticate_earth_engine(project)
        print("Earth Engine authentication completed.")
    if arguments.earthdata:
        username = input("NASA Earthdata username: ")
        password = getpass.getpass("NASA Earthdata password: ")
        destination = save_earthdata_credentials(username, password, arguments.credentials_path)
        print(f"Earthdata credentials saved to {destination}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
