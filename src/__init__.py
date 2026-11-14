"""Reusable climate visualisation helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path


_COMMON_FUNCTION_ROOT_ENV = os.environ.get("EARTH_DATA_COMMON_FUNCTION_ROOT", "").strip()
_COMMON_FUNCTION_ROOT_CANDIDATES = (
    Path(_COMMON_FUNCTION_ROOT_ENV) if _COMMON_FUNCTION_ROOT_ENV else None,
    Path(__file__).resolve().parents[2] / "common-files" / "function",
)
_COMMON_FUNCTION_ROOT = next(
    (path for path in _COMMON_FUNCTION_ROOT_CANDIDATES if path is not None and path.is_dir()),
    None,
)
if _COMMON_FUNCTION_ROOT is not None and str(_COMMON_FUNCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_FUNCTION_ROOT))
