"""Shared, pure assertion helpers.

No DB, no I/O — imported directly by tests (not used as fixtures).

``assert_json_safe`` guards the seam where a service hands a dict to a JSONB
column or a job ``result`` payload: numpy scalars, Decimals, DataFrames, and
NaN/inf all survive in-process but blow up (or silently corrupt) on encode.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

_JSON_SCALARS = (str, bool, int, float, type(None))
_FORBIDDEN = (Decimal, date, pd.DataFrame, pd.Series, bytes, set, frozenset)


def _assert_scalar_json_safe(value: Any, path: str) -> None:
    # numpy scalars/arrays first: np.float64 is a subclass of float, so the
    # _JSON_SCALARS check below would wrongly accept it.
    if isinstance(value, np.generic | np.ndarray):
        raise AssertionError(f"{path}: {type(value).__name__} (numpy) is not JSON-safe")
    if isinstance(value, _FORBIDDEN):
        raise AssertionError(f"{path}: {type(value).__name__} is not JSON-safe")
    if not isinstance(value, _JSON_SCALARS):
        raise AssertionError(f"{path}: {type(value).__name__} is not JSON-safe")


def _walk_values(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _walk_values(child, f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _walk_values(child, f"{path}[{index}]")
        return
    _assert_scalar_json_safe(value, path)


def assert_json_safe(payload: Any) -> None:
    """Recursively reject non-JSON types, then catch NaN/inf via ``json.dumps``."""
    _walk_values(payload, "root")
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"payload is not JSON-safe: {exc}") from exc
