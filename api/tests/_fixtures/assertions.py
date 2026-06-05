"""Shared, pure assertion helpers for API tests (issue #807).

No DB, no I/O — imported directly by tests (not used as fixtures).

``assert_json_safe`` is meaningful on a **pre-serialisation** dict (a
service-layer return value before FastAPI/Pydantic encodes it).  Run against
an already-parsed ``response.json()`` it is a near no-op, since the HTTP
round-trip can only yield JSON-native types — assert it on the dict the route
hands to the serialiser to catch numpy/Decimal/DataFrame leaks at the seam.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

_JSON_SCALARS = (str, bool, int, float, type(None))
_FORBIDDEN = (Decimal, date, pd.DataFrame, pd.Series, bytes, set, frozenset)
_CAMEL_CASE = re.compile(r"^[a-z][a-zA-Z0-9]*$")


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


def assert_response_matches_schema(
    response_json: Any, schema_class: type[BaseModel]
) -> BaseModel:
    """Validate ``response_json`` against ``schema_class``; return the model."""
    try:
        return schema_class.model_validate(response_json)
    except ValidationError as exc:
        raise AssertionError(
            f"response does not match {schema_class.__name__}: {exc}"
        ) from exc


def _walk_keys(value: Any, path: str, check: Callable[[str, str], None]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            check(key, f"{path}.{key}")
            _walk_keys(child, f"{path}.{key}", check)
        return
    if isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _walk_keys(child, f"{path}[{index}]", check)


def assert_camel_case_keys(payload: Any) -> None:
    """Recursively assert every dict key is camelCase."""

    def check(key: str, path: str) -> None:
        if not _CAMEL_CASE.match(key):
            raise AssertionError(f"{path}: key {key!r} is not camelCase")

    _walk_keys(payload, "root", check)


def assert_no_snake_case_keys(
    payload: Any, *, allow: frozenset[str] = frozenset()
) -> None:
    """Recursively assert no dict key is snake_case (contains ``_``)."""

    def check(key: str, path: str) -> None:
        if "_" in key and key not in allow:
            raise AssertionError(f"{path}: snake_case key {key!r} not allowed")

    _walk_keys(payload, "root", check)
