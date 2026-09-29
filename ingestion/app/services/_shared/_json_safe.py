"""Shared helpers for JSON-safe numeric conversion.

PostgreSQL's JSONB type (and strict-mode ``json.dumps``) reject
``NaN`` and ``Infinity`` literals. Services that persist computed
statistics into JSONB columns must coerce non-finite floats to ``None``
before handing the dict to SQLAlchemy.

This module provides a single narrow utility. ``app.repositories.yfinance_repository``
keeps its own ``_safe_val``/``_safe_float`` pair because it additionally
unwraps pandas/numpy scalar types, which is out of scope here.
"""

from __future__ import annotations

import math
from typing import Any


def safe_float(v: Any, ndigits: int | None = None) -> float | None:
    """Return ``v`` as a finite float, or ``None`` for NaN/Inf/None/unparsable.

    Args:
        v: Any value convertible to ``float``.
        ndigits: Optional rounding precision (passed to ``round``).

    Returns:
        A finite ``float`` (rounded if *ndigits* is given), or
        ``None`` when *v* is ``None``, non-finite, or cannot be cast.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, ndigits) if ndigits is not None else f
