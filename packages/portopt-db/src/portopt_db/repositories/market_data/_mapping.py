"""Turn yfinance values into column values, and shape rows for ``ON CONFLICT`` upserts.

``_refreshable_columns`` picks the columns an upsert may overwrite, and
``_last_wins`` keeps the last of the rows in a batch that share a conflict key.
"""

import math
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import String

from portopt_db.base import Base


def _safe_val(v: Any) -> Any:
    """Convert pandas/numpy types to Python natives, NaN/NaT/NA to None."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    item_fn = getattr(v, "item", None)  # numpy scalar
    if callable(item_fn):
        native = item_fn()
        # A numpy float32 NaN is not a float subclass, so it only shows as NaN
        # once unwrapped. Unwrapped once: an object whose item() returns another
        # such object would otherwise never stop.
        if native is None or (
            isinstance(native, float) and (math.isnan(native) or math.isinf(native))
        ):
            return None
        return native
    return v


def _safe_int(v: Any) -> int | None:
    v = _safe_val(v)
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _safe_float(v: Any) -> float | None:
    v = _safe_val(v)
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_percent_as_fraction(v: Any) -> float | None:
    """Convert a figure Yahoo quotes in percent (``0.44`` = 0.44%) to a fraction."""
    f = _safe_float(v)
    return None if f is None else f / 100.0


def _strip_nul(text: str) -> str:
    """Remove NUL characters, which PostgreSQL text cannot hold.

    psycopg2 rejects the whole multi-row INSERT when any value carries one, so a
    single stray NUL in scraped text would otherwise drop every row in the batch.
    """
    return text.replace("\x00", "")


def _safe_str(v: Any, max_len: int | None = None) -> str | None:
    v = _safe_val(v)
    if v is None:
        return None
    s = _strip_nul(str(v))
    if max_len:
        s = s[:max_len]
    return s


def _refreshable_columns(row: dict[str, Any], sticky: frozenset[str]) -> list[str]:
    """Columns an ON CONFLICT update may overwrite for a single-row snapshot.

    Every column except the conflict key, minus the ``sticky`` ones this row
    leaves empty. ``updated_at`` stays in so the row is still stamped.
    """
    columns = [
        name
        for name, value in row.items()
        if name != "instrument_id" and not (name in sticky and value is None)
    ]
    return [*columns, "updated_at"]


def _last_wins(rows: list[dict[str, Any]], *key_fields: str) -> list[dict[str, Any]]:
    """Collapse rows that share a conflict key, keeping the last occurrence.

    A single ``INSERT ... ON CONFLICT DO UPDATE`` cannot affect the same key
    twice ("cannot affect row a second time"), and the error aborts the whole
    statement, so a duplicate in one feed would drop that category for the
    ticker on every run. ``instrument_id`` is constant within a batch and is
    left out of ``key_fields``.
    """
    return list({tuple(r[k] for k in key_fields): r for r in rows}.values())


def _safe_date(v: Any) -> date | None:
    """Convert various date-like values to date."""
    v = _safe_val(v)
    if v is None:
        return None
    # pd.NaT / pd.NA are missing values but NaT subclasses datetime, so the
    # branch below would call NaT.date() -> NaT and leak a bogus date into a
    # NOT NULL column. Treat any pandas NA scalar as missing.
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v).date()
        except (ValueError, TypeError):
            pass
        # yfinance valuation-measure columns are labelled "M/D/YYYY" without
        # zero padding.
        try:
            return datetime.strptime(v, "%m/%d/%Y").date()
        except (ValueError, TypeError):
            return None
    if isinstance(v, int | float):
        # Yahoo epoch dates sit at 00:00 UTC, so reading them in the host's
        # timezone would shift the calendar day on any host west of UTC.
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc).date()
        except (ValueError, TypeError, OSError, OverflowError):
            return None
    return None


def truncate(model: type[Base], column: str, value: Any) -> str | None:
    """Convert ``value`` with ``_safe_str`` and cut it to the column's length.

    Equivalent to ``_safe_str(value, width)``, where ``width`` is the length of
    the type ``model`` declares for ``column``, or ``None`` when that type is not
    a ``String`` or has no length; a change to the model's column width therefore
    changes the cut with it.

    Args:
        model: Mapped class whose table holds the column.
        column: Name of the column the value is stored in.
        value: Raw value to store.

    Returns:
        ``None`` where ``_safe_str`` returns it, otherwise the text cut to the
        column's length; uncut when the column declares no length.

    Raises:
        KeyError: ``model``'s table has no column named ``column``.
    """
    column_type = model.__table__.c[column].type
    width = column_type.length if isinstance(column_type, String) else None
    return _safe_str(value, width)
