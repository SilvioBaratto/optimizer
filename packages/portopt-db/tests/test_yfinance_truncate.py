"""``truncate`` cuts text to the width the model declares for the column.

The yfinance repositories used to pass each width as a literal; ``truncate`` reads it
from ``Model.__table__.c[column].type.length`` instead, so a migration that widens a
column cannot leave a stale literal cutting values shorter than the column allows.
"""

from __future__ import annotations

import math

import pytest
from sqlalchemy import String

from portopt_db.models.market_data.yfinance_data import (
    InsiderTransaction,
    SecFiling,
    TickerNews,
    TickerProfile,
)
from portopt_db.repositories.market_data._mapping import _safe_str, truncate


def _width(model: type, column: str) -> int:
    column_type = model.__table__.c[column].type  # type: ignore[attr-defined]
    assert isinstance(column_type, String)
    assert column_type.length is not None
    return column_type.length


@pytest.mark.parametrize(
    ("model", "column"),
    [
        (TickerProfile, "symbol"),
        (TickerNews, "news_uuid"),
        (InsiderTransaction, "ownership"),
    ],
)
def test_truncate_uses_model_width(model: type, column: str) -> None:
    """A value longer than the column comes back cut to the column's length."""
    width = _width(model, column)

    assert truncate(model, column, "x" * (width + 25)) == "x" * width


def test_truncate_keeps_a_value_that_fits() -> None:
    """A value within the width is returned as given."""
    assert truncate(TickerProfile, "symbol", "AAPL") == "AAPL"


def test_truncate_passes_none_through() -> None:
    """``None`` stays ``None``, as ``_safe_str`` returns it for a missing value."""
    assert truncate(TickerProfile, "symbol", None) is None


def test_truncate_leaves_a_column_without_length_whole() -> None:
    """``SecFiling.url`` is ``Text``, so no width applies."""
    long_url = "https://example.com/" + "a" * 5000

    assert truncate(SecFiling, "url", long_url) == long_url


def test_truncate_rejects_an_unknown_column() -> None:
    """A misspelled column name raises instead of skipping the cut."""
    with pytest.raises(KeyError):
        truncate(TickerProfile, "no_such_column", "x")


@pytest.mark.parametrize(
    "raw", ["A" * 80, "nul\x00inside" * 10, 12345678901234567890, math.nan, None, ""]
)
def test_truncate_equals_safe_str_with_the_model_width(raw: object) -> None:
    """``truncate(M, c, v)`` matches the ``_safe_str(v, width)`` it replaces."""
    width = _width(TickerProfile, "symbol")

    assert truncate(TickerProfile, "symbol", raw) == _safe_str(raw, width)
