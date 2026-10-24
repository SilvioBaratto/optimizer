"""Unit tests for ``MetadataClient.fetch_valuation_measures``.

Covers the yfinance ``Ticker.valuation`` 9-metric panel, empty-frame handling,
the dating of its ``Current`` / ``M/D/YYYY`` string columns so the date-keyed
statement repository can store them, and ``MetadataClientProtocol`` conformance.
"""

from __future__ import annotations

import uuid
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest
from portopt_db.repositories.market_data.yfinance_repository import (
    YFinanceRepository,
)

from app.services.market_data.yfinance.protocols.interfaces import (
    MetadataClientProtocol,
)
from app.services.market_data.yfinance.ticker import metadata as metadata_module
from app.services.market_data.yfinance.ticker.metadata import MetadataClient


def _build_client(valuation: pd.DataFrame) -> MetadataClient:
    """Construct a ``MetadataClient`` whose cached ticker exposes *valuation*."""
    cache = MagicMock(name="cache")
    cache.get.return_value = MagicMock(valuation=valuation)
    rate_limiter = MagicMock(name="rate_limiter")
    circuit_breaker = MagicMock(name="circuit_breaker")
    circuit_breaker.check.return_value = None
    return MetadataClient(
        cache=cache,
        rate_limiter=rate_limiter,
        circuit_breaker=circuit_breaker,
    )


def _valuation_fixture() -> pd.DataFrame:
    """Return a 9-metric valuation panel including the asserted superset."""
    columns = [pd.Timestamp("2024-12-31"), pd.Timestamp("2025-03-31")]
    rows = {
        "MarketCap": [3_000_000_000_000.0, 3_100_000_000_000.0],
        "EnterpriseValue": [3_050_000_000_000.0, 3_150_000_000_000.0],
        "PeRatio": [28.5, 29.1],
        "ForwardPeRatio": [27.0, 27.5],
        "PegRatio": [2.1, 2.2],
        "PsRatio": [7.5, 7.7],
        "PbRatio": [45.0, 46.5],
        "EnterprisesValueRevenueRatio": [7.6, 7.8],
        "EnterprisesValueEBITDARatio": [22.0, 22.5],
    }
    return pd.DataFrame(rows, index=columns).T


def test_when_valuation_panel_present_then_dataframe_is_returned() -> None:
    fixture = _valuation_fixture()
    client = _build_client(valuation=fixture)

    result = client.fetch_valuation_measures("AAPL")

    assert isinstance(result, pd.DataFrame)
    assert result.shape == fixture.shape
    assert {"PeRatio", "PbRatio", "MarketCap"}.issubset(set(result.index))
    pd.testing.assert_frame_equal(result, fixture)


def test_when_valuation_panel_empty_then_none_is_returned() -> None:
    client = _build_client(valuation=pd.DataFrame())

    result = client.fetch_valuation_measures("AAPL")

    assert result is None


def test_when_metadata_client_constructed_then_protocol_is_satisfied() -> None:
    client = _build_client(valuation=_valuation_fixture())

    assert isinstance(client, MetadataClientProtocol)


_FETCH_DAY = date(2026, 7, 14)


@pytest.fixture
def fetch_day(monkeypatch: pytest.MonkeyPatch) -> date:
    """Pin the day the client stamps on the panel's ``Current`` column."""
    monkeypatch.setattr(metadata_module, "_utc_today", lambda: _FETCH_DAY)
    return _FETCH_DAY


def _yahoo_valuation_panel() -> pd.DataFrame:
    """Return the shape yfinance 1.6 emits: string labels, not dates."""
    return pd.DataFrame(
        {
            "Current": [28.5, 3.0e12],
            "6/30/2026": [27.0, 2.9e12],
            "3/31/2026": [26.0, 2.8e12],
        },
        index=["PeRatio", "MarketCap"],
    )


def test_when_panel_has_yahoo_string_labels_then_columns_are_dates(
    fetch_day: date,
) -> None:
    client = _build_client(valuation=_yahoo_valuation_panel())

    result = client.fetch_valuation_measures("AAPL")

    assert result is not None
    assert [pd.Timestamp(c) for c in result.columns] == [
        pd.Timestamp(fetch_day),
        pd.Timestamp("2026-06-30"),
        pd.Timestamp("2026-03-31"),
    ]
    assert result.loc["PeRatio"].tolist() == [28.5, 27.0, 26.0]


def test_when_panel_has_yahoo_string_labels_then_repository_stores_every_column(
    fetch_day: date,
) -> None:
    client = _build_client(valuation=_yahoo_valuation_panel())
    frame = client.fetch_valuation_measures("AAPL")
    repo = YFinanceRepository(MagicMock())
    captured: list[dict] = []

    def fake_upsert(model, rows, *, constraint_name, update_columns=None):
        captured.extend(rows)
        return len(rows)

    repo._upsert = fake_upsert  # type: ignore[method-assign]

    written = repo.upsert_financial_statements(
        uuid.uuid4(), frame, "valuation_measures", "point_in_time"
    )

    assert written == 6
    assert {r["period_date"] for r in captured} == {
        fetch_day,
        date(2026, 6, 30),
        date(2026, 3, 31),
    }


def test_when_a_period_column_falls_on_the_fetch_day_then_it_wins_the_date(
    fetch_day: date,
) -> None:
    panel = pd.DataFrame({"Current": [1.0], "7/14/2026": [2.0]}, index=["PeRatio"])
    client = _build_client(valuation=panel)

    result = client.fetch_valuation_measures("AAPL")

    assert result is not None
    assert result.shape == (1, 1)
    assert result.loc["PeRatio"].tolist() == [2.0]


def test_when_a_label_is_not_a_date_then_that_column_is_dropped(
    fetch_day: date,
) -> None:
    panel = pd.DataFrame({"Current": [1.0], "n/a": [2.0]}, index=["PeRatio"])
    client = _build_client(valuation=panel)

    result = client.fetch_valuation_measures("AAPL")

    assert result is not None
    assert result.shape == (1, 1)
