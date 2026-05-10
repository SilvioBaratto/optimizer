"""Unit tests for ``AnalysisClient.fetch_eps_trend`` and ``fetch_eps_revisions``.

Covers the yfinance 1.3.0 ``Ticker.eps_trend`` and ``Ticker.eps_revisions``
DataFrame passthroughs, empty-frame handling, and ``AnalysisClientProtocol``
conformance.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd

from app.services.market_data.yfinance.protocols.interfaces import (
    AnalysisClientProtocol,
)
from app.services.market_data.yfinance.ticker.analysis import AnalysisClient


def _build_client(**ticker_attrs: Any) -> AnalysisClient:
    """Construct an ``AnalysisClient`` whose cached ticker exposes *ticker_attrs*."""
    cache = MagicMock(name="cache")
    cache.get.return_value = MagicMock(**ticker_attrs)
    rate_limiter = MagicMock(name="rate_limiter")
    circuit_breaker = MagicMock(name="circuit_breaker")
    circuit_breaker.check.return_value = None
    return AnalysisClient(
        cache=cache,
        rate_limiter=rate_limiter,
        circuit_breaker=circuit_breaker,
    )


def _eps_trend_fixture() -> pd.DataFrame:
    columns = ["current", "7daysAgo", "30daysAgo", "60daysAgo", "90daysAgo"]
    rows = {
        "0q": [1.50, 1.51, 1.52, 1.55, 1.60],
        "+1q": [1.65, 1.66, 1.67, 1.70, 1.72],
        "0y": [6.20, 6.22, 6.25, 6.30, 6.40],
        "+1y": [7.10, 7.12, 7.15, 7.20, 7.25],
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


def _eps_revisions_fixture() -> pd.DataFrame:
    columns = ["upLast7days", "upLast30days", "downLast30days", "downLast90days"]
    rows = {
        "0q": [3, 5, 1, 2],
        "+1q": [2, 4, 0, 1],
        "0y": [4, 8, 2, 3],
        "+1y": [1, 3, 1, 2],
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


def test_when_eps_trend_panel_present_then_dataframe_is_returned() -> None:
    fixture = _eps_trend_fixture()
    client = _build_client(eps_trend=fixture)

    result = client.fetch_eps_trend("AAPL")

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, fixture)


def test_when_eps_trend_panel_empty_then_none_is_returned() -> None:
    client = _build_client(eps_trend=pd.DataFrame())

    result = client.fetch_eps_trend("AAPL")

    assert result is None


def test_when_eps_revisions_panel_present_then_dataframe_is_returned() -> None:
    fixture = _eps_revisions_fixture()
    client = _build_client(eps_revisions=fixture)

    result = client.fetch_eps_revisions("AAPL")

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, fixture)


def test_when_eps_revisions_panel_empty_then_none_is_returned() -> None:
    client = _build_client(eps_revisions=pd.DataFrame())

    result = client.fetch_eps_revisions("AAPL")

    assert result is None


def test_when_analysis_client_constructed_then_protocol_is_satisfied() -> None:
    client = _build_client(
        eps_trend=_eps_trend_fixture(),
        eps_revisions=_eps_revisions_fixture(),
    )

    assert isinstance(client, AnalysisClientProtocol)
