"""Unit tests for ``MetadataClient.fetch_valuation_measures``.

Covers the yfinance 1.3.0 ``Ticker.valuation`` 9-metric panel passthrough,
empty-frame handling, and ``MetadataClientProtocol`` conformance.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from app.services.yfinance.protocols.interfaces import MetadataClientProtocol
from app.services.yfinance.ticker.metadata import MetadataClient


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
