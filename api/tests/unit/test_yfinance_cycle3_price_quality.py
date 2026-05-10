"""Cycle 3 regression: kwarg forwarding + NVDA split-day continuity (hermetic).

Locks in the post-#598 contract:
* ``fetch_history`` forwards the eight price-quality kwargs to ``ticker.history``
  on both branches (period and start/end).
* ``bulk_download`` defaults to ``auto_adjust=True``.
* Continuity invariant on hand-authored adjusted vs raw NVDA fixtures
  around the 2024-06-10 10:1 split.

No live Yahoo calls. ``yf.download`` and ``ticker.history`` are mocked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.services.yfinance._facade import YFinanceClient
from app.services.yfinance.infrastructure import CircuitBreaker, LRUCache, RateLimiter

_HISTORY_KWARGS: dict[str, Any] = {
    "interval": "1d",
    "auto_adjust": True,
    "back_adjust": False,
    "repair": True,
    "actions": False,
    "prepost": False,
    "keepna": False,
    "rounding": False,
    "timeout": None,
}


def _history_fixture(rows: int = 12) -> pd.DataFrame:
    """Synthetic ``Ticker.history`` payload that passes the ``min_rows=10`` guard."""
    index = pd.date_range("2024-01-02", periods=rows, freq="B")
    return pd.DataFrame({"Close": [100.0 + i for i in range(rows)]}, index=index)


def _adjusted_nvda_fixture() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        ["2024-06-07", "2024-06-10", "2024-06-11", "2024-06-14"]
    )
    return pd.DataFrame({"Close": [120.0, 121.0, 119.0, 122.0]}, index=index)


def _raw_nvda_fixture() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        ["2024-06-07", "2024-06-10", "2024-06-11", "2024-06-14"]
    )
    return pd.DataFrame({"Close": [1200.0, 1198.0, 120.0, 122.0]}, index=index)


@pytest.fixture
def client() -> YFinanceClient:
    return YFinanceClient(
        cache=LRUCache(capacity=8, default_ttl=60.0),
        rate_limiter=RateLimiter(delay=0.0),
        circuit_breaker=CircuitBreaker(),
        default_max_retries=1,
    )


def _install_ticker(client: YFinanceClient, ticker: MagicMock) -> None:
    client.get_ticker = MagicMock(return_value=ticker)  # type: ignore[method-assign]


def test_when_calling_fetch_history_with_period_then_all_eight_kwargs_forward(
    client: YFinanceClient,
) -> None:
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _history_fixture()
    _install_ticker(client, mock_ticker)

    client.fetch_history("NVDA", period="5y")

    mock_ticker.history.assert_called_once_with(period="5y", **_HISTORY_KWARGS)


def test_when_calling_fetch_history_with_start_end_then_all_eight_kwargs_forward(
    client: YFinanceClient,
) -> None:
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _history_fixture()
    _install_ticker(client, mock_ticker)

    client.fetch_history("NVDA", start="2024-01-01", end="2024-12-31")

    mock_ticker.history.assert_called_once_with(
        start="2024-01-01", end="2024-12-31", **_HISTORY_KWARGS
    )


def test_when_caller_overrides_auto_adjust_false_then_override_reaches_ticker(
    client: YFinanceClient,
) -> None:
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _history_fixture()
    _install_ticker(client, mock_ticker)

    client.fetch_history("NVDA", period="5y", auto_adjust=False)

    call_kwargs = mock_ticker.history.call_args.kwargs
    assert call_kwargs["auto_adjust"] is False
    assert call_kwargs["repair"] is True


def test_when_calling_bulk_download_with_defaults_then_auto_adjust_is_true(
    client: YFinanceClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_download = MagicMock(return_value=_history_fixture())
    monkeypatch.setattr("app.services.yfinance._facade.yf.download", mock_download)

    client.bulk_download(["NVDA"])

    assert mock_download.call_args.kwargs["auto_adjust"] is True


def test_when_inspecting_adjusted_nvda_fixture_then_no_split_day_step() -> None:
    adjusted = _adjusted_nvda_fixture()
    ratio = adjusted["Close"].iloc[-1] / adjusted["Close"].iloc[0]
    assert 0.9 <= ratio <= 1.1


def test_when_inspecting_raw_nvda_fixture_then_pre_post_split_ratio_exceeds_five() -> None:
    raw = _raw_nvda_fixture()
    ratio = raw["Close"].iloc[0] / raw["Close"].iloc[-1]
    assert ratio > 5
