"""A full-history fetch must end before today, like the incremental one.

Today's bar is still forming while the exchange is open. The freshness gate treats
the newest stored date as final, so a partial bar stored by a full fetch (first
load, re-base) stays in ``price_history`` until the next day's incremental window
rewrites it.
"""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pandas as pd
import pytest

from app.services.market_data import yfinance_data_service as service_module
from app.services.market_data.yfinance_data_service import YFinanceDataService

_TODAY = date(2026, 11, 6)


@pytest.fixture(autouse=True)
def _fixed_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module, "_today", lambda: _TODAY)


def _history() -> pd.DataFrame:
    index = pd.date_range("2021-01-04", periods=40, freq="B")
    return pd.DataFrame({"Close": [10.0] * 40}, index=index)


def _full_fetch_kwargs(
    *, period: str, oldest: date | None, staleness: Any
) -> dict[str, Any]:
    repo = MagicMock(name="repo")
    repo.get_staleness_info.return_value = staleness
    stored = [SimpleNamespace(date=oldest, close=10.0)] if oldest else []
    repo.get_price_history.return_value = stored
    yf = MagicMock(name="yf_client")
    yf.fetch_history.return_value = _history()
    yf.circuit_breaker.is_active = False
    yf.get_ticker.return_value = MagicMock(news=[])

    YFinanceDataService(repo=repo, yf_client=yf).fetch_and_store(
        uuid4(), "AAPL", period=period, mode="full"
    )

    yf.fetch_history.assert_called_once()
    return dict(yf.fetch_history.call_args.kwargs)


class TestFullFetchEnd:
    """Every shape of the full-history window stops before today."""

    def test_first_load_by_period_ends_today(self) -> None:
        """Bound a period request so the open session's bar is not returned."""
        kwargs = _full_fetch_kwargs(period="5y", oldest=None, staleness=None)

        assert kwargs["period"] == "5y"
        assert kwargs["end"] == _TODAY.isoformat()

    def test_rebase_from_the_oldest_stored_bar_ends_today(self) -> None:
        """Bound a start-anchored request the same way."""
        kwargs = _full_fetch_kwargs(
            period="5y", oldest=_TODAY - timedelta(days=6 * 365), staleness=None
        )

        assert kwargs["start"] == (_TODAY - timedelta(days=6 * 365)).isoformat()
        assert kwargs["end"] == _TODAY.isoformat()

    def test_max_period_ends_today(self) -> None:
        """Keep period='max' open-ended at the start but closed at today."""
        kwargs = _full_fetch_kwargs(period="max", oldest=None, staleness=None)

        assert kwargs["period"] == "max"
        assert kwargs["end"] == _TODAY.isoformat()

    def test_year_to_date_is_anchored_by_start_because_yfinance_cannot_bound_it(
        self,
    ) -> None:
        """Turn 'ytd' into an explicit window; yfinance rejects ytd with an end."""
        kwargs = _full_fetch_kwargs(period="ytd", oldest=None, staleness=None)

        assert kwargs["start"] == date(_TODAY.year, 1, 1).isoformat()
        assert "period" not in kwargs
        assert kwargs["end"] == _TODAY.isoformat()
