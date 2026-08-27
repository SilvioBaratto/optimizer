"""SPEC A10 — full options-chain ingestion (own low-frequency step + gate).

yfinance and the DB session are mocked (no network, no real DB). The upsert is
PostgreSQL-only (ON CONFLICT), so persistence is verified at the service level
by asserting the repo upsert is invoked with flattened rows.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd

from app.services.market_data.options_service import (
    _flatten_chain,
    _is_fresh,
    run_bulk_options_fetch,
)

M = "app.services.market_data.options_service"


def _chain() -> SimpleNamespace:
    calls = pd.DataFrame(
        {
            "contractSymbol": ["AAPL260918C00150000"],
            "strike": [150.0],
            "lastPrice": [12.3],
            "bid": [12.0],
            "ask": [12.6],
            "volume": [100],
            "openInterest": [500],
            "impliedVolatility": [0.25],
            "inTheMoney": [True],
        }
    )
    puts = pd.DataFrame(
        {
            "contractSymbol": ["AAPL260918P00150000"],
            "strike": [150.0],
            "lastPrice": [8.1],
            "bid": [8.0],
            "ask": [8.3],
            "volume": [50],
            "openInterest": [300],
            "impliedVolatility": [0.28],
            "inTheMoney": [False],
        }
    )
    return SimpleNamespace(calls=calls, puts=puts, underlying={})


def _session_ctx() -> MagicMock:
    @contextmanager
    def _cm():
        yield MagicMock(name="session")

    dbm = MagicMock()
    dbm.get_session = _cm
    return dbm


def _yf_with_options() -> MagicMock:
    yf = MagicMock()
    yf.metadata.fetch_options_expirations.return_value = ("2026-09-18",)
    yf.metadata.fetch_option_chain.return_value = _chain()
    return yf


class TestModel:
    def test_option_contract_columns(self) -> None:
        from app.models.market_data.yfinance_data import OptionContract

        for col in (
            "expiry",
            "strike",
            "option_type",
            "implied_volatility",
            "in_the_money",
            "as_of",
        ):
            assert hasattr(OptionContract, col)


class TestFlatten:
    def test_flatten_yields_calls_and_puts(self) -> None:
        rows = _flatten_chain(_chain(), date(2026, 8, 27), date(2026, 9, 18))
        assert len(rows) == 2
        by_type = {r["option_type"]: r for r in rows}
        assert by_type["call"]["strike"] == 150.0
        assert by_type["call"]["open_interest"] == 500
        assert by_type["call"]["in_the_money"] is True
        assert by_type["put"]["implied_volatility"] == 0.28

    def test_in_the_money_nan_maps_to_none(self) -> None:
        calls = pd.DataFrame(
            {
                "contractSymbol": ["X"],
                "strike": [1.0],
                "inTheMoney": [float("nan")],
            }
        )
        chain = SimpleNamespace(calls=calls, puts=pd.DataFrame())
        rows = _flatten_chain(chain, date(2026, 8, 27), date(2026, 9, 18))
        assert rows[0]["in_the_money"] is None

    def test_flatten_skips_rows_without_symbol_or_strike(self) -> None:
        calls = pd.DataFrame({"contractSymbol": [None], "strike": [1.0]})
        chain = SimpleNamespace(calls=calls, puts=pd.DataFrame())
        assert _flatten_chain(chain, date(2026, 8, 27), date(2026, 9, 18)) == []


class TestIsFresh:
    def test_none_is_not_fresh(self) -> None:
        assert _is_fresh(None, 168, datetime.now(timezone.utc)) is False

    def test_recent_is_fresh(self) -> None:
        now = datetime(2026, 8, 27, tzinfo=timezone.utc)
        assert _is_fresh(date(2026, 8, 25), 168, now) is True

    def test_old_is_stale(self) -> None:
        now = datetime(2026, 8, 27, tzinfo=timezone.utc)
        assert _is_fresh(date(2026, 8, 1), 168, now) is False


class TestBulkFetch:
    def _repo(self, *, iid, as_of):
        repo = MagicMock(name="repo")
        repo.get_instruments_with_yfinance_ticker.return_value = [
            SimpleNamespace(id=iid, yfinance_ticker="AAPL")
        ]
        # Grouped gate query: present-and-fresh -> skip; absent -> stale/fetch.
        repo.get_options_as_of_bulk.return_value = {iid: as_of} if as_of else {}
        repo.upsert_option_chain.return_value = 2
        return repo

    def test_persists_when_stale(self) -> None:
        iid = uuid4()
        repo = self._repo(iid=iid, as_of=None)
        yf = _yf_with_options()
        with (
            patch(f"{M}.YFinanceRepository", return_value=repo),
            patch(f"{M}.call_with_timeout", lambda fn, _t: fn()),
            patch("app.database.database_manager", _session_ctx()),
        ):
            result = run_bulk_options_fetch(yf)

        repo.get_options_as_of_bulk.assert_called_once()
        repo.get_options_as_of.assert_not_called()  # no per-instrument query
        repo.upsert_option_chain.assert_called_once()
        assert result["contract_rows"] == 2
        assert result["instruments_processed"] == 1
        assert result["instruments_skipped_fresh"] == 0

    def test_skips_when_fresh(self) -> None:
        iid = uuid4()
        repo = self._repo(iid=iid, as_of=date.today())
        yf = _yf_with_options()
        with (
            patch(f"{M}.YFinanceRepository", return_value=repo),
            patch(f"{M}.call_with_timeout", lambda fn, _t: fn()),
            patch("app.database.database_manager", _session_ctx()),
        ):
            result = run_bulk_options_fetch(yf, staleness_hours=168)

        repo.upsert_option_chain.assert_not_called()
        yf.metadata.fetch_options_expirations.assert_not_called()
        assert result["instruments_skipped_fresh"] == 1

    def test_yfinance_calls_are_timeout_wrapped(self) -> None:
        iid = uuid4()
        repo = self._repo(iid=iid, as_of=None)
        yf = _yf_with_options()
        seen: list[float] = []

        def _capture(fn, timeout):
            seen.append(timeout)
            return fn()

        with (
            patch(f"{M}.YFinanceRepository", return_value=repo),
            patch(f"{M}.call_with_timeout", _capture),
            patch("app.database.database_manager", _session_ctx()),
        ):
            run_bulk_options_fetch(yf, request_timeout=7.5)

        # expirations + one option_chain both routed through the watchdog
        assert seen and all(t == 7.5 for t in seen)
