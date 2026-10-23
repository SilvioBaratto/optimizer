"""SPEC B3 — regional market summaries via Yahoo's marketSummary endpoint.

The client is exercised with only yfinance's HTTP layer replaced, so the real
request/parse path runs; the repo runs against real SQLite; the bulk service is
driven with a mocked repo + session.
"""

from __future__ import annotations

import datetime as dt
import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from portopt_db.repositories.market_data.market_summary_repository import (
    MarketSummaryRepository,
)
from yfinance.data import YfData

from app.services.market_data.market_summary_service import run_market_summary_fetch
from app.services.market_data.yfinance.market.market_summary import (
    MARKET_IDENTIFIERS,
    MarketClient,
)

_AS_OF = dt.date(2026, 8, 27)


class _Resp:
    """Minimal stand-in for the HTTP response yfinance's ``YfData.get`` returns."""

    status_code = 200

    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def _serve(*results: Any):
    """Patch yfinance's HTTP layer; each summary request pops the next ``result``.

    The last ``result`` is reused once the sequence runs out. Any other endpoint
    (the market-status call ``yf.Market`` makes) answers with an empty object.
    """
    queue = list(results)

    def fake_get(url: str, params: Any = None, timeout: float = 30) -> _Resp:
        if "marketSummary" not in url:
            return _Resp({})
        result = queue.pop(0) if len(queue) > 1 else queue[0]
        return _Resp({"marketSummaryResponse": {"result": result, "error": None}})

    return patch.object(YfData, "get", side_effect=fake_get)


@pytest.fixture(autouse=True)
def _fresh_yfinance_cache():
    YfData.cache_get.cache_clear()
    yield
    YfData.cache_get.cache_clear()


def _client() -> MarketClient:
    return MarketClient(
        rate_limiter=MagicMock(),
        circuit_breaker=MagicMock(),
        default_max_retries=1,
    )


class TestMarketClient:
    def test_fetch_summary_unwraps_raw_values(self) -> None:
        quotes = [
            {
                "symbol": "^GSPC",
                "exchange": "SNP",
                "shortName": "S&P 500",
                "regularMarketPrice": {"raw": 5600.5, "fmt": "5,600.50"},
                "regularMarketChangePercent": {"raw": 0.42},
                "marketState": "REGULAR",
            }
        ]
        with _serve(quotes):
            out = _client().fetch_summary("US")

        assert out is not None and len(out) == 1
        assert out[0]["symbol"] == "^GSPC"
        assert out[0]["price"] == 5600.5
        assert out[0]["change_percent"] == 0.42
        assert out[0]["market_state"] == "REGULAR"

    def test_symbol_comes_from_quote_not_exchange_code(self) -> None:
        quotes = [
            {
                "symbol": "^GSPC",
                "exchange": "SNP",
                "shortName": "S&P 500",
                "regularMarketPrice": {"raw": 5600.5},
            }
        ]
        with _serve(quotes):
            out = _client().fetch_summary("US")
        assert out[0]["symbol"] == "^GSPC"  # not "SNP"

    def test_symbol_falls_back_to_exchange_code_when_missing(self) -> None:
        quotes = [{"exchange": "XYZ", "shortName": "Thing", "regularMarketPrice": 1.0}]
        with _serve(quotes):
            out = _client().fetch_summary("US")
        assert out[0]["symbol"] == "XYZ"

    def test_quotes_sharing_an_exchange_code_are_all_returned(self) -> None:
        # Futures, FX and rates quotes share one exchange code ("CME"); keying
        # the summary by exchange silently kept only the last of each group.
        quotes = [
            {"symbol": "^GSPC", "exchange": "SNP", "regularMarketPrice": 5600.0},
            {"symbol": "^DJI", "exchange": "DJI", "regularMarketPrice": 41000.0},
            {"symbol": "ES=F", "exchange": "CME", "regularMarketPrice": 5610.0},
            {"symbol": "NQ=F", "exchange": "CME", "regularMarketPrice": 19800.0},
            {"symbol": "YM=F", "exchange": "CME", "regularMarketPrice": 41050.0},
        ]
        with _serve(quotes):
            out = _client().fetch_summary("US")

        assert out is not None
        assert [r["symbol"] for r in out] == [q["symbol"] for q in quotes]

    def test_repeat_fetch_requests_fresh_data(self) -> None:
        # yfinance's process-wide response cache would replay the first answer
        # for the same market on every later sweep.
        first = [{"symbol": "^GSPC", "exchange": "SNP", "regularMarketPrice": 1000}]
        second = [{"symbol": "^GSPC", "exchange": "SNP", "regularMarketPrice": 1001}]
        client = _client()
        with _serve(first, second):
            a = client.fetch_summary("US")
            b = client.fetch_summary("US")

        assert a is not None and b is not None
        assert (a[0]["price"], b[0]["price"]) == (1000, 1001)

    def test_missing_result_yields_empty(self) -> None:
        with _serve(None):
            assert _client().fetch_summary("EUROPE") == []

    def test_unknown_market_is_rejected_without_a_request(self) -> None:
        with _serve([]) as get:
            assert _client().fetch_summary("MARS") is None
        get.assert_not_called()

    def test_eight_documented_identifiers(self) -> None:
        assert MARKET_IDENTIFIERS == (
            "US",
            "GB",
            "ASIA",
            "EUROPE",
            "RATES",
            "COMMODITIES",
            "CURRENCIES",
            "CRYPTOCURRENCIES",
        )


class TestRepository:
    def test_upsert_idempotent(self, db_session) -> None:
        repo = MarketSummaryRepository(db_session)
        rows = [
            {
                "symbol": "^GSPC",
                "short_name": "S&P 500",
                "price": 5600.5,
                "change": 12.3,
                "change_percent": 0.42,
                "previous_close": 5588.2,
                "market_state": "REGULAR",
            }
        ]
        assert repo.upsert_summaries("US", _AS_OF, rows) == 1
        assert repo.upsert_summaries("US", _AS_OF, rows) == 1  # idempotent
        db_session.flush()

        from portopt_db.models.market_data.market_summary import MarketSummary

        got = db_session.query(MarketSummary).all()
        assert len(got) == 1
        assert float(got[0].price) == 5600.5


def _fake_dbm() -> MagicMock:
    @contextmanager
    def _cm():
        yield MagicMock(name="session")

    dbm = MagicMock()
    dbm.get_session = _cm
    return dbm


class TestBulkFetch:
    def test_sweeps_all_markets(self) -> None:
        repo = MagicMock(name="repo")
        repo.upsert_summaries.return_value = 3
        yf = MagicMock()
        yf.market.fetch_summary.return_value = [{"symbol": "^GSPC"}]
        with (
            patch(
                "portopt_db.repositories.market_data.market_summary_repository."
                "MarketSummaryRepository",
                return_value=repo,
            ),
            patch("app.database.database_manager", _fake_dbm()),
        ):
            result = run_market_summary_fetch(yf)

        assert yf.market.fetch_summary.call_count == len(MARKET_IDENTIFIERS)
        assert result["rows_total"] == 3 * len(MARKET_IDENTIFIERS)
        assert result["error_count"] == 0

    def test_market_failure_isolated(self) -> None:
        repo = MagicMock(name="repo")
        repo.upsert_summaries.return_value = 0
        yf = MagicMock()
        yf.market.fetch_summary.side_effect = RuntimeError("boom")
        with (
            patch(
                "portopt_db.repositories.market_data.market_summary_repository."
                "MarketSummaryRepository",
                return_value=repo,
            ),
            patch("app.database.database_manager", _fake_dbm()),
        ):
            result = run_market_summary_fetch(yf, markets=("US", "GB"))

        assert result["error_count"] == 2

    def test_unrecoverable_fetch_is_recorded_as_an_error(self) -> None:
        repo = MagicMock(name="repo")
        repo.upsert_summaries.return_value = 4
        yf = MagicMock()
        yf.market.fetch_summary.side_effect = lambda market: (
            None if market == "GB" else [{"symbol": "^GSPC"}]
        )
        progress = MagicMock()
        with (
            patch(
                "portopt_db.repositories.market_data.market_summary_repository."
                "MarketSummaryRepository",
                return_value=repo,
            ),
            patch("app.database.database_manager", _fake_dbm()),
        ):
            result = run_market_summary_fetch(
                yf, markets=("US", "GB", "ASIA"), on_progress=progress
            )

        assert result["error_count"] == 1
        assert "GB" not in result["counts"]
        assert result["rows_total"] == 8
        assert [c.args[0] for c in repo.upsert_summaries.call_args_list] == [
            "US",
            "ASIA",
        ]
        assert progress.call_args.kwargs["errors"] == ["GB: no data"]

    def test_empty_summary_is_not_an_error(self) -> None:
        repo = MagicMock(name="repo")
        repo.upsert_summaries.return_value = 0
        yf = MagicMock()
        yf.market.fetch_summary.return_value = []
        with (
            patch(
                "portopt_db.repositories.market_data.market_summary_repository."
                "MarketSummaryRepository",
                return_value=repo,
            ),
            patch("app.database.database_manager", _fake_dbm()),
        ):
            result = run_market_summary_fetch(yf, markets=("US",))

        assert result["error_count"] == 0
        assert result["counts"] == {"US": 0}


class TestSchedulerStep:
    def test_composes_run_step(self) -> None:
        M = "app.services.jobs.scheduler"
        with (
            patch(f"{M}._run_step", return_value=True) as run_step,
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            ),
        ):
            from app.services.jobs.scheduler import (
                _market_summary_jobs,
                run_market_summary_step,
            )

            assert run_market_summary_step() is True

        assert run_step.call_args.args[0] == "market_summary"
        assert run_step.call_args.args[1] is _market_summary_jobs
