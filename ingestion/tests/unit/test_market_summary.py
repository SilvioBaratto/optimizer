"""SPEC B3 — regional market summaries via yf.Market.

Client parsing patches ``yf.Market``; the repo runs against real SQLite; the
bulk service is driven with a mocked repo + session.
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from portopt_db.repositories.market_data.market_summary_repository import (
    MarketSummaryRepository,
)

from app.services.market_data.market_summary_service import run_market_summary_fetch
from app.services.market_data.yfinance.market.market_summary import (
    MARKET_IDENTIFIERS,
    MarketClient,
)

_AS_OF = dt.date(2026, 8, 27)


def _client() -> MarketClient:
    return MarketClient(
        rate_limiter=MagicMock(),
        circuit_breaker=MagicMock(),
        default_max_retries=1,
    )


class TestMarketClient:
    def test_fetch_summary_unwraps_raw_values(self) -> None:
        summary = {
            "^GSPC": {
                "shortName": "S&P 500",
                "regularMarketPrice": {"raw": 5600.5, "fmt": "5,600.50"},
                "regularMarketChangePercent": {"raw": 0.42},
                "marketState": "REGULAR",
            }
        }
        market = SimpleNamespace(summary=summary)
        with patch(
            "app.services.market_data.yfinance.market.market_summary.yf.Market",
            return_value=market,
        ):
            out = _client().fetch_summary("US")

        assert out is not None and len(out) == 1
        assert out[0]["symbol"] == "^GSPC"
        assert out[0]["price"] == 5600.5
        assert out[0]["change_percent"] == 0.42
        assert out[0]["market_state"] == "REGULAR"

    def test_symbol_comes_from_quote_not_exchange_key(self) -> None:
        # yfinance keys summary by exchange code; the real symbol is q["symbol"].
        summary = {
            "SNP": {
                "symbol": "^GSPC",
                "shortName": "S&P 500",
                "regularMarketPrice": {"raw": 5600.5},
            }
        }
        market = SimpleNamespace(summary=summary)
        with patch(
            "app.services.market_data.yfinance.market.market_summary.yf.Market",
            return_value=market,
        ):
            out = _client().fetch_summary("US")
        assert out[0]["symbol"] == "^GSPC"  # not "SNP"

    def test_symbol_falls_back_to_key_when_missing(self) -> None:
        summary = {"XYZ": {"shortName": "Thing", "regularMarketPrice": 1.0}}
        market = SimpleNamespace(summary=summary)
        with patch(
            "app.services.market_data.yfinance.market.market_summary.yf.Market",
            return_value=market,
        ):
            out = _client().fetch_summary("US")
        assert out[0]["symbol"] == "XYZ"

    def test_non_dict_summary_yields_empty(self) -> None:
        market = SimpleNamespace(summary=None)
        with patch(
            "app.services.market_data.yfinance.market.market_summary.yf.Market",
            return_value=market,
        ):
            assert _client().fetch_summary("EUROPE") == []

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
