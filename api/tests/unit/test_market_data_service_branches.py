"""Branch-coverage tests for yfinance_data_service.py and reference_index_seeder.py.

Targets in yfinance_data_service.py:
  - Lines 105, 108, 110: _is_fresh — None guard, naive-tz normalization, now-tz normalization
  - Line  133:           _timed with request_timeout set
  - Lines 165-179:       staleness query failure → fallback to full fetch
  - Lines 251-253:       lazy ticker holder initialisation
  - Line  268:           profile skip when incremental + fresh
  - Lines 307-351:       price history insufficient-history guard
    (existing_prices True → skip, existing_prices False → store young listing)
  - Lines 362, 366:      history None/empty after guard → counts["prices"] fallback
  - Lines 437-439:       outer financials exception handler
  - Lines 530, 556:      dividends/splits empty → count=0
  - Lines 582, 638:      recommendations/institutional empty → count=0
  - Lines 668, 698:      mutualfund/insider empty → count=0
  - Lines 721-731:       news article canonical URL parsing
  - Lines 735-737:       news article previewUrl/link fallback
  - Lines 798-799:       serial bulk — missing yfinance_ticker guard
  - Line  815:           serial bulk — per-ticker exception + rollback
  - Line  880:           _fetch_one_ticker — missing ticker guard

Targets in reference_index_seeder.py:
  - Lines 107-119:       instrument is None → create new Instrument, flush
  - Lines 134, 137:      fetch_result counts and errors accumulation

Pure unit: MagicMock repo/session, no DB, no real yfinance network calls.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared builder helpers
# ---------------------------------------------------------------------------


def _minimal_yf_client() -> MagicMock:
    """Return a MagicMock yfinance client with every fetcher returning None/empty."""
    c = MagicMock(name="yf_client")
    c.fetch_info.return_value = None
    c.fetch_history.return_value = None
    c.financials.fetch_income_stmt.return_value = None
    c.financials.fetch_balance_sheet.return_value = None
    c.financials.fetch_cashflow.return_value = None
    c.corporate_actions.fetch_dividends.return_value = None
    c.corporate_actions.fetch_splits.return_value = None
    c.analysis.fetch_recommendations_summary.return_value = None
    c.analysis.fetch_analyst_price_targets.return_value = None
    c.analysis.fetch_eps_trend.return_value = None
    c.analysis.fetch_eps_revisions.return_value = None
    c.holders.fetch_institutional_holders.return_value = None
    c.holders.fetch_mutualfund_holders.return_value = None
    c.holders.fetch_insider_transactions.return_value = None
    c.metadata.fetch_valuation_measures.return_value = None
    ticker_mock = MagicMock()
    ticker_mock.news = []
    c.get_ticker.return_value = ticker_mock
    return c


def _minimal_repo() -> MagicMock:
    r = MagicMock(name="repo")
    r.get_staleness_info.return_value = None
    r.upsert_profile.return_value = 0
    r.upsert_price_history.return_value = 5
    r.upsert_dividends.return_value = 0
    r.upsert_splits.return_value = 0
    r.upsert_recommendations.return_value = 0
    r.upsert_price_targets.return_value = 0
    r.upsert_institutional_holders.return_value = 0
    r.upsert_mutualfund_holders.return_value = 0
    r.upsert_insider_transactions.return_value = 0
    r.upsert_financial_statements.return_value = 0
    r.upsert_news.return_value = 0
    return r


def _service(repo=None, yf_client=None, request_timeout=None):
    from app.services.market_data.yfinance_data_service import YFinanceDataService

    return YFinanceDataService(
        repo=repo or _minimal_repo(),
        yf_client=yf_client or _minimal_yf_client(),
        request_timeout=request_timeout,
    )


def _run(service, *, mode="full", period="5y", exchange_name=None, currency_code=None):
    return service.fetch_and_store(
        instrument_id=uuid4(),
        yfinance_ticker="AAPL",
        period=period,
        mode=mode,
        exchange_name=exchange_name,
        currency_code=currency_code,
    )


def _small_history() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    return pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=dates)


def _large_history(n: int = 1000) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": [100.0] * n}, index=dates)


# ---------------------------------------------------------------------------
# _is_fresh — lines 104-111
# ---------------------------------------------------------------------------


class TestIsFresh:
    def test_when_updated_at_is_none_returns_false(self):
        """Line 104-105: updated_at None → immediately return False."""
        from app.services.market_data.yfinance_data_service import _is_fresh

        now = datetime.now(timezone.utc)
        assert _is_fresh(None, 24, now) is False

    def test_when_updated_at_is_naive_it_is_treated_as_utc(self):
        """Line 107-108: naive updated_at is normalized to UTC for comparison."""
        from app.services.market_data.yfinance_data_service import _is_fresh

        # A naive datetime that is only 1 hour old — should be fresh for 24h window
        naive_recent = datetime(2026, 6, 10, 20, 0, 0)  # naive
        aware_now = datetime(2026, 6, 10, 21, 0, 0, tzinfo=timezone.utc)

        assert _is_fresh(naive_recent, 24, aware_now) is True

    def test_when_now_is_naive_it_is_normalized_to_utc(self):
        """Lines 109-110: naive `now` is normalized to UTC."""
        from app.services.market_data.yfinance_data_service import _is_fresh

        aware_recent = datetime(2026, 6, 10, 20, 0, 0, tzinfo=timezone.utc)
        naive_now = datetime(2026, 6, 10, 21, 0, 0)  # naive

        assert _is_fresh(aware_recent, 24, naive_now) is True

    def test_stale_data_returns_false(self):
        """Normal stale path: updated_at older than threshold → False."""
        from app.services.market_data.yfinance_data_service import _is_fresh

        old = datetime(2024, 1, 1, tzinfo=timezone.utc)
        now = datetime(2026, 6, 10, tzinfo=timezone.utc)
        assert _is_fresh(old, 24, now) is False


# ---------------------------------------------------------------------------
# _timed — line 133
# ---------------------------------------------------------------------------


class TestTimed:
    def test_when_request_timeout_set_then_call_with_timeout_is_used(self):
        """Line 133: _timed wraps fn in call_with_timeout when timeout is set."""
        from app.services.market_data.yfinance_data_service import YFinanceDataService

        repo = _minimal_repo()
        yf_client = _minimal_yf_client()
        service = YFinanceDataService(
            repo=repo, yf_client=yf_client, request_timeout=5.0
        )

        sentinel = object()
        fn = lambda: sentinel  # noqa: E731

        with patch(
            "app.services.market_data.yfinance_data_service.call_with_timeout",
            return_value=sentinel,
        ) as mock_cwt:
            result = service._timed(fn)

        mock_cwt.assert_called_once_with(fn, 5.0)
        assert result is sentinel

    def test_when_request_timeout_none_then_fn_called_directly(self):
        """Line 131-132: _timed calls fn() directly when no timeout."""
        service = _service(request_timeout=None)
        called = []
        fn = lambda: called.append(1) or 42  # noqa: E731

        result = service._timed(fn)

        assert result == 42
        assert called == [1]


# ---------------------------------------------------------------------------
# Staleness query failure — lines 165-179
# ---------------------------------------------------------------------------


class TestStalenessQueryFailure:
    def test_when_staleness_query_raises_then_falls_back_to_full_fetch(self):
        """Lines 165-179: repo.get_staleness_info raises → staleness=None → full path."""
        repo = _minimal_repo()
        repo.get_staleness_info.side_effect = RuntimeError("connection refused")
        yf_client = _minimal_yf_client()
        service = _service(repo=repo, yf_client=yf_client)

        result = service.fetch_and_store(
            instrument_id=uuid4(),
            yfinance_ticker="MSFT",
            period="5y",
            mode="incremental",
        )

        # Should complete without raising; staleness None → full fetch attempted
        assert "errors" in result
        # profile fetch was attempted (not skipped) because staleness=None
        yf_client.fetch_info.assert_called_once()

    def test_when_staleness_query_raises_then_warning_is_logged(self, caplog):
        """Staleness query failure must be logged at WARNING."""
        import logging

        repo = _minimal_repo()
        repo.get_staleness_info.side_effect = RuntimeError("db gone")
        service = _service(repo=repo)

        with caplog.at_level(
            logging.WARNING,
            logger="app.services.market_data.yfinance_data_service",
        ):
            service.fetch_and_store(
                instrument_id=uuid4(),
                yfinance_ticker="AAPL",
                period="5y",
                mode="incremental",
            )

        assert any("Staleness query failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Profile skip in incremental mode — line 268
# ---------------------------------------------------------------------------


class TestProfileSkipIncremental:
    def test_when_profile_fresh_in_incremental_mode_then_profile_skipped(self):
        """Line 268: incremental + fresh profile → 'profile' in skipped."""
        repo = _minimal_repo()
        yf_client = _minimal_yf_client()

        now = datetime.now(timezone.utc)
        repo.get_staleness_info.return_value = {
            "profile_updated_at": now,  # fresh (0 seconds old)
            "price_max_date": None,
            "financials_updated_at": None,
            "dividends_updated_at": None,
            "splits_updated_at": None,
            "recommendations_updated_at": None,
            "price_targets_updated_at": None,
            "institutional_holders_updated_at": None,
            "mutualfund_holders_updated_at": None,
            "insider_transactions_updated_at": None,
        }
        service = _service(repo=repo, yf_client=yf_client)

        result = service.fetch_and_store(
            instrument_id=uuid4(),
            yfinance_ticker="AAPL",
            period="5y",
            mode="incremental",
        )

        assert "profile" in result["skipped"]
        yf_client.fetch_info.assert_not_called()


# ---------------------------------------------------------------------------
# Price history — lines 307-366
# ---------------------------------------------------------------------------


class TestPriceHistoryGuards:
    def test_when_history_insufficient_and_existing_prices_then_skip(self):
        """Lines 328-348: short history + prior data → skip, counts['prices']=0.

        mode="full" means staleness is never fetched upfront (staleness stays None).
        Inside the price guard, staleness is None so the secondary repo call fires.
        That secondary call returns a price_max_date → existing prices → skip.
        """
        repo = _minimal_repo()
        yf_client = _minimal_yf_client()
        yf_client.fetch_history.return_value = _small_history()

        # First call: upfront staleness query in incremental mode — not called in full.
        # The only get_staleness_info call is the secondary one inside the guard.
        secondary_staleness = {"price_max_date": "2025-01-01"}
        repo.get_staleness_info.return_value = secondary_staleness

        with patch(
            "app.services.market_data.yfinance_data_service.has_sufficient_history",
            return_value=(False, 1200, 1140),
        ):
            service = _service(repo=repo, yf_client=yf_client)
            result = service.fetch_and_store(
                instrument_id=uuid4(),
                yfinance_ticker="AAPL",
                period="5y",
                mode="full",  # staleness=None upfront → secondary call fires
                exchange_name="NASDAQ",
            )

        assert "prices:insufficient_history" in result["skipped"]
        assert result["counts"].get("prices", 0) == 0
        repo.upsert_price_history.assert_not_called()

    def test_when_history_insufficient_and_no_existing_prices_then_store(self):
        """Lines 350-360: short history + no prior data → young listing, store.

        Secondary get_staleness_info returns price_max_date=None → no existing prices
        → short history is stored as a young listing.
        """
        repo = _minimal_repo()
        yf_client = _minimal_yf_client()
        yf_client.fetch_history.return_value = _small_history()
        # Secondary call returns no existing prices
        repo.get_staleness_info.return_value = {"price_max_date": None}

        with patch(
            "app.services.market_data.yfinance_data_service.has_sufficient_history",
            return_value=(False, 1200, 1140),
        ):
            service = _service(repo=repo, yf_client=yf_client)
            result = service.fetch_and_store(
                instrument_id=uuid4(),
                yfinance_ticker="AAPL",
                period="5y",
                mode="full",
                exchange_name="NASDAQ",
            )

        # Short history stored because no existing prices → young listing
        repo.upsert_price_history.assert_called_once()
        assert result["counts"]["prices"] == 5

    def test_when_history_none_then_prices_count_defaults_to_zero(self):
        """Line 366: history is None → counts['prices'] = counts.get('prices', 0)."""
        repo = _minimal_repo()
        yf_client = _minimal_yf_client()
        yf_client.fetch_history.return_value = None

        service = _service(repo=repo, yf_client=yf_client)
        result = _run(service)

        assert result["counts"].get("prices", 0) == 0
        repo.upsert_price_history.assert_not_called()

    def test_when_history_raises_then_error_appended(self):
        """Price history exception → error appended, rest continues."""
        repo = _minimal_repo()
        yf_client = _minimal_yf_client()
        yf_client.fetch_history.side_effect = ConnectionError("network error")

        service = _service(repo=repo, yf_client=yf_client)
        result = _run(service)

        assert any("prices:" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Financials outer exception handler — lines 437-439
# ---------------------------------------------------------------------------


class TestFinancialsOuterException:
    def test_when_financials_loop_raises_outer_exception_then_error_appended(self):
        """Lines 437-439: unexpected outer exception in financials block."""
        repo = _minimal_repo()
        yf_client = _minimal_yf_client()

        # Make the entire financials sub-client explode at attribute access
        yf_client.financials = MagicMock()
        # Make iteration itself raise via the income_stmt fetch raising AttributeError
        # that propagates outside the inner try — simulate by making the loop-level
        # iterator raise. Easiest: patch the financials client to raise on access.
        yf_client.financials.fetch_income_stmt.side_effect = [
            Exception("db gone"),
            Exception("db gone"),
        ]
        yf_client.financials.fetch_balance_sheet.side_effect = Exception("outer")
        yf_client.financials.fetch_cashflow.side_effect = Exception("outer")

        service = _service(repo=repo, yf_client=yf_client)
        result = _run(service)

        # Inner-loop exceptions are caught per statement type, outer block handles rest
        assert any("financials" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Empty-data guards for dividends, splits, recs, holders (lines 530,556,582,638,668,698)
# ---------------------------------------------------------------------------


class TestEmptyDataGuards:
    def _empty_df(self) -> pd.DataFrame:
        return pd.DataFrame()

    def test_dividends_empty_df_sets_count_zero(self):
        """Line 530: dividends DataFrame is empty → counts['dividends'] = 0."""
        repo = _minimal_repo()
        yf_client = _minimal_yf_client()
        yf_client.corporate_actions.fetch_dividends.return_value = self._empty_df()

        service = _service(repo=repo, yf_client=yf_client)
        result = _run(service)

        assert result["counts"].get("dividends", 0) == 0
        repo.upsert_dividends.assert_not_called()

    def test_splits_empty_df_sets_count_zero(self):
        """Line 556: splits DataFrame is empty → counts['splits'] = 0."""
        repo = _minimal_repo()
        yf_client = _minimal_yf_client()
        yf_client.corporate_actions.fetch_splits.return_value = self._empty_df()

        service = _service(repo=repo, yf_client=yf_client)
        result = _run(service)

        assert result["counts"].get("splits", 0) == 0
        repo.upsert_splits.assert_not_called()

    def test_recommendations_empty_df_sets_count_zero(self):
        """Line 582: recommendations empty → counts['recommendations'] = 0."""
        repo = _minimal_repo()
        yf_client = _minimal_yf_client()
        yf_client.analysis.fetch_recommendations_summary.return_value = self._empty_df()

        service = _service(repo=repo, yf_client=yf_client)
        result = _run(service)

        assert result["counts"].get("recommendations", 0) == 0

    def test_institutional_holders_empty_df_sets_count_zero(self):
        """Line 638: institutional_holders empty → count = 0."""
        repo = _minimal_repo()
        yf_client = _minimal_yf_client()
        yf_client.holders.fetch_institutional_holders.return_value = self._empty_df()

        service = _service(repo=repo, yf_client=yf_client)
        result = _run(service)

        assert result["counts"].get("institutional_holders", 0) == 0

    def test_mutualfund_holders_empty_df_sets_count_zero(self):
        """Line 668: mutualfund_holders empty → count = 0."""
        repo = _minimal_repo()
        yf_client = _minimal_yf_client()
        yf_client.holders.fetch_mutualfund_holders.return_value = self._empty_df()

        service = _service(repo=repo, yf_client=yf_client)
        result = _run(service)

        assert result["counts"].get("mutualfund_holders", 0) == 0

    def test_insider_transactions_empty_df_sets_count_zero(self):
        """Line 698: insider_transactions empty → count = 0."""
        repo = _minimal_repo()
        yf_client = _minimal_yf_client()
        yf_client.holders.fetch_insider_transactions.return_value = self._empty_df()

        service = _service(repo=repo, yf_client=yf_client)
        result = _run(service)

        assert result["counts"].get("insider_transactions", 0) == 0


# ---------------------------------------------------------------------------
# News article parsing — lines 721-737
# ---------------------------------------------------------------------------


class TestNewsArticleParsing:
    """Lines 721-737: canonical URL dict, previewUrl fallback, link fallback."""

    def _make_ticker_with_news(self, articles: list[dict]) -> MagicMock:
        ticker = MagicMock()
        ticker.news = articles
        return ticker

    def _build_service(self, ticker_mock: MagicMock) -> Any:
        repo = _minimal_repo()
        yf_client = _minimal_yf_client()
        yf_client.get_ticker.return_value = ticker_mock
        return _service(repo=repo, yf_client=yf_client), repo

    def test_canonical_url_dict_is_used_as_link(self):
        """Lines 724-728: content has canonicalUrl dict → use its 'url' key."""
        article = {
            "content": {
                "canonicalUrl": {"url": "https://example.com/article"},
                "previewUrl": "https://preview.com/article",
            }
        }
        ticker = self._make_ticker_with_news([article])
        svc, repo = self._build_service(ticker)

        with patch(
            "app.services.market_data.yfinance.news.scraper.ArticleScraper"
        ) as MockScraper:
            mock_scraper = MockScraper.return_value
            mock_scraper.fetch.return_value = {"success": True, "content": "body text"}
            _run(svc)

        mock_scraper.fetch.assert_called_once_with("https://example.com/article")

    def test_preview_url_used_when_canonical_is_not_a_dict(self):
        """Line 729: canonicalUrl is a string (not dict) → fall back to previewUrl."""
        article = {
            "content": {
                "canonicalUrl": "https://not-a-dict.com",  # string, not dict
                "previewUrl": "https://preview.com/story",
            }
        }
        ticker = self._make_ticker_with_news([article])
        svc, repo = self._build_service(ticker)

        with patch(
            "app.services.market_data.yfinance.news.scraper.ArticleScraper"
        ) as MockScraper:
            mock_scraper = MockScraper.return_value
            mock_scraper.fetch.return_value = {"success": True, "content": "text"}
            _run(svc)

        mock_scraper.fetch.assert_called_once_with("https://preview.com/story")

    def test_article_link_used_when_no_canonical_or_preview(self):
        """Lines 730-731: no content sub-dict → fall back to article['link']."""
        article = {
            "content": {"canonicalUrl": None, "previewUrl": None},
            "link": "https://fallback.com/news",
        }
        ticker = self._make_ticker_with_news([article])
        svc, repo = self._build_service(ticker)

        with patch(
            "app.services.market_data.yfinance.news.scraper.ArticleScraper"
        ) as MockScraper:
            mock_scraper = MockScraper.return_value
            mock_scraper.fetch.return_value = {"success": True, "content": "body"}
            _run(svc)

        mock_scraper.fetch.assert_called_once_with("https://fallback.com/news")

    def test_when_scraper_returns_no_content_then_full_content_not_set(self):
        """Lines 731-732: scraper success=True but content empty → no full_content."""
        article = {
            "content": {"canonicalUrl": {"url": "https://example.com/a"}},
        }
        ticker = self._make_ticker_with_news([article])
        svc, repo = self._build_service(ticker)

        with patch(
            "app.services.market_data.yfinance.news.scraper.ArticleScraper"
        ) as MockScraper:
            mock_scraper = MockScraper.return_value
            mock_scraper.fetch.return_value = {"success": True, "content": ""}
            _run(svc)

        assert "full_content" not in article


# ---------------------------------------------------------------------------
# Serial bulk fetch — lines 798-799, 815
# ---------------------------------------------------------------------------


class TestSerialBulkFetch:
    """run_bulk_yfinance_fetch serial path."""

    @contextmanager
    def _fake_session_ctx(self, session: MagicMock):
        yield session

    def _make_instrument(self, *, yfinance_ticker: str | None) -> MagicMock:
        inst = MagicMock()
        inst.id = uuid4()
        inst.yfinance_ticker = yfinance_ticker
        inst.exchange_name = "NASDAQ"
        inst.currency_code = "USD"
        return inst

    def test_when_instrument_has_no_ticker_then_error_appended_and_continues(self):
        """Lines 798-799: instrument.yfinance_ticker is None/empty → error appended."""
        from app.services.market_data.yfinance_data_service import (
            run_bulk_yfinance_fetch,
        )

        session = MagicMock()
        inst_no_ticker = self._make_instrument(yfinance_ticker=None)

        request = MagicMock()
        request.workers = 1
        request.period = "5y"
        request.mode = "full"

        # _settings is a *local* import inside run_bulk_yfinance_fetch;
        # patch the canonical module attribute app.config.settings instead.
        with (
            patch("app.database.database_manager") as db_mgr,
            patch(
                "app.services.market_data.yfinance_data_service.YFinanceRepository"
            ) as MockRepo,
            patch("app.services.market_data.yfinance_data_service.YFinanceDataService"),
            patch("app.config.settings") as mock_settings,
        ):
            db_mgr.get_session.return_value = self._fake_session_ctx(session)
            mock_repo_inst = MockRepo.return_value
            mock_repo_inst.get_instruments_with_yfinance_ticker.return_value = [
                inst_no_ticker
            ]
            mock_settings.yfinance_request_timeout_seconds = 30

            result = run_bulk_yfinance_fetch(request, MagicMock())

        assert result["error_count"] >= 1

    def test_when_fetch_raises_then_rollback_called_and_continues(self):
        """Line 815: per-ticker exception triggers session.rollback() and continues."""
        from app.services.market_data.yfinance_data_service import (
            run_bulk_yfinance_fetch,
        )

        session = MagicMock()
        good_inst = self._make_instrument(yfinance_ticker="AAPL")

        request = MagicMock()
        request.workers = 1
        request.period = "5y"
        request.mode = "full"

        with (
            patch("app.database.database_manager") as db_mgr,
            patch(
                "app.services.market_data.yfinance_data_service.YFinanceRepository"
            ) as MockRepo,
            patch(
                "app.services.market_data.yfinance_data_service.YFinanceDataService"
            ) as MockSvc,
            patch("app.config.settings") as mock_settings,
        ):
            db_mgr.get_session.return_value = self._fake_session_ctx(session)
            mock_repo_inst = MockRepo.return_value
            mock_repo_inst.get_instruments_with_yfinance_ticker.return_value = [
                good_inst
            ]
            mock_svc_inst = MockSvc.return_value
            mock_svc_inst.fetch_and_store.side_effect = RuntimeError("crash")
            mock_settings.yfinance_request_timeout_seconds = 30

            result = run_bulk_yfinance_fetch(request, MagicMock())

        session.rollback.assert_called()
        assert result["error_count"] >= 1


# ---------------------------------------------------------------------------
# _fetch_one_ticker — line 880
# ---------------------------------------------------------------------------


class TestFetchOneTickerMissingTicker:
    def test_when_spec_has_empty_ticker_then_raises_value_error(self):
        """Line 880: empty yfinance_ticker → ValueError raised."""
        from app.services.market_data.yfinance_data_service import (
            _fetch_one_ticker,
            _InstrumentSpec,
        )

        spec = _InstrumentSpec(
            instrument_id=uuid4(),
            yfinance_ticker="",  # empty → should raise
            exchange_name=None,
            currency_code=None,
        )
        request = MagicMock(period="5y", mode="full")

        with pytest.raises(ValueError, match="missing yfinance_ticker"):
            _fetch_one_ticker(spec, request, MagicMock(), MagicMock())


# ---------------------------------------------------------------------------
# _aggregate_outcome — line 916 (skipped count)
# ---------------------------------------------------------------------------


class TestAggregateOutcome:
    def test_when_outcome_has_skipped_then_count_returned(self):
        """Line 916: skipped list length is returned as int."""
        from app.services.market_data.yfinance_data_service import (
            _aggregate_outcome,
            _InstrumentSpec,
        )

        spec = _InstrumentSpec(
            instrument_id=uuid4(),
            yfinance_ticker="SPY",
            exchange_name=None,
            currency_code=None,
        )
        outcome = {
            "counts": {"prices": 100, "profile": 1},
            "errors": [],
            "skipped": ["dividends", "splits", "recommendations"],
        }
        counts: dict[str, int] = {}
        errors: list[str] = []

        skipped_count = _aggregate_outcome(spec, outcome, counts, errors)

        assert skipped_count == 3
        assert counts["prices"] == 100
        assert counts["profile"] == 1

    def test_when_outcome_is_exception_then_error_appended_and_zero_returned(self):
        """Lines 910-911: Exception outcome → appended to errors, returns 0."""
        from app.services.market_data.yfinance_data_service import (
            _aggregate_outcome,
            _InstrumentSpec,
        )

        spec = _InstrumentSpec(
            instrument_id=uuid4(),
            yfinance_ticker="FAIL",
            exchange_name=None,
            currency_code=None,
        )
        counts: dict[str, int] = {}
        errors: list[str] = []

        skipped_count = _aggregate_outcome(spec, RuntimeError("boom"), counts, errors)

        assert skipped_count == 0
        assert any("FAIL" in e for e in errors)


# ---------------------------------------------------------------------------
# reference_index_seeder — lines 107-119, 134, 137
# ---------------------------------------------------------------------------


@contextmanager
def _fake_session_ctx(session: MagicMock):
    yield session


class TestReferenceIndexSeederInstrumentCreation:
    """Lines 107-119: instrument is None → create new Instrument, flush, increment."""

    def _build_nyse_session(self, nyse_id: int = 1) -> MagicMock:
        session = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = nyse_id
        return session

    def test_when_instrument_missing_then_new_row_is_created_and_flushed(self):
        """Lines 107-119: repo returns None → Instrument constructed and session.flush called."""
        from app.services.market_data import reference_index_seeder as mod

        session = self._build_nyse_session()
        repo = MagicMock()
        repo.get_instrument_by_yfinance_ticker.return_value = None  # not found

        # fetch_and_store returns minimal valid result
        service = MagicMock()
        service.fetch_and_store.return_value = {"counts": {"prices": 10}, "errors": []}

        with (
            patch("app.database.database_manager") as db_mgr,
            patch.object(mod, "YFinanceRepository", return_value=repo),
            patch.object(mod, "YFinanceDataService", return_value=service),
            patch.object(mod, "Instrument") as MockInstrument,
        ):
            db_mgr.get_session.return_value = _fake_session_ctx(session)
            mock_instrument_instance = MagicMock()
            mock_instrument_instance.id = uuid4()
            MockInstrument.return_value = mock_instrument_instance

            result = mod.seed_reference_indices(["SPY"], yf_client=MagicMock())

        session.add.assert_called_once_with(mock_instrument_instance)
        session.flush.assert_called_once()
        assert result.instruments_created == 1

    def test_when_instrument_exists_then_already_present_incremented(self):
        """Line 121: repo returns existing instrument → instruments_already_present += 1."""
        from app.services.market_data import reference_index_seeder as mod

        session = self._build_nyse_session()
        existing_instrument = MagicMock()
        existing_instrument.id = uuid4()
        repo = MagicMock()
        repo.get_instrument_by_yfinance_ticker.return_value = existing_instrument

        service = MagicMock()
        service.fetch_and_store.return_value = {"counts": {"prices": 5}, "errors": []}

        with (
            patch("app.database.database_manager") as db_mgr,
            patch.object(mod, "YFinanceRepository", return_value=repo),
            patch.object(mod, "YFinanceDataService", return_value=service),
        ):
            db_mgr.get_session.return_value = _fake_session_ctx(session)

            result = mod.seed_reference_indices(["SPY"], yf_client=MagicMock())

        assert result.instruments_already_present == 1
        assert result.instruments_created == 0

    def test_fetch_result_counts_are_accumulated_into_result(self):
        """Line 134: fetch_result['counts'] merged into result.counts."""
        from app.services.market_data import reference_index_seeder as mod

        session = self._build_nyse_session()
        existing = MagicMock()
        existing.id = uuid4()
        repo = MagicMock()
        repo.get_instrument_by_yfinance_ticker.return_value = existing

        service = MagicMock()
        service.fetch_and_store.return_value = {
            "counts": {"prices": 250, "profile": 1},
            "errors": [],
        }

        with (
            patch("app.database.database_manager") as db_mgr,
            patch.object(mod, "YFinanceRepository", return_value=repo),
            patch.object(mod, "YFinanceDataService", return_value=service),
        ):
            db_mgr.get_session.return_value = _fake_session_ctx(session)

            result = mod.seed_reference_indices(["SPY"], yf_client=MagicMock())

        assert result.counts["prices"] == 250
        assert result.counts["profile"] == 1

    def test_fetch_result_errors_are_accumulated_into_result(self):
        """Line 137: fetch_result['errors'] appended to all_errors → result.errors."""
        from app.services.market_data import reference_index_seeder as mod

        session = self._build_nyse_session()
        existing = MagicMock()
        existing.id = uuid4()
        repo = MagicMock()
        repo.get_instrument_by_yfinance_ticker.return_value = existing

        service = MagicMock()
        service.fetch_and_store.return_value = {
            "counts": {},
            "errors": ["profile: timeout", "prices: empty"],
        }

        with (
            patch("app.database.database_manager") as db_mgr,
            patch.object(mod, "YFinanceRepository", return_value=repo),
            patch.object(mod, "YFinanceDataService", return_value=service),
        ):
            db_mgr.get_session.return_value = _fake_session_ctx(session)

            result = mod.seed_reference_indices(["SPY"], yf_client=MagicMock())

        assert any("profile: timeout" in e for e in result.errors)
        assert any("prices: empty" in e for e in result.errors)

    def test_when_ticker_raises_during_seed_then_rollback_and_continues(self):
        """Lines 141-147: per-ticker exception → rollback, error recorded, next ticker."""
        from app.services.market_data import reference_index_seeder as mod

        session = self._build_nyse_session()
        repo = MagicMock()
        repo.get_instrument_by_yfinance_ticker.side_effect = RuntimeError("db down")

        service = MagicMock()

        with (
            patch("app.database.database_manager") as db_mgr,
            patch.object(mod, "YFinanceRepository", return_value=repo),
            patch.object(mod, "YFinanceDataService", return_value=service),
        ):
            db_mgr.get_session.return_value = _fake_session_ctx(session)

            result = mod.seed_reference_indices(["SPY", "QQQ"], yf_client=MagicMock())

        session.rollback.assert_called()
        assert len(result.errors) >= 1
