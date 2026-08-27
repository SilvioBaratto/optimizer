"""Branch coverage for YFinanceDataService residual paths.

Targets: app/services/market_data/yfinance_data_service.py

Branches covered:
  _is_fresh()
    - updated_at is None → False
    - updated_at naive (no tzinfo) → normalised to UTC before comparison
    - now naive → normalised to UTC before comparison

  YFinanceDataService._timed()
    - request_timeout set → delegates to call_with_timeout

  fetch_and_store() — incremental mode staleness query failure
    - get_staleness_info raises → staleness=None, falls back to full fetch

  fetch_and_store() — incremental price fetch path (start/end date-ranged)
    - mode=incremental, price_max_date exists → uses start= parameter

  fetch_and_store() — insufficient history protection branches
    - insufficient history + existing prices → skip, append prices:insufficient_history
    - insufficient history + no existing prices (young listing) → accept short history

  fetch_and_store() — news scraper branches
    - news list non-empty, link from canonicalUrl dict
    - news list non-empty, link from previewUrl fallback
    - scraper returns success=False → full_content not injected

  _aggregate_outcome()
    - outcome is Exception → appended to errors, returns 0 skipped

  run_bulk_yfinance_fetch() — missing yfinance_ticker guard
    - ticker is None/empty → skipped with error

All external I/O is mocked. No DB sessions are opened.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd

from app.services.market_data.yfinance_data_service import (
    YFinanceDataService,
    _aggregate_outcome,
    _InstrumentSpec,
    _is_fresh,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_repo(*, staleness: dict[str, Any] | None = None) -> MagicMock:
    repo = MagicMock(name="repo")
    repo.get_staleness_info.return_value = staleness
    repo.upsert_profile.return_value = 0
    repo.upsert_price_history.return_value = 0
    repo.upsert_financial_statements.return_value = 0
    repo.upsert_dividends.return_value = 0
    repo.upsert_splits.return_value = 0
    repo.upsert_recommendations.return_value = 0
    repo.upsert_price_targets.return_value = 0
    repo.upsert_institutional_holders.return_value = 0
    repo.upsert_mutualfund_holders.return_value = 0
    repo.upsert_insider_transactions.return_value = 0
    repo.upsert_news.return_value = 3
    return repo


def _make_yf_client_silent() -> MagicMock:
    yf_client = MagicMock(name="yf_client")
    yf_client.fetch_info.return_value = None
    yf_client.fetch_history.return_value = None
    yf_client.financials.fetch_income_stmt.return_value = None
    yf_client.financials.fetch_balance_sheet.return_value = None
    yf_client.financials.fetch_cashflow.return_value = None
    yf_client.corporate_actions.fetch_dividends.return_value = None
    yf_client.corporate_actions.fetch_splits.return_value = None
    yf_client.analysis.fetch_recommendations_summary.return_value = None
    yf_client.analysis.fetch_analyst_price_targets.return_value = None
    yf_client.analysis.fetch_eps_trend.return_value = None
    yf_client.analysis.fetch_eps_revisions.return_value = None
    yf_client.holders.fetch_institutional_holders.return_value = None
    yf_client.holders.fetch_mutualfund_holders.return_value = None
    yf_client.holders.fetch_insider_transactions.return_value = None
    yf_client.metadata.fetch_valuation_measures.return_value = None
    yf_client.analysis.fetch_earnings_estimate.return_value = None
    yf_client.analysis.fetch_revenue_estimate.return_value = None
    yf_client.analysis.fetch_growth_estimates.return_value = None
    ticker_mock = MagicMock()
    ticker_mock.news = []
    yf_client.get_ticker.return_value = ticker_mock
    return yf_client


class TestValuationEpsStalenessGate:
    """valuation/eps_trend/eps_revisions are staleness-gated on the shared
    financial_statements table (financials_updated_at) — no longer refetched every run."""

    def test_skipped_when_financials_fresh(self) -> None:
        now = datetime.now(timezone.utc)
        repo = _make_repo(
            staleness={"financials_updated_at": now, "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        result = _run(repo, yf, mode="incremental")
        assert "valuation_measures" in result["skipped"]
        assert "eps_trend" in result["skipped"]
        assert "eps_revisions" in result["skipped"]
        yf.metadata.fetch_valuation_measures.assert_not_called()
        yf.analysis.fetch_eps_trend.assert_not_called()
        yf.analysis.fetch_eps_revisions.assert_not_called()

    def test_fetched_when_financials_stale(self) -> None:
        repo = _make_repo(
            staleness={"financials_updated_at": None, "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        _run(repo, yf, mode="incremental")
        yf.metadata.fetch_valuation_measures.assert_called_once()
        yf.analysis.fetch_eps_trend.assert_called_once()
        yf.analysis.fetch_eps_revisions.assert_called_once()


class TestPriceUnit:
    """GBp/sub-unit prices (SPEC OQ2): the listing currency is stored as-is on
    price rows so the fx layer converts — no lossy transform at ingest."""

    def test_price_history_model_has_price_unit_column(self) -> None:
        from app.models.market_data.yfinance_data import PriceHistory

        assert hasattr(PriceHistory, "price_unit")

    def test_price_unit_passed_from_currency_code(self) -> None:
        hist = pd.DataFrame(
            {
                "Open": [1.0],
                "High": [1.0],
                "Low": [1.0],
                "Close": [1.0],
                "Volume": [10],
                "Dividends": [0.0],
                "Stock Splits": [0.0],
            },
            index=pd.to_datetime(["2024-01-02"]),
        )
        repo = _make_repo(
            staleness={
                "price_max_date": date(2024, 1, 1),
                "financials_updated_at": None,
            }
        )
        yf = _make_yf_client_silent()
        yf.fetch_history.return_value = hist
        _run(repo, yf, mode="incremental", currency_code="GBX")
        assert repo.upsert_price_history.call_args.kwargs.get("price_unit") == "GBX"


class TestAnalystEstimates:
    """earnings/revenue/growth estimates: dedicated tables, staleness-gated on
    the shared fundamentals window (SPEC A1 / OQ5)."""

    def test_models_have_expected_columns(self) -> None:
        from app.models.market_data.yfinance_data import (
            EarningsEstimate,
            GrowthEstimate,
            RevenueEstimate,
        )

        for col in ("period", "num_analysts", "avg", "low", "high", "growth"):
            assert hasattr(EarningsEstimate, col)
        assert hasattr(EarningsEstimate, "year_ago_eps")
        assert hasattr(RevenueEstimate, "year_ago_revenue")
        for col in ("stock_trend", "industry_trend", "sector_trend", "index_trend"):
            assert hasattr(GrowthEstimate, col)

    def test_skipped_when_financials_fresh(self) -> None:
        now = datetime.now(timezone.utc)
        repo = _make_repo(
            staleness={"financials_updated_at": now, "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        result = _run(repo, yf, mode="incremental")
        assert "earnings_estimate" in result["skipped"]
        assert "revenue_estimate" in result["skipped"]
        assert "growth_estimates" in result["skipped"]
        yf.analysis.fetch_earnings_estimate.assert_not_called()

    def test_persisted_when_stale(self) -> None:
        repo = _make_repo(
            staleness={"financials_updated_at": None, "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        est = pd.DataFrame(
            {"numberOfAnalysts": [5], "avg": [1.2], "low": [1.0], "high": [1.5]},
            index=["0q"],
        )
        yf.analysis.fetch_earnings_estimate.return_value = est
        yf.analysis.fetch_revenue_estimate.return_value = est
        yf.analysis.fetch_growth_estimates.return_value = pd.DataFrame(
            {"stockTrend": [0.1], "indexTrend": [0.08]}, index=["0q"]
        )
        _run(repo, yf, mode="incremental")
        repo.upsert_earnings_estimate.assert_called_once()
        repo.upsert_revenue_estimate.assert_called_once()
        repo.upsert_growth_estimates.assert_called_once()


def _run(
    repo: MagicMock,
    yf_client: MagicMock,
    *,
    mode: str = "full",
    exchange_name: str | None = None,
    currency_code: str | None = "USD",
    period: str = "5y",
) -> dict[str, Any]:
    service = YFinanceDataService(repo=repo, yf_client=yf_client)
    return service.fetch_and_store(
        instrument_id=uuid4(),
        yfinance_ticker="AAPL",
        period=period,
        mode=mode,
        exchange_name=exchange_name,
        currency_code=currency_code,
    )


def _stale_history() -> pd.DataFrame:
    """Return a minimal price history DataFrame with enough rows."""
    idx = pd.date_range("2020-01-01", periods=800, freq="B")
    return pd.DataFrame({"Close": [100.0] * 800}, index=idx)


def _short_history(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": [50.0] * n}, index=idx)


# ===========================================================================
# _is_fresh() — all branches
# ===========================================================================


class TestIsFresh:
    def test_when_updated_at_is_none_returns_false(self) -> None:
        now = datetime.now(timezone.utc)
        assert _is_fresh(None, 24, now) is False

    def test_when_both_aware_and_recent_returns_true(self) -> None:
        now = datetime.now(timezone.utc)
        updated_at = now - timedelta(hours=1)
        assert _is_fresh(updated_at, 24, now) is True

    def test_when_both_aware_and_stale_returns_false(self) -> None:
        now = datetime.now(timezone.utc)
        updated_at = now - timedelta(hours=25)
        assert _is_fresh(updated_at, 24, now) is False

    def test_when_updated_at_naive_then_normalised_to_utc(self) -> None:
        # Covers the `if updated_at.tzinfo is None` branch
        now = datetime.now(timezone.utc)
        naive_updated_at = datetime.utcnow() - timedelta(hours=1)
        # naive → treated as UTC → 1h ago < 24h threshold → fresh
        assert _is_fresh(naive_updated_at, 24, now) is True

    def test_when_now_naive_then_normalised_to_utc(self) -> None:
        # Covers the `if now.tzinfo is None` branch
        aware_updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
        naive_now = datetime.utcnow()
        assert _is_fresh(aware_updated_at, 24, naive_now) is True

    def test_when_both_naive_and_recent_returns_true(self) -> None:
        naive_now = datetime.utcnow()
        naive_updated = naive_now - timedelta(hours=2)
        assert _is_fresh(naive_updated, 24, naive_now) is True


# ===========================================================================
# YFinanceDataService._timed() — timeout path
# ===========================================================================


class TestTimedMethod:
    def test_when_request_timeout_set_delegates_to_call_with_timeout(self) -> None:
        repo = _make_repo()
        yf_client = _make_yf_client_silent()
        service = YFinanceDataService(
            repo=repo, yf_client=yf_client, request_timeout=30.0
        )

        captured: list[Any] = []

        def _fake_call_with_timeout(fn: Any, timeout: float) -> Any:
            captured.append(timeout)
            return fn()

        with patch(
            "app.services.market_data.yfinance_data_service.call_with_timeout",
            side_effect=_fake_call_with_timeout,
        ):
            service.fetch_and_store(
                instrument_id=uuid4(),
                yfinance_ticker="AAPL",
                mode="full",
            )

        assert 30.0 in captured

    def test_when_request_timeout_none_calls_fn_directly(self) -> None:
        repo = _make_repo()
        yf_client = _make_yf_client_silent()
        service = YFinanceDataService(
            repo=repo, yf_client=yf_client, request_timeout=None
        )

        with patch(
            "app.services.market_data.yfinance_data_service.call_with_timeout"
        ) as mock_cwt:
            service.fetch_and_store(
                instrument_id=uuid4(),
                yfinance_ticker="AAPL",
                mode="full",
            )

        mock_cwt.assert_not_called()


# ===========================================================================
# fetch_and_store() — incremental staleness query failure
# ===========================================================================


class TestIncrementalStalenessQueryFailure:
    def test_when_get_staleness_info_raises_then_staleness_is_none_full_fetch(
        self,
    ) -> None:
        repo = _make_repo()
        repo.get_staleness_info.side_effect = RuntimeError("db connection lost")
        yf_client = _make_yf_client_silent()

        result = _run(repo, yf_client, mode="incremental")

        # Falls back to full fetch — profile is attempted
        yf_client.fetch_info.assert_called_once()
        assert not any("staleness" in e for e in result["errors"])


# ===========================================================================
# fetch_and_store() — incremental price fetch via start/end
# ===========================================================================


class TestIncrementalPriceFetch:
    def test_when_price_max_date_exists_fetch_uses_start_end_parameters(self) -> None:
        existing_max = date(2025, 1, 15)
        staleness = {
            "profile_updated_at": None,
            "financials_updated_at": None,
            "dividends_updated_at": None,
            "splits_updated_at": None,
            "recommendations_updated_at": None,
            "price_targets_updated_at": None,
            "institutional_holders_updated_at": None,
            "mutualfund_holders_updated_at": None,
            "insider_transactions_updated_at": None,
            "price_max_date": existing_max,
        }
        repo = _make_repo(staleness=staleness)
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.return_value = None

        _run(repo, yf_client, mode="incremental")

        assert yf_client.fetch_history.call_count == 1
        call_kwargs = yf_client.fetch_history.call_args
        # Should have used start= rather than period=
        args_kwargs = call_kwargs[1] if call_kwargs[1] else {}
        all_args = {**args_kwargs}
        assert "start" in all_args or "period" not in all_args


# ===========================================================================
# fetch_and_store() — insufficient history protection
# ===========================================================================


class TestInsufficientHistoryBranches:
    def test_when_insufficient_history_and_existing_prices_then_skipped(self) -> None:
        """Short fetch + prior price data → protect existing series, skip upsert."""
        staleness = {
            "profile_updated_at": None,
            "financials_updated_at": None,
            "dividends_updated_at": None,
            "splits_updated_at": None,
            "recommendations_updated_at": None,
            "price_targets_updated_at": None,
            "institutional_holders_updated_at": None,
            "mutualfund_holders_updated_at": None,
            "insider_transactions_updated_at": None,
            "price_max_date": None,
        }
        repo = _make_repo(staleness=staleness)
        repo.get_staleness_info.return_value = {"price_max_date": date(2024, 12, 1)}
        yf_client = _make_yf_client_silent()

        # 100 rows is well below the expected minimum for "5y" on NYSE
        yf_client.fetch_history.return_value = _short_history(100)

        with patch(
            "app.services.market_data.yfinance_data_service.has_sufficient_history",
            return_value=(False, 1260, 1197),
        ):
            result = _run(
                repo, yf_client, mode="full", exchange_name="NYSE", period="5y"
            )

        assert "prices:insufficient_history" in result["skipped"]
        repo.upsert_price_history.assert_not_called()

    def test_when_insufficient_history_and_no_prior_prices_then_young_listing_stored(
        self,
    ) -> None:
        """Short fetch + no prior price data → young listing, accept and store."""
        repo = _make_repo(staleness=None)
        # get_staleness_info returns no price_max_date (young listing)
        repo.get_staleness_info.return_value = {"price_max_date": None}
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.return_value = _short_history(80)

        with patch(
            "app.services.market_data.yfinance_data_service.has_sufficient_history",
            return_value=(False, 1260, 1197),
        ):
            result = _run(
                repo, yf_client, mode="full", exchange_name="NYSE", period="5y"
            )

        assert "prices:insufficient_history" not in result["skipped"]
        repo.upsert_price_history.assert_called_once()

    def test_when_staleness_already_set_and_insufficient_history_uses_cached_price_max_date(
        self,
    ) -> None:
        """When staleness is already populated (not None), price_max_date read from it,
        not via a second get_staleness_info call."""
        staleness: dict[str, Any] = {
            "profile_updated_at": None,
            "financials_updated_at": None,
            "dividends_updated_at": None,
            "splits_updated_at": None,
            "recommendations_updated_at": None,
            "price_targets_updated_at": None,
            "institutional_holders_updated_at": None,
            "mutualfund_holders_updated_at": None,
            "insider_transactions_updated_at": None,
            "price_max_date": date(2024, 6, 1),  # existing prices
        }
        repo = _make_repo(staleness=staleness)
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.return_value = _short_history(50)

        with patch(
            "app.services.market_data.yfinance_data_service.has_sufficient_history",
            return_value=(False, 1260, 1197),
        ):
            result = _run(
                repo, yf_client, mode="full", exchange_name="LSE", period="5y"
            )

        # price_max_date=date(2024,6,1) → has_existing_prices=True → skip
        assert "prices:insufficient_history" in result["skipped"]


# ===========================================================================
# fetch_and_store() — news scraper branches
# ===========================================================================


class TestNewsScraperBranches:
    def _make_article_canonical_url(self, url: str) -> dict[str, Any]:
        return {
            "content": {
                "canonicalUrl": {"url": url},
                "previewUrl": "https://preview.example.com/article",
            }
        }

    def _make_article_preview_url(self, url: str) -> dict[str, Any]:
        return {
            "content": {
                "canonicalUrl": None,  # not a dict
                "previewUrl": url,
            }
        }

    def _make_article_link_fallback(self, url: str) -> dict[str, Any]:
        return {
            "link": url,
            "content": {
                "canonicalUrl": None,
                "previewUrl": None,
            },
        }

    def test_when_canonical_url_dict_link_is_used_and_scraper_succeeds(self) -> None:
        repo = _make_repo()
        yf_client = _make_yf_client_silent()
        article = self._make_article_canonical_url("https://example.com/news/1")
        ticker_mock = MagicMock()
        ticker_mock.news = [article]
        yf_client.get_ticker.return_value = ticker_mock

        # ArticleScraper is imported locally inside fetch_and_store — patch at source
        with patch(
            "app.services.market_data.yfinance.news.scraper.ArticleScraper"
        ) as MockScraper:
            scraper_instance = MagicMock()
            scraper_instance.fetch.return_value = {
                "success": True,
                "content": "Full article text",
            }
            MockScraper.return_value = scraper_instance

            result = _run(repo, yf_client, mode="full")

        assert "news" in result["counts"]
        assert result["counts"]["news"] == 3  # repo.upsert_news returns 3
        # Verify scraper was called with the canonical URL
        scraper_instance.fetch.assert_called_once_with("https://example.com/news/1")
        # full_content injected into article
        assert article.get("full_content") == "Full article text"

    def test_when_scraper_returns_success_false_full_content_not_injected(
        self,
    ) -> None:
        repo = _make_repo()
        yf_client = _make_yf_client_silent()
        article = self._make_article_canonical_url("https://example.com/news/2")
        ticker_mock = MagicMock()
        ticker_mock.news = [article]
        yf_client.get_ticker.return_value = ticker_mock

        with patch(
            "app.services.market_data.yfinance.news.scraper.ArticleScraper"
        ) as MockScraper:
            scraper_instance = MagicMock()
            scraper_instance.fetch.return_value = {"success": False, "content": None}
            MockScraper.return_value = scraper_instance

            result = _run(repo, yf_client, mode="full")

        assert "full_content" not in article
        assert result["counts"]["news"] == 3

    def test_when_preview_url_used_when_canonical_not_a_dict(self) -> None:
        repo = _make_repo()
        yf_client = _make_yf_client_silent()
        article = self._make_article_preview_url("https://preview.example.com/story")
        ticker_mock = MagicMock()
        ticker_mock.news = [article]
        yf_client.get_ticker.return_value = ticker_mock

        with patch(
            "app.services.market_data.yfinance.news.scraper.ArticleScraper"
        ) as MockScraper:
            scraper_instance = MagicMock()
            scraper_instance.fetch.return_value = {"success": True, "content": "text"}
            MockScraper.return_value = scraper_instance

            _run(repo, yf_client, mode="full")

        scraper_instance.fetch.assert_called_once_with(
            "https://preview.example.com/story"
        )

    def test_when_no_link_then_scraper_not_called(self) -> None:
        repo = _make_repo()
        yf_client = _make_yf_client_silent()
        # Article with no usable link
        article: dict[str, Any] = {
            "content": {"canonicalUrl": None, "previewUrl": None}
        }
        ticker_mock = MagicMock()
        ticker_mock.news = [article]
        yf_client.get_ticker.return_value = ticker_mock

        with patch(
            "app.services.market_data.yfinance.news.scraper.ArticleScraper"
        ) as MockScraper:
            scraper_instance = MagicMock()
            MockScraper.return_value = scraper_instance
            _run(repo, yf_client, mode="full")

        scraper_instance.fetch.assert_not_called()

    def test_when_news_list_empty_then_counts_news_is_zero(self) -> None:
        repo = _make_repo()
        yf_client = _make_yf_client_silent()
        # news=[] already set by _make_yf_client_silent

        result = _run(repo, yf_client, mode="full")

        assert result["counts"].get("news", 0) == 0


# ===========================================================================
# _aggregate_outcome() — exception branch
# ===========================================================================


class TestAggregateOutcome:
    def _make_spec(self) -> _InstrumentSpec:
        return _InstrumentSpec(
            instrument_id=uuid4(),
            yfinance_ticker="AAPL",
            exchange_name="NASDAQ",
            currency_code="USD",
        )

    def test_when_outcome_is_exception_error_appended_and_returns_zero(self) -> None:
        spec = self._make_spec()
        counts: dict[str, int] = {}
        errors: list[str] = []

        skipped = _aggregate_outcome(spec, RuntimeError("fetch failed"), counts, errors)

        assert skipped == 0
        assert len(errors) == 1
        assert "AAPL" in errors[0]
        assert "fetch failed" in errors[0]
        assert counts == {}

    def test_when_outcome_is_result_dict_counts_are_merged(self) -> None:
        spec = self._make_spec()
        counts: dict[str, int] = {"profile": 1}
        errors: list[str] = []
        outcome = {
            "counts": {"profile": 2, "prices": 5},
            "errors": ["prices: timeout"],
            "skipped": ["financials", "dividends"],
        }

        skipped = _aggregate_outcome(spec, outcome, counts, errors)

        assert counts["profile"] == 3  # 1 + 2
        assert counts["prices"] == 5
        assert skipped == 2
        assert len(errors) == 1
        assert "AAPL: prices: timeout" in errors[0]


# ===========================================================================
# run_bulk_yfinance_fetch() — missing yfinance_ticker guard
# ===========================================================================


class TestRunBulkYfinanceFetchMissingTicker:
    def test_when_instrument_has_no_ticker_then_error_logged_and_job_continues(
        self,
    ) -> None:
        from contextlib import contextmanager
        from types import SimpleNamespace

        import app.services.market_data.yfinance_data_service as svc_module

        instrument_with_ticker = SimpleNamespace(
            id=uuid4(),
            yfinance_ticker="AAPL",
            exchange_name="NASDAQ",
            currency_code="USD",
        )
        instrument_no_ticker = SimpleNamespace(
            id=uuid4(),
            yfinance_ticker=None,
            exchange_name="NYSE",
            currency_code="USD",
        )

        repo_mock = MagicMock()
        repo_mock.get_instruments_with_yfinance_ticker.return_value = [
            instrument_with_ticker,
            instrument_no_ticker,
        ]
        repo_mock.get_staleness_info.return_value = None
        repo_mock.upsert_profile.return_value = 0
        repo_mock.upsert_price_history.return_value = 0
        repo_mock.upsert_financial_statements.return_value = 0
        repo_mock.upsert_dividends.return_value = 0
        repo_mock.upsert_splits.return_value = 0
        repo_mock.upsert_recommendations.return_value = 0
        repo_mock.upsert_price_targets.return_value = 0
        repo_mock.upsert_institutional_holders.return_value = 0
        repo_mock.upsert_mutualfund_holders.return_value = 0
        repo_mock.upsert_insider_transactions.return_value = 0
        repo_mock.upsert_news.return_value = 0

        session_mock = MagicMock()
        db_mgr = MagicMock()

        @contextmanager
        def _get_session():
            yield session_mock

        db_mgr.get_session = _get_session

        request = SimpleNamespace(workers=1, period="5y", mode="full")
        yf_client = _make_yf_client_silent()

        with (
            patch.object(svc_module, "YFinanceRepository", return_value=repo_mock),
            patch("app.database.database_manager", db_mgr),
            patch("app.config.settings") as mock_settings,
        ):
            mock_settings.yfinance_request_timeout_seconds = 30
            result = svc_module.run_bulk_yfinance_fetch(request, yf_client)

        assert result["tickers_processed"] == 2
        assert result["error_count"] >= 1
