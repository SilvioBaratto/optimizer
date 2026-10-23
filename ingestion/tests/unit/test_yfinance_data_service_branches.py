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

from app.services.market_data.empty_fetch_memory import EmptyFetchMemory
from app.services.market_data.yfinance_data_service import (
    StalenessThresholds,
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
    repo.upsert_shares_outstanding.return_value = 0
    repo.upsert_recommendations.return_value = 0
    repo.upsert_price_targets.return_value = 0
    repo.upsert_institutional_holders.return_value = 0
    repo.upsert_mutualfund_holders.return_value = 0
    repo.upsert_insider_transactions.return_value = 0
    repo.upsert_major_holders.return_value = 0
    repo.upsert_insider_purchases.return_value = 0
    repo.upsert_insider_roster.return_value = 0
    repo.upsert_profile_extras.return_value = 0
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
    yf_client.corporate_actions.fetch_shares_full.return_value = None
    yf_client.analysis.fetch_recommendations_summary.return_value = None
    yf_client.analysis.fetch_upgrades_downgrades.return_value = None
    yf_client.financials.fetch_sec_filings.return_value = None
    yf_client.analysis.fetch_analyst_price_targets.return_value = None
    yf_client.analysis.fetch_eps_trend.return_value = None
    yf_client.analysis.fetch_eps_revisions.return_value = None
    yf_client.holders.fetch_institutional_holders.return_value = None
    yf_client.holders.fetch_mutualfund_holders.return_value = None
    yf_client.holders.fetch_insider_transactions.return_value = None
    yf_client.holders.fetch_major_holders.return_value = None
    yf_client.holders.fetch_insider_purchases.return_value = None
    yf_client.holders.fetch_insider_roster_holders.return_value = None
    yf_client.metadata.fetch_valuation_measures.return_value = None
    yf_client.metadata.fetch_isin.return_value = None
    yf_client.analysis.fetch_earnings_estimate.return_value = None
    yf_client.analysis.fetch_revenue_estimate.return_value = None
    yf_client.analysis.fetch_growth_estimates.return_value = None
    yf_client.analysis.fetch_earnings_history.return_value = None
    yf_client.metadata.fetch_earnings_dates.return_value = None
    ticker_mock = MagicMock()
    ticker_mock.news = []
    yf_client.get_ticker.return_value = ticker_mock
    return yf_client


# Categories refreshed on the fundamentals window; each is gated on the
# ``<name>_updated_at`` marker of its own table.
_FUNDAMENTAL_CATEGORIES = (
    "valuation_measures",
    "eps_trend",
    "eps_revisions",
    "earnings_estimate",
    "revenue_estimate",
    "growth_estimates",
    "earnings_history",
    "earnings_dates",
    "sec_filings",
    "shares_outstanding",
)


def _fresh_fundamentals(
    stamp: datetime, *, except_for: tuple[str, ...] = ()
) -> dict[str, Any]:
    return {
        f"{category}_updated_at": stamp
        for category in _FUNDAMENTAL_CATEGORIES
        if category not in except_for
    }


_STATEMENT_GROUPS = (
    "income_statement_annual",
    "income_statement_quarterly",
    "balance_sheet_annual",
    "balance_sheet_quarterly",
    "cashflow_annual",
    "cashflow_quarterly",
)


def _fully_fresh(
    now: datetime,
    *,
    except_for: tuple[str, ...] = (),
    except_groups: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Staleness of a ticker whose every gated category was stored at *now*.

    ``except_for`` leaves fundamentals categories unstored and ``except_groups``
    leaves statement groups unstored (their marker is ``None``).
    """
    staleness: dict[str, Any] = {
        "price_max_date": date.today(),
        "financials_updated_at": None if except_groups else now,
        **{
            f"{group}_updated_at": None if group in except_groups else now
            for group in _STATEMENT_GROUPS
        },
        **_fresh_fundamentals(now, except_for=except_for),
    }
    for category in (
        "profile",
        "dividends",
        "splits",
        "recommendations",
        "price_targets",
        "institutional_holders",
        "mutualfund_holders",
        "insider_transactions",
    ):
        staleness[f"{category}_updated_at"] = now
    return staleness


class TestValuationEpsStalenessGate:
    """valuation/eps_trend/eps_revisions are gated on their own tables' freshness
    markers: a fresh marker skips the fetch, a stale one triggers it."""

    def test_skipped_when_fundamentals_fresh(self) -> None:
        now = datetime.now(timezone.utc)
        repo = _make_repo(
            staleness={**_fresh_fundamentals(now), "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        result = _run(repo, yf, mode="incremental")
        assert "valuation_measures" in result["skipped"]
        assert "eps_trend" in result["skipped"]
        assert "eps_revisions" in result["skipped"]
        yf.metadata.fetch_valuation_measures.assert_not_called()
        yf.analysis.fetch_eps_trend.assert_not_called()
        yf.analysis.fetch_eps_revisions.assert_not_called()

    def test_fetched_when_fundamentals_stale(self) -> None:
        repo = _make_repo(
            staleness={"financials_updated_at": None, "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        _run(repo, yf, mode="incremental")
        yf.metadata.fetch_valuation_measures.assert_called_once()
        yf.analysis.fetch_eps_trend.assert_called_once()
        yf.analysis.fetch_eps_revisions.assert_called_once()


class TestStatementStalenessGate:
    """Each financial statement is gated on its own freshness.

    One stored statement must not make the whole block look fresh: statements
    that failed during a rate-limit burst are retried on the next run instead of
    waiting out the window.
    """

    def _staleness(self, *fresh_groups: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        staleness: dict[str, Any] = {"price_max_date": None}
        for group in fresh_groups:
            staleness[f"{group}_updated_at"] = now
        return staleness

    def test_only_the_stale_statements_are_refetched(self) -> None:
        repo = _make_repo(staleness=self._staleness("income_statement_annual"))
        yf = _make_yf_client_silent()

        result = _run(repo, yf, mode="incremental")

        income_flags = [
            c.kwargs["quarterly"]
            for c in yf.financials.fetch_income_stmt.call_args_list
        ]
        assert income_flags == [True]
        assert yf.financials.fetch_balance_sheet.call_count == 2
        assert yf.financials.fetch_cashflow.call_count == 2
        assert "financials" not in result["skipped"]

    def test_financials_skipped_when_every_statement_is_fresh(self) -> None:
        repo = _make_repo(staleness=self._staleness(*_STATEMENT_GROUPS))
        yf = _make_yf_client_silent()

        result = _run(repo, yf, mode="incremental")

        assert "financials" in result["skipped"]
        yf.financials.fetch_income_stmt.assert_not_called()
        yf.financials.fetch_balance_sheet.assert_not_called()
        yf.financials.fetch_cashflow.assert_not_called()

    def test_full_mode_refetches_every_statement_despite_fresh_markers(self) -> None:
        repo = _make_repo(staleness=self._staleness(*_STATEMENT_GROUPS))
        yf = _make_yf_client_silent()

        _run(repo, yf, mode="full")

        assert yf.financials.fetch_income_stmt.call_count == 2
        assert yf.financials.fetch_balance_sheet.call_count == 2
        assert yf.financials.fetch_cashflow.call_count == 2


class TestFundamentalsCategoryGate:
    """Each fundamentals category is gated on its own table's freshness.

    The gate is independent of ``financials_updated_at``: a category whose fetch
    failed while the statements were stored is retried, and a stored category
    stays skipped even when the ticker has no statements yet.
    """

    def test_ticker_with_every_category_fresh_skips_the_network(self) -> None:
        now = datetime.now(timezone.utc)
        repo = _make_repo(staleness=_fully_fresh(now))
        yf = _make_yf_client_silent()

        result = _run(repo, yf, mode="incremental")

        assert result["skipped"] == ["all:fresh"]
        yf.get_ticker.assert_not_called()

    def test_category_that_failed_is_retried_while_statements_are_fresh(self) -> None:
        now = datetime.now(timezone.utc)
        repo = _make_repo(staleness=_fully_fresh(now, except_for=("eps_trend",)))
        yf = _make_yf_client_silent()

        result = _run(repo, yf, mode="incremental")

        assert "all:fresh" not in result["skipped"]
        assert "eps_trend" not in result["skipped"]
        yf.analysis.fetch_eps_trend.assert_called_once()
        yf.analysis.fetch_eps_revisions.assert_not_called()
        yf.analysis.fetch_earnings_estimate.assert_not_called()
        yf.financials.fetch_sec_filings.assert_not_called()
        yf.corporate_actions.fetch_shares_full.assert_not_called()
        yf.metadata.fetch_valuation_measures.assert_not_called()

    def test_each_category_is_refetched_when_only_it_is_stale(self) -> None:
        now = datetime.now(timezone.utc)
        fetchers = {
            "valuation_measures": lambda yf: yf.metadata.fetch_valuation_measures,
            "eps_trend": lambda yf: yf.analysis.fetch_eps_trend,
            "eps_revisions": lambda yf: yf.analysis.fetch_eps_revisions,
            "earnings_estimate": lambda yf: yf.analysis.fetch_earnings_estimate,
            "revenue_estimate": lambda yf: yf.analysis.fetch_revenue_estimate,
            "growth_estimates": lambda yf: yf.analysis.fetch_growth_estimates,
            "earnings_history": lambda yf: yf.analysis.fetch_earnings_history,
            "earnings_dates": lambda yf: yf.metadata.fetch_earnings_dates,
            "sec_filings": lambda yf: yf.financials.fetch_sec_filings,
            "shares_outstanding": lambda yf: yf.corporate_actions.fetch_shares_full,
        }
        assert set(fetchers) == set(_FUNDAMENTAL_CATEGORIES)

        for stale, pick in fetchers.items():
            repo = _make_repo(staleness=_fully_fresh(now, except_for=(stale,)))
            yf = _make_yf_client_silent()

            _run(repo, yf, mode="incremental")

            pick(yf).assert_called_once()
            for other, pick_other in fetchers.items():
                if other != stale:
                    pick_other(yf).assert_not_called()

    def test_categories_stored_without_statements_are_not_refetched(self) -> None:
        now = datetime.now(timezone.utc)
        staleness = _fully_fresh(now)
        staleness["financials_updated_at"] = None
        for group in _STATEMENT_GROUPS:
            staleness[f"{group}_updated_at"] = None
        repo = _make_repo(staleness=staleness)
        yf = _make_yf_client_silent()

        result = _run(repo, yf, mode="incremental")

        assert yf.financials.fetch_income_stmt.call_count == 2
        for category in _FUNDAMENTAL_CATEGORIES:
            assert category in result["skipped"]
        yf.analysis.fetch_eps_trend.assert_not_called()
        yf.metadata.fetch_valuation_measures.assert_not_called()
        yf.financials.fetch_sec_filings.assert_not_called()


class TestEmptyAnswerMemory:
    """A category Yahoo answers with nothing is not re-requested on every run.

    Freshness markers advance only when rows are written, so a category or
    statement group Yahoo has no data for looks stale forever. The service
    remembers a clean empty answer and treats it as settled for
    ``no_data_hours``; failures and rate-limited answers stay retryable.
    """

    _INSTRUMENT = uuid4()

    def _calm_client(self) -> MagicMock:
        yf = _make_yf_client_silent()
        yf.circuit_breaker.is_active = False
        return yf

    def _fetch(
        self,
        staleness: dict[str, Any],
        yf: MagicMock,
        memory: EmptyFetchMemory,
        *,
        mode: str = "incremental",
        thresholds: StalenessThresholds | None = None,
    ) -> dict[str, Any]:
        service = YFinanceDataService(
            repo=_make_repo(staleness=staleness),
            yf_client=yf,
            empty_fetches=memory,
        )
        return service.fetch_and_store(
            instrument_id=self._INSTRUMENT,
            yfinance_ticker="AAPL",
            mode=mode,
            currency_code="USD",
            thresholds=thresholds,
        )

    def test_empty_category_is_not_requested_on_the_next_run(self) -> None:
        """Skip a category whose previous run came back empty."""
        now = datetime.now(timezone.utc)
        memory = EmptyFetchMemory()
        staleness = _fully_fresh(now, except_for=("eps_trend",))
        staleness["profile_updated_at"] = (
            None  # keeps the ticker past the short-circuit
        )

        self._fetch(staleness, self._calm_client(), memory)
        yf = self._calm_client()
        result = self._fetch(staleness, yf, memory)

        yf.analysis.fetch_eps_trend.assert_not_called()
        assert "eps_trend" in result["skipped"]

    def test_empty_category_is_requested_again_after_the_no_data_window(self) -> None:
        """Retry an empty category once the no-data window has passed."""
        now = datetime.now(timezone.utc)
        memory = EmptyFetchMemory()
        memory.remember(
            self._INSTRUMENT, "eps_trend_updated_at", now - timedelta(hours=73)
        )
        yf = self._calm_client()

        self._fetch(_fully_fresh(now, except_for=("eps_trend",)), yf, memory)

        yf.analysis.fetch_eps_trend.assert_called_once()

    def test_no_data_window_is_taken_from_the_thresholds(self) -> None:
        """Honour a caller-supplied no-data window."""
        now = datetime.now(timezone.utc)
        memory = EmptyFetchMemory()
        staleness = _fully_fresh(now, except_for=("eps_trend",))
        thresholds = StalenessThresholds(no_data_hours=0)

        self._fetch(staleness, self._calm_client(), memory, thresholds=thresholds)
        yf = self._calm_client()
        self._fetch(staleness, yf, memory, thresholds=thresholds)

        yf.analysis.fetch_eps_trend.assert_called_once()

    def test_failed_category_is_requested_again_on_the_next_run(self) -> None:
        """Never remember an exception as an empty answer."""
        now = datetime.now(timezone.utc)
        memory = EmptyFetchMemory()
        staleness = _fully_fresh(now, except_for=("eps_trend",))
        failing = self._calm_client()
        failing.analysis.fetch_eps_trend.side_effect = TimeoutError("hung")

        result = self._fetch(staleness, failing, memory)
        yf = self._calm_client()
        self._fetch(staleness, yf, memory)

        assert any(e.startswith("eps_trend:") for e in result["errors"])
        yf.analysis.fetch_eps_trend.assert_called_once()

    def test_empty_answer_during_a_rate_limit_is_not_remembered(self) -> None:
        """Treat an empty answer as a lost request while the breaker is open."""
        now = datetime.now(timezone.utc)
        memory = EmptyFetchMemory()
        staleness = _fully_fresh(now, except_for=("eps_trend",))
        throttled = self._calm_client()
        throttled.circuit_breaker.is_active = True

        self._fetch(staleness, throttled, memory)
        yf = self._calm_client()
        self._fetch(staleness, yf, memory)

        yf.analysis.fetch_eps_trend.assert_called_once()

    def test_category_that_stored_rows_is_not_remembered_as_empty(self) -> None:
        """Remember only empty answers; stored rows advance the DB marker."""
        now = datetime.now(timezone.utc)
        memory = EmptyFetchMemory()
        yf = self._calm_client()
        yf.analysis.fetch_eps_trend.return_value = pd.DataFrame({"current": [1.0]})
        repo = _make_repo(staleness=_fully_fresh(now, except_for=("eps_trend",)))
        repo.upsert_eps_trend.return_value = 4

        YFinanceDataService(
            repo=repo, yf_client=yf, empty_fetches=memory
        ).fetch_and_store(
            instrument_id=self._INSTRUMENT,
            yfinance_ticker="AAPL",
            mode="incremental",
            currency_code="USD",
        )

        assert not memory.is_recent(self._INSTRUMENT, "eps_trend_updated_at", 72, now)

    def test_full_mode_ignores_remembered_empty_answers(self) -> None:
        """Refetch everything in full mode, whatever the memory holds."""
        now = datetime.now(timezone.utc)
        memory = EmptyFetchMemory()
        memory.remember(self._INSTRUMENT, "eps_trend_updated_at", now)
        yf = self._calm_client()

        self._fetch(_fully_fresh(now), yf, memory, mode="full")

        yf.analysis.fetch_eps_trend.assert_called_once()

    def test_empty_statement_group_is_not_requested_on_the_next_run(self) -> None:
        """Gate each statement group on its own empty answer."""
        now = datetime.now(timezone.utc)
        memory = EmptyFetchMemory()
        staleness = _fully_fresh(now, except_groups=("cashflow_quarterly",))
        staleness["profile_updated_at"] = (
            None  # keeps the ticker past the short-circuit
        )

        self._fetch(staleness, self._calm_client(), memory)
        yf = self._calm_client()
        result = self._fetch(staleness, yf, memory)

        yf.financials.fetch_cashflow.assert_not_called()
        assert "financials" in result["skipped"]

    def test_ticker_left_with_only_empty_answers_skips_the_network(self) -> None:
        """Short-circuit once every remaining gap is a remembered empty answer."""
        now = datetime.now(timezone.utc)
        memory = EmptyFetchMemory()
        staleness = _fully_fresh(
            now, except_for=("eps_trend",), except_groups=("cashflow_quarterly",)
        )

        first = self._fetch(staleness, self._calm_client(), memory)
        yf = self._calm_client()
        second = self._fetch(staleness, yf, memory)

        assert "all:fresh" not in first["skipped"]
        assert second["skipped"] == ["all:fresh"]
        yf.get_ticker.assert_not_called()

    def test_services_do_not_share_memory_unless_it_is_injected(self) -> None:
        """Keep the default memory private to one service instance."""
        now = datetime.now(timezone.utc)
        staleness = _fully_fresh(now, except_for=("eps_trend",))

        for _ in range(2):
            yf = self._calm_client()
            YFinanceDataService(
                repo=_make_repo(staleness=staleness), yf_client=yf
            ).fetch_and_store(
                instrument_id=self._INSTRUMENT,
                yfinance_ticker="AAPL",
                mode="incremental",
                currency_code="USD",
            )
            yf.analysis.fetch_eps_trend.assert_called_once()


class TestPriceUnit:
    """GBp/sub-unit prices (SPEC OQ2): the listing currency is stored as-is on
    price rows so the fx layer converts — no lossy transform at ingest."""

    def test_price_history_model_has_price_unit_column(self) -> None:
        from portopt_db.models.market_data.yfinance_data import PriceHistory

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


class TestIsinBackfill:
    """ISIN is not carried by yfinance's .info, so the profile step fetches it
    separately (Ticker.isin) and injects it into the info dict before upsert —
    otherwise ticker_profiles.isin is always NULL."""

    def test_isin_fetched_and_injected_into_profile(self) -> None:
        repo = _make_repo(staleness=None)
        yf = _make_yf_client_silent()
        yf.fetch_info.return_value = {"symbol": "AAPL", "shortName": "Apple"}
        yf.metadata.fetch_isin.return_value = "US0378331005"
        _run(repo, yf, mode="full")
        yf.metadata.fetch_isin.assert_called_once()
        info_arg = repo.upsert_profile.call_args.args[1]
        assert info_arg["isin"] == "US0378331005"

    def test_isin_absent_leaves_profile_isin_unset(self) -> None:
        repo = _make_repo(staleness=None)
        yf = _make_yf_client_silent()
        yf.fetch_info.return_value = {"symbol": "AAPL", "shortName": "Apple"}
        yf.metadata.fetch_isin.return_value = None
        _run(repo, yf, mode="full")
        info_arg = repo.upsert_profile.call_args.args[1]
        assert info_arg.get("isin") is None

    def test_isin_not_fetched_when_no_profile_info(self) -> None:
        repo = _make_repo(staleness=None)
        yf = _make_yf_client_silent()  # fetch_info returns None -> profile skipped
        _run(repo, yf, mode="full")
        yf.metadata.fetch_isin.assert_not_called()


class TestAnalystEstimates:
    """earnings/revenue/growth estimates: dedicated tables, staleness-gated on
    the shared fundamentals window (SPEC A1 / OQ5)."""

    def test_models_have_expected_columns(self) -> None:
        from portopt_db.models.market_data.yfinance_data import (
            EarningsEstimate,
            GrowthEstimate,
            RevenueEstimate,
        )

        for col in ("period", "num_analysts", "avg", "low", "high", "growth"):
            assert hasattr(EarningsEstimate, col)
        assert hasattr(EarningsEstimate, "year_ago_eps")
        assert hasattr(RevenueEstimate, "year_ago_revenue")
        for col in ("stock_trend", "index_trend"):
            assert hasattr(GrowthEstimate, col)

    def test_skipped_when_fundamentals_fresh(self) -> None:
        now = datetime.now(timezone.utc)
        repo = _make_repo(
            staleness={**_fresh_fundamentals(now), "price_max_date": None}
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


class TestEarningsTimeline:
    """earnings_history + earnings_dates (SPEC A2)."""

    def test_models_have_expected_columns(self) -> None:
        from portopt_db.models.market_data.yfinance_data import (
            EarningsDate,
            EarningsHistory,
        )

        for col in (
            "period_date",
            "eps_estimate",
            "eps_actual",
            "eps_difference",
            "surprise_percent",
        ):
            assert hasattr(EarningsHistory, col)
        for col in ("earnings_date", "eps_estimate", "eps_actual", "surprise_percent"):
            assert hasattr(EarningsDate, col)

    def test_skipped_when_fundamentals_fresh(self) -> None:
        now = datetime.now(timezone.utc)
        repo = _make_repo(
            staleness={**_fresh_fundamentals(now), "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        result = _run(repo, yf, mode="incremental")
        assert "earnings_history" in result["skipped"]
        assert "earnings_dates" in result["skipped"]

    def test_persisted_when_stale(self) -> None:
        repo = _make_repo(
            staleness={"financials_updated_at": None, "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        yf.analysis.fetch_earnings_history.return_value = pd.DataFrame(
            {"epsEstimate": [1.0], "epsActual": [1.1], "surprisePercent": [10.0]},
            index=pd.to_datetime(["2024-03-31"]),
        )
        yf.metadata.fetch_earnings_dates.return_value = pd.DataFrame(
            {"EPS Estimate": [1.2], "Reported EPS": [1.3], "Surprise(%)": [8.0]},
            index=pd.to_datetime(["2024-07-25"]),
        )
        _run(repo, yf, mode="incremental")
        repo.upsert_earnings_history.assert_called_once()
        repo.upsert_earnings_dates.assert_called_once()


class TestAnalystActions:
    """analyst_actions (upgrades/downgrades) — SPEC A3, analyst-rating cadence."""

    def test_model_has_expected_columns(self) -> None:
        from portopt_db.models.market_data.yfinance_data import AnalystAction

        for col in ("action_date", "firm", "from_grade", "to_grade", "action"):
            assert hasattr(AnalystAction, col)

    def test_skipped_when_recommendations_fresh(self) -> None:
        now = datetime.now(timezone.utc)
        repo = _make_repo(
            staleness={"recommendations_updated_at": now, "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        result = _run(repo, yf, mode="incremental")
        assert "analyst_actions" in result["skipped"]
        yf.analysis.fetch_upgrades_downgrades.assert_not_called()

    def test_persisted_when_stale(self) -> None:
        repo = _make_repo(
            staleness={"recommendations_updated_at": None, "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        yf.analysis.fetch_upgrades_downgrades.return_value = pd.DataFrame(
            {
                "Firm": ["MS", "GS"],
                "ToGrade": ["Buy", "Hold"],
                "FromGrade": ["Hold", "Buy"],
                "Action": ["up", "down"],
            },
            index=pd.to_datetime(["2024-05-01", "2024-05-02"]),
        )
        _run(repo, yf, mode="incremental")
        repo.upsert_analyst_actions.assert_called_once()


class TestSecFilings:
    """sec_filings (SPEC A5) — list of filing dicts."""

    def test_model_has_expected_columns(self) -> None:
        from portopt_db.models.market_data.yfinance_data import SecFiling

        for col in ("filing_date", "form_type", "title", "url"):
            assert hasattr(SecFiling, col)

    def test_skipped_when_fundamentals_fresh(self) -> None:
        now = datetime.now(timezone.utc)
        repo = _make_repo(
            staleness={**_fresh_fundamentals(now), "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        result = _run(repo, yf, mode="incremental")
        assert "sec_filings" in result["skipped"]

    def test_persisted_when_stale(self) -> None:
        repo = _make_repo(
            staleness={"financials_updated_at": None, "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        yf.financials.fetch_sec_filings.return_value = [
            {
                "date": "2024-02-01",
                "type": "10-K",
                "title": "Annual report",
                "edgarUrl": "u",
            },
            {
                "date": "2024-05-01",
                "type": "10-Q",
                "title": "Quarterly",
                "edgarUrl": "u2",
            },
        ]
        _run(repo, yf, mode="incremental")
        repo.upsert_sec_filings.assert_called_once()


class TestCorpActionExtras:
    """Shares outstanding + capital gains + price-history CG column (SPEC A6)."""

    def test_shares_outstanding_model_columns(self) -> None:
        from portopt_db.models.market_data.yfinance_data import SharesOutstanding

        for col in ("date", "shares"):
            assert hasattr(SharesOutstanding, col)

    def test_price_history_has_capital_gains_column(self) -> None:
        from portopt_db.models.market_data.yfinance_data import PriceHistory

        assert hasattr(PriceHistory, "capital_gains")

    def test_shares_skipped_when_fundamentals_fresh(self) -> None:
        now = datetime.now(timezone.utc)
        repo = _make_repo(
            staleness={**_fresh_fundamentals(now), "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        result = _run(repo, yf, mode="incremental")
        assert "shares_outstanding" in result["skipped"]
        yf.corporate_actions.fetch_shares_full.assert_not_called()

    def test_shares_persisted_when_stale(self) -> None:
        repo = _make_repo(
            staleness={"financials_updated_at": None, "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        yf.corporate_actions.fetch_shares_full.return_value = pd.Series(
            [1_000_000, 1_050_000],
            index=pd.to_datetime(["2024-01-01", "2024-06-01"]),
        )
        _run(repo, yf, mode="incremental")
        repo.upsert_shares_outstanding.assert_called_once()


class TestHoldersExtras:
    """major_holders + insider_purchases + insider_roster (SPEC A7)."""

    def test_major_holders_model_columns(self) -> None:
        from portopt_db.models.market_data.yfinance_data import MajorHolders

        for col in ("insiders_percent_held", "institutions_percent_held"):
            assert hasattr(MajorHolders, col)

    def test_insider_purchase_model_columns(self) -> None:
        from portopt_db.models.market_data.yfinance_data import InsiderPurchaseSummary

        for col in ("net_shares", "total_insider_shares"):
            assert hasattr(InsiderPurchaseSummary, col)

    def test_insider_roster_model_columns(self) -> None:
        from portopt_db.models.market_data.yfinance_data import InsiderRosterHolder

        for col in ("insider_name", "position"):
            assert hasattr(InsiderRosterHolder, col)

    def test_skipped_when_holders_fresh(self) -> None:
        now = datetime.now(timezone.utc)
        repo = _make_repo(
            staleness={
                "institutional_holders_updated_at": now,
                "insider_transactions_updated_at": now,
                "price_max_date": None,
            }
        )
        yf = _make_yf_client_silent()
        result = _run(repo, yf, mode="incremental")
        assert "major_holders" in result["skipped"]
        assert "insider_purchases" in result["skipped"]
        assert "insider_roster" in result["skipped"]
        yf.holders.fetch_major_holders.assert_not_called()
        yf.holders.fetch_insider_purchases.assert_not_called()
        yf.holders.fetch_insider_roster_holders.assert_not_called()

    def test_persisted_when_stale(self) -> None:
        repo = _make_repo(
            staleness={
                "institutional_holders_updated_at": None,
                "insider_transactions_updated_at": None,
                "price_max_date": None,
            }
        )
        yf = _make_yf_client_silent()
        yf.holders.fetch_major_holders.return_value = pd.DataFrame(
            {"Value": [0.01, 0.60]},
            index=["insidersPercentHeld", "institutionsPercentHeld"],
        )
        yf.holders.fetch_insider_purchases.return_value = pd.DataFrame(
            {
                "Insider Purchases Last 6m": [
                    "Purchases",
                    "Sales",
                    "Net Shares Purchased (Sold)",
                    "Total Insider Shares Held",
                ],
                "Shares": [100, 40, 60, 5000],
                "Trans": [2, 1, 1, None],
            }
        )
        yf.holders.fetch_insider_roster_holders.return_value = pd.DataFrame(
            {
                "Name": ["JANE DOE"],
                "Position": ["CEO"],
                "Most Recent Transaction": ["Buy"],
                "Latest Transaction Date": ["2024-05-01"],
                "Shares Owned Directly": [1000],
            }
        )
        _run(repo, yf, mode="incremental")
        repo.upsert_major_holders.assert_called_once()
        repo.upsert_insider_purchases.assert_called_once()
        repo.upsert_insider_roster.assert_called_once()


class TestProfileExtras:
    """ticker_profile_extras (SPEC A9) — mapped from the already-fetched info."""

    def test_model_has_expected_columns(self) -> None:
        from portopt_db.models.market_data.yfinance_data import TickerProfileExtra

        for col in ("shares_short", "short_ratio", "overall_risk", "sector_key"):
            assert hasattr(TickerProfileExtra, col)

    def test_skipped_when_profile_fresh(self) -> None:
        now = datetime.now(timezone.utc)
        repo = _make_repo(staleness={"profile_updated_at": now, "price_max_date": None})
        yf = _make_yf_client_silent()
        result = _run(repo, yf, mode="incremental")
        assert "profile" in result["skipped"]
        repo.upsert_profile_extras.assert_not_called()

    def test_persisted_when_stale(self) -> None:
        repo = _make_repo(
            staleness={"profile_updated_at": None, "price_max_date": None}
        )
        yf = _make_yf_client_silent()
        yf.fetch_info.return_value = {
            "symbol": "AAPL",
            "sharesShort": 100_000,
            "shortRatio": 1.2,
            "overallRisk": 3,
            "sectorKey": "technology",
        }
        _run(repo, yf, mode="incremental")
        repo.upsert_profile_extras.assert_called_once()


class TestStalenessInjection:
    """A bulk sweep pre-loads staleness in one grouped query and injects it, so
    fetch_and_store must NOT run its own per-instrument staleness query."""

    def test_injected_staleness_skips_per_instrument_query(self) -> None:
        now = datetime.now(timezone.utc)
        repo = _make_repo(staleness=None)
        yf = _make_yf_client_silent()
        service = YFinanceDataService(repo=repo, yf_client=yf)
        result = service.fetch_and_store(
            instrument_id=uuid4(),
            yfinance_ticker="AAPL",
            mode="incremental",
            staleness={**_fresh_fundamentals(now), "price_max_date": None},
        )
        repo.get_staleness_info.assert_not_called()
        assert "valuation_measures" in result["skipped"]

    def test_missing_staleness_falls_back_to_query(self) -> None:
        repo = _make_repo(staleness={"price_max_date": None})
        yf = _make_yf_client_silent()
        service = YFinanceDataService(repo=repo, yf_client=yf)
        # No staleness kwarg → the internal query still runs.
        service.fetch_and_store(
            instrument_id=uuid4(), yfinance_ticker="AAPL", mode="incremental"
        )
        repo.get_staleness_info.assert_called_once()


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
        # actions=True so Dividends / Stock Splits land in price_history.
        assert all_args.get("actions") is True


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
