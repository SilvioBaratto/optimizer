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

  fetch_and_store() — adjustment-basis re-base
    - stored overlap closes disagree with the re-fetched window → full re-fetch
    - re-base fetch truncated or empty → the window is stored alone
    - young listing / thin name whose refetch reaches the first stored bar → stored
    - stored series older than the period → fetch starts at its oldest bar

  fetch_and_store() — per-category SAVEPOINT isolation
    - a failed category rolls back alone; earlier and later writes survive

  fetch_and_store() / run_bulk_yfinance_fetch() — unreachable Yahoo
    - transport faults are reported apart from "no data"
    - a sweep in which most requests went unanswered ends as a failed job

  fetch_and_store() — news scraper branches
    - news list non-empty, link from canonicalUrl dict
    - news list non-empty, link from previewUrl fallback
    - scraper returns success=False → full_content not injected

  _aggregate_outcome()
    - outcome is Exception → appended to errors, returns 0 skipped

  run_bulk_yfinance_fetch() — missing yfinance_ticker guard
    - ticker is None/empty → skipped with error

External I/O (Yahoo, the scraper) is mocked and most tests use a MagicMock
repository. The re-base and SAVEPOINT tests that depend on real write semantics
run the real repository on the SQLite ``db_session`` fixture.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest

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


class TestDividendPriceCurrency:
    """Dividends may arrive tagged with a payment currency; the repository can
    only keep those in the price currency, so the service must say which it is."""

    def test_listing_currency_is_passed_as_price_currency(self) -> None:
        repo = _make_repo(staleness=None)
        yf = _make_yf_client_silent()
        dividends = pd.DataFrame(
            {"Dividends": [0.25], "currency": ["GBp"]},
            index=pd.to_datetime(["2024-01-15"]),
        )
        yf.corporate_actions.fetch_dividends.return_value = dividends
        _run(repo, yf, mode="full", currency_code="GBX")
        repo.upsert_dividends.assert_called_once()
        assert repo.upsert_dividends.call_args.args[1] is dividends
        assert repo.upsert_dividends.call_args.kwargs["price_currency"] == "GBX"


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

    def test_isin_lookup_failure_does_not_discard_the_fetched_profile(self) -> None:
        """The ISIN comes from a separate third-party lookup; its timeout must not
        cost the profile that was already fetched."""
        repo = _make_repo(staleness=None)
        yf = _make_yf_client_silent()
        yf.fetch_info.return_value = {"symbol": "AAPL", "shortName": "Apple"}
        yf.metadata.fetch_isin.side_effect = TimeoutError("lookup stalled")
        repo.upsert_profile.return_value = 1

        result = _run(repo, yf, mode="full")

        repo.upsert_profile.assert_called_once()
        assert "isin" not in repo.upsert_profile.call_args.args[1]
        repo.upsert_profile_extras.assert_called_once()
        assert result["counts"]["profile"] == 1
        assert any(e.startswith("isin:") for e in result["errors"])
        assert not any(e.startswith("profile:") for e in result["errors"])

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

        def _fake_call_with_timeout(fn: Any, timeout: float, **_: Any) -> Any:
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
# fetch_and_store() — full refetch must cover every stored bar
# ===========================================================================


class TestFullModeCoversStoredHistory:
    """auto_adjust rescales the whole series on each split/dividend, so a full
    refetch that stops short of the oldest stored bar leaves that tail on a stale
    adjustment basis and the series gains a spurious step at the window edge."""

    @staticmethod
    def _history_kwargs(repo: MagicMock, **run_kwargs: Any) -> dict[str, Any]:
        yf_client = _make_yf_client_silent()
        _run(repo, yf_client, **run_kwargs)
        assert yf_client.fetch_history.call_count == 1
        return yf_client.fetch_history.call_args.kwargs

    def test_when_stored_history_older_than_period_then_refetch_starts_at_oldest_bar(
        self,
    ) -> None:
        oldest = date.today() - timedelta(days=6 * 365)
        repo = _make_repo(staleness=None)
        repo.get_price_history.side_effect = _stored_prices({oldest: 100.0})

        kwargs = self._history_kwargs(repo, mode="full", period="5y")

        assert kwargs["start"] == oldest.isoformat()

    def test_when_stored_history_within_period_then_period_is_requested(self) -> None:
        repo = _make_repo(staleness=None)
        repo.get_price_history.side_effect = _stored_prices(
            {date.today() - timedelta(days=2 * 365): 100.0}
        )

        kwargs = self._history_kwargs(repo, mode="full", period="5y")

        assert kwargs["period"] == "5y"
        assert kwargs.get("start") is None

    def test_when_no_stored_prices_then_period_is_requested(self) -> None:
        repo = _make_repo(staleness=None)
        repo.get_price_history.side_effect = _stored_prices({})

        kwargs = self._history_kwargs(repo, mode="full", period="5y")

        assert kwargs["period"] == "5y"
        assert kwargs.get("start") is None

    def test_when_period_is_max_then_period_is_requested(self) -> None:
        repo = _make_repo(staleness=None)
        repo.get_price_history.side_effect = _stored_prices(
            {date.today() - timedelta(days=9 * 365): 100.0}
        )

        kwargs = self._history_kwargs(repo, mode="full", period="max")

        assert kwargs["period"] == "max"
        assert kwargs.get("start") is None

    def test_when_oldest_bar_lookup_fails_then_period_is_requested_without_error(
        self,
    ) -> None:
        repo = _make_repo(staleness=None)
        repo.get_price_history.side_effect = RuntimeError("db gone")
        yf_client = _make_yf_client_silent()

        result = _run(repo, yf_client, mode="full", period="5y")

        kwargs = yf_client.fetch_history.call_args.kwargs
        assert kwargs["period"] == "5y"
        assert kwargs.get("start") is None
        assert not any(e.startswith("prices") for e in result["errors"])

    def test_when_incremental_first_load_then_oldest_bar_is_not_queried(self) -> None:
        repo = _make_repo(staleness={"price_max_date": None})

        kwargs = self._history_kwargs(repo, mode="incremental")

        assert kwargs["period"] == "5y"
        repo.get_price_history.assert_not_called()


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

    def test_when_young_listing_returns_complete_short_history_then_rewritten(
        self,
    ) -> None:
        """A listing younger than the period that already has stored prices must
        still have its full refetch stored: Yahoo re-bases every adjusted close
        after a split/dividend and only this rewrite heals the older rows."""
        idx = pd.bdate_range(end=date.today(), periods=503)
        repo = _make_repo(staleness=None)
        repo.get_staleness_info.return_value = {
            "price_max_date": date.today() - timedelta(days=3),
            "price_min_date": idx[0].date(),
        }
        yf_client = _make_yf_client_silent()
        # ~2 years of consecutive weekday bars: every session since the IPO.
        yf_client.fetch_history.return_value = pd.DataFrame(
            {"Close": [50.0] * len(idx)}, index=idx
        )

        result = _run(repo, yf_client, mode="full", exchange_name="NYSE", period="5y")

        assert "prices:insufficient_history" not in result["skipped"]
        repo.upsert_price_history.assert_called_once()

    def test_when_gappy_history_and_existing_prices_then_still_protected(
        self,
    ) -> None:
        """Dropping most sessions inside the stored window is still a glitchy
        refetch, so the protective skip must keep firing for it."""
        idx = pd.bdate_range(end=date.today(), periods=503)
        repo = _make_repo(staleness=None)
        repo.get_staleness_info.return_value = {
            "price_max_date": date.today() - timedelta(days=3),
            "price_min_date": idx[0].date(),
        }
        yf_client = _make_yf_client_silent()
        gappy = idx[::5]
        yf_client.fetch_history.return_value = pd.DataFrame(
            {"Close": [50.0] * len(gappy)}, index=gappy
        )

        result = _run(repo, yf_client, mode="full", exchange_name="NYSE", period="5y")

        assert "prices:insufficient_history" in result["skipped"]
        repo.upsert_price_history.assert_not_called()

    def test_when_refetch_covers_only_recent_tail_of_stored_series_then_protected(
        self,
    ) -> None:
        """A head-truncated refetch returns a contiguous run of recent bars, so
        measured from its own first bar it looks complete. The window must be
        anchored at the earliest stored bar instead, otherwise the truncated
        rows overwrite the tail while older rows keep their old adjustment."""
        stored = pd.bdate_range(end=date.today(), periods=503)
        repo = _make_repo(staleness=None)
        repo.get_staleness_info.return_value = {
            "price_max_date": date.today() - timedelta(days=3),
            "price_min_date": stored[0].date(),
        }
        yf_client = _make_yf_client_silent()
        tail = stored[-22:]
        yf_client.fetch_history.return_value = pd.DataFrame(
            {"Close": [50.0] * len(tail)}, index=tail
        )

        result = _run(repo, yf_client, mode="full", exchange_name="NYSE", period="5y")

        assert "prices:insufficient_history" in result["skipped"]
        repo.upsert_price_history.assert_not_called()

    def test_when_earliest_stored_bar_unknown_then_short_refetch_stays_protected(
        self,
    ) -> None:
        """Without the earliest stored date there is no safe way to tell a young
        listing from a truncated refetch, so the full-period window applies."""
        idx = pd.bdate_range(end=date.today(), periods=503)
        repo = _make_repo(staleness=None)
        repo.get_staleness_info.return_value = {
            "price_max_date": date.today() - timedelta(days=3)
        }
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.return_value = pd.DataFrame(
            {"Close": [50.0] * len(idx)}, index=idx
        )

        result = _run(repo, yf_client, mode="full", exchange_name="NYSE", period="5y")

        assert "prices:insufficient_history" in result["skipped"]
        repo.upsert_price_history.assert_not_called()

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
# fetch_and_store() — adjustment-basis re-base
# ===========================================================================


def _stored_prices(closes: dict[date, float]) -> Any:
    """Fake ``repo.get_price_history`` over a dict of stored closes.

    Honours the date window, ordering and limit so the service's reads are
    checked against the contract of the real repository method.
    """
    from types import SimpleNamespace

    bars = [SimpleNamespace(date=d, close=c) for d, c in sorted(closes.items())]

    def get_price_history(
        instrument_id: Any,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 5000,
        ascending: bool = False,
    ) -> list[Any]:
        picked = [
            b
            for b in bars
            if (start_date is None or b.date >= start_date)
            and (end_date is None or b.date <= end_date)
        ]
        picked.sort(key=lambda b: b.date, reverse=not ascending)
        return picked[:limit]

    return get_price_history


_OVERLAP_STALENESS_MAX_DATE = date(2025, 1, 15)


def _incremental_staleness() -> dict[str, Any]:
    return {
        "profile_updated_at": None,
        "financials_updated_at": None,
        "dividends_updated_at": None,
        "splits_updated_at": None,
        "recommendations_updated_at": None,
        "price_targets_updated_at": None,
        "institutional_holders_updated_at": None,
        "mutualfund_holders_updated_at": None,
        "insider_transactions_updated_at": None,
        "price_max_date": _OVERLAP_STALENESS_MAX_DATE,
    }


def _window_history(close: float) -> pd.DataFrame:
    idx = pd.date_range("2025-01-10", "2025-01-16", freq="B")
    return pd.DataFrame({"Close": [close] * len(idx)}, index=idx)


def _full_history(close: float, periods: int = 800) -> pd.DataFrame:
    idx = pd.date_range(end="2025-01-16", periods=periods, freq="B")
    return pd.DataFrame({"Close": [close] * len(idx)}, index=idx)


def _history_by_request(window: Any, full: Any) -> Any:
    """``fetch_history`` fake: date-ranged call -> *window*, period call -> *full*."""

    def fetch_history(symbol: str, **kwargs: Any) -> Any:
        return window if kwargs.get("start") is not None else full

    return fetch_history


_FROZEN_TODAY = date(2025, 1, 16)


@pytest.fixture
def frozen_today(monkeypatch: pytest.MonkeyPatch) -> date:
    """Pin the service clock to the day the fixture histories end on."""
    monkeypatch.setattr(
        "app.services.market_data.yfinance_data_service._today",
        lambda: _FROZEN_TODAY,
    )
    return _FROZEN_TODAY


def _years_back(day: date, years: int) -> date:
    return day.replace(year=day.year - years)


def _yahoo_serving(bars: pd.DataFrame, today: date) -> Any:
    """``fetch_history`` fake that answers like Yahoo.

    A ``period`` reaches back from *today*, a ``start`` from that date; every
    answer is cut from *bars*, i.e. carries one adjustment basis.
    """

    def fetch_history(symbol: str, **kwargs: Any) -> pd.DataFrame:
        if kwargs.get("start") is not None:
            floor = pd.Timestamp(kwargs["start"])
        else:
            floor = pd.Timestamp(_years_back(today, int(kwargs["period"][:-1])))
        end = kwargs.get("end")
        ceiling = pd.Timestamp(end) if end else pd.Timestamp.max
        return bars[(bars.index >= floor) & (bars.index < ceiling)]

    return fetch_history


@pytest.mark.usefixtures("frozen_today")
class TestAdjustmentBasisRebase:
    """yfinance prices are split/dividend adjusted as of fetch time, so a new
    split or ex-dividend rescales every earlier bar. The 5-day incremental
    overlap rewrites only the window; the older stored bars must be re-based by
    a full-period fetch, or the series carries a phantom step at the window
    start that ``prices_to_returns`` turns into a fake return."""

    def test_incremental_refetches_full_history_when_overlap_was_readjusted(
        self,
    ) -> None:
        repo = _make_repo(staleness=_incremental_staleness())
        # Stored before a 2:1 split: 200. Yahoo now reports the same days at 100.
        repo.get_price_history.side_effect = _stored_prices(
            {
                date(2025, 1, 10): 200.0,
                date(2025, 1, 13): 200.0,
                date(2025, 1, 14): 200.0,
                date(2025, 1, 15): 200.0,
            }
        )
        window, full = _window_history(100.0), _full_history(100.0)
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.side_effect = _history_by_request(window, full)

        _run(repo, yf_client, mode="incremental")

        requests = [c.kwargs for c in yf_client.fetch_history.call_args_list]
        assert [r.get("period") for r in requests if "start" not in r] == ["5y"]
        stored_frame = repo.upsert_price_history.call_args.args[1]
        assert len(stored_frame) == len(full)

    def test_incremental_keeps_window_only_when_overlap_unchanged(self) -> None:
        repo = _make_repo(staleness=_incremental_staleness())
        repo.get_price_history.side_effect = _stored_prices(
            {date(2025, 1, 10): 100.0, date(2025, 1, 13): 100.0}
        )
        window = _window_history(100.0)
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.side_effect = _history_by_request(
            window, _full_history(100.0)
        )

        _run(repo, yf_client, mode="incremental")

        assert yf_client.fetch_history.call_count == 1
        assert repo.upsert_price_history.call_args.args[1] is window

    def test_incremental_ignores_provisional_last_bar_when_comparing(self) -> None:
        """The newest stored bar can be an intraday snapshot (exchanges still
        open at the 07:00 UTC run); its final close differing is not a re-base."""
        repo = _make_repo(staleness=_incremental_staleness())
        repo.get_price_history.side_effect = _stored_prices(
            {
                date(2025, 1, 10): 100.0,
                date(2025, 1, 13): 100.0,
                date(2025, 1, 15): 97.3,
            }
        )
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.side_effect = _history_by_request(
            _window_history(100.0), _full_history(100.0)
        )

        _run(repo, yf_client, mode="incremental")

        assert yf_client.fetch_history.call_count == 1

    def test_incremental_falls_back_to_window_when_rebase_fetch_is_truncated(
        self,
    ) -> None:
        repo = _make_repo(staleness=_incremental_staleness())
        repo.get_price_history.side_effect = _stored_prices(
            {
                date(2023, 1, 3): 180.0,
                date(2025, 1, 10): 200.0,
                date(2025, 1, 13): 200.0,
                date(2025, 1, 15): 200.0,
            }
        )
        window = _window_history(100.0)
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.side_effect = _history_by_request(
            window, _full_history(100.0, periods=60)
        )

        with patch(
            "app.services.market_data.yfinance_data_service.has_sufficient_history",
            return_value=(False, 1260, 1197),
        ):
            result = _run(repo, yf_client, mode="incremental", exchange_name="NYSE")

        assert "prices:insufficient_history" in result["skipped"]
        assert repo.upsert_price_history.call_args.args[1] is window

    def test_incremental_falls_back_to_window_when_rebase_fetch_returns_nothing(
        self,
    ) -> None:
        repo = _make_repo(staleness=_incremental_staleness())
        repo.get_price_history.side_effect = _stored_prices(
            {date(2025, 1, 10): 200.0, date(2025, 1, 13): 200.0}
        )
        window = _window_history(100.0)
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.side_effect = _history_by_request(window, None)

        _run(repo, yf_client, mode="incremental")

        assert repo.upsert_price_history.call_args.args[1] is window

    def test_full_refetch_of_young_listing_with_stored_prices_is_stored(
        self,
    ) -> None:
        """A listing younger than the 5y session floor never passes the row-count
        guard; refusing its refetch would leave it on a stale basis forever. The
        refetch reaches back to the first stored bar, so it re-bases everything."""
        repo = _make_repo(staleness=None)
        repo.get_staleness_info.return_value = {"price_max_date": date(2024, 4, 10)}
        history = _short_history(80)
        repo.get_price_history.side_effect = _stored_prices(
            {history.index[0].date(): 49.0, date(2024, 4, 10): 49.5}
        )
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.return_value = history

        with patch(
            "app.services.market_data.yfinance_data_service.has_sufficient_history",
            return_value=(False, 1260, 1197),
        ):
            result = _run(
                repo, yf_client, mode="full", exchange_name="NYSE", period="5y"
            )

        assert "prices:insufficient_history" not in result["skipped"]
        assert repo.upsert_price_history.call_args.args[1] is history

    def test_full_refetch_shorter_than_stored_series_is_still_protected(
        self,
    ) -> None:
        repo = _make_repo(staleness=None)
        repo.get_staleness_info.return_value = {"price_max_date": date(2024, 12, 1)}
        history = _short_history(100)
        repo.get_price_history.side_effect = _stored_prices(
            {date(2022, 1, 3): 40.0, date(2024, 12, 1): 50.0}
        )
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.return_value = history

        with patch(
            "app.services.market_data.yfinance_data_service.has_sufficient_history",
            return_value=(False, 1260, 1197),
        ):
            result = _run(
                repo, yf_client, mode="full", exchange_name="NYSE", period="5y"
            )

        assert "prices:insufficient_history" in result["skipped"]
        repo.upsert_price_history.assert_not_called()

    def test_full_refetch_of_thin_name_older_than_the_period_is_stored(
        self,
    ) -> None:
        """A thinly traded name never reaches 95% of the exchange sessions. Once
        its stored series is older than the period, the refetch starts at the
        oldest stored bar and still reaches it, so refusing it would freeze the
        adjustment basis for good."""
        repo = _make_repo(staleness=None)
        repo.get_staleness_info.return_value = {"price_max_date": date(2025, 1, 15)}
        yahoo = pd.DataFrame(
            {"Close": 50.0},
            index=pd.bdate_range("2017-06-01", _FROZEN_TODAY)[::2],
        )
        stored = {
            stamp.date(): 49.5 for stamp in yahoo.index[yahoo.index >= "2018-01-02"]
        }
        repo.get_price_history.side_effect = _stored_prices(stored)
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.side_effect = _yahoo_serving(yahoo, _FROZEN_TODAY)

        with patch(
            "app.services.market_data.yfinance_data_service.has_sufficient_history",
            return_value=(False, 1260, 1197),
        ):
            result = _run(
                repo, yf_client, mode="full", exchange_name="NYSE", period="5y"
            )

        assert "prices:insufficient_history" not in result["skipped"]
        refetched = repo.upsert_price_history.call_args.args[1]
        assert refetched.index.min().date() == min(stored)

    def test_full_refetch_missing_stored_bars_inside_the_window_is_protected(
        self,
    ) -> None:
        repo = _make_repo(staleness=None)
        repo.get_staleness_info.return_value = {"price_max_date": date(2025, 1, 15)}
        idx = pd.date_range("2023-01-02", "2025-01-16", freq="B")[::2]
        history = pd.DataFrame({"Close": [50.0] * len(idx)}, index=idx)
        stored = {date(2018, 1, 2): 30.0, date(2020, 3, 2): 35.0}
        stored.update({stamp.date(): 49.5 for stamp in idx})
        repo.get_price_history.side_effect = _stored_prices(stored)
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.return_value = history

        with patch(
            "app.services.market_data.yfinance_data_service.has_sufficient_history",
            return_value=(False, 1260, 1197),
        ):
            result = _run(
                repo, yf_client, mode="full", exchange_name="NYSE", period="5y"
            )

        assert "prices:insufficient_history" in result["skipped"]
        repo.upsert_price_history.assert_not_called()

    def test_full_refetch_stopping_at_the_period_window_is_protected(self) -> None:
        """The series is older than the period, so the fetch asked for its oldest
        bar. A response that stops at the period window would re-base only that
        span and leave the older bars on the previous basis."""
        repo = _make_repo(staleness=None)
        repo.get_staleness_info.return_value = {"price_max_date": date(2025, 1, 15)}
        window_only = pd.DataFrame(
            {"Close": 50.0},
            index=pd.bdate_range("2020-01-17", _FROZEN_TODAY)[::2],
        )
        stored = {date(2018, 1, 2): 30.0, date(2019, 6, 3): 35.0}
        stored.update({stamp.date(): 49.5 for stamp in window_only.index})
        repo.get_price_history.side_effect = _stored_prices(stored)
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.return_value = window_only

        with patch(
            "app.services.market_data.yfinance_data_service.has_sufficient_history",
            return_value=(False, 1260, 1197),
        ):
            result = _run(
                repo, yf_client, mode="full", exchange_name="NYSE", period="5y"
            )

        assert "prices:insufficient_history" in result["skipped"]
        repo.upsert_price_history.assert_not_called()


class TestAdjustmentBasisChanged:
    def test_rounding_noise_and_missing_values_are_not_a_basis_change(self) -> None:
        from decimal import Decimal
        from types import SimpleNamespace

        from app.services.market_data.yfinance_data_service import (
            _adjustment_basis_changed,
        )

        stored = [
            SimpleNamespace(date=date(2025, 1, 10), close=Decimal("123.456789")),
            SimpleNamespace(date=date(2025, 1, 13), close=None),
            SimpleNamespace(date=date(2025, 1, 14), close=Decimal("50.0")),
        ]
        fresh = pd.DataFrame(
            {"Close": [123.4567890004, 77.0, float("nan")]},
            index=pd.DatetimeIndex(
                ["2025-01-10", "2025-01-13", "2025-01-14"], tz="America/New_York"
            ),
        )
        fresh = pd.concat(
            [fresh, pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex([pd.NaT]))]
        )

        assert _adjustment_basis_changed(stored, fresh) is False

    def test_a_dividend_sized_gap_is_a_basis_change(self) -> None:
        from types import SimpleNamespace

        from app.services.market_data.yfinance_data_service import (
            _adjustment_basis_changed,
        )

        stored = [SimpleNamespace(date=date(2025, 1, 10), close=100.0)]
        fresh = pd.DataFrame({"Close": [99.5]}, index=pd.to_datetime(["2025-01-10"]))

        assert _adjustment_basis_changed(stored, fresh) is True


@pytest.mark.usefixtures("frozen_today")
class TestRebaseSpansWholeStoredSeries:
    """Nothing prunes price_history, so the stored series outgrows any fixed
    fetch period. A re-base that reaches back only ``period`` leaves every older
    bar on the previous adjustment basis, and ``prices_to_returns`` turns that
    seam into a fake return; the full fetch has to start at the oldest stored bar.
    """

    def _full_fetch_request(
        self, stored: dict[date, float], *, period: str = "5y"
    ) -> dict[str, Any]:
        repo = _make_repo(staleness=None)
        repo.get_staleness_info.return_value = {"price_max_date": date(2025, 1, 15)}
        repo.get_price_history.side_effect = _stored_prices(stored)
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.return_value = _full_history(100.0)

        _run(repo, yf_client, mode="full", period=period)

        yf_client.fetch_history.assert_called_once()
        return dict(yf_client.fetch_history.call_args.kwargs)

    def test_full_fetch_starts_at_the_oldest_stored_bar_older_than_the_period(
        self,
    ) -> None:
        request = self._full_fetch_request(
            {date(2018, 1, 2): 30.0, date(2024, 6, 3): 90.0}
        )

        assert request["start"] == "2018-01-02"
        assert "period" not in request

    def test_full_fetch_starts_at_a_stored_bar_sitting_on_the_period_edge(
        self,
    ) -> None:
        """Yahoo's clock is a day or two off ours, so a bar on the edge may fall
        just outside a period fetch."""
        request = self._full_fetch_request(
            {date(2020, 1, 17): 40.0, date(2024, 6, 3): 90.0}
        )

        assert request["start"] == "2020-01-17"
        assert "period" not in request

    def test_full_fetch_keeps_the_period_when_the_series_is_younger_than_it(
        self,
    ) -> None:
        """The period still backfills bars that were never stored."""
        request = self._full_fetch_request(
            {date(2022, 3, 1): 40.0, date(2024, 6, 3): 90.0}
        )

        assert request["period"] == "5y"
        assert "start" not in request

    @pytest.mark.parametrize("period", ["max", "6mo"])
    def test_periods_other_than_whole_years_are_left_to_yfinance(
        self, period: str
    ) -> None:
        request = self._full_fetch_request({date(2010, 1, 4): 20.0}, period=period)

        assert request["period"] == period
        assert "start" not in request

    def test_incremental_rebase_leaves_a_series_older_than_the_period_on_one_basis(
        self, db_session: Any, frozen_today: date
    ) -> None:
        from portopt_db.repositories.market_data.yfinance_repository import (
            YFinanceRepository,
        )

        from tests._fixtures import seed_universe

        instrument = seed_universe(db_session).instrument
        repo = YFinanceRepository(db_session)
        stored_span = pd.bdate_range("2018-01-02", "2025-01-15")
        repo.upsert_price_history(
            instrument.id, pd.DataFrame({"Close": 200.0}, index=stored_span)
        )
        yahoo = pd.DataFrame(
            {"Close": 100.0}, index=pd.bdate_range("2017-06-01", frozen_today)
        )
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.side_effect = _yahoo_serving(yahoo, frozen_today)

        YFinanceDataService(repo=repo, yf_client=yf_client).fetch_and_store(
            instrument_id=instrument.id,
            yfinance_ticker="AAPL",
            mode="incremental",
            staleness={
                **_incremental_staleness(),
                "price_max_date": stored_span[-1].date(),
            },
        )

        closes = [
            float(bar.close)
            for bar in repo.get_price_history(instrument.id, limit=10_000)
        ]
        assert len(closes) == len(stored_span) + 1
        assert set(closes) == {100.0}


@pytest.mark.usefixtures("frozen_today")
class TestAdjustmentBasisRebaseAgainstRealRepository:
    def test_incremental_run_leaves_every_stored_close_on_the_new_basis(
        self, db_session: Any
    ) -> None:
        from portopt_db.models.market_data.yfinance_data import PriceHistory
        from portopt_db.repositories.market_data.yfinance_repository import (
            YFinanceRepository,
        )

        from tests._fixtures import seed_universe

        instrument = seed_universe(db_session).instrument
        repo = YFinanceRepository(db_session)
        old_basis = pd.DataFrame(
            {"Close": [200.0] * 10},
            index=pd.date_range("2025-01-02", periods=10, freq="B"),
        )
        repo.upsert_price_history(instrument.id, old_basis)
        max_date = old_basis.index[-1].date()

        new_basis = pd.DataFrame(
            {"Close": [100.0] * 12},
            index=pd.date_range("2025-01-02", periods=12, freq="B"),
        )
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.side_effect = _history_by_request(
            new_basis.loc[new_basis.index >= pd.Timestamp("2025-01-10")], new_basis
        )

        YFinanceDataService(repo=repo, yf_client=yf_client).fetch_and_store(
            instrument_id=instrument.id,
            yfinance_ticker="AAPL",
            mode="incremental",
            staleness={**_incremental_staleness(), "price_max_date": max_date},
        )

        closes = {
            float(r.close)
            for r in db_session.query(PriceHistory).filter_by(
                instrument_id=instrument.id
            )
        }
        assert closes == {100.0}


# ===========================================================================
# fetch_and_store() — per-category SAVEPOINT isolation
# ===========================================================================


class _PgLikeSession:
    """Models the PostgreSQL rule that defeats per-category error handling.

    A failed statement aborts the whole transaction: every later statement
    raises until the enclosing SAVEPOINT is rolled back, and COMMIT of an
    aborted transaction silently turns into ROLLBACK instead of raising.
    """

    def __init__(self) -> None:
        self.committed: list[str] = []
        self._pending: list[str] = []
        self._aborted = False

    def write(self, label: str) -> None:
        if self._aborted:
            raise RuntimeError("InFailedSqlTransaction: current transaction is aborted")
        self._pending.append(label)

    def fail(self, error: Exception) -> None:
        self._aborted = True
        raise error

    @contextmanager
    def begin_nested(self) -> Any:
        if self._aborted:
            raise RuntimeError("InFailedSqlTransaction: current transaction is aborted")
        mark = len(self._pending)
        try:
            yield
        except BaseException:
            del self._pending[mark:]
            self._aborted = False
            raise

    def commit(self) -> None:
        if not self._aborted:
            self.committed.extend(self._pending)
        self._pending.clear()
        self._aborted = False


class TestCategoryWriteIsolation:
    """One failing category must not take the ticker's other writes with it."""

    def test_failed_category_does_not_discard_earlier_or_later_writes(self) -> None:
        session = _PgLikeSession()
        repo = _make_repo(staleness=None)
        repo.session = session

        def ok(label: str, count: int) -> Any:
            def write(*args: Any, **kwargs: Any) -> int:
                session.write(label)
                return count

            return write

        def duplicate_key(*args: Any, **kwargs: Any) -> int:
            session.fail(
                RuntimeError(
                    "CardinalityViolation: ON CONFLICT DO UPDATE command "
                    "cannot affect row a second time"
                )
            )
            return 0

        repo.upsert_profile.side_effect = ok("profile", 1)
        repo.upsert_price_history.side_effect = ok("prices", 3)
        repo.upsert_dividends.side_effect = duplicate_key
        repo.upsert_splits.side_effect = ok("splits", 1)
        yf_client = _make_yf_client_silent()
        yf_client.fetch_info.return_value = {"longName": "Apple Inc."}
        yf_client.fetch_history.return_value = _stale_history()
        yf_client.corporate_actions.fetch_dividends.return_value = pd.Series(
            [0.2, 0.2], index=pd.to_datetime(["2024-02-01", "2024-02-01"])
        )
        yf_client.corporate_actions.fetch_splits.return_value = pd.Series(
            [2.0], index=pd.to_datetime(["2024-03-01"])
        )

        result = _run(repo, yf_client, mode="full")
        session.commit()

        assert {"profile", "prices", "splits"} <= set(session.committed)
        assert "dividends" not in result["counts"]
        assert len(result["errors"]) == 1
        assert result["errors"][0].startswith("dividends:")

    def test_failed_category_rolls_back_alone_on_a_real_session(
        self, db_session: Any
    ) -> None:
        from portopt_db.models.market_data.yfinance_data import Dividend, PriceHistory
        from portopt_db.repositories.market_data.yfinance_repository import (
            YFinanceRepository,
        )

        from tests._fixtures import seed_universe

        class _DividendsFailAfterWriting(YFinanceRepository):
            def upsert_dividends(
                self, instrument_id: Any, dividends: Any, **kw: Any
            ) -> int:
                super().upsert_dividends(instrument_id, dividends, **kw)
                raise RuntimeError("statement timeout")

        instrument = seed_universe(db_session).instrument
        repo = _DividendsFailAfterWriting(db_session)
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.return_value = _stale_history()
        yf_client.corporate_actions.fetch_dividends.return_value = pd.Series(
            [0.2], index=pd.to_datetime(["2024-02-01"])
        )

        result = YFinanceDataService(repo=repo, yf_client=yf_client).fetch_and_store(
            instrument_id=instrument.id, yfinance_ticker="AAPL", mode="full"
        )
        db_session.commit()

        assert db_session.query(PriceHistory).count() == 800
        assert db_session.query(Dividend).count() == 0
        assert result["errors"] == ["dividends: statement timeout"]

    def test_no_repository_write_bypasses_the_savepoint_helper(self) -> None:
        import ast
        import inspect

        import app.services.market_data.yfinance_data_service as svc_module

        tree = ast.parse(inspect.getsource(svc_module))
        direct_writes = sorted(
            f"line {node.lineno}: {node.func.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr.startswith("upsert_")
        )

        assert direct_writes == []


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


# ===========================================================================
# fetch_and_store() — an unreachable Yahoo must not look like "no data"
# ===========================================================================


class _DownTicker:
    """A ``yf.Ticker`` stand-in for which every request fails with *error*."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.news = []

    def history(self, **kwargs: Any) -> Any:
        raise self._error

    def __getattr__(self, name: str) -> Any:
        raise self._error


def _real_yf_client(ticker: Any) -> Any:
    """The real client and sub-clients over *ticker*, with inert infrastructure."""
    from app.services.market_data.yfinance import YFinanceClient

    cache = MagicMock()
    cache.get.return_value = ticker
    return YFinanceClient(
        cache=cache,
        rate_limiter=MagicMock(),
        circuit_breaker=MagicMock(),
        default_max_retries=2,
    )


def _fetch_through_real_client(
    ticker: Any,
    *,
    mode: str = "full",
    staleness: Any = None,
    request_timeout: float | None = None,
) -> dict[str, Any]:
    repo = _make_repo(staleness=staleness)
    repo.get_price_history.return_value = []
    service = YFinanceDataService(
        repo=repo,
        yf_client=_real_yf_client(ticker),
        request_timeout=request_timeout,
    )
    with patch("app.services.infrastructure.retry._full_jitter", return_value=0.0):
        return service.fetch_and_store(
            instrument_id=uuid4(),
            yfinance_ticker="AAPL",
            mode=mode,
            staleness=staleness if mode == "incremental" else None,
        )


class TestUnreachableYahooIsNotNoData:
    """Exhausted retries on network faults used to collapse into the same ``None``
    as Yahoo's "no data" answer, so a whole outage stored nothing, raised nothing
    and left the ticker's result indistinguishable from a quiet weekend."""

    def test_network_faults_are_reported_as_transport_failures(self) -> None:
        result = _fetch_through_real_client(
            _DownTicker(ConnectionError("network down"))
        )

        assert result["transport_failures"] > 0
        assert any(e.startswith("transport:") for e in result["errors"])

    def test_yahoo_no_data_answers_are_not_transport_failures(self) -> None:
        from yfinance.exceptions import YFDataException

        result = _fetch_through_real_client(
            _DownTicker(YFDataException("No data found"))
        )

        assert result["transport_failures"] == 0
        assert not any(e.startswith("transport:") for e in result["errors"])

    def test_requests_under_the_timeout_watchdog_are_tracked(self) -> None:
        """The watchdog runs each call in its own thread; a failure there must
        still reach the ticker's tally."""
        down = ConnectionError("network down")

        watched = _fetch_through_real_client(_DownTicker(down), request_timeout=5.0)
        unwatched = _fetch_through_real_client(_DownTicker(down))

        assert unwatched["transport_failures"] > 1
        assert watched["transport_failures"] == unwatched["transport_failures"]

    def test_a_watchdog_timeout_is_a_transport_failure(self) -> None:
        import time

        from app.services.market_data.yfinance import track_request_outcomes
        from app.services.market_data.yfinance_data_service import call_with_timeout

        with track_request_outcomes() as outcomes:
            with pytest.raises(TimeoutError):
                call_with_timeout(lambda: time.sleep(0.5), 0.05)

        assert len(outcomes.failures) == 1
        assert "within" in outcomes.failures[0]

    def test_answered_requests_are_counted_beside_the_failures(self) -> None:
        from yfinance.exceptions import YFDataException

        result = _fetch_through_real_client(
            _DownTicker(YFDataException("No data found"))
        )

        assert result["requests_answered"] > 0
        assert result["transport_failures"] == 0

    def test_unparseable_payloads_are_not_transport_failures(self) -> None:
        """A parse error is Yahoo's content, not the connection."""
        result = _fetch_through_real_client(_DownTicker(KeyError("quoteSummary")))

        assert result["transport_failures"] == 0

    def test_a_fresh_ticker_makes_no_request_and_reports_no_failure(self) -> None:
        now = datetime.now(timezone.utc)
        result = _fetch_through_real_client(
            _DownTicker(ConnectionError("network down")),
            mode="incremental",
            staleness=_fully_fresh(now),
        )

        assert result["skipped"] == ["all:fresh"]
        assert result["transport_failures"] == 0

    def test_empty_price_answer_for_a_stored_series_is_flagged(self) -> None:
        """yfinance swallows network faults inside ``history()`` and returns an
        empty frame, so an empty answer for a ticker whose series is stored is
        the only trace such an outage leaves."""
        staleness = _incremental_staleness()
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.return_value = None
        repo = _make_repo(staleness=staleness)

        result = _run_with_staleness(repo, yf_client, staleness)

        assert "prices:unavailable" in result["skipped"]

    def test_empty_price_answer_for_an_unstored_series_is_not_flagged(self) -> None:
        staleness = {**_incremental_staleness(), "price_max_date": None}
        yf_client = _make_yf_client_silent()
        yf_client.fetch_history.return_value = None
        repo = _make_repo(staleness=staleness)

        result = _run_with_staleness(repo, yf_client, staleness)

        assert "prices:unavailable" not in result["skipped"]


def _run_with_staleness(
    repo: MagicMock, yf_client: MagicMock, staleness: dict[str, Any]
) -> dict[str, Any]:
    service = YFinanceDataService(repo=repo, yf_client=yf_client)
    return service.fetch_and_store(
        instrument_id=uuid4(),
        yfinance_ticker="AAPL",
        mode="incremental",
        staleness=staleness,
    )


# ===========================================================================
# run_bulk_yfinance_fetch() — a sweep Yahoo never answered is a failed job
# ===========================================================================


def _sweep_result(*, failed: int, answered: int) -> dict[str, Any]:
    return {
        "counts": {},
        "errors": [f"transport: {failed} request(s) failed"] if failed else [],
        "skipped": [],
        "transport_failures": failed,
        "requests_answered": answered,
    }


def _run_serial_sweep(
    per_ticker: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, BaseException | None, list[dict[str, Any]]]:
    """Run the serial sweep over tickers whose fetches return *per_ticker*."""
    from types import SimpleNamespace

    import app.services.market_data.yfinance_data_service as svc_module

    instruments = [
        SimpleNamespace(
            id=uuid4(),
            yfinance_ticker=f"T{i}",
            exchange_name="NASDAQ",
            currency_code="USD",
        )
        for i in range(len(per_ticker))
    ]
    repo_mock = MagicMock()
    repo_mock.get_instruments_with_yfinance_ticker.return_value = instruments
    service_mock = MagicMock()
    service_mock.fetch_and_store.side_effect = per_ticker

    @contextmanager
    def _get_session():
        yield MagicMock()

    db_mgr = MagicMock()
    db_mgr.get_session = _get_session
    progress: list[dict[str, Any]] = []
    request = SimpleNamespace(workers=1, period="5y", mode="full")

    result: dict[str, Any] | None = None
    raised: BaseException | None = None
    with (
        patch.object(svc_module, "YFinanceRepository", return_value=repo_mock),
        patch.object(svc_module, "YFinanceDataService", return_value=service_mock),
        patch.object(svc_module, "_backfill_instrument_isin_from_profiles"),
        patch("app.database.database_manager", db_mgr),
        patch("app.config.settings") as mock_settings,
    ):
        mock_settings.yfinance_request_timeout_seconds = 30
        try:
            result = svc_module.run_bulk_yfinance_fetch(
                request, MagicMock(), on_progress=lambda **kw: progress.append(kw)
            )
        except Exception as exc:
            raised = exc
    return result, raised, progress


class TestSerialSweepOutageDetection:
    """An outage used to end as a "completed" job with zero counts, so the daily
    pipeline carried on as if prices had updated."""

    def test_a_sweep_yahoo_never_answered_raises_instead_of_completing(self) -> None:
        """The raise lets the step runner mark the job failed exactly once, so no
        terminal status is written here; the evidence stays on the job row."""
        from app.services.market_data.yfinance_data_service import (
            YahooUnreachableError,
        )

        _, raised, progress = _run_serial_sweep(
            [_sweep_result(failed=20, answered=0)] * 5
        )

        assert isinstance(raised, YahooUnreachableError)
        assert "100 of 100" in str(raised)
        assert not any(
            call.get("status") in ("completed", "failed") for call in progress
        )
        assert progress[-1]["result"]["transport_failures"] == 100
        assert progress[-1]["errors"]

    def test_a_few_failed_requests_leave_the_job_completed(self) -> None:
        result, raised, progress = _run_serial_sweep(
            [_sweep_result(failed=1, answered=19)] * 5
        )

        assert raised is None
        assert result is not None
        assert result["transport_failures"] == 5
        assert result["requests_answered"] == 95
        assert progress[-1]["status"] == "completed"

    def test_half_the_requests_unanswered_is_an_outage(self) -> None:
        _, raised, _ = _run_serial_sweep([_sweep_result(failed=10, answered=10)])

        assert raised is not None

    def test_just_under_half_unanswered_is_not_an_outage(self) -> None:
        _, raised, _ = _run_serial_sweep([_sweep_result(failed=9, answered=11)])

        assert raised is None

    def test_a_sweep_that_made_no_request_completes(self) -> None:
        """Every ticker fresh: nothing was asked, so nothing can have failed."""
        result, raised, progress = _run_serial_sweep(
            [_sweep_result(failed=0, answered=0)] * 3
        )

        assert raised is None
        assert progress[-1]["status"] == "completed"
