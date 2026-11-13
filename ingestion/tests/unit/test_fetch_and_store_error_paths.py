"""Characterization of ``YFinanceDataService._fetch_and_store``, category by category.

Validates ``app/services/market_data/yfinance_data_service.py``
(``YFinanceDataService.fetch_and_store`` and the ``_fetch_and_store`` sweep it
wraps). No requirement id is registered for this function yet.

The sweep runs one block per data category: check freshness, fetch from Yahoo,
store a non-empty answer inside a savepoint, and turn an exception into an error
string. These tests pin what the returned dict reports for every category when
its fetch fails, when Yahoo answers empty and when incremental freshness skips
it, plus how empty answers are remembered between runs. They are written
against the function as it stands and must pass unchanged once it is split, so
they assert on the returned dict and on the empty-answer memory only, never on
how the service reaches its collaborators.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, PropertyMock
from uuid import UUID, uuid4

import pandas as pd
import pytest

from app.services.market_data.empty_fetch_memory import EmptyFetchMemory
from app.services.market_data.yfinance.infrastructure import CircuitBreaker
from app.services.market_data.yfinance_data_service import (
    StalenessThresholds,
    YFinanceDataService,
)
from tests.unit.test_yfinance_data_service_branches import (
    _fully_fresh,
    _make_repo,
    _make_yf_client_silent,
)

# Every repository write reports this many rows, so a count in the result proves
# the category's write ran and its answer was propagated.
_ROWS = 3

_REFUSAL = "Yahoo refused the request"

# Windows a few hours apart: a category wired to a sibling's window lands on the
# wrong side of its own boundary when stamped one hour inside or outside it.
_THRESHOLDS = StalenessThresholds(
    profile_hours=11,
    financials_hours=200,
    dividends_hours=13,
    splits_hours=17,
    recommendations_hours=19,
    price_targets_hours=23,
    institutional_holders_hours=29,
    mutualfund_holders_hours=31,
    insider_transactions_hours=37,
)

_STATEMENT_MARKERS = (
    "income_statement_annual_updated_at",
    "income_statement_quarterly_updated_at",
    "balance_sheet_annual_updated_at",
    "balance_sheet_quarterly_updated_at",
    "cashflow_annual_updated_at",
    "cashflow_quarterly_updated_at",
)

_EVENT_MARKERS = (
    "dividends_updated_at",
    "splits_updated_at",
    "recommendations_updated_at",
    "price_targets_updated_at",
    "institutional_holders_updated_at",
    "mutualfund_holders_updated_at",
    "insider_transactions_updated_at",
)

_FUNDAMENTAL_MARKERS = (
    "valuation_measures_updated_at",
    "eps_trend_updated_at",
    "eps_revisions_updated_at",
    "earnings_estimate_updated_at",
    "revenue_estimate_updated_at",
    "growth_estimates_updated_at",
    "earnings_history_updated_at",
    "earnings_dates_updated_at",
    "sec_filings_updated_at",
    "shares_outstanding_updated_at",
)

_WRITES = (
    "upsert_profile",
    "upsert_profile_extras",
    "upsert_price_history",
    "upsert_financial_statements",
    "upsert_eps_trend",
    "upsert_eps_revisions",
    "upsert_earnings_estimate",
    "upsert_revenue_estimate",
    "upsert_growth_estimates",
    "upsert_earnings_history",
    "upsert_earnings_dates",
    "upsert_analyst_actions",
    "upsert_sec_filings",
    "upsert_shares_outstanding",
    "upsert_dividends",
    "upsert_splits",
    "upsert_recommendations",
    "upsert_price_targets",
    "upsert_institutional_holders",
    "upsert_mutualfund_holders",
    "upsert_insider_transactions",
    "upsert_major_holders",
    "upsert_insider_purchases",
    "upsert_insider_roster",
    "upsert_news",
)

Answer = Callable[[MagicMock, Any], None]
Failure = Callable[[MagicMock, MagicMock, Exception], None]


def _client_method(client: MagicMock, path: str) -> MagicMock:
    target = client
    for part in path.split("."):
        target = getattr(target, part)
    return target


def _returned_by(*paths: str) -> Answer:
    def _answer(client: MagicMock, payload: Any) -> None:
        for path in paths:
            _client_method(client, path).return_value = payload

    return _answer


def _raised_by(*paths: str) -> Failure:
    def _fail(client: MagicMock, _repo: MagicMock, exc: Exception) -> None:
        for path in paths:
            _client_method(client, path).side_effect = exc

    return _fail


def _news_feed(client: MagicMock, payload: Any) -> None:
    client.get_ticker.return_value.news = payload


def _news_write_rejected(_client: MagicMock, repo: MagicMock, exc: Exception) -> None:
    # The news request runs under the shared retry, which turns any exception
    # into "no articles"; a failure reaches the news handler only after the
    # request, from the stored-article lookup or the write.
    repo.upsert_news.side_effect = exc


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"current": [1.0, 2.0]}, index=["0q", "+1q"])


def _statement() -> pd.DataFrame:
    return pd.DataFrame(
        {pd.Timestamp("2025-12-31"): [100.0, 10.0]},
        index=["Total Revenue", "Net Income"],
    )


def _valuation() -> pd.DataFrame:
    return pd.DataFrame(
        {"Current": [1e9, 20.0], "3/31/2026": [9e8, 18.0]},
        index=["Market Cap", "Trailing P/E"],
    )


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": [10.0, 11.0], "Close": [10.5, 11.5], "Volume": [100, 200]},
        index=pd.to_datetime(["2026-11-10", "2026-11-11"]),
    )


def _events() -> pd.Series:
    return pd.Series([0.25], index=pd.to_datetime(["2026-06-01"]))


_EMPTY_FRAMES = (None, pd.DataFrame())
_EMPTY_SERIES = (None, pd.Series(dtype=float))


@dataclass(frozen=True)
class Category:
    """One data category of the sweep and the levers that drive it.

    Attributes:
        name: Key of the category in ``counts`` and its label in ``skipped``.
        markers: Staleness keys that skip the category in incremental mode when
            every one of them is inside the window; empty for a category the
            sweep fetches on every run.
        window_hours: Freshness window of the markers under ``_THRESHOLDS``.
        sample: Builds an answer the category stores.
        empties: Answers the category treats as Yahoo having nothing.
        answer: Makes the client answer the category with a payload.
        fail: Makes the category raise: its fetch, or its write for news.
        stored: Counts reported when the category stores ``sample``.
        failure_errors: Errors reported, in order, when ``fail`` raises
            ``_REFUSAL``.
        counts_when_failed: Counts still reported when ``fail`` raises.
        counts_when_empty: Counts reported for an empty answer.
        settled_by_empty_answer: Whether a recent empty answer remembered for
            the markers skips the category the way a fresh marker does.
    """

    name: str
    markers: tuple[str, ...]
    window_hours: int | None
    sample: Callable[[], Any]
    empties: tuple[Any, ...]
    answer: Answer
    fail: Failure
    stored: dict[str, int]
    failure_errors: tuple[str, ...]
    counts_when_failed: dict[str, int]
    counts_when_empty: dict[str, int]
    settled_by_empty_answer: bool = True


def _block(
    name: str,
    fetcher: str,
    marker: str,
    window_hours: int,
    sample: Callable[[], Any] = _frame,
    empties: tuple[Any, ...] = _EMPTY_FRAMES,
) -> Category:
    return Category(
        name=name,
        markers=(marker,),
        window_hours=window_hours,
        sample=sample,
        empties=empties,
        answer=_returned_by(fetcher),
        fail=_raised_by(fetcher),
        stored={name: _ROWS},
        failure_errors=(f"{name}: {_REFUSAL}",),
        counts_when_failed={},
        counts_when_empty={name: 0},
    )


def _fundamental(name: str, fetcher: str, **kwargs: Any) -> Category:
    return _block(
        name, fetcher, f"{name}_updated_at", _THRESHOLDS.financials_hours, **kwargs
    )


_STATEMENT_FETCHERS = (
    "financials.fetch_income_stmt",
    "financials.fetch_balance_sheet",
    "financials.fetch_cashflow",
)

CATEGORIES: list[Category] = [
    Category(
        name="profile",
        markers=("profile_updated_at",),
        window_hours=_THRESHOLDS.profile_hours,
        sample=lambda: {"longName": "Example Corp", "financialCurrency": "USD"},
        empties=(None, {}),
        answer=_returned_by("fetch_info"),
        fail=_raised_by("fetch_info"),
        stored={"profile": _ROWS, "profile_extras": _ROWS},
        failure_errors=(f"profile: {_REFUSAL}",),
        counts_when_failed={},
        counts_when_empty={"profile": 0},
        settled_by_empty_answer=False,
    ),
    Category(
        name="prices",
        markers=(),
        window_hours=None,
        sample=_history,
        empties=_EMPTY_FRAMES,
        answer=_returned_by("fetch_history"),
        fail=_raised_by("fetch_history"),
        stored={"prices": _ROWS},
        failure_errors=(f"prices: {_REFUSAL}",),
        counts_when_failed={},
        counts_when_empty={"prices": 0},
    ),
    Category(
        name="financials",
        markers=_STATEMENT_MARKERS,
        window_hours=_THRESHOLDS.financials_hours,
        sample=_statement,
        empties=_EMPTY_FRAMES,
        answer=_returned_by(*_STATEMENT_FETCHERS),
        fail=_raised_by(*_STATEMENT_FETCHERS),
        stored={"financials": len(_STATEMENT_MARKERS) * _ROWS},
        failure_errors=(
            f"financials.income_statement.annual: {_REFUSAL}",
            f"financials.income_statement.quarterly: {_REFUSAL}",
            f"financials.balance_sheet.annual: {_REFUSAL}",
            f"financials.balance_sheet.quarterly: {_REFUSAL}",
            f"financials.cashflow.annual: {_REFUSAL}",
            f"financials.cashflow.quarterly: {_REFUSAL}",
        ),
        counts_when_failed={"financials": 0},
        counts_when_empty={"financials": 0},
    ),
    _fundamental(
        "valuation_measures", "metadata.fetch_valuation_measures", sample=_valuation
    ),
    _fundamental("eps_trend", "analysis.fetch_eps_trend"),
    _fundamental("eps_revisions", "analysis.fetch_eps_revisions"),
    _fundamental("earnings_estimate", "analysis.fetch_earnings_estimate"),
    _fundamental("revenue_estimate", "analysis.fetch_revenue_estimate"),
    _fundamental("growth_estimates", "analysis.fetch_growth_estimates"),
    _fundamental("earnings_history", "analysis.fetch_earnings_history"),
    _fundamental("earnings_dates", "metadata.fetch_earnings_dates"),
    _block(
        "analyst_actions",
        "analysis.fetch_upgrades_downgrades",
        "recommendations_updated_at",
        _THRESHOLDS.recommendations_hours,
    ),
    _fundamental(
        "sec_filings",
        "financials.fetch_sec_filings",
        sample=lambda: [{"type": "10-K", "date": "2026-02-01"}],
        empties=(None, []),
    ),
    _fundamental(
        "shares_outstanding",
        "corporate_actions.fetch_shares_full",
        sample=lambda: pd.Series([1e9], index=pd.to_datetime(["2026-11-01"])),
        empties=_EMPTY_SERIES,
    ),
    _block(
        "dividends",
        "corporate_actions.fetch_dividends",
        "dividends_updated_at",
        _THRESHOLDS.dividends_hours,
        sample=_events,
        empties=_EMPTY_SERIES,
    ),
    _block(
        "splits",
        "corporate_actions.fetch_splits",
        "splits_updated_at",
        _THRESHOLDS.splits_hours,
        sample=_events,
        empties=_EMPTY_SERIES,
    ),
    _block(
        "recommendations",
        "analysis.fetch_recommendations_summary",
        "recommendations_updated_at",
        _THRESHOLDS.recommendations_hours,
    ),
    _block(
        "price_targets",
        "analysis.fetch_analyst_price_targets",
        "price_targets_updated_at",
        _THRESHOLDS.price_targets_hours,
        sample=lambda: {"current": 100.0, "mean": 120.0},
        empties=(None, {}),
    ),
    _block(
        "institutional_holders",
        "holders.fetch_institutional_holders",
        "institutional_holders_updated_at",
        _THRESHOLDS.institutional_holders_hours,
    ),
    _block(
        "mutualfund_holders",
        "holders.fetch_mutualfund_holders",
        "mutualfund_holders_updated_at",
        _THRESHOLDS.mutualfund_holders_hours,
    ),
    _block(
        "insider_transactions",
        "holders.fetch_insider_transactions",
        "insider_transactions_updated_at",
        _THRESHOLDS.insider_transactions_hours,
    ),
    _block(
        "major_holders",
        "holders.fetch_major_holders",
        "institutional_holders_updated_at",
        _THRESHOLDS.institutional_holders_hours,
    ),
    _block(
        "insider_purchases",
        "holders.fetch_insider_purchases",
        "insider_transactions_updated_at",
        _THRESHOLDS.insider_transactions_hours,
    ),
    _block(
        "insider_roster",
        "holders.fetch_insider_roster_holders",
        "insider_transactions_updated_at",
        _THRESHOLDS.insider_transactions_hours,
    ),
    Category(
        name="news",
        markers=(),
        window_hours=None,
        # No link, so the sweep never hands the article to the page scraper.
        sample=lambda: [{"id": "news-1", "title": "Example Corp beats estimates"}],
        empties=(None, []),
        answer=_news_feed,
        fail=_news_write_rejected,
        stored={"news": _ROWS},
        failure_errors=(f"news: {_REFUSAL}",),
        counts_when_failed={},
        counts_when_empty={"news": 0},
    ),
]

_GATED = [category for category in CATEGORIES if category.markers]
_SETTLED_BY_MEMORY = [c for c in _GATED if c.settled_by_empty_answer]


def _name(category: Category) -> str:
    return category.name


def _storing_repo() -> MagicMock:
    repo = _make_repo()
    for write in _WRITES:
        getattr(repo, write).return_value = _ROWS
    repo.get_price_history.return_value = []
    repo.get_scraped_news_uuids.return_value = set()
    return repo


def _answering_client() -> MagicMock:
    client = _make_yf_client_silent()
    client.circuit_breaker = CircuitBreaker()
    for category in CATEGORIES:
        category.answer(client, category.sample())
    return client


def _sweep(
    client: MagicMock,
    mode: str,
    *,
    repo: MagicMock | None = None,
    memory: EmptyFetchMemory | None = None,
    instrument_id: UUID | None = None,
    **options: Any,
) -> dict[str, Any]:
    service = YFinanceDataService(
        repo if repo is not None else _storing_repo(), client, empty_fetches=memory
    )
    return service.fetch_and_store(
        instrument_id if instrument_id is not None else uuid4(),
        "EXMPL",
        mode=mode,
        thresholds=_THRESHOLDS,
        **options,
    )


def _stored_counts(*excluding: str) -> dict[str, int]:
    return {
        key: rows
        for category in CATEGORIES
        if category.name not in excluding
        for key, rows in category.stored.items()
    }


def _sharing_markers(category: Category) -> set[str]:
    return {other.name for other in CATEGORIES if other.markers == category.markers}


def _stamped(category: Category, hours_past_window: int) -> dict[str, Any]:
    age = timedelta(hours=(category.window_hours or 0) + hours_past_window)
    stamp = datetime.now(timezone.utc) - age
    return {"price_max_date": None, **dict.fromkeys(category.markers, stamp)}


def _remembered(
    memory: EmptyFetchMemory, instrument_id: UUID, markers: Iterable[str]
) -> set[str]:
    now = datetime.now(timezone.utc)
    return {
        marker
        for marker in markers
        if memory.is_recent(instrument_id, marker, _THRESHOLDS.no_data_hours, now)
    }


@pytest.mark.parametrize("category", CATEGORIES, ids=_name)
def test_category_failure_is_recorded_and_other_categories_still_store(
    category: Category,
) -> None:
    """Report a failed category by its own error and keep storing every other one."""
    client = _answering_client()
    repo = _storing_repo()
    category.fail(client, repo, RuntimeError(_REFUSAL))

    result = _sweep(client, "full", repo=repo)

    assert result["errors"] == list(category.failure_errors)
    assert result["counts"] == {
        **_stored_counts(category.name),
        **category.counts_when_failed,
    }
    assert result["skipped"] == []
    assert result["transport_failures"] == 0


_EMPTY_ANSWERS = [
    pytest.param(category, empty, id=f"{category.name}-{type(empty).__name__}")
    for category in CATEGORIES
    for empty in category.empties
]


@pytest.mark.parametrize(("category", "empty"), _EMPTY_ANSWERS)
def test_empty_answer_is_counted_as_zero_not_errored(
    category: Category, empty: Any
) -> None:
    """Count an empty answer as nothing stored, neither an error nor a skip."""
    client = _answering_client()
    category.answer(client, empty)

    result = _sweep(client, "full")

    assert result["errors"] == []
    assert result["skipped"] == []
    assert result["counts"] == {
        **_stored_counts(category.name),
        **category.counts_when_empty,
    }


@pytest.mark.parametrize("category", _GATED, ids=_name)
def test_incremental_fresh_category_is_skipped(category: Category) -> None:
    """Skip a category whose markers sit inside its window.

    Its fetch is set to raise, so a fetch made for the category despite the fresh
    markers would surface as its error. Categories gated on the same markers are
    skipped with it.
    """
    client = _answering_client()
    repo = _storing_repo()
    category.fail(client, repo, RuntimeError(_REFUSAL))
    sharing = _sharing_markers(category)

    result = _sweep(client, "incremental", repo=repo, staleness=_stamped(category, -1))

    assert sorted(result["skipped"]) == sorted(sharing)
    assert result["errors"] == []
    assert result["counts"] == _stored_counts(*sharing)


@pytest.mark.parametrize("category", _GATED, ids=_name)
def test_incremental_category_past_its_window_is_fetched(category: Category) -> None:
    """Fetch and store a category whose markers are just past its own window."""
    result = _sweep(_answering_client(), "incremental", staleness=_stamped(category, 1))

    assert result["skipped"] == []
    assert result["errors"] == []
    assert result["counts"] == _stored_counts()


@pytest.mark.parametrize("category", _SETTLED_BY_MEMORY, ids=_name)
def test_recent_empty_answer_settles_the_category_like_a_fresh_marker(
    category: Category,
) -> None:
    """Skip a category whose markers Yahoo recently answered empty for.

    The profile is left out: it is gated on its marker alone, and no step ever
    remembers an empty profile.
    """
    instrument_id = uuid4()
    memory = EmptyFetchMemory()
    for marker in category.markers:
        memory.remember(instrument_id, marker, datetime.now(timezone.utc))
    client = _answering_client()
    repo = _storing_repo()
    category.fail(client, repo, RuntimeError(_REFUSAL))
    sharing = _sharing_markers(category)

    result = _sweep(
        client,
        "incremental",
        repo=repo,
        memory=memory,
        instrument_id=instrument_id,
        staleness={"price_max_date": None},
    )

    assert sorted(result["skipped"]) == sorted(sharing)
    assert result["errors"] == []
    assert result["counts"] == _stored_counts(*sharing)


def test_all_fresh_short_circuits_without_network() -> None:
    """Return before touching the client when every category and price is current.

    The client has no attributes at all, so any request would raise.
    """
    repo = _make_repo(staleness=_fully_fresh(datetime.now(timezone.utc)))
    unreachable_client = MagicMock(spec=[])

    result = YFinanceDataService(repo, unreachable_client).fetch_and_store(
        uuid4(), "EXMPL", mode="incremental"
    )

    assert result == {
        "counts": {},
        "errors": [],
        "skipped": ["all:fresh"],
        "transport_failures": 0,
        "requests_answered": 0,
    }


def test_unreachable_statements_client_fails_the_block_as_one_error() -> None:
    """Report one ``financials`` error, not one per group, when the sub-client fails.

    The statement fetchers are looked up before the per-group handlers run, so
    the block's outer handler catches the failure. SEC filings use the same
    sub-client and fail on their own.
    """
    client = _answering_client()
    type(client).financials = PropertyMock(side_effect=RuntimeError(_REFUSAL))

    result = _sweep(client, "full")

    assert result["errors"] == [f"financials: {_REFUSAL}", f"sec_filings: {_REFUSAL}"]
    assert result["counts"] == _stored_counts("financials", "sec_filings")


def test_isin_failure_is_reported_without_costing_the_profile() -> None:
    """Report a failed ISIN lookup on its own and still store the profile."""
    client = _answering_client()
    client.metadata.fetch_isin.side_effect = RuntimeError(_REFUSAL)

    result = _sweep(client, "full")

    assert result["errors"] == [f"isin: {_REFUSAL}"]
    assert result["counts"] == _stored_counts()


def test_statement_groups_remember_empty_answers_group_by_group() -> None:
    """Remember each empty statement group on its own, even beside a failed one.

    A failed group and a group that stored rows are not remembered; an empty
    group is, whatever its sibling period did.
    """
    instrument_id = uuid4()
    memory = EmptyFetchMemory()
    client = _answering_client()

    def _income(_symbol: str, quarterly: bool = False) -> pd.DataFrame:
        if quarterly:
            return pd.DataFrame()
        raise RuntimeError(_REFUSAL)

    client.financials.fetch_income_stmt.side_effect = _income
    client.financials.fetch_cashflow.return_value = None

    result = _sweep(client, "full", memory=memory, instrument_id=instrument_id)

    assert result["errors"] == [f"financials.income_statement.annual: {_REFUSAL}"]
    assert result["counts"]["financials"] == 2 * _ROWS
    assert _remembered(memory, instrument_id, _STATEMENT_MARKERS) == {
        "income_statement_quarterly_updated_at",
        "cashflow_annual_updated_at",
        "cashflow_quarterly_updated_at",
    }


def test_fundamental_categories_remember_only_answered_empty_ones() -> None:
    """Remember a fundamentals category that answered without rows, not a failed one.

    A valuation panel whose columns resolve to no date stores nothing and is
    remembered like an empty answer.
    """
    instrument_id = uuid4()
    memory = EmptyFetchMemory()
    client = _answering_client()
    client.analysis.fetch_eps_trend.return_value = None
    client.analysis.fetch_eps_revisions.side_effect = RuntimeError(_REFUSAL)
    client.metadata.fetch_valuation_measures.return_value = pd.DataFrame(
        {"5 years ago": [1e9]}, index=["Market Cap"]
    )

    result = _sweep(client, "full", memory=memory, instrument_id=instrument_id)

    assert result["errors"] == [f"eps_revisions: {_REFUSAL}"]
    assert result["counts"]["valuation_measures"] == 0
    assert _remembered(memory, instrument_id, _FUNDAMENTAL_MARKERS) == {
        "eps_trend_updated_at",
        "valuation_measures_updated_at",
    }


def test_event_marker_is_remembered_only_when_all_its_categories_are_empty() -> None:
    """Remember a shared marker only when every category behind it answered empty.

    One failed category, or one that stored rows, keeps the marker retryable for
    the categories that came back empty.
    """
    instrument_id = uuid4()
    memory = EmptyFetchMemory()
    client = _answering_client()
    client.analysis.fetch_recommendations_summary.return_value = None
    client.analysis.fetch_upgrades_downgrades.side_effect = RuntimeError(_REFUSAL)
    client.holders.fetch_institutional_holders.return_value = None
    client.holders.fetch_major_holders.return_value = None
    client.holders.fetch_insider_transactions.return_value = None
    client.holders.fetch_insider_roster_holders.return_value = None
    client.corporate_actions.fetch_dividends.return_value = None

    result = _sweep(client, "full", memory=memory, instrument_id=instrument_id)

    assert result["errors"] == [f"analyst_actions: {_REFUSAL}"]
    assert _remembered(memory, instrument_id, _EVENT_MARKERS) == {
        "institutional_holders_updated_at",
        "dividends_updated_at",
    }
