"""Unit tests of the per-category runner of the yfinance sweep.

Validates ``app/services/market_data/yfinance/categories.py`` (``run_category``,
``is_settled``, ``CategorySpec``, ``FetchResult.to_response``,
``default_is_empty`` and ``is_empty_collection``) and the financial-statement
registry of ``YFinanceDataService`` in
``app/services/market_data/yfinance_data_service.py``. No requirement id is
registered for this module yet.

The savepoint tests run against the in-memory SQLite session of ``db_session``,
so a rolled-back write is checked by reading the table back, not by inspecting
calls.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pytest
from portopt_db.models.universe.universe import Exchange
from portopt_db.repositories.market_data.yfinance_repository import (
    FINANCIAL_STATEMENT_GROUPS,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.services.market_data.empty_fetch_memory import EmptyFetchMemory
from app.services.market_data.yfinance.categories import (
    CategorySpec,
    FetchResult,
    default_is_empty,
    is_empty_collection,
    is_settled,
    run_category,
)
from app.services.market_data.yfinance.fetch_plan import (
    FetchContext,
    FetchMode,
    FetchPlan,
    StalenessThresholds,
)
from app.services.market_data.yfinance_data_service import YFinanceDataService
from tests._fixtures.yfinance_fakes import make_repo, make_silent_yf_client

_NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
_THRESHOLDS = StalenessThresholds()
_INSTRUMENT = UUID("00000000-0000-0000-0000-0000000000a1")

_NAME = "eps_trend"
_MARKER = "eps_trend_updated_at"
_OTHER_MARKER = "eps_revisions_updated_at"

_REFUSAL = "Yahoo refused the request"
_ROWS = 7

_FRESH = {_MARKER: _NOW - timedelta(hours=1)}
_STALE = {_MARKER: _NOW - timedelta(hours=_THRESHOLDS.financials_hours + 1)}

_KEPT = "KEPT-EXCHANGE"
_ROLLED_BACK = "ROLLED-BACK-EXCHANGE"


class _NoDatabase:
    """Session for tests whose store touches no table: a savepoint that does nothing."""

    class _Savepoint:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_exc: object) -> None:
            return None

    def begin_nested(self) -> _Savepoint:
        """Return a savepoint that lets every exception of its block through."""
        return self._Savepoint()


def _ctx() -> FetchContext:
    return FetchContext(
        instrument_id=_INSTRUMENT,
        yfinance_ticker="EXMPL",
        period="5y",
        mode=FetchMode.INCREMENTAL,
        exchange_name=None,
        currency_code="USD",
        asset_class="equity",
        instrument_type="STOCK",
        now=_NOW,
        thresholds=_THRESHOLDS,
    )


def _plan(
    staleness: Mapping[str, Any] | None,
    mode: FetchMode = FetchMode.INCREMENTAL,
) -> FetchPlan:
    return FetchPlan(mode=mode, staleness=staleness, thresholds=_THRESHOLDS, now=_NOW)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"current": [1.0, 2.0]}, index=["0q", "+1q"])


def _fetched_frame(_ctx: FetchContext) -> pd.DataFrame:
    return _frame()


def _stored_rows(_payload: Any, _ctx: FetchContext) -> int:
    return _ROWS


def _must_not_fetch(_ctx: FetchContext) -> Any:
    raise AssertionError("a settled category was fetched")


def _must_not_store(_payload: Any, _ctx: FetchContext) -> int:
    raise AssertionError("an empty answer was stored")


def _refused(_ctx: FetchContext) -> Any:
    raise RuntimeError(_REFUSAL)


def _spec(
    fetch: Callable[[FetchContext], Any] = _fetched_frame,
    store: Callable[[Any, FetchContext], int] = _stored_rows,
    **options: Any,
) -> CategorySpec[Any]:
    return CategorySpec(name=_NAME, marker=_MARKER, fetch=fetch, store=store, **options)


def _run(
    spec: CategorySpec[Any],
    plan: FetchPlan | None = None,
    *,
    session: Any = None,
    recently_empty: Callable[[str], bool] | None = None,
) -> FetchResult:
    result = FetchResult()
    run_category(
        spec,
        _ctx(),
        plan if plan is not None else _plan(None, FetchMode.FULL),
        result,
        session=session if session is not None else _NoDatabase(),
        recently_empty=recently_empty,
    )
    return result


def _remembering(*markers: str) -> Callable[[str], bool]:
    memory = EmptyFetchMemory()
    for marker in markers:
        memory.remember(_INSTRUMENT, marker, _NOW - timedelta(hours=1))

    def _recently_empty(marker: str) -> bool:
        return memory.is_recent(_INSTRUMENT, marker, _THRESHOLDS.no_data_hours, _NOW)

    return _recently_empty


def _exchange_names(session: Session) -> set[str]:
    query = select(Exchange.name).where(Exchange.name.in_([_KEPT, _ROLLED_BACK]))
    return set(session.scalars(query))


# ---------------------------------------------------------------------------
# run_category: due check, fetch, empty check, store, error
# ---------------------------------------------------------------------------


def test_settled_category_is_skipped_without_a_fetch() -> None:
    """List a category whose marker is fresh under its name and never fetch it."""
    result = _run(_spec(fetch=_must_not_fetch), _plan(_FRESH))

    assert result.skipped == [_NAME]
    assert result.errors == []
    assert result.counts == {}


def test_failed_fetch_is_reported_by_name_and_not_counted() -> None:
    """Report a fetch that raises as ``"<name>: <message>"`` and count nothing."""
    result = _run(_spec(fetch=_refused))

    assert result.errors == [f"{_NAME}: {_REFUSAL}"]
    assert result.counts == {}
    assert result.skipped == []


@pytest.mark.parametrize("empty", [None, pd.DataFrame()], ids=["none", "no-rows"])
def test_empty_answer_counts_zero_rows_and_is_not_stored(empty: Any) -> None:
    """Count an empty answer as zero rows stored, neither an error nor a skip.

    The sweep has always reported such a category as ``0`` rather than leaving it
    out, and the empty-answer memory reads that zero, so it is kept.
    """
    result = _run(_spec(fetch=lambda _ctx: empty, store=_must_not_store))

    assert result.counts == {_NAME: 0}
    assert result.errors == []
    assert result.skipped == []


def test_stored_rows_are_counted_under_the_category_name() -> None:
    """Count what the store reports, and hand it the fetched answer and context."""
    received: list[tuple[pd.DataFrame, FetchContext]] = []

    def _store(payload: pd.DataFrame, ctx: FetchContext) -> int:
        received.append((payload, ctx))
        return _ROWS

    result = _run(_spec(store=_store))

    assert result.counts == {_NAME: _ROWS}
    assert result.errors == []
    ((payload, ctx),) = received
    pd.testing.assert_frame_equal(payload, _frame())
    assert ctx == _ctx()


def test_failed_store_rolls_back_its_savepoint_and_keeps_earlier_writes(
    db_session: Session,
) -> None:
    """Undo only the failed category's writes and report it by its own name."""
    db_session.add(Exchange(name=_KEPT))
    db_session.flush()

    def _store_then_fail(_payload: Any, _ctx: FetchContext) -> int:
        db_session.add(Exchange(name=_ROLLED_BACK))
        db_session.flush()
        raise RuntimeError(_REFUSAL)

    result = _run(_spec(store=_store_then_fail), session=db_session)

    assert result.errors == [f"{_NAME}: {_REFUSAL}"]
    assert result.counts == {}
    assert _exchange_names(db_session) == {_KEPT}


def test_rejected_statement_is_reported_by_its_cause_without_the_sql(
    db_session: Session,
) -> None:
    """Report a write the database refused by the driver's cause, not the SQL text."""
    db_session.add(Exchange(name=_KEPT))
    db_session.flush()

    def _store_duplicate(_payload: Any, _ctx: FetchContext) -> int:
        db_session.add(Exchange(name=_KEPT))
        db_session.flush()
        return 1

    result = _run(_spec(store=_store_duplicate), session=db_session)

    assert result.errors == [
        f"{_NAME}: IntegrityError: UNIQUE constraint failed: exchanges.name"
    ]
    assert result.counts == {}
    assert _exchange_names(db_session) == {_KEPT}


def test_transform_reshapes_the_answer_before_it_is_stored() -> None:
    """Store the transformed answer, not the fetched one."""
    received: list[pd.DataFrame] = []

    def _store(payload: pd.DataFrame, _ctx: FetchContext) -> int:
        received.append(payload)
        return len(payload)

    result = _run(_spec(store=_store, transform=lambda frame: frame.head(1)))

    assert result.counts == {_NAME: 1}
    pd.testing.assert_frame_equal(received[0], _frame().head(1))


def test_answer_the_transform_empties_counts_zero_rows() -> None:
    """Count zero rows and store nothing when the transform leaves nothing."""
    result = _run(_spec(store=_must_not_store, transform=lambda frame: frame.iloc[0:0]))

    assert result.counts == {_NAME: 0}
    assert result.errors == []


def test_failed_transform_is_reported_by_name() -> None:
    """Report a transform that raises like a failed fetch."""

    def _unreadable(_frame: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("no column resolved to a date")

    result = _run(_spec(store=_must_not_store, transform=_unreadable))

    assert result.errors == [f"{_NAME}: no column resolved to a date"]
    assert result.counts == {}


@pytest.mark.parametrize(
    ("answer", "expected"),
    [([], {_NAME: 0}), (None, {_NAME: 0}), ([{"type": "10-K"}], {_NAME: _ROWS})],
    ids=["empty-list", "none", "one-filing"],
)
def test_spec_decides_what_counts_as_an_empty_answer(
    answer: Any, expected: dict[str, int]
) -> None:
    """Use the spec's own emptiness rule for answers that are not frames."""
    spec = _spec(fetch=lambda _ctx: answer, is_empty=is_empty_collection)

    assert _run(spec).counts == expected


def test_unknown_marker_raises_instead_of_reporting_a_category_error() -> None:
    """Let a misspelt marker fail the sweep rather than pass as a category error."""
    spec = CategorySpec(
        name=_NAME,
        marker="eps_trnd_updated_at",
        fetch=_fetched_frame,
        store=_stored_rows,
    )

    with pytest.raises(KeyError, match="eps_trnd_updated_at"):
        _run(spec)


def test_categories_report_in_the_order_they_run() -> None:
    """Keep counts, errors and skips in the order the categories ran."""
    result = FetchResult()
    plan = _plan(_FRESH | {_OTHER_MARKER: None})
    specs = [
        CategorySpec("eps_trend", _MARKER, _must_not_fetch, _stored_rows),
        CategorySpec("eps_revisions", _OTHER_MARKER, _refused, _stored_rows),
        CategorySpec(
            "sec_filings", "sec_filings_updated_at", _fetched_frame, _stored_rows
        ),
        CategorySpec(
            "earnings_dates", "earnings_dates_updated_at", _refused, _stored_rows
        ),
        CategorySpec(
            "shares_outstanding",
            "shares_outstanding_updated_at",
            _fetched_frame,
            _stored_rows,
        ),
    ]

    for spec in specs:
        run_category(
            spec, _ctx(), plan, result, session=_NoDatabase(), recently_empty=None
        )

    assert list(result.counts) == ["sec_filings", "shares_outstanding"]
    assert result.errors == [
        f"eps_revisions: {_REFUSAL}",
        f"earnings_dates: {_REFUSAL}",
    ]
    assert result.skipped == ["eps_trend"]


# ---------------------------------------------------------------------------
# Memory of empty answers
# ---------------------------------------------------------------------------


def test_recent_empty_answer_settles_a_category_past_its_window() -> None:
    """Skip a stale category Yahoo recently answered empty, as if it were fresh."""
    result = _run(
        _spec(fetch=_must_not_fetch),
        _plan(_STALE),
        recently_empty=_remembering(_MARKER),
    )

    assert result.skipped == [_NAME]
    assert result.errors == []


def test_empty_answer_memory_is_ignored_when_the_plan_fetches_everything() -> None:
    """Fetch every category in full mode, whatever Yahoo answered before."""
    result = _run(
        _spec(),
        _plan(_STALE, FetchMode.FULL),
        recently_empty=_remembering(_MARKER),
    )

    assert result.counts == {_NAME: _ROWS}
    assert result.skipped == []


def test_empty_answer_of_another_marker_does_not_settle_the_category() -> None:
    """Only the category's own marker is looked up in the memory."""
    result = _run(
        _spec(),
        _plan(_STALE),
        recently_empty=_remembering(_OTHER_MARKER),
    )

    assert result.counts == {_NAME: _ROWS}
    assert result.skipped == []


@pytest.mark.parametrize(
    ("plan", "remembered", "settled", "reads"),
    [
        (_plan(_FRESH), True, True, []),
        (_plan(_STALE), True, True, [_MARKER]),
        (_plan(_STALE), False, False, [_MARKER]),
        (_plan(None), True, False, []),
        (_plan(_STALE, FetchMode.FULL), True, False, []),
    ],
    ids=[
        "fresh-marker",
        "stale-remembered",
        "stale-not-remembered",
        "no-staleness",
        "full-mode",
    ],
)
def test_memory_is_read_only_for_a_due_marker_of_an_incremental_plan(
    plan: FetchPlan, remembered: bool, settled: bool, reads: list[str]
) -> None:
    """Read the memory lazily: never for a fresh marker or a plan that fetches all.

    The sweep adds to the memory while it runs, so a read that is not needed
    must not happen.
    """
    asked: list[str] = []

    def _recently_empty(marker: str) -> bool:
        asked.append(marker)
        return remembered

    assert is_settled(plan, _MARKER, _recently_empty) is settled
    assert asked == reads


def test_without_a_memory_only_freshness_settles_a_category() -> None:
    """Treat a missing memory as no empty answer on record."""
    assert is_settled(_plan(_FRESH), _MARKER) is True
    assert is_settled(_plan(_STALE), _MARKER) is False


# ---------------------------------------------------------------------------
# Emptiness rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (None, True),
        (pd.DataFrame(), True),
        (pd.Series(dtype=float), True),
        (_frame(), False),
        (pd.Series([0.25]), False),
    ],
    ids=["none", "empty-frame", "empty-series", "frame", "series"],
)
def test_default_is_empty_reads_frames_and_series(payload: Any, expected: bool) -> None:
    """Treat ``None`` and a frame or series without rows as nothing to store."""
    assert default_is_empty(payload) is expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [(None, True), ([], True), ({}, True), ([{"a": 1}], False), ({"mean": 1.0}, False)],
    ids=["none", "empty-list", "empty-dict", "list", "dict"],
)
def test_is_empty_collection_reads_lists_and_dicts(
    payload: Any, expected: bool
) -> None:
    """Treat ``None`` and an empty list or dict as nothing to store."""
    assert is_empty_collection(payload) is expected


# ---------------------------------------------------------------------------
# FetchResult.to_response
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (FetchResult(), {"counts": {}, "errors": [], "skipped": []}),
        (
            FetchResult(
                counts={"profile": 1, "financials": 0},
                errors=[f"eps_trend: {_REFUSAL}"],
                skipped=["dividends", "splits"],
            ),
            {
                "counts": {"profile": 1, "financials": 0},
                "errors": [f"eps_trend: {_REFUSAL}"],
                "skipped": ["dividends", "splits"],
            },
        ),
    ],
    ids=["empty", "populated"],
)
def test_to_response_returns_the_dict_fetch_and_store_has_always_returned(
    result: FetchResult, expected: dict[str, Any]
) -> None:
    """Return the same keys in the same order, every key present even when empty."""
    response = result.to_response()

    assert list(response) == ["counts", "errors", "skipped"]
    assert repr(response) == repr(expected)
    assert json.dumps(response) == json.dumps(expected)


# ---------------------------------------------------------------------------
# Financial-statement registry of the sweep
# ---------------------------------------------------------------------------

# Which client request serves which statement, written out from Yahoo's naming
# rather than read from the service.
_STATEMENT_REQUESTS = {
    "fetch_income_stmt": "income_statement",
    "fetch_balance_sheet": "balance_sheet",
    "fetch_cashflow": "cashflow",
}


def _statement_answer(statement: str) -> Callable[..., pd.DataFrame]:
    def _answer(_symbol: str, quarterly: bool = False) -> pd.DataFrame:
        frame = pd.DataFrame(
            {pd.Timestamp("2025-12-31"): [100.0]}, index=["Total Revenue"]
        )
        frame.attrs["source"] = (statement, "quarterly" if quarterly else "annual")
        return frame

    return _answer


def test_each_statement_group_is_stored_under_the_group_it_was_fetched_for() -> None:
    """Store every statement frame under the statement and period it came from.

    Every fetcher answers a frame tagged with its own statement and period, so a
    fetcher paired with the wrong statement type, or an annual request stored as
    quarterly, shows up as a mismatched pair.
    """
    client = make_silent_yf_client()
    for request, statement in _STATEMENT_REQUESTS.items():
        getattr(client.financials, request).side_effect = _statement_answer(statement)
    stored: list[tuple[tuple[str, str], tuple[str, str]]] = []

    def _record(
        _instrument_id: UUID,
        frame: pd.DataFrame,
        statement_type: str,
        period_type: str,
        **_options: Any,
    ) -> int:
        stored.append((frame.attrs["source"], (statement_type, period_type)))
        return 1

    repo = make_repo()
    repo.upsert_financial_statements.side_effect = _record

    result = YFinanceDataService(repo, client).fetch_and_store(
        uuid4(), "EXMPL", mode="full"
    )

    assert stored == [(group, group) for group in FINANCIAL_STATEMENT_GROUPS]
    assert result["counts"]["financials"] == len(FINANCIAL_STATEMENT_GROUPS)
