"""Unit tests of the freshness decision of the yfinance sweep.

Validates ``app/services/market_data/yfinance/fetch_plan.py`` (``FetchPlan.is_due``,
``FetchPlan.all_fresh``, ``FetchPlan.uses_staleness``, ``FetchMode`` and
``FetchContext``). No requirement id is registered for this module yet.

Every expected window comes from ``_WINDOW_BY_MARKER``, written out from the
thresholds below, never from the module under test. The windows are a few hours
apart, so a marker wired to a sibling's window lands on the wrong side of its own
boundary when stamped one hour inside or outside it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import pytest
from portopt_db.repositories.market_data.yfinance_repository import (
    FINANCIAL_STATEMENT_GROUPS,
    FUNDAMENTAL_CATEGORIES,
)

from app.services.market_data.yfinance.fetch_plan import (
    FetchContext,
    FetchMode,
    FetchPlan,
    StalenessThresholds,
)

_NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)

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

_STATEMENT_MARKERS = (
    "income_statement_annual_updated_at",
    "income_statement_quarterly_updated_at",
    "balance_sheet_annual_updated_at",
    "balance_sheet_quarterly_updated_at",
    "cashflow_annual_updated_at",
    "cashflow_quarterly_updated_at",
)

_WINDOW_BY_MARKER: dict[str, int] = {
    "profile_updated_at": 11,
    "dividends_updated_at": 13,
    "splits_updated_at": 17,
    "recommendations_updated_at": 19,
    "price_targets_updated_at": 23,
    "institutional_holders_updated_at": 29,
    "mutualfund_holders_updated_at": 31,
    "insider_transactions_updated_at": 37,
    **dict.fromkeys(_FUNDAMENTAL_MARKERS, 200),
    **dict.fromkeys(_STATEMENT_MARKERS, 200),
}

_MARKERS = tuple(_WINDOW_BY_MARKER)

_UNKNOWN_MARKER = "news_updated_at"


def _plan(
    staleness: Mapping[str, Any] | None,
    mode: FetchMode = FetchMode.INCREMENTAL,
) -> FetchPlan:
    """Build a plan over the test thresholds at the fixed reference time."""
    return FetchPlan(mode=mode, staleness=staleness, thresholds=_THRESHOLDS, now=_NOW)


def _aged(hours: float) -> datetime:
    """Return the stamp of a write made *hours* before the reference time."""
    return _NOW - timedelta(hours=hours)


def _all_stamped(age_hours: float) -> dict[str, datetime]:
    """Return a staleness snapshot with every marker written *age_hours* ago."""
    return dict.fromkeys(_MARKERS, _aged(age_hours))


def test_marker_table_covers_every_category_the_repository_reports() -> None:
    """The test's marker table stays in step with the repository's categories."""
    assert set(_FUNDAMENTAL_MARKERS) == {
        f"{category}_updated_at" for category in FUNDAMENTAL_CATEGORIES
    }
    assert set(_STATEMENT_MARKERS) == {
        f"{statement}_{period}_updated_at"
        for statement, period in FINANCIAL_STATEMENT_GROUPS
    }


@pytest.mark.parametrize("marker", _MARKERS)
def test_full_mode_makes_every_marker_due(marker: str) -> None:
    """A full fetch refreshes a category even when it was just written."""
    plan = _plan(_all_stamped(0), mode=FetchMode.FULL)

    assert plan.is_due(marker) is True


@pytest.mark.parametrize("marker", _MARKERS)
def test_incremental_without_staleness_makes_every_marker_due(marker: str) -> None:
    """With no staleness snapshot nothing is known to be fresh."""
    assert _plan(None).is_due(marker) is True


@pytest.mark.parametrize("marker", _MARKERS)
def test_marker_inside_its_window_is_not_due(marker: str) -> None:
    """A marker written one hour inside its own window is skipped."""
    staleness = {marker: _aged(_WINDOW_BY_MARKER[marker] - 1)}

    assert _plan(staleness).is_due(marker) is False


@pytest.mark.parametrize("marker", _MARKERS)
def test_marker_past_its_window_is_due(marker: str) -> None:
    """A marker written one hour past its own window is fetched again."""
    staleness = {marker: _aged(_WINDOW_BY_MARKER[marker] + 1)}

    assert _plan(staleness).is_due(marker) is True


@pytest.mark.parametrize("marker", _MARKERS)
def test_marker_exactly_at_its_window_is_due(marker: str) -> None:
    """The window is open at its end: a write exactly that old is stale."""
    staleness = {marker: _aged(_WINDOW_BY_MARKER[marker])}

    assert _plan(staleness).is_due(marker) is True


@pytest.mark.parametrize(
    "staleness",
    [{}, {"dividends_updated_at": None}],
    ids=["marker-absent", "marker-none"],
)
def test_marker_without_a_stored_timestamp_is_due(
    staleness: dict[str, Any],
) -> None:
    """A category never stored has no timestamp and is always fetched."""
    assert _plan(staleness).is_due("dividends_updated_at") is True


def test_naive_timestamp_is_read_as_utc() -> None:
    """A stamp without a timezone compares as UTC against an aware reference."""
    plan = _plan({"dividends_updated_at": _aged(1).replace(tzinfo=None)})

    assert plan.is_due("dividends_updated_at") is False


def test_naive_reference_time_is_read_as_utc() -> None:
    """A reference time without a timezone compares as UTC against aware stamps."""
    plan = FetchPlan(
        mode=FetchMode.INCREMENTAL,
        staleness={"dividends_updated_at": _aged(1)},
        thresholds=_THRESHOLDS,
        now=_NOW.replace(tzinfo=None),
    )

    assert plan.is_due("dividends_updated_at") is False


@pytest.mark.parametrize(
    ("age_hours", "fresh_hours", "expected"),
    [
        (20, 24, False),
        (20, None, True),
        (5, 4, True),
        (5, None, False),
    ],
    ids=["longer-window-keeps-it", "own-window-expired", "shorter-window", "own"],
)
def test_fresh_hours_replaces_the_marker_window(
    age_hours: int, fresh_hours: int | None, expected: bool
) -> None:
    """An explicit window overrides the marker's own 13 h dividends window."""
    plan = _plan({"dividends_updated_at": _aged(age_hours)})

    assert plan.is_due("dividends_updated_at", fresh_hours) is expected


@pytest.mark.parametrize(
    ("mode", "staleness"),
    [
        (FetchMode.INCREMENTAL, {_UNKNOWN_MARKER: _NOW}),
        (FetchMode.INCREMENTAL, None),
        (FetchMode.FULL, None),
    ],
    ids=["incremental", "incremental-no-staleness", "full"],
)
def test_unknown_marker_raises_key_error_naming_it(
    mode: FetchMode, staleness: dict[str, Any] | None
) -> None:
    """A marker with no window is a programming error in every mode."""
    with pytest.raises(KeyError, match=_UNKNOWN_MARKER):
        _plan(staleness, mode=mode).is_due(_UNKNOWN_MARKER)


def test_fresh_hours_does_not_admit_an_unknown_marker() -> None:
    """An explicit window does not stand in for a marker the plan does not know."""
    with pytest.raises(KeyError, match=_UNKNOWN_MARKER):
        _plan({}).is_due(_UNKNOWN_MARKER, fresh_hours=24)


@pytest.mark.parametrize(
    ("staleness", "markers", "expected"),
    [
        (_all_stamped(1), _MARKERS, True),
        (
            {**_all_stamped(1), "splits_updated_at": _aged(18)},
            _MARKERS,
            False,
        ),
        ({**_all_stamped(1), "splits_updated_at": None}, _MARKERS, False),
        (
            {**_all_stamped(1), "splits_updated_at": _aged(18)},
            ("dividends_updated_at", "recommendations_updated_at"),
            True,
        ),
        ({}, (), True),
    ],
    ids=[
        "every-marker-fresh",
        "one-marker-stale",
        "one-marker-never-stored",
        "stale-marker-not-listed",
        "nothing-listed",
    ],
)
def test_all_fresh_only_when_every_listed_marker_is_fresh(
    staleness: dict[str, Any], markers: tuple[str, ...], expected: bool
) -> None:
    """``all_fresh`` holds only when no listed marker is due."""
    assert _plan(staleness).all_fresh(markers) is expected


def test_all_fresh_is_false_in_full_mode() -> None:
    """A full fetch never counts a listed marker as fresh."""
    plan = _plan(_all_stamped(0), mode=FetchMode.FULL)

    assert plan.all_fresh(_MARKERS) is False


def test_all_fresh_rejects_an_unknown_marker_after_a_due_one() -> None:
    """Every listed marker is checked, so a typo surfaces even after a due one."""
    plan = _plan({})

    with pytest.raises(KeyError, match=_UNKNOWN_MARKER):
        plan.all_fresh(("dividends_updated_at", _UNKNOWN_MARKER))


@pytest.mark.parametrize(
    ("mode", "staleness", "expected"),
    [
        (FetchMode.INCREMENTAL, {}, True),
        (FetchMode.INCREMENTAL, None, False),
        (FetchMode.FULL, {}, False),
    ],
    ids=["incremental-with-staleness", "incremental-no-staleness", "full"],
)
def test_uses_staleness_only_for_an_incremental_plan_with_a_snapshot(
    mode: FetchMode, staleness: dict[str, Any] | None, expected: bool
) -> None:
    """Only an incremental plan with a snapshot can find a category fresh."""
    assert _plan(staleness, mode=mode).uses_staleness is expected


@pytest.mark.parametrize(
    ("value", "member"),
    [("full", FetchMode.FULL), ("incremental", FetchMode.INCREMENTAL)],
)
def test_fetch_mode_matches_the_mode_strings_callers_pass(
    value: str, member: FetchMode
) -> None:
    """The modes keep the strings the sweep and the CLI already pass."""
    assert FetchMode(value) is member
    assert member == value


def test_fetch_plan_is_immutable() -> None:
    """A plan cannot be changed once the sweep has started deciding with it."""
    plan = _plan({})

    with pytest.raises(FrozenInstanceError):
        plan.mode = FetchMode.FULL  # type: ignore[misc]


def test_fetch_context_carries_its_fields_and_is_immutable() -> None:
    """The context keeps what it was built with and cannot be changed."""
    instrument_id = uuid4()
    context = FetchContext(
        instrument_id=instrument_id,
        yfinance_ticker="SHEL.L",
        period="5y",
        mode=FetchMode.INCREMENTAL,
        exchange_name="LSE",
        currency_code="GBX",
        asset_class="equity",
        instrument_type="STOCK",
        now=_NOW,
        thresholds=_THRESHOLDS,
    )

    assert context.instrument_id == instrument_id
    assert context.mode is FetchMode.INCREMENTAL
    assert context.thresholds is _THRESHOLDS
    with pytest.raises(FrozenInstanceError):
        context.period = "1y"  # type: ignore[misc]
