"""Unit tests of the price-storage decision of the yfinance sweep.

Validates ``app/services/market_data/yfinance/price_policy.py``
(``decide_price_storage``, ``spans_stored_series``, ``PriceBasis`` and
``PriceStorageDecision``). No requirement id is registered for this module yet.

A full-period price history replaces the stored series only when that keeps every
stored close on one adjustment basis. The cases below pin which history is stored
and which is refused, including the one that used to be lost for good: a young
listing with no stored prices, whose short history the sufficiency check alone
discarded on every run.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError
from datetime import date, timedelta

import pytest

from app.services.market_data.yfinance import price_policy
from app.services.market_data.yfinance.price_policy import (
    PriceBasis,
    PriceStorageDecision,
    PriceStorageReason,
    decide_price_storage,
    spans_stored_series,
)

_SHORT = PriceBasis(sufficient=False, expected_sessions=1260, minimum_sessions=1197)
_SHORT_SPANNING = PriceBasis(
    sufficient=False,
    expected_sessions=1260,
    minimum_sessions=1197,
    spans_stored_series=True,
)
_SUFFICIENT = PriceBasis(sufficient=True, expected_sessions=1260, minimum_sessions=1197)

_FIRST_STORED = date(2024, 3, 1)


def test_young_listing_with_no_stored_prices_is_stored() -> None:
    """A short first load is a young listing's whole history, not a truncated one."""
    decision = decide_price_storage(100, young_listing=True, basis=_SHORT)

    assert decision == PriceStorageDecision(
        store=True, reason=PriceStorageReason.YOUNG_LISTING
    )


def test_sufficient_history_is_stored() -> None:
    """A history with the sessions the calendar expects replaces the stored one."""
    decision = decide_price_storage(1197, young_listing=False, basis=_SUFFICIENT)

    assert decision == PriceStorageDecision(
        store=True, reason=PriceStorageReason.SUFFICIENT_HISTORY
    )


@pytest.mark.parametrize(
    ("young_listing", "basis"),
    [
        (False, _SUFFICIENT),
        (True, _SHORT),
        (False, _SHORT_SPANNING),
        (False, _SHORT),
    ],
    ids=["sufficient", "young-listing", "spans-stored-series", "short"],
)
def test_empty_history_is_never_stored(young_listing: bool, basis: PriceBasis) -> None:
    """Nothing fetched leaves the stored series alone, whatever it is judged against."""
    decision = decide_price_storage(0, young_listing=young_listing, basis=basis)

    assert decision == PriceStorageDecision(
        store=False, reason=PriceStorageReason.NOTHING_FETCHED
    )


def test_one_row_is_judged_not_dropped_as_empty() -> None:
    """The emptiness boundary sits at zero rows: one row is a history to judge."""
    decision = decide_price_storage(1, young_listing=True, basis=_SHORT)

    assert decision.store is True
    assert decision.reason is PriceStorageReason.YOUNG_LISTING


def test_negative_length_is_rejected() -> None:
    """A history cannot have fewer than zero rows; a caller passing one has a bug."""
    with pytest.raises(ValueError, match="history_len"):
        decide_price_storage(-1, young_listing=False, basis=_SUFFICIENT)


def test_short_refetch_of_an_established_series_is_refused() -> None:
    """A truncated refetch would re-base only part of the stored series."""
    decision = decide_price_storage(800, young_listing=False, basis=_SHORT)

    assert decision == PriceStorageDecision(
        store=False, reason=PriceStorageReason.PROTECTS_STORED_SERIES
    )


def test_short_history_spanning_every_stored_bar_is_stored() -> None:
    """A history from the first stored bar on re-bases the whole series at once."""
    decision = decide_price_storage(800, young_listing=False, basis=_SHORT_SPANNING)

    assert decision == PriceStorageDecision(
        store=True, reason=PriceStorageReason.SPANS_STORED_SERIES
    )


@pytest.mark.parametrize(
    ("young_listing", "basis", "reason"),
    [
        (True, _SUFFICIENT, PriceStorageReason.SUFFICIENT_HISTORY),
        (
            False,
            PriceBasis(sufficient=True, spans_stored_series=True),
            PriceStorageReason.SUFFICIENT_HISTORY,
        ),
        (True, _SHORT_SPANNING, PriceStorageReason.YOUNG_LISTING),
    ],
    ids=[
        "sufficient-before-young-listing",
        "sufficient-before-spanning",
        "young-listing-before-spanning",
    ],
)
def test_the_first_reason_to_store_names_the_decision(
    young_listing: bool, basis: PriceBasis, reason: PriceStorageReason
) -> None:
    """Sufficiency is checked first, then a young listing, then the stored span."""
    decision = decide_price_storage(500, young_listing=young_listing, basis=basis)

    assert decision == PriceStorageDecision(store=True, reason=reason)


@pytest.mark.parametrize(
    ("opens", "expected"),
    [
        (_FIRST_STORED - timedelta(days=30), True),
        (_FIRST_STORED, True),
        (_FIRST_STORED + timedelta(days=7), True),
        (_FIRST_STORED + timedelta(days=8), False),
        (_FIRST_STORED + timedelta(days=365), False),
        (None, False),
    ],
    ids=[
        "opens-before-first-bar",
        "opens-on-first-bar",
        "opens-a-week-after",
        "opens-eight-days-after",
        "opens-a-year-after",
        "no-first-bar-date",
    ],
)
def test_history_spans_the_stored_series_up_to_a_week_after_its_first_bar(
    opens: date | None, expected: bool
) -> None:
    """A holiday or first-day gap may push the opening bar back by up to a week."""
    assert spans_stored_series(opens, _FIRST_STORED) is expected


def test_basis_defaults_to_no_span_and_no_session_counts() -> None:
    """A basis built from sufficiency alone claims no span and no calendar counts."""
    basis = PriceBasis(sufficient=True)

    assert basis.spans_stored_series is False
    assert basis.expected_sessions is None
    assert basis.minimum_sessions is None


@pytest.mark.parametrize(
    ("value", "attribute"),
    [
        (_SHORT, "sufficient"),
        (
            PriceStorageDecision(store=True, reason=PriceStorageReason.YOUNG_LISTING),
            "store",
        ),
    ],
    ids=["basis", "decision"],
)
def test_basis_and_decision_are_frozen(value: object, attribute: str) -> None:
    """Neither value can be changed once the decision is made."""
    with pytest.raises(FrozenInstanceError):
        setattr(value, attribute, True)


def test_policy_imports_nothing_that_reaches_a_clock_database_or_log() -> None:
    """The decision stays pure: only value types from the standard library."""
    tree = ast.parse(inspect.getsource(price_policy))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    clock_reads = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in {"today", "now", "utcnow"}
    ]

    assert imported <= {"__future__", "dataclasses", "datetime", "enum"}
    assert clock_reads == []
