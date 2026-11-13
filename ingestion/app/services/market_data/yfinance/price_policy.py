"""Whether a fetched price history may replace the stored price series.

yfinance adjusts every close for splits and dividends as of the fetch date, so the
stored series stays on one adjustment basis only while each fetch rewrites all of
it. A full-period history that comes back short therefore must not be stored over
an established series: it would re-base only its own span. Two short histories are
still safe to store: a young listing's first load, which is its whole history, and
one that reaches back to the first stored bar.

The decision is pure. The sweep looks up the facts (the exchange calendar, the
stored bars) and this module says what to do with the history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

# A history may open a few sessions after the first stored bar (a holiday, a
# first-day gap) and still rewrite the whole stored series.
REBASE_START_TOLERANCE = timedelta(days=7)


class PriceStorageReason(str, Enum):
    """Why a fetched full-period history is stored or refused."""

    NOTHING_FETCHED = "nothing_fetched"
    SUFFICIENT_HISTORY = "sufficient_history"
    YOUNG_LISTING = "young_listing"
    SPANS_STORED_SERIES = "spans_stored_series"
    PROTECTS_STORED_SERIES = "protects_stored_series"


@dataclass(frozen=True)
class PriceBasis:
    """How a fetched full-period history measures up to what it would replace.

    Attributes:
        sufficient: The history holds the share of sessions the exchange calendar
            expects over the window it is judged on, or no calendar judges it. The
            window is the period, or the stored series' lifetime when the period
            alone finds the history short.
        spans_stored_series: The history opens no later than
            ``REBASE_START_TOLERANCE`` after the first stored bar, so storing it
            rewrites every stored close on one basis.
        expected_sessions: Sessions the calendar schedules over the judged
            window; ``None`` when no calendar judged it.
        minimum_sessions: Fewest rows that count as sufficient; ``None`` when no
            calendar judged it.
    """

    sufficient: bool
    spans_stored_series: bool = False
    expected_sessions: int | None = None
    minimum_sessions: int | None = None


@dataclass(frozen=True)
class PriceStorageDecision:
    """What the sweep does with a fetched full-period history.

    Attributes:
        store: Whether the history replaces the stored series.
        reason: The first rule that decided it.
    """

    store: bool
    reason: PriceStorageReason


def spans_stored_series(history_start: date | None, first_stored: date) -> bool:
    """Report whether a history reaches back to the first stored bar.

    Args:
        history_start: Date of the history's opening bar; ``None`` when it has no
            readable date.
        first_stored: Date of the oldest bar already stored.

    Returns:
        True when the history opens on or before *first_stored* plus
        ``REBASE_START_TOLERANCE``.
    """
    return (
        history_start is not None
        and history_start <= first_stored + REBASE_START_TOLERANCE
    )


def decide_price_storage(
    history_len: int, young_listing: bool, basis: PriceBasis
) -> PriceStorageDecision:
    """Decide whether a fetched full-period history replaces the stored series.

    The rules are checked in order and the first that applies decides: nothing
    fetched is never stored; a sufficient history is stored; a young listing's
    short history is stored, because it is the listing's whole history; a short
    history that spans the stored series is stored; any other short history is
    refused, so it cannot re-base part of an established series.

    Args:
        history_len: Rows in the fetched history.
        young_listing: True when no price is stored for the instrument yet, so a
            short history is taken for a young listing's whole history rather than
            a truncated refetch. Consulted only for a short history.
        basis: How the history measures up to what it would replace.

    Returns:
        Whether to store the history, and the rule that decided it.

    Raises:
        ValueError: *history_len* is negative.
    """
    if history_len < 0:
        raise ValueError(f"history_len must not be negative, got {history_len}")
    if history_len == 0:
        return PriceStorageDecision(False, PriceStorageReason.NOTHING_FETCHED)
    if basis.sufficient:
        return PriceStorageDecision(True, PriceStorageReason.SUFFICIENT_HISTORY)
    # Judged on sufficiency alone, a recent listing's first load was discarded on
    # every run, which kept ~102 instruments out of price_history without an error.
    if young_listing:
        return PriceStorageDecision(True, PriceStorageReason.YOUNG_LISTING)
    if basis.spans_stored_series:
        return PriceStorageDecision(True, PriceStorageReason.SPANS_STORED_SERIES)
    return PriceStorageDecision(False, PriceStorageReason.PROTECTS_STORED_SERIES)
