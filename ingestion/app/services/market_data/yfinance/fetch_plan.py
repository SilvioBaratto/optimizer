"""Freshness decision of the yfinance sweep: which data categories are due.

The sweep stamps every stored category with ``updated_at`` and, in incremental
mode, skips a category whose latest stamp is still inside its window. This module
holds that decision and nothing else. A plan carries the staleness snapshot and
the reference time, so it reads no clock, database or network, and asking it the
same question twice gives the same answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from portopt_db.repositories.market_data.yfinance_repository import (
    FINANCIAL_STATEMENT_GROUPS,
    FUNDAMENTAL_CATEGORIES,
    statement_staleness_key,
)


@dataclass
class StalenessThresholds:
    """Configurable freshness thresholds per data category."""

    profile_hours: int = 24
    financials_hours: int = 168  # 7 days
    dividends_hours: int = 24
    splits_hours: int = 24
    recommendations_hours: int = 24
    price_targets_hours: int = 24
    institutional_holders_hours: int = 24
    mutualfund_holders_hours: int = 24
    insider_transactions_hours: int = 24
    etf_metadata_hours: int = 168  # 7 days
    # How long a clean empty answer for a fundamentals category or statement
    # settles it. Shorter than financials_hours because the client cannot tell
    # "no data" from "retries exhausted": a fetch that merely failed is retried
    # within days, not after a week.
    no_data_hours: int = 72
    # news: always fetched (no threshold)
    # prices: always fetched (date-ranged)
    price_overlap_days: int = 5
    min_history_tolerance: float = 0.95


class FetchMode(str, Enum):
    """How much of a ticker one sweep refreshes.

    The values are the mode strings the scheduler and the CLI already pass.
    """

    FULL = "full"
    INCREMENTAL = "incremental"


def is_fresh(updated_at: datetime | None, threshold_hours: int, now: datetime) -> bool:
    """Report whether a write is still inside its freshness window.

    Args:
        updated_at: Time of the latest write; ``None`` when nothing was stored.
            A value without a timezone is read as UTC.
        threshold_hours: Length of the window, measured back from *now*.
        now: Reference time; a value without a timezone is read as UTC.

    Returns:
        True when *updated_at* lies less than *threshold_hours* before *now*. A
        write exactly that old, or no write at all, is not fresh.
    """
    if updated_at is None:
        return False
    # A naive and an aware time cannot be subtracted, and SQLite returns stored
    # stamps without their zone.
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - updated_at) < timedelta(hours=threshold_hours)


def _window_hours_by_marker(thresholds: StalenessThresholds) -> dict[str, int]:
    """Return the freshness window of every marker the sweep gates on."""
    # Corporate-action, analyst and holder categories refresh on their own,
    # shorter windows; every fundamentals category and statement group shares
    # the fundamentals one.
    windows = {
        "profile_updated_at": thresholds.profile_hours,
        "dividends_updated_at": thresholds.dividends_hours,
        "splits_updated_at": thresholds.splits_hours,
        "recommendations_updated_at": thresholds.recommendations_hours,
        "price_targets_updated_at": thresholds.price_targets_hours,
        "institutional_holders_updated_at": thresholds.institutional_holders_hours,
        "mutualfund_holders_updated_at": thresholds.mutualfund_holders_hours,
        "insider_transactions_updated_at": thresholds.insider_transactions_hours,
    }
    for category in FUNDAMENTAL_CATEGORIES:
        windows[f"{category}_updated_at"] = thresholds.financials_hours
    for statement, period in FINANCIAL_STATEMENT_GROUPS:
        windows[statement_staleness_key(statement, period)] = (
            thresholds.financials_hours
        )
    return windows


def _window_hours(thresholds: StalenessThresholds, marker: str) -> int:
    """Return the freshness window of *marker*, or raise ``KeyError`` naming it."""
    windows = _window_hours_by_marker(thresholds)
    if marker not in windows:
        raise KeyError(f"no freshness window for staleness marker {marker!r}")
    return windows[marker]


@dataclass(frozen=True)
class FetchContext:
    """What one ticker's sweep runs with, fixed before its first request.

    Attributes:
        instrument_id: Database id the fetched rows are stored under.
        yfinance_ticker: Yahoo symbol every request is made for.
        period: Price-history period of a full fetch, e.g. ``"5y"``.
        mode: Whether categories still inside their window are skipped.
        exchange_name: Exchange whose calendar judges whether the stored prices
            are complete; ``None`` when unknown.
        currency_code: Listing currency the prices are quoted in, e.g. ``"GBX"``;
            ``None`` when unknown.
        asset_class: Asset class, consulted only when ``instrument_type`` is
            unknown to tell a bond or multi-asset fund.
        instrument_type: Yahoo instrument type such as ``"ETF"``; ``None`` when
            unknown.
        now: Reference time every freshness decision of the sweep is made at.
        thresholds: Freshness window per category.
    """

    instrument_id: UUID
    yfinance_ticker: str
    period: str
    mode: FetchMode
    exchange_name: str | None
    currency_code: str | None
    asset_class: str | None
    instrument_type: str | None
    now: datetime
    thresholds: StalenessThresholds


@dataclass(frozen=True)
class FetchPlan:
    """Which categories of one ticker are due for a fetch and which are fresh.

    Built once per ticker from the staleness snapshot taken before the first
    request. Only freshness is decided here: the sweep also counts a category
    Yahoo recently answered empty as settled, and that memory stays with the
    sweep because the sweep itself adds to it while it runs.

    Attributes:
        mode: Full fetches every category; incremental skips the fresh ones.
        staleness: Latest write per marker key, as
            ``YFinanceRepository.get_staleness_info`` reports it. ``None`` when
            unknown, which makes every category due.
        thresholds: Freshness window per category.
        now: Reference time every decision is measured against.
    """

    mode: FetchMode
    staleness: Mapping[str, Any] | None
    thresholds: StalenessThresholds
    now: datetime

    @property
    def uses_staleness(self) -> bool:
        """Whether a category can be skipped as fresh at all.

        True only for an incremental plan with a staleness snapshot. When False,
        every marker is due.
        """
        return self.mode == FetchMode.INCREMENTAL and self.staleness is not None

    def is_due(self, marker: str) -> bool:
        """Report whether the category behind *marker* has to be fetched.

        Args:
            marker: Staleness key of the category, e.g. ``"dividends_updated_at"``
                or ``"balance_sheet_annual_updated_at"``.

        Returns:
            False only when the plan uses staleness and the marker's latest write
            lies inside the window. A marker never stored is always due.

        Raises:
            KeyError: *marker* is not one the sweep gates on. Raised in every
                mode, so a misspelt marker never passes for a due one.
        """
        window = _window_hours(self.thresholds, marker)
        staleness = self.staleness
        if self.mode != FetchMode.INCREMENTAL or staleness is None:
            return True
        return not is_fresh(staleness.get(marker), window, self.now)

    def all_fresh(self, markers: Sequence[str]) -> bool:
        """Report whether none of the categories behind *markers* is due.

        Args:
            markers: Staleness keys to check, each against its own window.

        Returns:
            True when no listed marker is due, and for an empty sequence. A
            non-empty sequence is never fresh in full mode.

        Raises:
            KeyError: A listed marker is not one the sweep gates on. Every marker
                is checked, so one listed after a due marker still raises.
        """
        due = [self.is_due(marker) for marker in markers]
        return not any(due)
