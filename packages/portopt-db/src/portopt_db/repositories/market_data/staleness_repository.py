"""Repository for an instrument's stored price dates and category write stamps."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import func, select

from portopt_db.models.market_data.yfinance_data import (
    AnalystPriceTarget,
    AnalystRecommendation,
    Dividend,
    EarningsDate,
    EarningsEstimate,
    EarningsHistory,
    EpsRevisions,
    EpsTrend,
    FinancialStatement,
    GrowthEstimate,
    InsiderTransaction,
    InstitutionalHolder,
    MutualFundHolder,
    PriceHistory,
    RevenueEstimate,
    SecFiling,
    SharesOutstanding,
    StockSplit,
    TickerNews,
    TickerProfile,
)
from portopt_db.repository import RepositoryBase

# (statement_type, period_type) pairs written by the six statement fetches.
FINANCIAL_STATEMENT_GROUPS: tuple[tuple[str, str], ...] = (
    ("income_statement", "annual"),
    ("income_statement", "quarterly"),
    ("balance_sheet", "annual"),
    ("balance_sheet", "quarterly"),
    ("cashflow", "annual"),
    ("cashflow", "quarterly"),
)


# valuation_measures rows share financial_statements with the statements but
# are refreshed independently, so they get their own marker.
VALUATION_MEASURES_STATEMENT_TYPE = "valuation_measures"
VALUATION_MEASURES_PERIOD_TYPE = "point_in_time"

_FUNDAMENTALS_CATEGORY_MODELS: tuple[tuple[str, type], ...] = (
    ("eps_trend", EpsTrend),
    ("eps_revisions", EpsRevisions),
    ("earnings_estimate", EarningsEstimate),
    ("revenue_estimate", RevenueEstimate),
    ("growth_estimates", GrowthEstimate),
    ("earnings_history", EarningsHistory),
    ("earnings_dates", EarningsDate),
    ("sec_filings", SecFiling),
    ("shares_outstanding", SharesOutstanding),
)


# Categories refreshed on the fundamentals window. Each one reports its own
# ``{category}_updated_at`` so a failed or empty fetch of one category is not
# hidden by the others being fresh.
FUNDAMENTAL_CATEGORIES: tuple[str, ...] = (
    "valuation_measures",
    *(category for category, _ in _FUNDAMENTALS_CATEGORY_MODELS),
)


_STALENESS_CATEGORY_MODELS: tuple[tuple[str, type], ...] = (
    ("profile", TickerProfile),
    ("dividends", Dividend),
    ("splits", StockSplit),
    ("recommendations", AnalystRecommendation),
    ("price_targets", AnalystPriceTarget),
    ("institutional_holders", InstitutionalHolder),
    ("mutualfund_holders", MutualFundHolder),
    ("insider_transactions", InsiderTransaction),
    ("news", TickerNews),
    *_FUNDAMENTALS_CATEGORY_MODELS,
)


def statement_staleness_key(statement_type: str, period_type: str) -> str:
    """Return the staleness-dict key for one financial statement group.

    Args:
        statement_type: Statement label, e.g. ``"balance_sheet"``.
        period_type: ``"annual"`` or ``"quarterly"``.

    Returns:
        The key under which :meth:`YFinanceRepository.get_staleness_info`
        reports that group's last write, e.g. ``"balance_sheet_annual_updated_at"``.
    """
    return f"{statement_type}_{period_type}_updated_at"


def _statement_staleness(
    latest: Mapping[tuple[str, str], datetime | None],
) -> dict[str, Any]:
    """Build the financial_statements entries of a staleness dict.

    Args:
        latest: ``MAX(updated_at)`` per ``(statement_type, period_type)`` group.

    Returns:
        One marker per statement group, ``financials_updated_at`` and
        ``valuation_measures_updated_at``. ``financials_updated_at`` is the
        *oldest* of the six statement groups, and ``None`` while any group has
        never been stored: a statement that failed during a rate-limit burst
        must not be hidden by one that succeeded.
    """
    entries: dict[str, Any] = {
        statement_staleness_key(stmt, period): latest.get((stmt, period))
        for stmt, period in FINANCIAL_STATEMENT_GROUPS
    }
    stamps = list(entries.values())
    entries["financials_updated_at"] = (
        None if any(stamp is None for stamp in stamps) else min(stamps)
    )
    entries["valuation_measures_updated_at"] = latest.get(
        (VALUATION_MEASURES_STATEMENT_TYPE, VALUATION_MEASURES_PERIOD_TYPE)
    )
    return entries


class StalenessRepository(RepositoryBase):
    """Price date range and category last-write stamps for incremental fetches."""

    def get_staleness_info(self, instrument_id: UUID) -> dict[str, Any]:
        """Return staleness metadata for incremental fetch decisions.

        Returns a dict with:
          - price_max_date: MAX(date) from price_history (date or None)
          - price_min_date: MIN(date) from price_history (date or None)
          - {category}_updated_at: MAX(updated_at) from each non-price table (datetime or None)
          - {statement_type}_{period_type}_updated_at: MAX(updated_at) of each of
            the six financial statement groups
          - financials_updated_at: the oldest of those six, ``None`` while any
            group is missing
          - valuation_measures_updated_at: MAX(updated_at) of the valuation rows
            stored alongside the statements
        """
        result: dict[str, Any] = {}

        price_max, price_min = self.session.execute(
            select(func.max(PriceHistory.date), func.min(PriceHistory.date)).where(
                PriceHistory.instrument_id == instrument_id
            )
        ).one()
        result["price_max_date"] = price_max
        result["price_min_date"] = price_min

        for category, model in _STALENESS_CATEGORY_MODELS:
            val = self.session.execute(
                select(func.max(cast(Any, model).updated_at)).where(
                    cast(Any, model).instrument_id == instrument_id
                )
            ).scalar_one_or_none()
            result[f"{category}_updated_at"] = val

        latest_by_group: dict[tuple[str, str], datetime | None] = {
            (stmt, period): stamp
            for stmt, period, stamp in self.session.execute(
                select(
                    FinancialStatement.statement_type,
                    FinancialStatement.period_type,
                    func.max(FinancialStatement.updated_at),
                )
                .where(FinancialStatement.instrument_id == instrument_id)
                .group_by(
                    FinancialStatement.statement_type,
                    FinancialStatement.period_type,
                )
            ).all()
        }
        result.update(_statement_staleness(latest_by_group))

        return result

    def get_staleness_info_bulk(
        self, instrument_ids: Sequence[UUID]
    ) -> dict[UUID, dict[str, Any]]:
        """Batched :meth:`get_staleness_info` for a whole sweep.

        Returns one full staleness dict per requested id (all-None for an
        instrument with no rows yet), matching the single-query shape exactly —
        but via one grouped query per category instead of one per instrument.
        """
        ids = list(instrument_ids)
        result: dict[UUID, dict[str, Any]] = {
            iid: {
                "price_max_date": None,
                "price_min_date": None,
                **{f"{cat}_updated_at": None for cat, _ in _STALENESS_CATEGORY_MODELS},
                **_statement_staleness({}),
            }
            for iid in ids
        }
        if not ids:
            return result

        for iid, mx, mn in self.session.execute(
            select(
                PriceHistory.instrument_id,
                func.max(PriceHistory.date),
                func.min(PriceHistory.date),
            )
            .where(PriceHistory.instrument_id.in_(ids))
            .group_by(PriceHistory.instrument_id)
        ).all():
            result[iid]["price_max_date"] = mx
            result[iid]["price_min_date"] = mn

        for category, model in _STALENESS_CATEGORY_MODELS:
            m = cast(Any, model)
            for iid, mx in self.session.execute(
                select(m.instrument_id, func.max(m.updated_at))
                .where(m.instrument_id.in_(ids))
                .group_by(m.instrument_id)
            ).all():
                result[iid][f"{category}_updated_at"] = mx

        latest_by_instrument: dict[UUID, dict[tuple[str, str], datetime | None]] = {}
        for iid, stmt, period, stamp in self.session.execute(
            select(
                FinancialStatement.instrument_id,
                FinancialStatement.statement_type,
                FinancialStatement.period_type,
                func.max(FinancialStatement.updated_at),
            )
            .where(FinancialStatement.instrument_id.in_(ids))
            .group_by(
                FinancialStatement.instrument_id,
                FinancialStatement.statement_type,
                FinancialStatement.period_type,
            )
        ).all():
            latest_by_instrument.setdefault(iid, {})[(stmt, period)] = stamp
        for iid, latest in latest_by_instrument.items():
            result[iid].update(_statement_staleness(latest))

        return result
