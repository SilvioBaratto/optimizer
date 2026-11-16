"""Repository for yfinance data access with PostgreSQL upsert support."""

import logging
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any, cast
from uuid import UUID

from sqlalchemy import and_, exists, func, select
from sqlalchemy.orm import Session, aliased, joinedload

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
from portopt_db.models.universe.pinned_instrument import PinnedInstrument
from portopt_db.models.universe.universe import Instrument
from portopt_db.ports import InstrumentSummary, PriceBar, ProfileSummary

# Re-exported: tests import these helpers from this module.
from portopt_db.repositories.market_data._mapping import _safe_date as _safe_date
from portopt_db.repositories.market_data._mapping import _safe_float as _safe_float
from portopt_db.repositories.market_data._mapping import _safe_int as _safe_int
from portopt_db.repositories.market_data._mapping import _safe_str as _safe_str
from portopt_db.repositories.market_data._mapping import _safe_val as _safe_val
from portopt_db.repositories.market_data.analyst_repository import AnalystRepository
from portopt_db.repositories.market_data.fundamentals_repository import (
    FundamentalsRepository,
)
from portopt_db.repositories.market_data.news_repository import NewsRepository
from portopt_db.repositories.market_data.options_repository import OptionsRepository

# Re-exported: tests import it from this module.
from portopt_db.repositories.market_data.ownership_repository import (
    _SENTINEL_DATE as _SENTINEL_DATE,
)
from portopt_db.repositories.market_data.ownership_repository import (
    OwnershipRepository,
)
from portopt_db.repositories.market_data.prices_repository import PricesRepository

logger = logging.getLogger(__name__)


# A delisted instrument that Yahoo has priced nothing for in this long is no
# longer worth a request in every sweep.
DORMANT_AFTER_DAYS = 365


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


def _optional_float(value: float | None) -> float | None:
    # ``Numeric`` columns load as ``Decimal`` although the models annotate ``float``.
    return None if value is None else float(value)


def _price_bar(row: PriceHistory) -> PriceBar:
    """Copy a ``price_history`` row into the DTO the ``PriceReader`` port returns."""
    return PriceBar(
        date=row.date,
        open=_optional_float(row.open),
        high=_optional_float(row.high),
        low=_optional_float(row.low),
        close=_optional_float(row.close),
        volume=row.volume,
        dividends=_optional_float(row.dividends),
        stock_splits=_optional_float(row.stock_splits),
        capital_gains=_optional_float(row.capital_gains),
        price_unit=row.price_unit,
    )


# The aggregate repositories are base classes, not attributes, so ``self`` in
# their methods is the facade instance: tests that replace ``_upsert`` on a
# facade instance still see every aggregate's writes.
class YFinanceRepository(
    PricesRepository,
    FundamentalsRepository,
    AnalystRepository,
    OwnershipRepository,
    NewsRepository,
    OptionsRepository,
):
    """Sync repository for yfinance data. Uses PostgreSQL ON CONFLICT upsert."""

    def __init__(self, session: Session):
        super().__init__(session)

    # ------------------------------------------------------------------
    # Ticker Profile
    # ------------------------------------------------------------------

    def profile_for(self, instrument_id: UUID) -> ProfileSummary | None:
        """Return the profile :meth:`get_profile` reads, as a DTO.

        Args:
            instrument_id: The instrument's primary key.

        Returns:
            The profile's sector and long name, or ``None`` when none is stored.
        """
        profile = self.get_profile(instrument_id)
        if profile is None:
            return None
        return ProfileSummary(sector=profile.sector, long_name=profile.long_name)

    # ------------------------------------------------------------------
    # Price History
    # ------------------------------------------------------------------

    def price_bars_for(
        self,
        instrument_id: UUID,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int | None = None,
        ascending: bool = False,
    ) -> list[PriceBar]:
        """Return the bars :meth:`get_price_history` reads, as DTOs.

        Args:
            instrument_id: The instrument's primary key.
            start_date: Inclusive lower bound; ``None`` leaves it open.
            end_date: Inclusive upper bound; ``None`` leaves it open.
            limit: Keep at most this many bars from the front of the order; ``None``
                keeps every bar in the window.
            ascending: Order the bars oldest first instead of newest first.

        Returns:
            The bars in the requested order, decimal columns as ``float``; empty when
            none is stored.
        """
        rows = self.get_price_history(
            instrument_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            ascending=ascending,
        )
        return [_price_bar(row) for row in rows]

    def newest_close_per_instrument(self, start: date, end: date) -> list[date]:
        """Return each instrument's newest bar date with a close inside a window.

        Args:
            start: Inclusive lower bound on the bar date.
            end: Inclusive upper bound on the bar date.

        Returns:
            One date per instrument with a non-NULL close in the window, in no
            guaranteed order; empty when no instrument has one.
        """
        stmt = (
            select(func.max(PriceHistory.date))
            .where(
                PriceHistory.close.is_not(None),
                PriceHistory.date >= start,
                PriceHistory.date <= end,
            )
            .group_by(PriceHistory.instrument_id)
        )
        return list(self.session.execute(stmt).scalars())

    # ------------------------------------------------------------------
    # Staleness info (for incremental fetch)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Instrument queries (used by yfinance_data endpoint)
    # ------------------------------------------------------------------

    def get_instruments_with_yfinance_ticker(
        self, *, today: date | None = None
    ) -> Sequence[Instrument]:
        """Return the instruments to ingest: those with a non-empty yfinance_ticker.

        A delisted row whose symbol is now carried by an active row (the old side
        of a venue transfer) is left out, so one symbol is never fetched and
        stored under two instrument ids. A delisted row with no active successor
        stays, because ``delisted_at`` records when a universe build stopped
        seeing a name, not when it stopped trading. It leaves once it has been
        delisted for ``DORMANT_AFTER_DAYS`` and has no stored bar from that
        period, i.e. Yahoo has priced nothing for it since; a pinned symbol never
        leaves. Its stored history stays. The exchange is eager-loaded.

        Args:
            today: Reference date for the dormancy cutoff; defaults to the
                current date.
        """
        cutoff = (today or date.today()) - timedelta(days=DORMANT_AFTER_DAYS)
        successor = aliased(Instrument)
        superseded = and_(
            Instrument.delisted_at.is_not(None),
            exists().where(
                successor.yfinance_ticker == Instrument.yfinance_ticker,
                successor.delisted_at.is_(None),
            ),
        )
        dormant = and_(
            Instrument.delisted_at.is_not(None),
            Instrument.delisted_at < cutoff,
            ~exists().where(
                PriceHistory.instrument_id == Instrument.id,
                PriceHistory.date >= cutoff,
            ),
            ~exists().where(PinnedInstrument.ticker == Instrument.yfinance_ticker),
        )
        return (
            self.session.execute(
                select(Instrument)
                .options(joinedload(Instrument.exchange))
                .where(Instrument.yfinance_ticker.isnot(None))
                .where(Instrument.yfinance_ticker != "")
                .where(~superseded)
                .where(~dormant)
            )
            .scalars()
            .unique()
            .all()
        )

    def get_instrument_by_yfinance_ticker(
        self, yfinance_ticker: str
    ) -> Instrument | None:
        """Return the instrument for a yfinance_ticker, with exchange eager-loaded.

        ``instruments`` is unique on ``(ticker, exchange)`` and ``yfinance_ticker``
        is indexed but not unique, so a symbol that moved venues leaves a delisted
        row beside its active one. The active listing wins; among several
        candidates the most recently delisted (or, for active rows, the most
        recently created) does, so the result is deterministic and a genuinely
        delisted holding still resolves.

        Args:
            yfinance_ticker: The Yahoo Finance symbol to look up.

        Returns:
            The chosen instrument, or ``None`` when no row has that symbol.
        """
        return (
            self.session.execute(
                select(Instrument)
                .options(joinedload(Instrument.exchange))
                .where(Instrument.yfinance_ticker == yfinance_ticker)
                .order_by(
                    Instrument.delisted_at.is_(None).desc(),
                    Instrument.delisted_at.desc(),
                    Instrument.created_at.desc(),
                    Instrument.ticker,
                    Instrument.id,
                )
                .limit(1)
            )
            .scalars()
            .first()
        )

    def instrument_for(self, yfinance_ticker: str) -> InstrumentSummary | None:
        """Return the row :meth:`get_instrument_by_yfinance_ticker` picks, as a DTO.

        Args:
            yfinance_ticker: The Yahoo Finance symbol to look up.

        Returns:
            The chosen instrument's summary, or ``None`` when no row has that symbol.
        """
        instrument = self.get_instrument_by_yfinance_ticker(yfinance_ticker)
        if instrument is None:
            return None
        return InstrumentSummary(
            id=instrument.id,
            name=instrument.name,
            instrument_type=instrument.instrument_type,
            currency_code=instrument.currency_code,
            delisted_at=instrument.delisted_at,
        )

    def get_benchmark_coverage(
        self,
        tickers: list[str],
    ) -> dict[str, tuple[int, date | None]]:
        """Return ``{ticker: (price_row_count, latest_price_date)}``.

        Tickers without an instrument row (or with zero prices) come back as
        ``(0, None)``. Drives the startup bootstrap and the scheduler's
        reference-index refresh, which re-seed anything missing or stale.
        """
        if not tickers:
            return {}

        stmt = (
            select(
                Instrument.yfinance_ticker,
                func.count(PriceHistory.id).label("price_rows"),
                func.max(PriceHistory.date).label("latest"),
            )
            .outerjoin(PriceHistory, PriceHistory.instrument_id == Instrument.id)
            .where(Instrument.yfinance_ticker.in_(tickers))
            .group_by(Instrument.yfinance_ticker)
        )
        rows = self.session.execute(stmt).all()
        coverage: dict[str, tuple[int, date | None]] = dict.fromkeys(tickers, (0, None))
        for ticker, count, latest in rows:
            coverage[ticker] = (int(count or 0), latest)
        return coverage
