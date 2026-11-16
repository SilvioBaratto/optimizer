"""YFinanceRepository aggregates and the modules that hold them.

- profile, financial statements, share counts, SEC filings: fundamentals_repository
- price history, dividends, stock splits: prices_repository
- estimates, earnings, analyst actions, recommendations, targets: analyst_repository
- institutional, mutual-fund and major holders, insiders: ownership_repository
- ticker news: news_repository
- option chains: options_repository
- staleness readers: staleness_repository
- instrument lookups by Yahoo symbol: universe.universe_repository
- PriceReader port DTOs: this module
"""

from collections.abc import Sequence
from datetime import date
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Instrument
from portopt_db.ports import InstrumentSummary, PriceBar, ProfileSummary

# ``as`` re-exports: tests import these helpers from this module.
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

# ``as`` re-export: a test imports it from this module.
from portopt_db.repositories.market_data.ownership_repository import (
    _SENTINEL_DATE as _SENTINEL_DATE,
)
from portopt_db.repositories.market_data.ownership_repository import (
    OwnershipRepository,
)
from portopt_db.repositories.market_data.prices_repository import PricesRepository

# ``as`` re-exports: ingestion imports these names from this module.
from portopt_db.repositories.market_data.staleness_repository import (
    FINANCIAL_STATEMENT_GROUPS as FINANCIAL_STATEMENT_GROUPS,
)
from portopt_db.repositories.market_data.staleness_repository import (
    FUNDAMENTAL_CATEGORIES as FUNDAMENTAL_CATEGORIES,
)
from portopt_db.repositories.market_data.staleness_repository import (
    VALUATION_MEASURES_PERIOD_TYPE as VALUATION_MEASURES_PERIOD_TYPE,
)
from portopt_db.repositories.market_data.staleness_repository import (
    VALUATION_MEASURES_STATEMENT_TYPE as VALUATION_MEASURES_STATEMENT_TYPE,
)
from portopt_db.repositories.market_data.staleness_repository import (
    StalenessRepository,
)
from portopt_db.repositories.market_data.staleness_repository import (
    statement_staleness_key as statement_staleness_key,
)

# ``as`` re-export: it was public here before the lookup that reads it moved.
from portopt_db.repositories.universe.universe_repository import (
    DORMANT_AFTER_DAYS as DORMANT_AFTER_DAYS,
)
from portopt_db.repositories.universe.universe_repository import UniverseRepository


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
# their methods is the facade instance: a test that replaces ``_upsert`` on a
# facade instance captures the rows the aggregates' upserts pass to it.
class YFinanceRepository(
    PricesRepository,
    FundamentalsRepository,
    AnalystRepository,
    OwnershipRepository,
    NewsRepository,
    OptionsRepository,
    StalenessRepository,
):
    """Sync repository for yfinance data. Uses PostgreSQL ON CONFLICT upsert."""

    def __init__(self, session: Session):
        super().__init__(session)

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

    def get_instruments_with_yfinance_ticker(
        self, *, today: date | None = None
    ) -> Sequence[Instrument]:
        """Return what :meth:`UniverseRepository.get_instruments_with_yfinance_ticker` does.

        Args:
            today: Reference date for the dormancy cutoff; defaults to the
                current date.
        """
        return UniverseRepository(self.session).get_instruments_with_yfinance_ticker(
            today=today
        )

    def get_instrument_by_yfinance_ticker(
        self, yfinance_ticker: str
    ) -> Instrument | None:
        """Return what :meth:`UniverseRepository.get_instrument_by_yfinance_ticker` does.

        Args:
            yfinance_ticker: The Yahoo Finance symbol to look up.
        """
        return UniverseRepository(self.session).get_instrument_by_yfinance_ticker(
            yfinance_ticker
        )

    def get_benchmark_coverage(
        self,
        tickers: list[str],
    ) -> dict[str, tuple[int, date | None]]:
        """Return what :meth:`UniverseRepository.get_benchmark_coverage` does.

        Args:
            tickers: Yahoo Finance symbols to report on.
        """
        return UniverseRepository(self.session).get_benchmark_coverage(tickers)

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
