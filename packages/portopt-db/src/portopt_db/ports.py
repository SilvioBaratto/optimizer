"""Read ports over the repositories, returning DTOs instead of ORM entities.

Each protocol holds the reads ``fund`` runs through one repository, and each
method returns builtins or a frozen dataclass from this module. ``YFinanceRepository``
implements :class:`PriceReader`, ``FxRatesRepository`` :class:`FxRateReader` and
``UniverseRepository`` :class:`UniverseReader`; ``tests/test_ports.py`` holds them to
the same signatures. The module imports only the standard library, so a consumer can
type against a port without importing a repository.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class InstrumentSummary:
    """The ``instruments`` columns that identify and classify one listing.

    Attributes:
        id: The instrument's primary key, which the other ``PriceReader`` methods
            take.
        name: ``instruments.name`` as stored, or ``None``.
        instrument_type: ``instruments.instrument_type`` as stored, or ``None``.
        currency_code: ``instruments.currency_code`` as stored, or ``None``.
        delisted_at: ``instruments.delisted_at`` as stored, or ``None`` for a listing
            not marked delisted.
    """

    id: uuid.UUID
    name: str | None
    instrument_type: str | None
    currency_code: str | None
    delisted_at: dt.date | None


@dataclass(frozen=True)
class ProfileSummary:
    """The ``ticker_profiles`` columns that classify an instrument.

    Attributes:
        sector: ``ticker_profiles.sector`` (Yahoo's label) as stored, or ``None``.
        long_name: ``ticker_profiles.long_name`` as stored, or ``None``.
    """

    sector: str | None
    long_name: str | None


@dataclass(frozen=True)
class PriceBar:
    """One stored daily ``price_history`` row.

    The price columns stay in ``price_unit``, which can be a minor unit such as
    ``GBp``.

    Attributes:
        date: Trading date of the bar.
        open: Opening price, or ``None``.
        high: Highest price, or ``None``.
        low: Lowest price, or ``None``.
        close: Closing price, or ``None``.
        volume: Traded volume, or ``None``.
        dividends: Dividend recorded on this date, or ``None``.
        stock_splits: Split ratio recorded on this date, or ``None``.
        capital_gains: Fund capital-gain distribution recorded on this date, or
            ``None``.
        price_unit: Currency or minor unit the prices are quoted in, or ``None``.
    """

    date: dt.date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: int | None
    dividends: float | None
    stock_splits: float | None
    capital_gains: float | None
    price_unit: str | None


@dataclass(frozen=True)
class InstrumentClass:
    """The ``instruments`` columns that classify one listing of a Yahoo symbol.

    Attributes:
        yfinance_ticker: The Yahoo symbol the row carries.
        id: The row's primary key.
        instrument_type: ``instruments.instrument_type`` as stored, or ``None``.
        asset_class: ``instruments.asset_class`` as stored.
        fi_subclass: ``instruments.fi_subclass`` as stored, or ``None``.
    """

    yfinance_ticker: str
    id: uuid.UUID
    instrument_type: str | None
    asset_class: str
    fi_subclass: str | None


@dataclass(frozen=True)
class FxRatePoint:
    """One daily rate, expressed in the base currency the caller asked for.

    Attributes:
        quote_currency: ISO-4217 code of the foreign unit being priced.
        date: Calendar date the rate applies to.
        rate: Base-currency units per one unit of ``quote_currency``.
    """

    quote_currency: str
    date: dt.date
    rate: float


@runtime_checkable
class PriceReader(Protocol):
    """Instrument, profile and daily-bar reads.

    Each read is keyed by Yahoo symbol or by instrument, except
    :meth:`newest_close_dates`, which spans every instrument.
    """

    def instrument_for(self, yfinance_ticker: str) -> InstrumentSummary | None:
        """Return the instrument listed under ``yfinance_ticker``.

        Args:
            yfinance_ticker: The Yahoo Finance symbol to look up.

        Returns:
            The listing's summary, an active listing chosen over a delisted one, or
            ``None`` when no instrument carries the symbol.
        """
        ...

    def profile_for(self, instrument_id: uuid.UUID) -> ProfileSummary | None:
        """Return the stored profile of an instrument.

        Args:
            instrument_id: The instrument's primary key.

        Returns:
            The profile's summary, or ``None`` when none is stored.
        """
        ...

    def price_bars_for(
        self,
        instrument_id: uuid.UUID,
        start_date: dt.date | None = None,
        end_date: dt.date | None = None,
        limit: int | None = None,
        ascending: bool = False,
    ) -> list[PriceBar]:
        """Return an instrument's stored daily bars within a date window.

        Args:
            instrument_id: The instrument's primary key.
            start_date: Inclusive lower bound; ``None`` leaves it open.
            end_date: Inclusive upper bound; ``None`` leaves it open.
            limit: Keep at most this many bars from the front of the order; ``None``
                keeps every bar in the window.
            ascending: Order the bars oldest first instead of newest first.

        Returns:
            The bars in the requested order; empty when none is stored.
        """
        ...

    def newest_close_dates(self, start: dt.date, end: dt.date) -> list[dt.date]:
        """Return each instrument's newest bar date with a close inside a window.

        Args:
            start: Inclusive lower bound on the bar date.
            end: Inclusive upper bound on the bar date.

        Returns:
            One date per instrument with a non-NULL close in the window, in no
            guaranteed order; empty when no instrument has one.
        """
        ...


@runtime_checkable
class FxRateReader(Protocol):
    """Daily FX-rate reads for one base currency."""

    def rates_for(
        self,
        base: str,
        quotes: Sequence[str] | None = None,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> list[FxRatePoint]:
        """Return the stored rates expressed in ``base``.

        Args:
            base: Base currency the rates are expressed in.
            quotes: Restrict to these quote currencies; ``None`` returns all.
            start: Inclusive lower date bound; ``None`` leaves it open.
            end: Inclusive upper date bound; ``None`` leaves it open.

        Returns:
            The rates ordered by quote currency, then date; empty when none matches.
        """
        ...


@runtime_checkable
class UniverseReader(Protocol):
    """Reads of the instrument universe: classification and the Trading 212 overlay."""

    def instrument_classes_for(self, tickers: Collection[str]) -> list[InstrumentClass]:
        """Return the classification of every row that carries one of ``tickers``.

        A symbol that moved venues leaves a delisted row beside its live one, so a
        symbol can have several entries.

        Args:
            tickers: Yahoo symbols to look up; an empty collection matches no row.

        Returns:
            One entry per matching row, sorted by these keys in turn: not delisted
            before delisted, newer ``delisted_at`` first, newer ``created_at``
            first, then ``ticker`` and ``id``; empty when no row matches.
        """
        ...

    def get_t212_untradeable_tickers(
        self, tickers: Collection[str] | None = None
    ) -> list[str] | None:
        """Return the Yahoo symbols the overlay found absent from the broker.

        Args:
            tickers: Restrict the answer to these symbols. ``None`` asks about every
                instrument; an empty collection asks about none.

        Returns:
            Sorted symbols flagged untradeable on every active row, or ``None`` when
            the overlay has evaluated no row at all.
        """
        ...

    def get_t212_unevaluated_tickers(self, tickers: Collection[str]) -> list[str]:
        """Return the given Yahoo symbols whose broker status the overlay never judged.

        Args:
            tickers: Symbols to check; an empty collection asks about none.

        Returns:
            Sorted symbols that have an active row and no active row with a flag.
        """
        ...


__all__ = [
    "FxRatePoint",
    "FxRateReader",
    "InstrumentClass",
    "InstrumentSummary",
    "PriceBar",
    "PriceReader",
    "ProfileSummary",
    "UniverseReader",
]
