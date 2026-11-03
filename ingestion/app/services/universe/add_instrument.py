"""Add one hand-picked instrument to the universe after validating it on Yahoo.

The Screener-driven build classifies funds from their names and only sees the head
of each venue, so an operator who holds an ETF it misses or mislabels adds it by
hand. This service does that safely: it reads the listing from Yahoo (type, venue,
currency, names), checks that prices exist, and upserts one ``instruments`` row
shaped like a build row.

It writes through the caller's session and never commits; the CLI owns the
transaction. It claims no job slot because it touches a single ticker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID

from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.universe.universe_repository import UniverseRepository
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.services.market_data.yfinance import YFinanceClient
from app.services.universe.trading212.enums import (
    AssetClass,
    DurationBucket,
    FiSubclass,
)
from app.services.universe.yfinance_source import (
    resolve_exchange_name,
    supported_exchange_names,
)

logger = logging.getLogger(__name__)

_QUOTE_TYPE_TO_INSTRUMENT_TYPE = {"EQUITY": "STOCK", "ETF": "ETF"}
_SHORT_NAME_MAX = 100


class AddInstrumentError(ValueError):
    """The ticker cannot be added; the message says what to change."""


@dataclass(frozen=True)
class AddInstrumentResult:
    """Outcome of one :func:`add_instrument` call.

    Attributes:
        instrument_id: Primary key of the written ``instruments`` row.
        created: True when the row is new, False when an existing row was updated.
        warnings: Non-fatal findings for the operator, such as another
            instrument carrying the same ISIN.
    """

    instrument_id: UUID
    created: bool
    warnings: list[str] = field(default_factory=list)


def _choices(enum_type: type[Enum]) -> str:
    return ", ".join(str(member.value) for member in enum_type)


def _validated_classification(
    instrument_type: str,
    asset_class: str | None,
    fi_subclass: str | None,
    duration_bucket: str | None,
) -> str:
    """Check the classification inputs and return the asset class to store.

    Raises:
        AddInstrumentError: A value is not a known member, an ETF has no asset
            class, or a fixed-income detail accompanies another class.
    """
    if asset_class is None:
        if instrument_type == "ETF":
            raise AddInstrumentError(
                "an ETF needs an explicit asset class (--asset-class), one of: "
                f"{_choices(AssetClass)}"
            )
        asset_class = AssetClass.EQUITY.value
    elif asset_class not in {m.value for m in AssetClass}:
        raise AddInstrumentError(
            f"unknown asset class {asset_class!r}; choose one of: {_choices(AssetClass)}"
        )

    if fi_subclass is not None and fi_subclass not in {m.value for m in FiSubclass}:
        raise AddInstrumentError(
            f"unknown fixed-income subclass {fi_subclass!r}; "
            f"choose one of: {_choices(FiSubclass)}"
        )
    if duration_bucket is not None and duration_bucket not in {
        m.value for m in DurationBucket
    }:
        raise AddInstrumentError(
            f"unknown duration bucket {duration_bucket!r}; "
            f"choose one of: {_choices(DurationBucket)}"
        )
    if asset_class != AssetClass.FIXED_INCOME.value and (
        fi_subclass is not None or duration_bucket is not None
    ):
        raise AddInstrumentError(
            "fi_subclass and duration_bucket apply only to the fixed_income "
            f"asset class, not {asset_class!r}"
        )
    return asset_class


def _resolve_exchange(
    repo: UniverseRepository, yahoo_code: str | None, override: str | None
) -> Exchange:
    """Find the ``exchanges`` row for the listing; never creates one."""
    supported = ", ".join(supported_exchange_names())
    if override is not None:
        name = override
    elif not yahoo_code:
        raise AddInstrumentError(
            "Yahoo reports no exchange for this ticker; pass --exchange NAME, "
            f"one of: {supported}"
        )
    else:
        resolved = resolve_exchange_name(yahoo_code)
        if resolved is None:
            raise AddInstrumentError(
                f"Yahoo exchange code {yahoo_code!r} is not a supported venue; pass "
                f"--exchange NAME, one of: {supported}"
            )
        name = resolved
    exchange = repo.get_exchange_by_name(name)
    if exchange is None:
        raise AddInstrumentError(
            f"exchange {name!r} is not in the universe; supported venues: {supported}"
        )
    return exchange


def _first_non_empty(*values: Any) -> str | None:
    return next((v for v in values if isinstance(v, str) and v.strip()), None)


def _other_listings_of_isin(
    session: Session, isin: str | None, own_id: UUID
) -> list[str]:
    if not isin:
        return []
    rows = session.execute(
        select(Instrument.ticker, Instrument.yfinance_ticker)
        .where(Instrument.isin == isin)
        .where(Instrument.id != own_id)
    ).all()
    return sorted({yf or ticker for ticker, yf in rows})


def add_instrument(
    session: Session,
    yf_client: YFinanceClient,
    *,
    yahoo_ticker: str,
    asset_class: str | None = None,
    isin: str | None = None,
    fi_subclass: str | None = None,
    duration_bucket: str | None = None,
    t212_ticker: str | None = None,
    exchange: str | None = None,
    name: str | None = None,
) -> AddInstrumentResult:
    """Validate ``yahoo_ticker`` on Yahoo and upsert its ``instruments`` row.

    The row is keyed on ``(ticker, exchange)`` like a build row, with
    ``yfinance_ticker`` set to the same symbol and ``currency_code`` stored as
    Yahoo's raw listing currency (``GBp`` stays pence). Re-running is idempotent:
    it updates the same row, clears any delisting, and keeps a stored ``name``,
    ``isin``, ``t212_ticker`` or ``currency_code`` when neither the call nor
    Yahoo supplies a replacement (the ``save_instruments_batch`` coalescing
    rule). Asset-class details, by contrast, are overwritten so the operator can
    correct them.

    Args:
        session: Open session; flushed but not committed.
        yf_client: Yahoo client providing ``fetch_info`` and ``fetch_history``.
        yahoo_ticker: Yahoo symbol such as ``VWCE.DE``; stripped and upper-cased.
        asset_class: One of :class:`AssetClass`. Required for an ETF; a stock
            defaults to ``equity``.
        isin: ISIN to store; ``None`` keeps a stored one.
        fi_subclass: :class:`FiSubclass` value, only with ``fixed_income``.
        duration_bucket: :class:`DurationBucket` value, only with ``fixed_income``.
        t212_ticker: Trading 212 ticker to store; ``None`` keeps a stored one.
        exchange: ``exchanges.name`` overriding the one Yahoo's code resolves to.
        name: Display name overriding Yahoo's ``longName``.

    Returns:
        The row id, whether it was created, and any non-fatal warnings.

    Raises:
        AddInstrumentError: Yahoo has no usable profile or prices, the quote type
            is neither EQUITY nor ETF, the classification is invalid, or the
            exchange cannot be resolved to an existing ``exchanges`` row.
    """
    symbol = yahoo_ticker.strip().upper()
    if not symbol:
        raise AddInstrumentError("the Yahoo ticker is empty")

    info = yf_client.fetch_info(symbol)
    if not info:
        raise AddInstrumentError(
            f"Yahoo returned no profile for {symbol}: unknown ticker or no answer"
        )

    quote_type = str(info.get("quoteType") or "").upper()
    instrument_type = _QUOTE_TYPE_TO_INSTRUMENT_TYPE.get(quote_type)
    if instrument_type is None:
        raise AddInstrumentError(
            f"{symbol} has Yahoo quote type {quote_type or 'unknown'!r}; "
            "only EQUITY and ETF listings can be added"
        )

    resolved_class = _validated_classification(
        instrument_type, asset_class, fi_subclass, duration_bucket
    )
    repo = UniverseRepository(session)
    exchange_row = _resolve_exchange(repo, info.get("exchange"), exchange)

    history = yf_client.fetch_history(symbol, period="1mo", min_rows=1)
    if history is None or history.empty:
        raise AddInstrumentError(f"Yahoo has no price history for {symbol}")

    warnings: list[str] = []
    currency = _first_non_empty(info.get("currency"))
    if currency is None:
        warnings.append(
            f"Yahoo reports no currency for {symbol}; currency_code left empty"
        )

    row = session.execute(
        select(Instrument)
        .where(Instrument.ticker == symbol)
        .where(Instrument.exchange_id == exchange_row.id)
    ).scalar_one_or_none()
    created = row is None
    if row is None:
        row = Instrument(ticker=symbol, exchange_id=exchange_row.id)
        session.add(row)

    row.short_name = (
        _first_non_empty(info.get("shortName"), row.short_name, symbol) or symbol
    )[:_SHORT_NAME_MAX]
    row.name = _first_non_empty(name, info.get("longName"), row.name)
    row.isin = isin or row.isin
    row.t212_ticker = t212_ticker or row.t212_ticker
    row.instrument_type = instrument_type
    row.currency_code = currency or row.currency_code
    row.yfinance_ticker = symbol
    row.asset_class = resolved_class
    row.fi_subclass = fi_subclass
    row.duration_bucket = duration_bucket
    row.delisted_at = None
    row.delisting_return = None
    session.flush()

    others = _other_listings_of_isin(session, row.isin, row.id)
    if others:
        warnings.append(
            f"ISIN {row.isin} is also carried by {', '.join(others)}; "
            "two listings of one fund both enter the universe"
        )

    logger.info(
        "add_instrument: %s %s as %s on %s",
        "created" if created else "updated",
        symbol,
        resolved_class,
        exchange_row.name,
    )
    return AddInstrumentResult(row.id, created, warnings)
