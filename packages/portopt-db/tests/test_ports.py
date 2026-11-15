"""Contract tests for ``portopt_db.ports`` and the repositories that implement them.

A ``runtime_checkable`` ``isinstance`` check only proves the method names exist, so
the conformance test also compares each port method's parameters and resolved type
hints with the repository's. The DTO tests check that what crosses a port carries
the stored values and no ORM entity.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import inspect
import types
import typing
import uuid
from collections.abc import Callable, Iterator
from decimal import Decimal

import pytest
from _fixtures import add_and_flush, get_or_create_exchange
from sqlalchemy.orm import Session

from portopt_db.base import Base
from portopt_db.models.market_data.fx_rates import FxRate
from portopt_db.models.market_data.yfinance_data import PriceHistory, TickerProfile
from portopt_db.models.universe.universe import Instrument
from portopt_db.ports import (
    FxRatePoint,
    FxRateReader,
    InstrumentSummary,
    PriceBar,
    PriceReader,
    ProfileSummary,
    UniverseReader,
)
from portopt_db.repositories.market_data.fx_rates_repository import FxRatesRepository
from portopt_db.repositories.market_data.yfinance_repository import YFinanceRepository
from portopt_db.repositories.universe.universe_repository import UniverseRepository

_PORT_ADAPTERS: list[tuple[type, type]] = [
    (PriceReader, YFinanceRepository),
    (FxRateReader, FxRatesRepository),
    (UniverseReader, UniverseRepository),
]

# Leaf types a port may return besides frozen dataclasses.
_VALUE_TYPES: frozenset[object] = frozenset(
    {str, int, float, bool, type(None), dt.date, dt.datetime, uuid.UUID, Decimal}
)
_CONTAINER_ORIGINS: frozenset[object] = frozenset(
    {list, tuple, dict, set, frozenset, types.UnionType, typing.Union}
)


def _port_methods(port: type) -> list[str]:
    return sorted(
        name
        for name, member in vars(port).items()
        if inspect.isfunction(member) and not name.startswith("_")
    )


def _parameters(method: Callable[..., object]) -> list[tuple[str, object, object]]:
    return [
        (param.name, param.kind, param.default)
        for param in inspect.signature(method).parameters.values()
    ]


def _reachable_types(hint: object) -> Iterator[object]:
    """Yield ``hint``, its type arguments and the field types of any dataclass."""
    yield hint
    for argument in typing.get_args(hint):
        yield from _reachable_types(argument)
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        for field_hint in typing.get_type_hints(hint).values():
            yield from _reachable_types(field_hint)


def _port_hints() -> Iterator[tuple[str, str, object]]:
    for port, _ in _PORT_ADAPTERS:
        for name in _port_methods(port):
            for role, hint in typing.get_type_hints(getattr(port, name)).items():
                yield f"{port.__name__}.{name}", role, hint


def _is_plain_class(candidate: object) -> bool:
    return isinstance(candidate, type) and typing.get_origin(candidate) is None


def _is_frozen_dataclass(candidate: object) -> bool:
    return (
        isinstance(candidate, type)
        and dataclasses.is_dataclass(candidate)
        and candidate.__dataclass_params__.frozen
    )


@pytest.mark.parametrize(("port", "repository"), _PORT_ADAPTERS)
def test_concrete_repositories_satisfy_ports(
    port: type, repository: type, db_session: Session
) -> None:
    """Each repository is its port's instance, with matching signatures and hints."""
    assert isinstance(repository(db_session), port)
    assert _port_methods(port), f"{port.__name__} declares no method"
    for name in _port_methods(port):
        port_method = getattr(port, name)
        concrete = getattr(repository, name)
        assert _parameters(concrete) == _parameters(port_method), name
        assert typing.get_type_hints(concrete) == typing.get_type_hints(port_method), (
            name
        )


def test_ports_return_no_orm_entities() -> None:
    """No port method names an ORM model in its parameters or return type."""
    orm_hits = [
        (method, role, candidate)
        for method, role, hint in _port_hints()
        for candidate in _reachable_types(hint)
        if _is_plain_class(candidate) and issubclass(candidate, Base)
    ]

    assert orm_hits == []


def test_ports_return_only_builtins_and_frozen_dataclasses() -> None:
    """A port returns builtins, stdlib value types and frozen dataclasses only."""
    offenders = []
    for method, role, hint in _port_hints():
        if role != "return":
            continue
        for candidate in _reachable_types(hint):
            origin = typing.get_origin(candidate)
            allowed = (
                origin in _CONTAINER_ORIGINS
                if origin is not None
                else candidate in _VALUE_TYPES or _is_frozen_dataclass(candidate)
            )
            if not allowed:
                offenders.append((method, candidate))

    assert offenders == []


def _instrument(
    session: Session,
    ticker: str,
    *,
    yfinance_ticker: str | None = None,
    delisted_at: dt.date | None = None,
) -> Instrument:
    return add_and_flush(
        session,
        Instrument(
            ticker=ticker,
            short_name=ticker,
            name=f"{ticker} Fund",
            instrument_type="ETF",
            currency_code="GBp",
            yfinance_ticker=yfinance_ticker or ticker,
            delisted_at=delisted_at,
            exchange=get_or_create_exchange(session, "Test Exchange"),
        ),
    )


def test_instrument_for_returns_the_listing_the_symbol_lookup_picks(
    db_session: Session,
) -> None:
    """The summary describes the row ``get_instrument_by_yfinance_ticker`` picks."""
    _instrument(
        db_session, "OLD", yfinance_ticker="VUSA.L", delisted_at=dt.date(2025, 1, 2)
    )
    active = _instrument(db_session, "VUSA", yfinance_ticker="VUSA.L")
    repository = YFinanceRepository(db_session)

    summary = repository.instrument_for("VUSA.L")

    assert summary == InstrumentSummary(
        id=active.id,
        name="VUSA Fund",
        instrument_type="ETF",
        currency_code="GBp",
        delisted_at=None,
    )
    picked = repository.get_instrument_by_yfinance_ticker("VUSA.L")
    assert picked is not None
    assert summary.id == picked.id


def test_instrument_for_an_unknown_symbol_is_none(db_session: Session) -> None:
    """A symbol no instrument carries has no summary."""
    assert YFinanceRepository(db_session).instrument_for("NOPE") is None


def test_profile_for_copies_sector_and_long_name(db_session: Session) -> None:
    """The profile summary copies the stored sector and long name."""
    instrument = _instrument(db_session, "AAA")
    add_and_flush(
        db_session,
        TickerProfile(
            instrument_id=instrument.id,
            symbol="AAA",
            sector="Technology",
            long_name="AAA Holdings plc",
        ),
    )
    repository = YFinanceRepository(db_session)

    assert repository.profile_for(instrument.id) == ProfileSummary(
        sector="Technology", long_name="AAA Holdings plc"
    )
    assert repository.profile_for(uuid.uuid4()) is None


def _seed_bars(session: Session, instrument: Instrument, days: int) -> None:
    for offset in range(days):
        add_and_flush(
            session,
            PriceHistory(
                instrument_id=instrument.id,
                date=dt.date(2026, 1, 1) + dt.timedelta(days=offset),
                open=Decimal("10.5"),
                high=Decimal("11.25"),
                low=Decimal("9.75"),
                close=Decimal(10 + offset),
                volume=1_000 + offset,
                dividends=None,
                stock_splits=Decimal(0),
                capital_gains=None,
                price_unit="GBp",
            ),
        )


def test_price_bars_for_carries_every_column_as_floats(db_session: Session) -> None:
    """Every stored column reaches the bar; the decimal columns arrive as ``float``."""
    instrument = _instrument(db_session, "AAA")
    _seed_bars(db_session, instrument, 1)

    (bar,) = YFinanceRepository(db_session).price_bars_for(instrument.id)

    assert bar == PriceBar(
        date=dt.date(2026, 1, 1),
        open=10.5,
        high=11.25,
        low=9.75,
        close=10.0,
        volume=1_000,
        dividends=None,
        stock_splits=0.0,
        capital_gains=None,
        price_unit="GBp",
    )
    assert all(
        isinstance(value, float)
        for value in (bar.open, bar.high, bar.low, bar.close, bar.stock_splits)
    )


@pytest.mark.parametrize(
    "window",
    [
        {},
        {"ascending": True},
        {"limit": 2},
        {"limit": 2, "ascending": True},
        {"start_date": dt.date(2026, 1, 2), "end_date": dt.date(2026, 1, 3)},
    ],
)
def test_price_bars_for_reads_the_same_bars_as_get_price_history(
    db_session: Session, window: dict[str, typing.Any]
) -> None:
    """The bars follow ``get_price_history``'s window, order and limit."""
    instrument = _instrument(db_session, "AAA")
    _seed_bars(db_session, instrument, 4)
    repository = YFinanceRepository(db_session)

    bars = repository.price_bars_for(instrument.id, **window)

    rows = repository.get_price_history(instrument.id, **window)
    assert [(bar.date, bar.close) for bar in bars] == [
        (row.date, float(row.close)) for row in rows if row.close is not None
    ]


def _add_rate(session: Session, base: str, quote: str, day: dt.date, rate: str) -> None:
    add_and_flush(
        session,
        FxRate(base_currency=base, quote_currency=quote, date=day, rate=Decimal(rate)),
    )


def test_rates_for_returns_float_rates_in_the_get_rates_order(
    db_session: Session,
) -> None:
    """Rates keep ``get_rates``' filters and order, as ``float``."""
    _add_rate(db_session, "EUR", "USD", dt.date(2026, 10, 2), "0.92")
    _add_rate(db_session, "EUR", "GBP", dt.date(2026, 10, 1), "1.17")
    _add_rate(db_session, "EUR", "USD", dt.date(2026, 10, 1), "0.91")
    _add_rate(db_session, "USD", "EUR", dt.date(2026, 10, 1), "1.08")
    repository = FxRatesRepository(db_session)

    points = repository.rates_for("EUR", ["USD", "GBP"], end=dt.date(2026, 10, 2))

    assert points == [
        FxRatePoint(quote_currency="GBP", date=dt.date(2026, 10, 1), rate=1.17),
        FxRatePoint(quote_currency="USD", date=dt.date(2026, 10, 1), rate=0.91),
        FxRatePoint(quote_currency="USD", date=dt.date(2026, 10, 2), rate=0.92),
    ]
    assert all(isinstance(point.rate, float) for point in points)
    assert repository.rates_for("EUR", start=dt.date(2026, 10, 2)) == [points[-1]]
