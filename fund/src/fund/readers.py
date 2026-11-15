"""Composition point for the ``portopt_db`` read ports.

Each factory wraps a session in the repository that implements a port and returns
it typed as that port, so a module that reads through a port depends on the
protocol, not on the repository class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from portopt_db.repositories.market_data.fx_rates_repository import FxRatesRepository
from portopt_db.repositories.market_data.yfinance_repository import YFinanceRepository
from portopt_db.repositories.universe.universe_repository import UniverseRepository

if TYPE_CHECKING:
    from portopt_db.ports import FxRateReader, PriceReader, UniverseReader
    from sqlalchemy.orm import Session


def price_reader(session: Session) -> PriceReader:
    """Return the instrument, profile and price-bar reader over ``session``.

    Args:
        session: The open ``portopt_db`` session the reader queries.
    """
    return YFinanceRepository(session)


def fx_rate_reader(session: Session) -> FxRateReader:
    """Return the FX-rate reader over ``session``.

    Args:
        session: The open ``portopt_db`` session the reader queries.
    """
    return FxRatesRepository(session)


def universe_reader(session: Session) -> UniverseReader:
    """Return the Trading 212 overlay reader over ``session``.

    Args:
        session: The open ``portopt_db`` session the reader queries.
    """
    return UniverseRepository(session)


__all__ = ["fx_rate_reader", "price_reader", "universe_reader"]
