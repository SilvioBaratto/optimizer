"""Per-domain DB seed builders for the ingestion test suite.

Each domain module exports one result ``NamedTuple`` and one
``seed_<domain>(session, *, ...)`` builder.  Builders insert via
``session.add`` + ``session.flush()`` and never ``commit()`` — the
SAVEPOINT ``db_session`` fixture owns the transaction boundary.

Usage::

    def test_x(db_session):
        seed = seed_market_data(db_session)
        ...
"""

from __future__ import annotations

from tests._fixtures.macro import MacroSeed, seed_macro
from tests._fixtures.market_data import MarketDataSeed, seed_market_data
from tests._fixtures.universe import UniverseSeed, seed_universe

__all__ = [
    "MacroSeed",
    "MarketDataSeed",
    "UniverseSeed",
    "seed_macro",
    "seed_market_data",
    "seed_universe",
]
