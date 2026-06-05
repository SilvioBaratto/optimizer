"""Per-domain DB seed builders for API tests (issue #804).

Each domain module exports one result ``NamedTuple`` and one
``seed_<domain>(session, *, ...)`` builder.  Builders insert via
``session.add`` + ``session.flush()`` and never ``commit()`` — the
SAVEPOINT ``db_session`` fixture owns the transaction boundary.

Usage::

    def test_x(db_session):
        seed = seed_portfolio(db_session)
        ...
"""

from __future__ import annotations

from tests._fixtures.assertions import (
    assert_camel_case_keys,
    assert_json_safe,
    assert_no_snake_case_keys,
    assert_response_matches_schema,
)
from tests._fixtures.factors import FactorsSeed, seed_factors
from tests._fixtures.macro import MacroSeed, seed_macro
from tests._fixtures.market_data import MarketDataSeed, seed_market_data
from tests._fixtures.portfolio import PortfolioSeed, seed_portfolio
from tests._fixtures.rebalancing import RebalancingSeed, seed_rebalancing
from tests._fixtures.risk import RiskSeed, seed_risk
from tests._fixtures.universe import UniverseSeed, seed_universe

__all__ = [
    "FactorsSeed",
    "assert_camel_case_keys",
    "assert_json_safe",
    "assert_no_snake_case_keys",
    "assert_response_matches_schema",
    "MacroSeed",
    "MarketDataSeed",
    "PortfolioSeed",
    "RebalancingSeed",
    "RiskSeed",
    "UniverseSeed",
    "seed_factors",
    "seed_macro",
    "seed_market_data",
    "seed_portfolio",
    "seed_rebalancing",
    "seed_risk",
    "seed_universe",
]
