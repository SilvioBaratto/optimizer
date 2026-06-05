"""Smoke tests for the conftest seed fixtures (issue #805).

Proves each ``seeded_*`` fixture returns its builder's ``*Seed`` NamedTuple
unchanged, the row is queryable via ``db_session``, FK-dependent fixtures
chain off ``seeded_portfolio``, and the per-test SAVEPOINT rolls back
between tests (no row leakage).
"""

from __future__ import annotations

from app.models.factors.factor import FactorScore
from app.models.macro.macro_regime import EconomicIndicator
from app.models.market_data.yfinance_data import TickerProfile
from app.models.portfolio.portfolio import Portfolio
from app.models.rebalancing.rebalancing import RebalancingPolicy
from app.models.risk.risk import RiskLimit
from app.models.universe.universe import Exchange, Instrument
from tests._fixtures import (
    FactorsSeed,
    MacroSeed,
    MarketDataSeed,
    PortfolioSeed,
    RebalancingSeed,
    RiskSeed,
    UniverseSeed,
)


class TestSeededFixtureReturnTypes:
    def test_when_seeded_portfolio_requested_portfolio_seed_is_returned(
        self, seeded_portfolio
    ):
        assert isinstance(seeded_portfolio, PortfolioSeed)

    def test_when_seeded_universe_requested_universe_seed_is_returned(
        self, seeded_universe
    ):
        assert isinstance(seeded_universe, UniverseSeed)

    def test_when_seeded_market_data_requested_market_data_seed_is_returned(
        self, seeded_market_data
    ):
        assert isinstance(seeded_market_data, MarketDataSeed)

    def test_when_seeded_factors_requested_factors_seed_is_returned(
        self, seeded_factors
    ):
        assert isinstance(seeded_factors, FactorsSeed)

    def test_when_seeded_macro_requested_macro_seed_is_returned(self, seeded_macro):
        assert isinstance(seeded_macro, MacroSeed)

    def test_when_seeded_risk_requested_risk_seed_is_returned(self, seeded_risk):
        assert isinstance(seeded_risk, RiskSeed)

    def test_when_seeded_rebalancing_requested_rebalancing_seed_is_returned(
        self, seeded_rebalancing
    ):
        assert isinstance(seeded_rebalancing, RebalancingSeed)


class TestSeededFixtureRowsQueryable:
    def test_when_seeded_portfolio_requested_row_is_queryable(
        self, db_session, seeded_portfolio
    ):
        assert db_session.get(Portfolio, seeded_portfolio.portfolio.id) is not None

    def test_when_seeded_universe_requested_rows_are_queryable(
        self, db_session, seeded_universe
    ):
        assert db_session.get(Exchange, seeded_universe.exchange.id) is not None
        assert db_session.get(Instrument, seeded_universe.instrument.id) is not None

    def test_when_seeded_market_data_requested_profile_is_queryable(
        self, db_session, seeded_market_data
    ):
        assert db_session.get(TickerProfile, seeded_market_data.profile.id) is not None

    def test_when_seeded_factors_requested_score_is_queryable(
        self, db_session, seeded_factors
    ):
        assert db_session.get(FactorScore, seeded_factors.score.id) is not None

    def test_when_seeded_macro_requested_indicator_is_queryable(
        self, db_session, seeded_macro
    ):
        assert db_session.get(EconomicIndicator, seeded_macro.indicator.id) is not None


class TestFkDependentFixturesChainOffPortfolio:
    def test_when_seeded_risk_requested_limit_fk_matches_seeded_portfolio(
        self, db_session, seeded_portfolio, seeded_risk
    ):
        limit = db_session.get(RiskLimit, seeded_risk.limit.id)
        assert limit is not None
        assert limit.portfolio_id == seeded_portfolio.portfolio.id

    def test_when_seeded_rebalancing_requested_policy_fk_matches_seeded_portfolio(
        self, db_session, seeded_portfolio, seeded_rebalancing
    ):
        policy = db_session.get(RebalancingPolicy, seeded_rebalancing.policy.id)
        assert policy is not None
        assert policy.portfolio_id == seeded_portfolio.portfolio.id


class TestSavepointRollbackBetweenTests:
    """Two tests each seed one portfolio; if SAVEPOINT rollback works, each
    sees exactly one row (no leakage from the sibling test)."""

    def test_first_test_sees_single_portfolio(self, db_session, seeded_portfolio):
        assert db_session.query(Portfolio).count() == 1

    def test_second_test_sees_single_portfolio(self, db_session, seeded_portfolio):
        assert db_session.query(Portfolio).count() == 1
