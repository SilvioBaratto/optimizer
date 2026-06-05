"""Smoke tests for the ``tests._fixtures`` per-domain DB seed builders (issue #804).

Each builder must insert and round-trip against the function-scoped
``db_session`` fixture (SAVEPOINT pattern) without ever calling
``session.commit()``.  FK-dependent builders must guard parent existence.
"""

from __future__ import annotations

import uuid

import pytest

from app.models.portfolio.portfolio import Portfolio, PortfolioSnapshot
from app.models.universe.universe import Exchange, Instrument
from tests._fixtures import (
    seed_factors,
    seed_macro,
    seed_market_data,
    seed_portfolio,
    seed_rebalancing,
    seed_risk,
    seed_universe,
)


class TestUniverseSeed:
    def test_when_seed_universe_called_exchange_and_instrument_persist(
        self, db_session
    ):
        seed = seed_universe(db_session)
        fetched = db_session.get(Instrument, seed.instrument.id)
        assert fetched is not None
        assert fetched.exchange.name == seed.exchange.name


class TestMarketDataSeed:
    def test_when_seed_market_data_called_ticker_profile_persists(self, db_session):
        seed = seed_market_data(db_session)
        assert seed.profile.instrument_id == seed.instrument.id
        assert seed.profile.sector is not None

    def test_when_universe_and_market_data_share_defaults_no_unique_violation(
        self, db_session
    ):
        seed_universe(db_session)
        seed_market_data(db_session)
        assert db_session.query(Exchange).count() == 1
        assert db_session.query(Instrument).count() == 1


class TestPortfolioSeed:
    def test_when_seed_portfolio_called_weights_and_sectors_round_trip(
        self, db_session
    ):
        seed = seed_portfolio(db_session)
        fetched = db_session.get(PortfolioSnapshot, seed.snapshot.id)
        assert fetched.weights == {"AAPL": 0.6, "MSFT": 0.4}
        assert fetched.sector_mapping == {"AAPL": "Technology", "MSFT": "Technology"}
        assert seed.position.portfolio_id == seed.portfolio.id
        assert seed.account.portfolio_id == seed.portfolio.id

    def test_when_seed_portfolio_called_server_default_columns_are_populated(
        self, db_session
    ):
        portfolio = seed_portfolio(db_session).portfolio
        assert portfolio.currency == "EUR"
        assert portfolio.benchmark_ticker == "SPY"
        assert portfolio.is_active is True


class TestFactorsSeed:
    def test_when_seed_factors_called_score_and_report_persist(self, db_session):
        seed = seed_factors(db_session)
        assert seed.score.id is not None
        assert seed.report.id is not None


class TestRiskSeed:
    def test_when_seed_risk_called_with_portfolio_limit_persists(self, db_session):
        portfolio = seed_portfolio(db_session).portfolio
        seed = seed_risk(db_session, portfolio)
        assert seed.limit.portfolio_id == portfolio.id

    def test_when_seed_risk_called_with_unflushed_portfolio_assertion_raised(
        self, db_session
    ):
        ghost = Portfolio(name="ghost")
        ghost.id = uuid.uuid4()
        with pytest.raises(AssertionError):
            seed_risk(db_session, ghost)


class TestMacroSeed:
    def test_when_seed_macro_called_indicator_calibration_fred_persist(
        self, db_session
    ):
        seed = seed_macro(db_session)
        assert seed.indicator.id is not None
        assert seed.calibration.id is not None
        assert seed.fred.id is not None


class TestRebalancingSeed:
    def test_when_seed_rebalancing_called_with_portfolio_policy_persists(
        self, db_session
    ):
        portfolio = seed_portfolio(db_session).portfolio
        seed = seed_rebalancing(db_session, portfolio)
        assert seed.policy.portfolio_id == portfolio.id

    def test_when_seed_rebalancing_called_with_unflushed_portfolio_assertion_raised(
        self, db_session
    ):
        ghost = Portfolio(name="ghost")
        ghost.id = uuid.uuid4()
        with pytest.raises(AssertionError):
            seed_rebalancing(db_session, ghost)
