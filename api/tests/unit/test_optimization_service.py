"""Unit tests for optimization_service module.

Covers:
  - resolve_optimizer: each registered type maps to correct factory
  - extract_results: weights/metrics/risk_contributions extraction
  - efficient_frontier: computed for MeanRisk/RobustMeanRisk, null otherwise
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# TestResolveOptimizer
# ---------------------------------------------------------------------------


class TestResolveOptimizer:
    """resolve_optimizer delegates to the correct factory per optimizer_type."""

    def test_mean_risk_resolves(self) -> None:
        from app.services.optimization_service import resolve_optimizer
        from skfolio.optimization import MeanRisk

        optimizer = resolve_optimizer("mean_risk", {})

        assert isinstance(optimizer, MeanRisk)

    def test_hrp_resolves(self) -> None:
        from app.services.optimization_service import resolve_optimizer
        from skfolio.optimization import HierarchicalRiskParity

        optimizer = resolve_optimizer("hrp", {})

        assert isinstance(optimizer, HierarchicalRiskParity)

    def test_herc_resolves(self) -> None:
        from app.services.optimization_service import resolve_optimizer
        from skfolio.optimization import HierarchicalEqualRiskContribution

        optimizer = resolve_optimizer("herc", {})

        assert isinstance(optimizer, HierarchicalEqualRiskContribution)

    def test_nco_resolves(self) -> None:
        from app.services.optimization_service import resolve_optimizer
        from skfolio.optimization import NestedClustersOptimization

        optimizer = resolve_optimizer("nco", {})

        assert isinstance(optimizer, NestedClustersOptimization)

    def test_equal_weighted_resolves(self) -> None:
        from app.services.optimization_service import resolve_optimizer
        from skfolio.optimization import EqualWeighted

        optimizer = resolve_optimizer("equal_weighted", {})

        assert isinstance(optimizer, EqualWeighted)

    def test_inverse_volatility_resolves(self) -> None:
        from app.services.optimization_service import resolve_optimizer
        from skfolio.optimization import InverseVolatility

        optimizer = resolve_optimizer("inverse_volatility", {})

        assert isinstance(optimizer, InverseVolatility)

    def test_risk_budgeting_resolves(self) -> None:
        from app.services.optimization_service import resolve_optimizer
        from skfolio.optimization import RiskBudgeting

        optimizer = resolve_optimizer("risk_budgeting", {})

        assert isinstance(optimizer, RiskBudgeting)

    def test_max_diversification_resolves(self) -> None:
        from app.services.optimization_service import resolve_optimizer
        from skfolio.optimization import MaximumDiversification

        optimizer = resolve_optimizer("max_diversification", {})

        assert isinstance(optimizer, MaximumDiversification)

    def test_unknown_type_raises_value_error(self) -> None:
        from app.services.optimization_service import resolve_optimizer

        with pytest.raises(ValueError, match="Unknown optimizer_type"):
            resolve_optimizer("does_not_exist", {})

    def test_mean_risk_has_efficient_frontier_size_set(self) -> None:
        """MeanRisk resolves with efficient_frontier_size set for frontier computation."""
        from app.services.optimization_service import (
            _FRONTIER_SIZE,
            resolve_optimizer,
        )

        optimizer = resolve_optimizer("mean_risk", {})

        assert optimizer.efficient_frontier_size == _FRONTIER_SIZE

    def test_hrp_has_no_efficient_frontier_size(self) -> None:
        """Non-MeanRisk types do NOT have efficient_frontier_size."""
        from app.services.optimization_service import resolve_optimizer

        optimizer = resolve_optimizer("hrp", {})

        # HRP does not have the parameter — resolver must not set it
        assert not hasattr(optimizer, "efficient_frontier_size")

    def test_mean_risk_with_risk_measure_in_config_does_not_raise(self) -> None:
        """Regression for #419: config dict holding risk_measure must not collide.

        Previously the service did ``factory(**config)`` which let
        ``risk_measure`` land in ``**kwargs`` and collide with the
        factory's own explicit ``risk_measure=`` argument inside the
        ``MeanRisk(...)`` constructor call.
        """
        from skfolio.measures import RiskMeasure

        from app.services.optimization_service import resolve_optimizer

        optimizer = resolve_optimizer("mean_risk", {"risk_measure": "cvar"})

        assert optimizer.risk_measure == RiskMeasure.CVAR

    @pytest.mark.parametrize(
        ("config_dict", "expected_field", "expected_value"),
        [
            ({"risk_measure": "variance"}, "risk_measure", "VARIANCE"),
            ({"risk_measure": "standard_deviation"}, "risk_measure", "STANDARD_DEVIATION"),
            ({"objective": "maximize_ratio"}, "objective_function", "MAXIMIZE_RATIO"),
            ({"min_weights": -0.2, "max_weights": 0.6}, "min_weights", -0.2),
            ({"budget": 0.8}, "budget", 0.8),
        ],
    )
    def test_mean_risk_honors_config_dict_fields(
        self,
        config_dict: dict,
        expected_field: str,
        expected_value: Any,
    ) -> None:
        """Service must translate config-dict primitives into typed MeanRiskConfig."""
        from app.services.optimization_service import resolve_optimizer

        optimizer = resolve_optimizer("mean_risk", config_dict)

        actual = getattr(optimizer, expected_field)
        if hasattr(actual, "name"):
            assert actual.name == expected_value
        else:
            assert actual == expected_value


# ---------------------------------------------------------------------------
# TestExtractResults
# ---------------------------------------------------------------------------


def _make_mock_portfolio(
    tickers: list[str] | None = None,
    sharpe: float = 1.2,
) -> MagicMock:
    """Build a mock skfolio Portfolio with realistic attributes."""
    tickers = tickers or ["AAPL", "MSFT", "GOOG"]
    weights = {t: 1.0 / len(tickers) for t in tickers}
    portfolio = MagicMock()
    portfolio.weights_dict = weights
    portfolio.annualized_mean = 0.10
    portfolio.annualized_standard_deviation = 0.15
    portfolio.annualized_sharpe_ratio = sharpe
    portfolio.contribution.return_value = np.full(len(tickers), 0.05)
    return portfolio


def _make_mock_pipeline(portfolio_or_population: Any) -> MagicMock:
    """Build a mock fitted Pipeline whose predict() returns the given object."""
    pipeline = MagicMock()
    pipeline.predict.return_value = portfolio_or_population
    return pipeline


class TestExtractResults:
    """extract_results extracts weights, metrics, contributions from a fitted pipeline."""

    def test_returns_weights_dict(self) -> None:
        from app.services.optimization_service import extract_results

        tickers = ["AAPL", "MSFT", "GOOG"]
        portfolio = _make_mock_portfolio(tickers)
        pipeline = _make_mock_pipeline(portfolio)
        returns_df = pd.DataFrame(
            np.random.randn(100, 3) * 0.01, columns=tickers
        )

        result = extract_results(pipeline, returns_df)

        assert result["weights"] == {t: pytest.approx(1 / 3, abs=1e-9) for t in tickers}

    def test_returns_metrics_with_sharpe(self) -> None:
        from app.services.optimization_service import extract_results

        portfolio = _make_mock_portfolio(sharpe=2.5)
        pipeline = _make_mock_pipeline(portfolio)
        returns_df = pd.DataFrame(np.random.randn(100, 3) * 0.01)

        result = extract_results(pipeline, returns_df)

        assert result["metrics"]["annualized_sharpe_ratio"] == pytest.approx(2.5)

    def test_returns_risk_contributions(self) -> None:
        from app.services.optimization_service import extract_results

        tickers = ["AAPL", "MSFT"]
        portfolio = _make_mock_portfolio(tickers)
        portfolio.contribution.return_value = np.array([0.03, 0.07])
        pipeline = _make_mock_pipeline(portfolio)
        returns_df = pd.DataFrame(np.random.randn(100, 2) * 0.01, columns=tickers)

        result = extract_results(pipeline, returns_df)

        assert result["risk_contributions"]["AAPL"] == pytest.approx(0.03)
        assert result["risk_contributions"]["MSFT"] == pytest.approx(0.07)

    def test_efficient_frontier_null_when_portfolio_not_population(self) -> None:
        """Non-frontier optimizers return null efficient_frontier."""
        from app.services.optimization_service import extract_results

        portfolio = _make_mock_portfolio()
        pipeline = _make_mock_pipeline(portfolio)
        returns_df = pd.DataFrame(np.random.randn(100, 3) * 0.01)

        result = extract_results(pipeline, returns_df)

        assert result["efficient_frontier"] is None

    def test_efficient_frontier_computed_when_prediction_is_population(self) -> None:
        """When predict returns Population, efficient_frontier is a list."""
        from skfolio import Population

        from app.services.optimization_service import extract_results

        p1 = _make_mock_portfolio(sharpe=0.8)
        p2 = _make_mock_portfolio(sharpe=1.5)
        # Mock a Population-like object
        population = MagicMock(spec=Population)
        population.__iter__ = MagicMock(return_value=iter([p1, p2]))
        population.__len__ = MagicMock(return_value=2)
        # max(...) uses __iter__, so we need to make it iterable
        # We also need isinstance check to pass
        pipeline = _make_mock_pipeline(population)
        returns_df = pd.DataFrame(np.random.randn(100, 3) * 0.01)

        result = extract_results(pipeline, returns_df)

        assert isinstance(result["efficient_frontier"], list)
        assert len(result["efficient_frontier"]) == 2

    def test_optimal_portfolio_is_max_sharpe_from_population(self) -> None:
        """When Population returned, optimal portfolio has max annualized_sharpe_ratio."""
        from skfolio import Population

        from app.services.optimization_service import extract_results

        low_sharpe = _make_mock_portfolio(sharpe=0.5)
        high_sharpe = _make_mock_portfolio(sharpe=2.0)
        high_sharpe.annualized_mean = 0.20
        high_sharpe.annualized_standard_deviation = 0.10

        population = MagicMock(spec=Population)
        population.__iter__ = MagicMock(return_value=iter([low_sharpe, high_sharpe]))
        population.__len__ = MagicMock(return_value=2)

        pipeline = _make_mock_pipeline(population)
        returns_df = pd.DataFrame(np.random.randn(100, 3) * 0.01)

        result = extract_results(pipeline, returns_df)

        assert result["metrics"]["annualized_sharpe_ratio"] == pytest.approx(2.0)
