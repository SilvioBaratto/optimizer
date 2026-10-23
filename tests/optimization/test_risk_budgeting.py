"""Tests for the RiskBudgeting factory and configuration."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import RiskBudgeting

from optimizer.exceptions import ConfigurationError
from optimizer.moments import MomentEstimationConfig
from optimizer.optimization import (
    RiskBudgetingConfig,
    build_risk_budgeting,
)


@pytest.fixture(scope="module")
def returns():
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    return prices_to_returns(prices)


class TestRiskBudgetingConfig:
    def test_when_default_then_no_custom_budgets(self) -> None:
        cfg = RiskBudgetingConfig()
        assert cfg.budgets is None

    def test_when_default_then_min_weights_zero(self) -> None:
        cfg = RiskBudgetingConfig()
        assert cfg.min_weights == 0.0

    def test_when_default_then_max_weights_one(self) -> None:
        cfg = RiskBudgetingConfig()
        assert cfg.max_weights == 1.0

    def test_when_constructed_then_frozen(self) -> None:
        cfg = RiskBudgetingConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.min_weights = 0.05  # type: ignore[misc]

    def test_when_budgets_dont_sum_to_one_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="sum to 1"):
            RiskBudgetingConfig(
                budgets=(("AAPL", 0.5), ("MSFT", 0.3)),
            )


class TestPresets:
    def test_when_for_equal_risk_contribution_then_default_config(self) -> None:
        cfg = RiskBudgetingConfig.for_equal_risk_contribution()
        assert cfg.budgets is None

    def test_when_for_custom_budgets_then_budgets_stored(self) -> None:
        budgets = {"AAPL": 0.5, "MSFT": 0.5}
        cfg = RiskBudgetingConfig.for_custom_budgets(budgets)
        assert cfg.budgets == (("AAPL", 0.5), ("MSFT", 0.5))

    def test_when_for_custom_budgets_dont_sum_to_one_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="sum to 1"):
            RiskBudgetingConfig.for_custom_budgets({"AAPL": 0.5, "MSFT": 0.4})


class TestBuildRiskBudgeting:
    def test_when_built_then_returns_skfolio_class(self) -> None:
        est = build_risk_budgeting(RiskBudgetingConfig())
        assert isinstance(est, RiskBudgeting)

    def test_when_default_then_risk_budget_is_none(self) -> None:
        est = build_risk_budgeting(RiskBudgetingConfig())
        assert est.risk_budget is None

    def test_when_min_weights_set_then_forwarded(self) -> None:
        cfg = RiskBudgetingConfig(min_weights=0.01)
        est = build_risk_budgeting(cfg)
        assert est.min_weights == 0.01

    def test_when_max_weights_set_then_forwarded(self) -> None:
        cfg = RiskBudgetingConfig(max_weights=0.10)
        est = build_risk_budgeting(cfg)
        assert est.max_weights == 0.10

    def test_when_transaction_costs_set_then_forwarded(self) -> None:
        cfg = RiskBudgetingConfig(transaction_costs=0.001)
        est = build_risk_budgeting(cfg)
        assert est.transaction_costs == 0.001

    def test_when_explicit_prior_estimator_then_forwarded(self) -> None:
        from skfolio.prior import EmpiricalPrior

        prior = EmpiricalPrior()
        est = build_risk_budgeting(RiskBudgetingConfig(), prior_estimator=prior)
        assert est.prior_estimator is prior

    def test_when_prior_config_then_inner_prior_built(self) -> None:
        from skfolio.prior import EmpiricalPrior

        cfg = RiskBudgetingConfig(prior_config=MomentEstimationConfig())
        est = build_risk_budgeting(cfg)
        assert isinstance(est.prior_estimator, EmpiricalPrior)

    def test_when_kwargs_passed_then_forwarded(self) -> None:
        est = build_risk_budgeting(RiskBudgetingConfig(), risk_free_rate=0.02)
        assert est.risk_free_rate == 0.02

    def test_when_config_none_then_default_used(self) -> None:
        est = build_risk_budgeting(None)
        assert isinstance(est, RiskBudgeting)
        assert est.risk_budget is None


class TestIntegration:
    def test_when_erc_then_fits_and_sums_to_one(self, returns) -> None:
        cfg = RiskBudgetingConfig.for_equal_risk_contribution()
        est = build_risk_budgeting(cfg)
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-6)

    def test_when_custom_budgets_then_fits(self, returns) -> None:
        n = returns.shape[1]
        budgets = dict.fromkeys(returns.columns, 1.0 / n)
        cfg = RiskBudgetingConfig.for_custom_budgets(budgets)
        est = build_risk_budgeting(cfg)
        est.fit(returns)
        assert est.weights_.shape == (n,)
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-6)

    def test_when_column_order_differs_then_budgets_attach_by_ticker(
        self, returns
    ) -> None:
        budgets = {"AAPL": 0.6, "MSFT": 0.3, "JPM": 0.1}
        cfg = RiskBudgetingConfig.for_custom_budgets(budgets)
        aligned = ["AAPL", "MSFT", "JPM"]
        shuffled = ["JPM", "AAPL", "MSFT"]

        ref = build_risk_budgeting(cfg).fit(returns[aligned])
        est = build_risk_budgeting(cfg).fit(returns[shuffled])

        ref_w = pd.Series(ref.weights_, index=aligned)
        est_w = pd.Series(est.weights_, index=shuffled).reindex(aligned)
        np.testing.assert_allclose(est_w.to_numpy(), ref_w.to_numpy(), atol=1e-4)

        cov = returns[shuffled].cov().to_numpy()
        marginal = cov @ est.weights_
        contributions = pd.Series(
            est.weights_ * marginal / (est.weights_ @ marginal), index=shuffled
        )
        for ticker, share in budgets.items():
            assert contributions[ticker] == pytest.approx(share, abs=1e-3)

    def test_when_budgeted_ticker_absent_from_x_then_fits(self, returns) -> None:
        budgets = {"AAPL": 0.6, "MSFT": 0.3, "JPM": 0.1}
        cfg = RiskBudgetingConfig.for_custom_budgets(budgets)
        est = build_risk_budgeting(cfg).fit(returns[["AAPL", "MSFT"]])
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-6)
        assert est.weights_[0] > est.weights_[1] > 0

    def test_when_budget_covers_subset_of_x_then_fits(self, returns) -> None:
        cfg = RiskBudgetingConfig.for_custom_budgets({"AAPL": 0.7, "MSFT": 0.3})
        est = build_risk_budgeting(cfg).fit(returns[["JPM", "MSFT", "AAPL"]])
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-6)
