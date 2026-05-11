"""Cycle-3 §7.4 robust-path `_select_optimizer` tests (issue #540)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import MeanRisk
from skfolio.uncertainty_set import EmpiricalMuUncertaintySet

from optimizer.optimization import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
    RobustMeanRiskConfig,
    build_mean_risk,
    build_robust_mean_risk,
)
from research import stock_selection_pipeline as ssp
from research.optimization._config import _select_optimizer


def _hard_config() -> MeanRiskConfig:
    return MeanRiskConfig(
        objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
        risk_measure=RiskMeasureType.VARIANCE,
        min_weights=0.02,
        max_weights=0.10,
        max_sector_weight=0.15,
        budget=1.0,
        l2_coef=0.05,
        solver="CLARABEL",
    )


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    return prices_to_returns(load_sp500_dataset())


class TestSelectOptimizerNonRobust:
    def test_when_robust_false_then_returns_mean_risk_without_uncertainty_set(
        self,
    ) -> None:
        opt = _select_optimizer(
            _hard_config(),
            robust=False,
            uncertainty_level=0.95,
            sector_mapping={"AAPL": "Tech"},
        )
        assert isinstance(opt, MeanRisk)
        assert opt.mu_uncertainty_set_estimator is None

    def test_when_robust_false_then_min_sector_weights_forwarded(self) -> None:
        opt = _select_optimizer(
            _hard_config(),
            robust=False,
            uncertainty_level=0.95,
            sector_mapping={"AAPL": "Tech", "JPM": "Financials"},
            min_sector_weights={"Tech": 0.05},
        )
        constraints = opt.linear_constraints or []
        assert any("Tech >= 0.05" in c for c in constraints)

    def test_when_robust_false_then_previous_weights_forwarded(self) -> None:
        prev = np.array([0.5, 0.5])
        opt = _select_optimizer(
            _hard_config(),
            robust=False,
            uncertainty_level=0.95,
            sector_mapping={"AAPL": "Tech", "JPM": "Financials"},
            previous_weights=prev,
        )
        assert opt.previous_weights is not None
        np.testing.assert_array_equal(opt.previous_weights, prev)


class TestSelectOptimizerRobust:
    def test_when_robust_true_then_mu_uncertainty_set_attached(self) -> None:
        opt = _select_optimizer(
            _hard_config(),
            robust=True,
            uncertainty_level=0.95,
            sector_mapping={"AAPL": "Tech"},
        )
        assert isinstance(opt, MeanRisk)
        assert isinstance(opt.mu_uncertainty_set_estimator, EmpiricalMuUncertaintySet)

    def test_when_robust_true_with_custom_level_then_confidence_matches(self) -> None:
        with patch(
            "research.optimization._config.build_robust_mean_risk",
            wraps=build_robust_mean_risk,
        ) as mock_build:
            _select_optimizer(
                _hard_config(),
                robust=True,
                uncertainty_level=0.90,
                sector_mapping={"AAPL": "Tech"},
            )
        forwarded_cfg = mock_build.call_args.args[0]
        assert isinstance(forwarded_cfg, RobustMeanRiskConfig)
        assert forwarded_cfg.mu_uncertainty_set_config is not None
        assert forwarded_cfg.mu_uncertainty_set_config.confidence_level == 0.90

    def test_when_robust_true_then_underlying_mean_risk_config_preserved(
        self,
    ) -> None:
        with patch(
            "research.optimization._config.build_robust_mean_risk",
            wraps=build_robust_mean_risk,
        ) as mock_build:
            _select_optimizer(
                _hard_config(),
                robust=True,
                uncertainty_level=0.95,
                sector_mapping={"AAPL": "Tech"},
            )
        forwarded_cfg = mock_build.call_args.args[0]
        # The hard-constrained mean_risk_config must travel through unchanged
        assert (
            forwarded_cfg.mean_risk_config.objective
            == ObjectiveFunctionType.MAXIMIZE_RATIO
        )
        assert forwarded_cfg.mean_risk_config.max_weights == pytest.approx(0.10)

    def test_when_robust_true_then_min_sector_weights_forwarded(self) -> None:
        opt = _select_optimizer(
            _hard_config(),
            robust=True,
            uncertainty_level=0.95,
            sector_mapping={"AAPL": "Tech", "JPM": "Financials"},
            min_sector_weights={"Tech": 0.05},
        )
        constraints = opt.linear_constraints or []
        assert any("Tech >= 0.05" in c for c in constraints)

    def test_when_robust_true_then_previous_weights_forwarded(self) -> None:
        prev = np.array([0.5, 0.5])
        opt = _select_optimizer(
            _hard_config(),
            robust=True,
            uncertainty_level=0.95,
            sector_mapping={"AAPL": "Tech", "JPM": "Financials"},
            previous_weights=prev,
        )
        assert opt.previous_weights is not None
        np.testing.assert_array_equal(opt.previous_weights, prev)


class TestSelectOptimizerFallbackContract:
    """Robust path with uncertainty configs None recovers plain MeanRisk."""

    def test_when_uncertainty_configs_none_then_weights_match_atol_1e_8(
        self, returns: pd.DataFrame
    ) -> None:
        cfg = MeanRiskConfig.for_min_variance()
        plain = build_mean_risk(cfg)
        plain.fit(returns)

        # Equivalent of `_select_optimizer(robust=True)` but with both uncertainty
        # configs forced None (bypassing for_moderate) — recovers plain MeanRisk
        # to atol=1e-8 per RobustMeanRisk fallback contract.
        robust_cfg = RobustMeanRiskConfig(mean_risk_config=cfg)
        robust = build_robust_mean_risk(robust_cfg)
        robust.fit(returns)

        np.testing.assert_allclose(plain.weights_, robust.weights_, atol=1e-8)


class TestOptimizePortfolioRobustSignature:
    """`optimize_portfolio` accepts `robust` and `uncertainty_level` parameters."""

    def _stub_assembly(self) -> Any:
        from types import SimpleNamespace

        idx = pd.bdate_range("2020-01-01", periods=10)
        tickers = ["AAA", "BBB"]
        prices = pd.DataFrame(100.0, index=idx, columns=tickers)
        volumes = pd.DataFrame(1_000_000.0, index=idx, columns=tickers)
        return SimpleNamespace(
            prices=prices,
            volumes=volumes,
            macro_data=pd.DataFrame({"gdp_growth": [1.0]}, index=[idx[0]]),
            regime_data=pd.DataFrame({"pmi": [55.0]}, index=[idx[0]]),
            fundamentals=pd.DataFrame({"market_cap": [1e9, 2e9]}, index=tickers),
            analyst_data=pd.DataFrame(),
            insider_data=pd.DataFrame(),
            sector_mapping={"AAA": "Tech", "BBB": "Finance"},
            risk_free_rate=0.0,
            delisting_returns={},
            currency_map={"AAA": "USD", "BBB": "USD"},
            fx_rates=pd.DataFrame(),
        )

    def _capture(self, *, robust: bool, uncertainty_level: float = 0.95) -> Any:
        from types import SimpleNamespace

        captured: dict[str, Any] = {}

        def fake_pipeline(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                summary={"sharpe_ratio": 1.0},
                net_sharpe_ratio=None,
                weights=pd.Series([0.5, 0.5], index=["AAA", "BBB"]),
                net_returns=None,
            )

        with patch.object(
            ssp, "run_full_pipeline_with_selection", side_effect=fake_pipeline
        ):
            ssp.optimize_portfolio(
                assembly=self._stub_assembly(),
                investable=pd.Index(["AAA", "BBB"]),
                ic_history=None,
                n_selected=2,
                robust=robust,
                uncertainty_level=uncertainty_level,
            )
        return captured

    def test_when_robust_false_then_optimizer_has_no_uncertainty_set(self) -> None:
        captured = self._capture(robust=False)
        assert captured["optimizer"].mu_uncertainty_set_estimator is None

    def test_when_robust_true_then_optimizer_has_uncertainty_set(self) -> None:
        captured = self._capture(robust=True, uncertainty_level=0.95)
        assert isinstance(
            captured["optimizer"].mu_uncertainty_set_estimator,
            EmpiricalMuUncertaintySet,
        )
