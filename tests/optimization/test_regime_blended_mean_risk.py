"""Tests for build_regime_blended_mean_risk and RegimeBlendedMeanRiskConfig."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from skfolio.model_selection import WalkForward
from skfolio.portfolio import Portfolio
from sklearn.pipeline import Pipeline

from optimizer.exceptions import ConfigurationError
from optimizer.optimization import (
    RegimeBlendedMeanRiskConfig,
    build_regime_blended_mean_risk,
)

# ---------------------------------------------------------------------------
# Module-scoped fixtures — computed once per test session
# ---------------------------------------------------------------------------

_IDX = pd.bdate_range("2020-01-02", periods=400, freq="B")
_COLS = [f"TICK_{i:02d}" for i in range(15)]
_RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def returns_df() -> pd.DataFrame:
    data = _RNG.normal(loc=0.001, scale=0.02, size=(400, 15))
    return pd.DataFrame(data, index=_IDX, columns=_COLS)


@pytest.fixture(scope="module")
def factor_returns_df(returns_df: pd.DataFrame) -> pd.DataFrame:
    noise = _RNG.normal(0, 0.001, (400, 5))
    data = returns_df.iloc[:, :5].to_numpy() + noise
    cols = [f"F{i}" for i in range(5)]
    return pd.DataFrame(data, index=returns_df.index, columns=cols)


@pytest.fixture(scope="module")
def regime_probs_df(returns_df: pd.DataFrame) -> pd.DataFrame:
    t = np.linspace(0, 2 * np.pi * 400 / 160, 400)
    raw = np.column_stack([np.sin(t), -np.sin(t)])
    exp_raw = np.exp(raw)
    probs = exp_raw / exp_raw.sum(axis=1, keepdims=True)
    return pd.DataFrame(probs, index=returns_df.index, columns=["expansion", "stress"])


# ---------------------------------------------------------------------------
# 1. Pipeline construction and basic fit/predict
# ---------------------------------------------------------------------------


class TestFactoryProducesFittablePipeline:
    def test_returns_pipeline_and_walk_forward(
        self,
        factor_returns_df: pd.DataFrame,
        regime_probs_df: pd.DataFrame,
    ) -> None:
        result = build_regime_blended_mean_risk(
            factor_returns=factor_returns_df,
            regime_probabilities=regime_probs_df,
        )
        pipeline, cv = result
        assert isinstance(result, tuple)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(cv, WalkForward)

    def test_fit_predict_produces_valid_portfolio(
        self,
        returns_df: pd.DataFrame,
        factor_returns_df: pd.DataFrame,
        regime_probs_df: pd.DataFrame,
    ) -> None:
        pipeline, _ = build_regime_blended_mean_risk(
            factor_returns=factor_returns_df,
            regime_probabilities=regime_probs_df,
        )
        pipeline.fit(returns_df, y=factor_returns_df)
        portfolio = pipeline.predict(returns_df)

        assert isinstance(portfolio, Portfolio)
        assert isinstance(portfolio.weights, np.ndarray)
        assert sum(portfolio.weights) == pytest.approx(1.0, rel=1e-4)
        assert np.all(portfolio.weights >= 0.0 - 1e-8)
        assert np.all(portfolio.weights <= 1.0 + 1e-8)


# ---------------------------------------------------------------------------
# 2. Pipeline parameter exposure for hyperparameter tuning
# ---------------------------------------------------------------------------


class TestPipelineExposesStableTuningKeys:
    def test_expected_param_keys_present(
        self,
        factor_returns_df: pd.DataFrame,
        regime_probs_df: pd.DataFrame,
    ) -> None:
        pipeline, _ = build_regime_blended_mean_risk(
            factor_returns=factor_returns_df,
            regime_probabilities=regime_probs_df,
        )
        params = pipeline.get_params()
        assert "optimizer__cvar_beta" in params
        assert "optimizer__l2_coef" in params
        assert "optimizer__max_weights" in params
        assert "drop_correlated__threshold" in params


# ---------------------------------------------------------------------------
# 3. Date alignment: success and failure on overlap
# ---------------------------------------------------------------------------


class TestFactoryAlignsOnIntersectingDates:
    def test_partial_regime_probs_succeeds(
        self,
        returns_df: pd.DataFrame,
        factor_returns_df: pd.DataFrame,
        regime_probs_df: pd.DataFrame,
    ) -> None:
        truncated = regime_probs_df.iloc[30:]
        pipeline, _ = build_regime_blended_mean_risk(
            factor_returns=factor_returns_df,
            regime_probabilities=truncated,
        )
        pipeline.fit(returns_df, y=factor_returns_df)

    def test_too_little_overlap_raises_configuration_error(
        self,
        returns_df: pd.DataFrame,
        factor_returns_df: pd.DataFrame,
        regime_probs_df: pd.DataFrame,
    ) -> None:
        too_short = regime_probs_df.iloc[-20:]
        pipeline, _ = build_regime_blended_mean_risk(
            factor_returns=factor_returns_df,
            regime_probabilities=too_short,
        )
        with pytest.raises(ConfigurationError):
            pipeline.fit(returns_df, y=factor_returns_df)


# ---------------------------------------------------------------------------
# 4. Degenerate regime: constant stress probability
# ---------------------------------------------------------------------------


class TestConstantStressRegime:
    def test_constant_stress_fits_without_error(
        self,
        returns_df: pd.DataFrame,
        factor_returns_df: pd.DataFrame,
    ) -> None:
        probs = pd.DataFrame(
            {"stress": np.ones(400), "expansion": np.zeros(400)},
            index=returns_df.index,
        )
        pipeline, _ = build_regime_blended_mean_risk(
            factor_returns=factor_returns_df,
            regime_probabilities=probs,
        )
        pipeline.fit(returns_df, y=factor_returns_df)
        portfolio = pipeline.predict(returns_df)

        assert not np.any(np.isnan(portfolio.weights))
        assert np.all(portfolio.weights >= 0.0 - 1e-8)
        assert sum(portfolio.weights) == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# 5. EntropyPooling with irreconcilable view raises
# ---------------------------------------------------------------------------


class TestEntropyPoolingViewConflictRaises:
    def test_extreme_mean_view_raises(
        self,
        returns_df: pd.DataFrame,
        factor_returns_df: pd.DataFrame,
        regime_probs_df: pd.DataFrame,
    ) -> None:
        from skfolio.prior import EmpiricalPrior, EntropyPooling, TimeSeriesFactorModel

        from optimizer.optimization._factory import build_mean_risk
        from optimizer.optimization._regime_blended_mean_risk import (
            _ExternallyControlledRegimeCovariance,
        )
        from optimizer.pipeline._builder import build_portfolio_pipeline

        regime_cov = _ExternallyControlledRegimeCovariance(
            regime_probabilities=regime_probs_df
        )
        empirical_prior = EmpiricalPrior(covariance_estimator=regime_cov)
        factor_model = TimeSeriesFactorModel(factor_prior_estimator=empirical_prior)
        ep = EntropyPooling(
            prior_estimator=factor_model, mean_views=["TICK_00 == 99.0"]
        )
        optimizer = build_mean_risk(prior_estimator=ep)
        pipeline = build_portfolio_pipeline(optimizer)

        with pytest.raises(Exception):  # noqa: B017
            pipeline.fit(returns_df, y=factor_returns_df)


# ---------------------------------------------------------------------------
# 6. Config serialisability
# ---------------------------------------------------------------------------


class TestConfigIsSerialisable:
    def test_asdict_produces_primitives_only(self) -> None:
        raw = dataclasses.asdict(RegimeBlendedMeanRiskConfig())

        def _assert_leaves_are_primitives(obj: object) -> None:
            if isinstance(obj, dict):
                for v in obj.values():
                    _assert_leaves_are_primitives(v)
            elif isinstance(obj, list | tuple):
                for item in obj:
                    _assert_leaves_are_primitives(item)
            else:
                assert isinstance(obj, int | float | str | bool | type(None))

        _assert_leaves_are_primitives(raw)

    def test_asdict_succeeds_without_error(self) -> None:
        result = dataclasses.asdict(RegimeBlendedMeanRiskConfig())
        assert isinstance(result, dict)
        assert "mean_risk_config" in result
        assert "walk_forward_config" in result
        assert result["half_life"] == 40


# ---------------------------------------------------------------------------
# 7. Factory input validation (empty-frame guards)
# ---------------------------------------------------------------------------


class TestFactoryInputValidation:
    def test_empty_factor_returns_raises(self, regime_probs_df: pd.DataFrame) -> None:
        with pytest.raises(ConfigurationError, match="factor_returns"):
            build_regime_blended_mean_risk(
                factor_returns=pd.DataFrame(), regime_probabilities=regime_probs_df
            )

    def test_empty_regime_probabilities_raises(
        self, factor_returns_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ConfigurationError, match="regime_probabilities"):
            build_regime_blended_mean_risk(
                factor_returns=factor_returns_df, regime_probabilities=pd.DataFrame()
            )


# ---------------------------------------------------------------------------
# 8. Explicit config propagation and previous-weights forwarding
# ---------------------------------------------------------------------------


class TestExplicitConfigAndPreviousWeights:
    def test_explicit_config_half_life_propagates(
        self, factor_returns_df: pd.DataFrame, regime_probs_df: pd.DataFrame
    ) -> None:
        """An explicit config reaches the regime covariance estimator."""
        cfg = RegimeBlendedMeanRiskConfig(half_life=20)
        pipeline, _ = build_regime_blended_mean_risk(
            config=cfg,
            factor_returns=factor_returns_df,
            regime_probabilities=regime_probs_df,
        )
        # White-box walk of the prior chain (depends on skfolio attribute
        # names): MeanRisk → TimeSeriesFactorModel → EmpiricalPrior → regime cov.
        covariance = pipeline[
            -1
        ].prior_estimator.factor_prior_estimator.covariance_estimator
        assert covariance.half_life == 20

    def test_previous_weights_forwarded_to_optimizer(
        self, factor_returns_df: pd.DataFrame, regime_probs_df: pd.DataFrame
    ) -> None:
        """previous_weights are forwarded into the MeanRisk optimizer."""
        prev = np.full(len(_COLS), 1.0 / len(_COLS))
        pipeline, _ = build_regime_blended_mean_risk(
            factor_returns=factor_returns_df,
            regime_probabilities=regime_probs_df,
            previous_weights=prev,
        )
        np.testing.assert_array_equal(pipeline[-1].previous_weights, prev)


# ---------------------------------------------------------------------------
# 9. Direct fit() guards on the regime covariance estimator
# ---------------------------------------------------------------------------


class TestRegimeCovarianceFitGuards:
    def test_fit_without_regime_probabilities_raises(self) -> None:
        from optimizer.optimization._regime_blended_mean_risk import (
            _ExternallyControlledRegimeCovariance,
        )

        cov = _ExternallyControlledRegimeCovariance(regime_probabilities=None)
        with pytest.raises(ConfigurationError, match="regime_probabilities must be"):
            cov.fit(pd.DataFrame({"A": [0.01, 0.02], "B": [0.0, 0.01]}))

    def test_fit_with_non_dataframe_x_raises(
        self, regime_probs_df: pd.DataFrame
    ) -> None:
        from optimizer.optimization._regime_blended_mean_risk import (
            _ExternallyControlledRegimeCovariance,
        )

        cov = _ExternallyControlledRegimeCovariance(
            regime_probabilities=regime_probs_df
        )
        with pytest.raises(ConfigurationError, match="pandas DataFrame"):
            cov.fit(np.zeros((50, 3)))
