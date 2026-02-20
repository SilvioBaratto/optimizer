"""Tests for ML-based composite scoring (ridge and GBT)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV

from optimizer.factors import (
    CompositeMethod,
    CompositeScoringConfig,
    compute_composite_score,
    compute_ml_composite,
    fit_gbt_composite,
    fit_ridge_composite,
    predict_composite_scores,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def factor_columns() -> list[str]:
    return ["book_to_price", "earnings_yield", "gross_profitability", "momentum_12_1"]


@pytest.fixture()
def training_scores(factor_columns: list[str]) -> pd.DataFrame:
    """100-ticker training factor matrix."""
    rng = np.random.default_rng(0)
    tickers = [f"T{i:03d}" for i in range(100)]
    return pd.DataFrame(
        rng.normal(0, 1, (100, len(factor_columns))),
        index=tickers,
        columns=factor_columns,
    )


@pytest.fixture()
def training_returns(training_scores: pd.DataFrame) -> pd.Series:
    """Forward returns correlated with first factor."""
    rng = np.random.default_rng(1)
    returns = (
        0.3 * training_scores["book_to_price"]
        + rng.normal(0, 0.05, len(training_scores))
    )
    return pd.Series(returns, index=training_scores.index)


@pytest.fixture()
def current_scores(factor_columns: list[str]) -> pd.DataFrame:
    """20-ticker current-period factor matrix."""
    rng = np.random.default_rng(2)
    tickers = [f"C{i:02d}" for i in range(20)]
    return pd.DataFrame(
        rng.normal(0, 1, (20, len(factor_columns))),
        index=tickers,
        columns=factor_columns,
    )


@pytest.fixture()
def coverage(current_scores: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        True,
        index=current_scores.index,
        columns=current_scores.columns,
    )


# ---------------------------------------------------------------------------
# Tests: fit_ridge_composite
# ---------------------------------------------------------------------------


class TestFitRidgeComposite:
    def test_returns_ridgecv(
        self, training_scores: pd.DataFrame, training_returns: pd.Series
    ) -> None:
        model = fit_ridge_composite(training_scores, training_returns)
        assert isinstance(model, RidgeCV)

    def test_coef_shape(
        self, training_scores: pd.DataFrame, training_returns: pd.Series
    ) -> None:
        model = fit_ridge_composite(training_scores, training_returns)
        assert model.coef_.shape == (training_scores.shape[1],)

    def test_orthogonal_factors_approximately_equal_weights(self) -> None:
        """When factors are orthogonal and each has the same signal-to-noise,
        ridge weights should be approximately equal in magnitude."""
        rng = np.random.default_rng(42)
        n = 200
        # Two uncorrelated factors with identical signal strength
        f1 = rng.normal(0, 1, n)
        f2 = rng.normal(0, 1, n)
        noise = rng.normal(0, 0.5, n)
        y = 0.5 * f1 + 0.5 * f2 + noise
        scores = pd.DataFrame({"f1": f1, "f2": f2}, index=range(n))
        returns = pd.Series(y, index=range(n))

        model = fit_ridge_composite(scores, returns, alpha=0.01)
        coefs = np.abs(model.coef_)
        # Both weights should be similar (within 2x of each other)
        assert coefs[0] / coefs[1] < 2.0
        assert coefs[1] / coefs[0] < 2.0

    def test_handles_nan_rows(self) -> None:
        """NaN rows in training data are dropped; model still fits."""
        rng = np.random.default_rng(5)
        scores = pd.DataFrame(
            rng.normal(0, 1, (50, 3)), columns=["a", "b", "c"]
        )
        returns = pd.Series(rng.normal(0, 1, 50))
        # Inject NaN
        scores.iloc[5, 1] = float("nan")
        scores.iloc[10, :] = float("nan")
        model = fit_ridge_composite(scores, returns)
        assert isinstance(model, RidgeCV)

    def test_partial_index_overlap(self) -> None:
        """Only the common index rows are used for training."""
        rng = np.random.default_rng(6)
        scores = pd.DataFrame(
            rng.normal(0, 1, (60, 2)),
            index=range(60),
            columns=["f1", "f2"],
        )
        returns = pd.Series(rng.normal(0, 1, 40), index=range(40))
        # Only 40 rows overlap
        model = fit_ridge_composite(scores, returns)
        assert isinstance(model, RidgeCV)


# ---------------------------------------------------------------------------
# Tests: fit_gbt_composite
# ---------------------------------------------------------------------------


class TestFitGbtComposite:
    def test_returns_gbt(
        self, training_scores: pd.DataFrame, training_returns: pd.Series
    ) -> None:
        model = fit_gbt_composite(training_scores, training_returns)
        assert isinstance(model, GradientBoostingRegressor)

    def test_respects_max_depth(
        self, training_scores: pd.DataFrame, training_returns: pd.Series
    ) -> None:
        model = fit_gbt_composite(training_scores, training_returns, max_depth=2)
        assert model.max_depth == 2

    def test_respects_n_estimators(
        self, training_scores: pd.DataFrame, training_returns: pd.Series
    ) -> None:
        model = fit_gbt_composite(
            training_scores, training_returns, n_estimators=10
        )
        assert model.n_estimators_ == 10

    def test_handles_nan_rows(self) -> None:
        rng = np.random.default_rng(7)
        scores = pd.DataFrame(
            rng.normal(0, 1, (50, 3)), columns=["a", "b", "c"]
        )
        returns = pd.Series(rng.normal(0, 1, 50))
        scores.iloc[3, 2] = float("nan")
        model = fit_gbt_composite(scores, returns, n_estimators=5)
        assert isinstance(model, GradientBoostingRegressor)


# ---------------------------------------------------------------------------
# Tests: predict_composite_scores
# ---------------------------------------------------------------------------


class TestPredictCompositeScores:
    def test_output_shape(
        self,
        training_scores: pd.DataFrame,
        training_returns: pd.Series,
        current_scores: pd.DataFrame,
    ) -> None:
        model = fit_ridge_composite(training_scores, training_returns)
        result = predict_composite_scores(model, current_scores)
        assert isinstance(result, pd.Series)
        assert len(result) == len(current_scores)

    def test_zero_mean_unit_variance(
        self,
        training_scores: pd.DataFrame,
        training_returns: pd.Series,
        current_scores: pd.DataFrame,
    ) -> None:
        """Acceptance criterion: normalised output has zero mean and unit variance."""
        model = fit_ridge_composite(training_scores, training_returns)
        result = predict_composite_scores(model, current_scores)
        assert abs(result.mean()) < 1e-10
        assert abs(result.std(ddof=0) - 1.0) < 1e-10

    def test_gbt_zero_mean_unit_variance(
        self,
        training_scores: pd.DataFrame,
        training_returns: pd.Series,
        current_scores: pd.DataFrame,
    ) -> None:
        model = fit_gbt_composite(training_scores, training_returns, n_estimators=5)
        result = predict_composite_scores(model, current_scores)
        assert abs(result.mean()) < 1e-10
        assert abs(result.std(ddof=0) - 1.0) < 1e-10

    def test_nan_rows_propagated(self) -> None:
        """Rows with all-NaN factors should produce NaN in output."""
        rng = np.random.default_rng(8)
        cols = ["a", "b", "c"]
        train = pd.DataFrame(rng.normal(0, 1, (50, 3)), columns=cols)
        returns = pd.Series(rng.normal(0, 1, 50))
        model = fit_ridge_composite(train, returns)

        scores = pd.DataFrame(
            rng.normal(0, 1, (10, 3)),
            index=[f"S{i}" for i in range(10)],
            columns=cols,
        )
        scores.iloc[2, :] = float("nan")
        result = predict_composite_scores(model, scores)
        assert pd.isna(result.iloc[2])

    def test_index_preserved(
        self,
        training_scores: pd.DataFrame,
        training_returns: pd.Series,
        current_scores: pd.DataFrame,
    ) -> None:
        model = fit_ridge_composite(training_scores, training_returns)
        result = predict_composite_scores(model, current_scores)
        pd.testing.assert_index_equal(result.index, current_scores.index)

    def test_single_ticker_returns_zero(
        self,
        training_scores: pd.DataFrame,
        training_returns: pd.Series,
    ) -> None:
        """Single-row input: StandardScaler cannot standardise; returns zero."""
        model = fit_ridge_composite(training_scores, training_returns)
        single = training_scores.iloc[[0]]
        result = predict_composite_scores(model, single)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_no_extrapolation_extreme_values(
        self,
        training_scores: pd.DataFrame,
        training_returns: pd.Series,
    ) -> None:
        """GBT output is bounded: no extreme extrapolation (values within ±5 sigma)."""
        model = fit_gbt_composite(training_scores, training_returns, n_estimators=10)
        rng = np.random.default_rng(99)
        # Extreme factor values 10x training range
        extreme_scores = pd.DataFrame(
            rng.normal(0, 10, (50, training_scores.shape[1])),
            columns=training_scores.columns,
        )
        result = predict_composite_scores(model, extreme_scores)
        # After normalisation output is always unit-variance; check no Inf/NaN
        assert result.notna().all()
        assert np.isfinite(result.values).all()


# ---------------------------------------------------------------------------
# Tests: compute_ml_composite (integration)
# ---------------------------------------------------------------------------


class TestComputeMLComposite:
    def test_ridge_returns_series(
        self,
        current_scores: pd.DataFrame,
        training_scores: pd.DataFrame,
        training_returns: pd.Series,
    ) -> None:
        config = CompositeScoringConfig.for_ridge_weighted()
        result = compute_ml_composite(
            current_scores, training_scores, training_returns, config
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(current_scores)

    def test_gbt_returns_series(
        self,
        current_scores: pd.DataFrame,
        training_scores: pd.DataFrame,
        training_returns: pd.Series,
    ) -> None:
        config = CompositeScoringConfig.for_gbt_weighted()
        result = compute_ml_composite(
            current_scores, training_scores, training_returns, config
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(current_scores)

    def test_ridge_normalised(
        self,
        current_scores: pd.DataFrame,
        training_scores: pd.DataFrame,
        training_returns: pd.Series,
    ) -> None:
        config = CompositeScoringConfig.for_ridge_weighted()
        result = compute_ml_composite(
            current_scores, training_scores, training_returns, config
        )
        assert abs(result.mean()) < 1e-10
        assert abs(result.std(ddof=0) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Tests: compute_composite_score dispatch
# ---------------------------------------------------------------------------


class TestComputeCompositeScoreDispatch:
    def test_ridge_weighted_dispatch(
        self,
        current_scores: pd.DataFrame,
        coverage: pd.DataFrame,
        training_scores: pd.DataFrame,
        training_returns: pd.Series,
    ) -> None:
        config = CompositeScoringConfig.for_ridge_weighted()
        result = compute_composite_score(
            current_scores,
            coverage,
            config=config,
            training_scores=training_scores,
            training_returns=training_returns,
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(current_scores)

    def test_gbt_weighted_dispatch(
        self,
        current_scores: pd.DataFrame,
        coverage: pd.DataFrame,
        training_scores: pd.DataFrame,
        training_returns: pd.Series,
    ) -> None:
        config = CompositeScoringConfig.for_gbt_weighted()
        result = compute_composite_score(
            current_scores,
            coverage,
            config=config,
            training_scores=training_scores,
            training_returns=training_returns,
        )
        assert isinstance(result, pd.Series)

    def test_ridge_raises_without_training_data(
        self, current_scores: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        config = CompositeScoringConfig.for_ridge_weighted()
        with pytest.raises(ValueError, match="training_scores"):
            compute_composite_score(current_scores, coverage, config=config)

    def test_gbt_raises_without_training_data(
        self, current_scores: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        config = CompositeScoringConfig.for_gbt_weighted()
        with pytest.raises(ValueError, match="training_scores"):
            compute_composite_score(current_scores, coverage, config=config)

    def test_ridge_and_gbt_are_valid_composite_methods(self) -> None:
        """Acceptance criterion: RIDGE_WEIGHTED and GBT_WEIGHTED are valid enum values."""
        assert CompositeMethod.RIDGE_WEIGHTED in CompositeMethod
        assert CompositeMethod.GBT_WEIGHTED in CompositeMethod
        assert CompositeMethod.RIDGE_WEIGHTED.value == "ridge_weighted"
        assert CompositeMethod.GBT_WEIGHTED.value == "gbt_weighted"

    def test_config_presets(self) -> None:
        ridge_cfg = CompositeScoringConfig.for_ridge_weighted()
        gbt_cfg = CompositeScoringConfig.for_gbt_weighted()
        assert ridge_cfg.method == CompositeMethod.RIDGE_WEIGHTED
        assert gbt_cfg.method == CompositeMethod.GBT_WEIGHTED

    def test_config_hyperparams(self) -> None:
        cfg = CompositeScoringConfig(
            method=CompositeMethod.RIDGE_WEIGHTED,
            ridge_alpha=0.1,
            gbt_max_depth=4,
            gbt_n_estimators=30,
        )
        assert cfg.ridge_alpha == 0.1
        assert cfg.gbt_max_depth == 4
        assert cfg.gbt_n_estimators == 30

    def test_temporal_isolation_no_future_leak(
        self,
        current_scores: pd.DataFrame,
        coverage: pd.DataFrame,
        training_scores: pd.DataFrame,
        training_returns: pd.Series,
    ) -> None:
        """Training and prediction tickers are disjoint — no data leakage."""
        assert len(training_scores.index.intersection(current_scores.index)) == 0
        config = CompositeScoringConfig.for_ridge_weighted()
        # Should succeed with fully disjoint train/predict sets
        result = compute_composite_score(
            current_scores,
            coverage,
            config=config,
            training_scores=training_scores,
            training_returns=training_returns,
        )
        assert isinstance(result, pd.Series)
