"""Tests for factor integration with optimization."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    FactorExposureConstraints,
    build_factor_bl_views,
    build_factor_exposure_constraints,
    estimate_factor_premia,
)
from optimizer.optimization import build_mean_risk


@pytest.fixture()
def factor_scores() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    tickers = [f"T{i:02d}" for i in range(20)]
    return pd.DataFrame(
        rng.normal(0, 1, (20, 3)),
        index=tickers,
        columns=["value", "momentum", "profitability"],
    )


class TestBuildFactorBLViews:
    def test_returns_views_and_confidences(self, factor_scores: pd.DataFrame) -> None:
        premia = {"value": 0.04, "momentum": 0.06}
        views, confidences = build_factor_bl_views(
            factor_scores, premia, factor_scores.index
        )
        assert isinstance(views, list)
        assert isinstance(confidences, list)
        assert len(views) == len(confidences)

    def test_empty_premia(self, factor_scores: pd.DataFrame) -> None:
        views, confidences = build_factor_bl_views(
            factor_scores, {}, factor_scores.index
        )
        assert len(views) == 0


class TestBuildFactorExposureConstraints:
    def test_returns_dataclass(self, factor_scores: pd.DataFrame) -> None:
        result = build_factor_exposure_constraints(
            factor_scores, bounds=(-0.5, 0.5)
        )
        assert isinstance(result, FactorExposureConstraints)

    def test_no_longer_returns_strings(self, factor_scores: pd.DataFrame) -> None:
        """Acceptance criterion: old text-output path is gone."""
        result = build_factor_exposure_constraints(
            factor_scores, bounds=(-0.5, 0.5)
        )
        assert not isinstance(result, list)
        with pytest.raises(TypeError):
            # Iterating and expecting strings must fail
            for item in result:  # type: ignore[misc]
                assert "exposure" in item

    def test_matrix_shapes(self, factor_scores: pd.DataFrame) -> None:
        n_assets, n_factors = factor_scores.shape
        result = build_factor_exposure_constraints(
            factor_scores, bounds=(-0.5, 0.5)
        )
        assert result.left_inequality.shape == (2 * n_factors, n_assets)
        assert result.right_inequality.shape == (2 * n_factors,)

    def test_factor_names_preserved(self, factor_scores: pd.DataFrame) -> None:
        result = build_factor_exposure_constraints(
            factor_scores, bounds=(-0.5, 0.5)
        )
        assert result.factor_names == list(factor_scores.columns)

    def test_bounds_stored(self, factor_scores: pd.DataFrame) -> None:
        result = build_factor_exposure_constraints(
            factor_scores, bounds=(-0.3, 0.4)
        )
        assert np.all(result.lower_bounds == -0.3)
        assert np.all(result.upper_bounds == 0.4)

    def test_per_factor_bounds(self, factor_scores: pd.DataFrame) -> None:
        bounds = {
            "value": (-0.2, 0.2),
            "momentum": (-0.5, 0.5),
            "profitability": (-0.1, 0.8),
        }
        result = build_factor_exposure_constraints(factor_scores, bounds=bounds)
        assert result.lower_bounds[0] == pytest.approx(-0.2)
        assert result.upper_bounds[1] == pytest.approx(0.5)

    def test_missing_factor_in_bounds_dict_raises(
        self, factor_scores: pd.DataFrame
    ) -> None:
        bounds = {"value": (-0.2, 0.2)}  # missing momentum, profitability
        with pytest.raises(KeyError, match="momentum"):
            build_factor_exposure_constraints(factor_scores, bounds=bounds)

    def test_feasibility_warning_when_ew_outside_bounds(self) -> None:
        """Warn when equal-weight exposure violates bounds."""
        tickers = [f"T{i}" for i in range(10)]
        # Scores all strongly positive → equal-weight exposure >> 0
        scores = pd.DataFrame(
            np.ones((10, 1)) * 5.0, index=tickers, columns=["value"]
        )
        with pytest.warns(UserWarning, match="equal-weight exposure"):
            build_factor_exposure_constraints(scores, bounds=(-0.1, 0.1))

    def test_no_warning_when_feasible(self, factor_scores: pd.DataFrame) -> None:
        """No warning when equal-weight exposure is within wide bounds."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Wide bounds; z scores have near-zero cross-sectional mean
            build_factor_exposure_constraints(
                factor_scores, bounds=(-10.0, 10.0)
            )

    def test_constraint_encodes_correct_inequality(self) -> None:
        """A @ w <= b encodes lb <= z @ w <= ub."""
        rng = np.random.default_rng(7)
        tickers = [f"T{i}" for i in range(5)]
        scores = pd.DataFrame(
            rng.normal(0, 1, (5, 1)), index=tickers, columns=["value"]
        )
        lb, ub = -0.5, 0.5
        result = build_factor_exposure_constraints(scores, bounds=(lb, ub))
        z = scores["value"].values
        # Row 0: -z @ w <= -lb  =>  z @ w >= lb
        np.testing.assert_array_almost_equal(result.left_inequality[0], -z)
        assert result.right_inequality[0] == pytest.approx(-lb)
        # Row 1: z @ w <= ub
        np.testing.assert_array_almost_equal(result.left_inequality[1], z)
        assert result.right_inequality[1] == pytest.approx(ub)

    def test_passable_to_mean_risk(self, factor_scores: pd.DataFrame) -> None:
        """Constraints can be passed to MeanRisk without raising."""
        fec = build_factor_exposure_constraints(
            factor_scores, bounds=(-2.0, 2.0)
        )
        optimizer = build_mean_risk(factor_exposure_constraints=fec)
        rng = np.random.default_rng(10)
        X = pd.DataFrame(
            rng.normal(0, 0.01, (100, len(factor_scores))),
            columns=factor_scores.index,
        )
        optimizer.fit(X)
        assert optimizer.weights_ is not None


class TestBuildMeanRiskWithFactorExposure:
    """Integration test: portfolio satisfies lb <= z@w <= ub."""

    def test_exposure_within_bounds(self) -> None:
        """Acceptance criterion: Σ w_i z_{i,g} is within [lb, ub] to 1e-4."""
        rng = np.random.default_rng(99)
        n = 30
        tickers = [f"S{i:02d}" for i in range(n)]

        # Return data
        X = pd.DataFrame(
            rng.normal(0, 0.01, (252, n)), columns=tickers
        )

        # Factor scores: standardised (zero mean, unit variance cross-sectionally)
        raw = rng.normal(0, 1, n)
        z = (raw - raw.mean()) / raw.std()
        factor_scores = pd.DataFrame(
            z.reshape(n, 1), index=tickers, columns=["value"]
        )

        lb, ub = -0.2, 0.2
        fec = build_factor_exposure_constraints(
            factor_scores, bounds=(lb, ub)
        )
        optimizer = build_mean_risk(factor_exposure_constraints=fec)
        optimizer.fit(X)

        exposure = float(np.dot(optimizer.weights_, z))
        assert lb - 1e-4 <= exposure <= ub + 1e-4, (
            f"Exposure {exposure:.6f} outside [{lb}, {ub}]"
        )

    def test_explicit_kwargs_override_constraints(self) -> None:
        """Explicit left/right_inequality kwargs override FactorExposureConstraints."""
        rng = np.random.default_rng(5)
        n = 10
        tickers = [f"A{i}" for i in range(n)]
        z = rng.normal(0, 1, n)
        factor_scores = pd.DataFrame(
            z.reshape(n, 1), index=tickers, columns=["value"]
        )
        fec = build_factor_exposure_constraints(
            factor_scores, bounds=(-100.0, 100.0)
        )

        # Pass a tighter explicit constraint: z @ w <= 0.01
        explicit_A = np.array([z])
        explicit_b = np.array([0.01])
        optimizer = build_mean_risk(
            factor_exposure_constraints=fec,
            left_inequality=explicit_A,
            right_inequality=explicit_b,
        )
        X = pd.DataFrame(rng.normal(0, 0.01, (100, n)), columns=tickers)
        optimizer.fit(X)
        exposure = float(np.dot(optimizer.weights_, z))
        assert exposure <= 0.01 + 1e-4


class TestEstimateFactorPremia:
    def test_returns_dict(self) -> None:
        rng = np.random.default_rng(42)
        fmp_returns = pd.DataFrame(
            rng.normal(0.0002, 0.01, (252, 3)),
            columns=["value", "momentum", "quality"],
        )
        result = estimate_factor_premia(fmp_returns)
        assert isinstance(result, dict)
        assert "value" in result
        assert "momentum" in result
        # Annualized premia should be reasonable
        for v in result.values():
            assert -1.0 < v < 1.0
