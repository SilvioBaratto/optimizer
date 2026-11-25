"""Tests for the PCA residual (eigenportfolio) statistical-arbitrage model.

The golden numbers come from ``np.linalg.eigvalsh`` on the correlation matrix, from
the closed-form eigenvalue of an equicorrelated market, from least-squares and AR(1)
formulas written out in this file, and from the Avellaneda and Lee (2010) trading
thresholds worked through by hand.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.statarb import (
    PcaResidualConfig,
    PcaResidualModel,
    fit_pca_residual_model,
    pca_signals,
)
from optimizer.statarb._pca import _ou_summary

_PERIODS_PER_YEAR = 252
# Avellaneda and Lee (2010) trade only residuals that revert faster than 30 days,
# kappa above 252 / 30 = 8.4 per year.
_PAPER_MIN_KAPPA = 8.4
_SIGMA = 0.01
# Values that are not whole-number settings: NaN, infinity, fractions, integral
# floats, booleans (an int subclass), numpy floats and text.
_NOT_INTEGERS = [math.nan, math.inf, 2.5, 3.0, np.float64(3.0), True, "3"]


def _path_returns(
    rng: np.random.Generator, size: int, coefficient: float | None, scale: float
) -> np.ndarray:
    """Return idiosyncratic returns whose running sum is an AR(1) path.

    ``coefficient=None`` gives white-noise returns, so the running sum is a random
    walk. Otherwise the path follows ``x[t] = b x[t-1] + shock`` and the returns are
    its first differences, with the shock sized so the returns have variance
    ``(scale * _SIGMA) ** 2``.
    """
    volatility = scale * _SIGMA
    if coefficient is None:
        return rng.normal(scale=volatility, size=size)
    shock_sd = volatility * math.sqrt((1.0 + coefficient) / 2.0)
    shocks = rng.normal(scale=shock_sd, size=size)
    path = np.zeros(size)
    previous = 0.0
    for t in range(size):
        previous = coefficient * previous + shocks[t]
        path[t] = previous
    return np.diff(np.concatenate([[0.0], path]))


def _market(
    seed: int,
    size: int,
    coefficients: Sequence[float | None],
    scales: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Return returns driven by one common factor plus idiosyncratic OU paths.

    ``scales`` multiplies each asset's idiosyncratic volatility (default 1).
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(scale=_SIGMA, size=size)
    sizes = scales if scales is not None else [1.0] * len(coefficients)
    columns = {
        f"A{i:02d}": common + _path_returns(rng, size, coefficient, scale)
        for i, (coefficient, scale) in enumerate(zip(coefficients, sizes, strict=True))
    }
    index = pd.date_range("2018-01-01", periods=size, freq="B")
    return pd.DataFrame(columns, index=index)


def _correlation(returns: pd.DataFrame) -> np.ndarray:
    """Return the sample correlation matrix of the columns."""
    return np.corrcoef(returns.to_numpy().T)


def _ar1_by_hand(path: np.ndarray) -> tuple[float, float, float]:
    """Return intercept, slope and residual variance of ``x[t+1] = a + b x[t]``."""
    lagged, following = path[:-1], path[1:]
    centred = lagged - lagged.mean()
    slope = float((centred * (following - following.mean())).sum() / (centred**2).sum())
    intercept = float(following.mean() - slope * lagged.mean())
    errors = following - intercept - slope * lagged
    return intercept, slope, float((errors**2).sum() / (len(errors) - 2))


@pytest.fixture(scope="module")
def equicorrelated() -> pd.DataFrame:
    """Twenty assets with pairwise correlation 0.5 and OU residuals (b = 0.8)."""
    return _market(seed=0, size=3000, coefficients=[0.8] * 20)


@pytest.fixture(scope="module")
def equicorrelated_model(equicorrelated: pd.DataFrame) -> PcaResidualModel:
    """The one-factor model fitted to ``equicorrelated``."""
    return fit_pca_residual_model(equicorrelated, n_components=1)


@pytest.fixture(scope="module")
def small_market() -> pd.DataFrame:
    """Twelve assets of mixed speeds, used for the algebraic identities."""
    return _market(seed=3, size=400, coefficients=[0.5, 0.7, 0.8, 0.9] * 3)


class TestEigenStructure:
    """The principal components of the standardised returns."""

    def test_eigenvalues_equal_those_of_the_correlation_matrix(
        self, small_market: pd.DataFrame
    ) -> None:
        """All twelve eigenvalues match ``eigvalsh`` of ``corrcoef``, largest first."""
        model = fit_pca_residual_model(small_market, n_components=3)
        expected = np.linalg.eigvalsh(_correlation(small_market))[::-1]
        np.testing.assert_allclose(model.eigenvalues.to_numpy(), expected, rtol=1e-9)
        assert model.eigenvalues.is_monotonic_decreasing

    def test_explained_variance_ratio_is_the_eigenvalue_share(
        self, small_market: pd.DataFrame
    ) -> None:
        """The trace of a correlation matrix is the number of assets."""
        model = fit_pca_residual_model(small_market, n_components=3)
        np.testing.assert_allclose(
            model.explained_variance_ratio.to_numpy(),
            model.eigenvalues.to_numpy() / 12.0,
            rtol=1e-12,
        )
        assert model.explained_variance_ratio.sum() == pytest.approx(1.0)

    def test_first_eigenvalue_share_matches_one_factor_theory(
        self, equicorrelated_model: PcaResidualModel
    ) -> None:
        """Equicorrelation 0.5 over 20 assets gives a share of (1 + 19 / 2) / 20."""
        # With a common-factor variance equal to the idiosyncratic one, the pairwise
        # correlation is 0.5 and the top eigenvalue of (1 - r) I + r 11' is
        # 1 + (N - 1) r = 10.5, a share of 0.525. Sampling error over 3000 days is
        # about 0.01, so 0.03 is three standard errors.
        share = equicorrelated_model.explained_variance_ratio.iloc[0]
        assert share == pytest.approx(0.525, abs=0.03)

    def test_loadings_solve_the_eigenvalue_equation(
        self, small_market: pd.DataFrame
    ) -> None:
        """``corr @ v = lambda v``, unit length, mutually orthogonal."""
        model = fit_pca_residual_model(small_market, n_components=3)
        corr = _correlation(small_market)
        loadings = model.loadings.to_numpy()
        for j in range(3):
            np.testing.assert_allclose(
                corr @ loadings[:, j],
                model.eigenvalues.iloc[j] * loadings[:, j],
                atol=1e-10,
            )
        np.testing.assert_allclose(loadings.T @ loadings, np.eye(3), atol=1e-10)

    def test_the_market_component_loads_equally_on_every_asset(
        self, equicorrelated_model: PcaResidualModel
    ) -> None:
        """The leading eigenvector of an equicorrelated market is 1 / sqrt(N)."""
        loadings = equicorrelated_model.loadings["PC1"]
        assert (loadings > 0).all()
        np.testing.assert_allclose(loadings, 1.0 / math.sqrt(20), atol=0.03)

    def test_factor_returns_are_eigenvectors_over_volatilities_times_returns(
        self, small_market: pd.DataFrame
    ) -> None:
        """``F[t, j] = sum_i v[i, j] / sigma_i * R[t, i]`` on the raw returns."""
        model = fit_pca_residual_model(small_market, n_components=2)
        raw = small_market.to_numpy()
        volatility = [np.std(raw[:, i], ddof=1) for i in range(12)]
        expected = np.zeros((len(raw), 2))
        for j in range(2):
            for i in range(12):
                weight = model.loadings.iloc[i, j] / volatility[i]
                expected[:, j] += weight * raw[:, i]
        np.testing.assert_allclose(
            model.factor_returns.to_numpy(), expected, rtol=1e-10
        )

    def test_factor_returns_are_uncorrelated_with_variance_equal_to_eigenvalue(
        self, small_market: pd.DataFrame
    ) -> None:
        """Projections on orthogonal eigenvectors have covariance ``diag(lambda)``."""
        model = fit_pca_residual_model(small_market, n_components=3)
        covariance = np.cov(model.factor_returns.to_numpy().T)
        np.testing.assert_allclose(
            covariance, np.diag(model.eigenvalues.iloc[:3]), atol=1e-9
        )

    def test_labels_and_shapes(self, small_market: pd.DataFrame) -> None:
        """Frames are labelled by asset, date and component."""
        model = fit_pca_residual_model(small_market, n_components=3)
        components = ["PC1", "PC2", "PC3"]
        assert list(model.loadings.columns) == components
        assert list(model.loadings.index) == list(small_market.columns)
        assert list(model.factor_returns.columns) == components
        assert model.factor_returns.index.equals(small_market.index)
        assert model.betas.shape == (12, 3)
        assert list(model.betas.index) == list(small_market.columns)
        assert model.residuals.shape == small_market.shape
        assert model.residuals.index.equals(small_market.index)
        assert list(model.ou_parameters.index) == list(small_market.columns)
        assert list(model.s_scores.index) == list(small_market.columns)
        assert len(model.eigenvalues) == 12

    def test_the_fit_is_reproducible(self, small_market: pd.DataFrame) -> None:
        """The same returns give identical outputs, signs included."""
        first = fit_pca_residual_model(small_market, n_components=3)
        second = fit_pca_residual_model(small_market.copy(), n_components=3)
        pd.testing.assert_frame_equal(first.loadings, second.loadings)
        pd.testing.assert_frame_equal(first.ou_parameters, second.ou_parameters)

    def test_result_is_immutable(self, small_market: pd.DataFrame) -> None:
        """The model is a frozen dataclass."""
        model = fit_pca_residual_model(small_market, n_components=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            model.betas = model.loadings  # type: ignore[misc]


class TestRegression:
    """Per-asset regression on the eigenportfolio returns."""

    def test_residuals_satisfy_the_normal_equations(
        self, small_market: pd.DataFrame
    ) -> None:
        """Residuals have zero mean and are orthogonal to every factor."""
        model = fit_pca_residual_model(small_market, n_components=3)
        residuals = model.residuals.to_numpy()
        np.testing.assert_allclose(residuals.mean(axis=0), 0.0, atol=1e-15)
        factors = model.factor_returns.to_numpy()
        np.testing.assert_allclose(
            (factors - factors.mean(axis=0)).T @ residuals, 0.0, atol=1e-12
        )

    def test_fit_plus_residual_is_constant_over_time(
        self, small_market: pd.DataFrame
    ) -> None:
        """``R - F beta' - residual`` is the regression constant, one per asset."""
        model = fit_pca_residual_model(small_market, n_components=3)
        gap = (
            small_market.to_numpy()
            - model.factor_returns.to_numpy() @ model.betas.to_numpy().T
            - model.residuals.to_numpy()
        )
        np.testing.assert_allclose(gap.std(axis=0), 0.0, atol=1e-15)

    def test_betas_equal_the_closed_form_slope_for_one_factor(
        self, small_market: pd.DataFrame
    ) -> None:
        """With one factor the beta is ``cov(R_i, F) / var(F)``."""
        model = fit_pca_residual_model(small_market, n_components=1)
        factor = model.factor_returns["PC1"].to_numpy()
        for asset in small_market.columns:
            returns = small_market[asset].to_numpy()
            slope = np.cov(returns, factor)[0, 1] / factor.var(ddof=1)
            assert model.betas.loc[asset, "PC1"] == pytest.approx(slope, rel=1e-9)


class TestOuParameters:
    """The AR(1) fit of each asset's cumulative residual."""

    def test_parameters_equal_an_independent_ar1_fit(
        self, small_market: pd.DataFrame
    ) -> None:
        """kappa, m, sigma_eq, half-life and the s-score follow the AR(1) formulas."""
        model = fit_pca_residual_model(small_market, n_components=3)
        for asset in small_market.columns:
            path = np.cumsum(model.residuals[asset].to_numpy())
            intercept, slope, noise_variance = _ar1_by_hand(path)
            row = model.ou_parameters.loc[asset]
            assert 0.0 < slope < 1.0
            assert row["kappa"] == pytest.approx(
                -math.log(slope) * _PERIODS_PER_YEAR, rel=1e-9
            )
            level = intercept / (1.0 - slope)
            sigma_eq = math.sqrt(noise_variance / (1.0 - slope**2))
            assert row["m"] == pytest.approx(level, rel=1e-9)
            assert row["sigma_eq"] == pytest.approx(sigma_eq, rel=1e-9)
            assert row["half_life"] == pytest.approx(
                -math.log(2.0) / math.log(slope), rel=1e-9
            )
            assert bool(row["mean_reverting"]) is True
            assert model.s_scores[asset] == pytest.approx(
                (path[-1] - level) / sigma_eq, rel=1e-8, abs=1e-10
            )

    def test_periods_per_year_only_rescales_kappa(
        self, small_market: pd.DataFrame
    ) -> None:
        """Kappa is per year, so 52 periods a year scales it by 52 / 252."""
        daily = fit_pca_residual_model(small_market, n_components=2)
        weekly = fit_pca_residual_model(
            small_market, 2, PcaResidualConfig(periods_per_year=52)
        )
        np.testing.assert_allclose(
            weekly.ou_parameters["kappa"], daily.ou_parameters["kappa"] * 52 / 252
        )
        pd.testing.assert_series_equal(weekly.s_scores, daily.s_scores)
        np.testing.assert_allclose(
            weekly.ou_parameters["half_life"], daily.ou_parameters["half_life"]
        )

    def test_planted_reversion_speed_is_recovered_up_to_the_known_bias(self) -> None:
        """Mean kappa over 20 assets with b = 0.8 lies within 25% of the truth."""
        returns = _market(seed=1, size=2000, coefficients=[0.8] * 20)
        model = fit_pca_residual_model(returns, n_components=1)
        planted = -math.log(0.8) * _PERIODS_PER_YEAR
        # Regressing on a constant forces the residuals to sum to zero, so the
        # running sum is pulled back to zero at the end of the window. That raises
        # the estimated AR(1) coefficient above the planted 0.80 and so lowers kappa
        # below the planted 56.2: seeds 0 to 9 give means between 50.0 and 55.5.
        assert model.ou_parameters["mean_reverting"].all()
        assert model.ou_parameters["kappa"].mean() == pytest.approx(planted, rel=0.25)

    def test_faster_planted_reversion_gives_a_larger_kappa(self) -> None:
        """Assets planted at b = 0.6 revert faster than assets planted at b = 0.9."""
        returns = _market(seed=2, size=2000, coefficients=[0.6] * 10 + [0.9] * 10)
        kappa = fit_pca_residual_model(returns, n_components=1).ou_parameters["kappa"]
        assert kappa.iloc[:10].min() > kappa.iloc[10:].max()

    def test_a_random_walk_residual_is_slow_and_flagged_by_the_speed_screen(
        self,
    ) -> None:
        """A unit-root asset reverts at kappa near 2 to 3, far below the paper's 8.4."""
        coefficients: list[float | None] = [None] + [0.8] * 15
        returns = _market(seed=4, size=2000, coefficients=coefficients)
        unscreened = fit_pca_residual_model(returns, n_components=1)
        screened = fit_pca_residual_model(
            returns, 1, PcaResidualConfig(min_kappa=_PAPER_MIN_KAPPA)
        )
        walker = "A00"
        others = unscreened.ou_parameters.drop(index=walker)
        # Seed 4 gives 2.96; across seeds 0 to 29 the largest value is also 2.96.
        # The planted OU assets sit near 45.
        assert 0.0 < unscreened.ou_parameters.loc[walker, "kappa"] < _PAPER_MIN_KAPPA
        assert (
            unscreened.ou_parameters.loc[walker, "kappa"]
            < 0.2 * others["kappa"].median()
        )
        assert unscreened.ou_parameters.loc[walker, "half_life"] > 5 * (
            others["half_life"].median()
        )
        # The cumulative residual of a random walk has a coefficient just below one,
        # so the bare 0 < b < 1 rule keeps it and only the speed screen drops it.
        assert bool(unscreened.ou_parameters.loc[walker, "mean_reverting"]) is True
        assert bool(screened.ou_parameters.loc[walker, "mean_reverting"]) is False
        assert math.isnan(screened.s_scores[walker])
        assert screened.ou_parameters.drop(index=walker)["mean_reverting"].all()
        assert screened.s_scores.drop(index=walker).notna().all()

    def test_an_anti_persistent_residual_is_not_mean_reverting(self) -> None:
        """Seed 0 gives an AR(1) coefficient near -0.5 and no OU parameters."""
        returns = _market(
            seed=0,
            size=500,
            coefficients=[-0.9] + [0.8] * 15,
            scales=[20.0] + [1.0] * 15,
        )
        model = fit_pca_residual_model(returns, n_components=1)
        flagged = model.ou_parameters.loc["A00"]
        assert bool(flagged["mean_reverting"]) is False
        assert flagged[["kappa", "m", "sigma_eq", "half_life"]].isna().all()
        assert math.isnan(model.s_scores["A00"])
        assert model.ou_parameters.drop(index="A00")["mean_reverting"].all()


class TestOuSummary:
    """The AR(1) to OU conversion on cumulative paths built by hand.

    A cumulative residual from the factor regression ends at zero, which keeps its
    AR(1) coefficient below one, so these paths exercise the cases the public fit
    can hardly reach.
    """

    @staticmethod
    def _summary(paths: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
        """Return the OU table and s-scores for ``paths`` with default settings."""
        columns = pd.Index([f"p{i}" for i in range(paths.shape[1])])
        return _ou_summary(paths, columns, PcaResidualConfig())

    def test_coefficients_outside_zero_to_one_are_flagged(self) -> None:
        """Growth (b = 1.05), alternation (b = -0.9) and a flat path are rejected."""
        t = np.arange(60)
        growth = 1.05**t
        alternating = (-0.9) ** t
        flat = np.ones(60)
        steady = 0.8**t + 0.01 * np.random.default_rng(0).normal(size=60)
        table, scores = self._summary(
            np.column_stack([growth, alternating, flat, steady])
        )
        assert table["mean_reverting"].tolist() == [False, False, False, True]
        rejected = table.iloc[:3]
        assert rejected[["kappa", "m", "sigma_eq", "half_life"]].isna().all().all()
        assert scores.iloc[:3].isna().all()
        assert np.isfinite(scores.iloc[3])

    def test_an_exact_geometric_decay_has_no_noise_and_is_not_scored(self) -> None:
        """A noiseless path has zero AR(1) residual variance, so no s-score."""
        table, scores = self._summary((0.8 ** np.arange(60))[:, None])
        assert not table["mean_reverting"].iloc[0]
        assert math.isnan(scores.iloc[0])

    def test_oscillating_path_alone_gives_a_negative_slope(self) -> None:
        """The lower bound is checked on its own: slope -0.5 is below zero."""
        rng = np.random.default_rng(1)
        shocks = rng.normal(size=400)
        path = np.zeros(400)
        for t in range(1, 400):
            path[t] = -0.5 * path[t - 1] + shocks[t]
        table, _ = self._summary(path[:, None])
        assert not table["mean_reverting"].iloc[0]
        assert math.isnan(table["kappa"].iloc[0])


class TestPcaResidualValidation:
    """Every rejected setting or input of ``fit_pca_residual_model``."""

    @pytest.mark.parametrize("components", [0, -1, 12, 13])
    def test_n_components_must_leave_at_least_one_residual_dimension(
        self, small_market: pd.DataFrame, components: int
    ) -> None:
        """``1 <= n_components < n_assets``; all of them would fit every asset."""
        with pytest.raises(ConfigurationError, match="n_components"):
            fit_pca_residual_model(small_market, n_components=components)

    def test_non_dataframe_input_raises(self) -> None:
        """Only a DataFrame carries the asset and date labels."""
        with pytest.raises(DataError, match="DataFrame"):
            fit_pca_residual_model(np.ones((100, 5)))  # type: ignore[arg-type]

    def test_missing_values_raise(self, small_market: pd.DataFrame) -> None:
        """A NaN return cannot be standardised or regressed."""
        broken = small_market.copy()
        broken.iloc[5, 3] = np.nan
        with pytest.raises(DataError, match="finite"):
            fit_pca_residual_model(broken, n_components=2)

    def test_too_few_observations_raise(self, small_market: pd.DataFrame) -> None:
        """Fewer rows than ``min_observations`` is rejected."""
        with pytest.raises(DataError, match="at least 60"):
            fit_pca_residual_model(small_market.iloc[:59], n_components=2)

    def test_a_window_no_longer_than_the_factors_raises(self) -> None:
        """With ``T <= n_components + 1`` the factors would fit every asset exactly."""
        returns = _market(seed=7, size=11, coefficients=[0.8] * 12)
        with pytest.raises(DataError, match="observations"):
            fit_pca_residual_model(returns, 10, PcaResidualConfig(min_observations=10))

    def test_non_numeric_returns_raise(self, small_market: pd.DataFrame) -> None:
        """Text in the returns is a data error, not a raw ValueError."""
        text = small_market.astype(object)
        text.iloc[3, 2] = "n/a"
        with pytest.raises(DataError, match="returns must contain only numbers"):
            fit_pca_residual_model(text, n_components=2)

    def test_a_constant_asset_raises(self, small_market: pd.DataFrame) -> None:
        """An asset with no variance cannot be standardised."""
        flat = small_market.copy()
        flat["A04"] = 0.001
        with pytest.raises(DataError, match="A04"):
            fit_pca_residual_model(flat, n_components=2)

    def test_a_single_asset_raises(self) -> None:
        """There is nothing to take a common factor out of."""
        returns = _market(seed=7, size=100, coefficients=[0.8])
        with pytest.raises(DataError, match="at least 2 assets"):
            fit_pca_residual_model(returns, n_components=1)

    @pytest.mark.parametrize("periods", [0, -252, *_NOT_INTEGERS])
    def test_periods_per_year_must_be_a_positive_integer(self, periods: object) -> None:
        """Kappa is annualised; NaN would turn every kappa into NaN without a word."""
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            PcaResidualConfig(periods_per_year=periods)  # type: ignore[arg-type]

    @pytest.mark.parametrize("floor", [9, *_NOT_INTEGERS, None])
    def test_min_observations_must_be_an_integer_above_the_floor(
        self, floor: object
    ) -> None:
        """The AR(1) fit of a short window is not an estimate, and NaN is no floor."""
        with pytest.raises(ConfigurationError, match="min_observations"):
            PcaResidualConfig(min_observations=floor)  # type: ignore[arg-type]

    @pytest.mark.parametrize("kappa", [-1.0, math.nan, math.inf, True, "8.4", None])
    def test_min_kappa_must_be_finite_and_non_negative(self, kappa: object) -> None:
        """A speed floor below zero or not a number is meaningless."""
        with pytest.raises(ConfigurationError, match="min_kappa"):
            PcaResidualConfig(min_kappa=kappa)  # type: ignore[arg-type]

    def test_numpy_integers_and_integer_kappa_are_accepted(self) -> None:
        """Settings read from arrays behave like Python numbers."""
        config = PcaResidualConfig(
            periods_per_year=np.int64(52), min_observations=np.int32(60), min_kappa=8
        )
        assert config.periods_per_year == 52

    @pytest.mark.parametrize("components", [2.5, 3.0, math.nan, math.inf, True, "2"])
    def test_n_components_must_be_an_integer(
        self, small_market: pd.DataFrame, components: object
    ) -> None:
        """A fractional or NaN count would fail deep inside a slice."""
        with pytest.raises(ConfigurationError, match="n_components"):
            fit_pca_residual_model(small_market, n_components=components)  # type: ignore[arg-type]

    def test_a_numpy_integer_n_components_is_accepted(
        self, small_market: pd.DataFrame
    ) -> None:
        """A count read from an array gives the same model as a Python int."""
        numpy_model = fit_pca_residual_model(small_market, n_components=np.int64(2))
        plain_model = fit_pca_residual_model(small_market, n_components=2)
        pd.testing.assert_frame_equal(numpy_model.betas, plain_model.betas)

    def test_config_is_immutable(self) -> None:
        """The configuration is a frozen dataclass."""
        config = PcaResidualConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.min_kappa = 1.0  # type: ignore[misc]


def _scores(values: list[float]) -> pd.Series:
    """Return s-scores for assets ``a, b, c, ...``."""
    return pd.Series(values, index=[chr(ord("a") + i) for i in range(len(values))])


class TestPcaSignals:
    """The Avellaneda and Lee (2010) thresholds applied to one date."""

    def test_opening_rules_without_previous_positions(self) -> None:
        """Buy below -1.25, sell above +1.25, nothing in between."""
        scores = _scores([-1.5, -1.0, -0.4, 0.0, 0.8, 1.3, math.nan])
        positions = pca_signals(scores)
        assert positions.tolist() == [1, 0, 0, 0, 0, -1, 0]
        assert positions.index.equals(scores.index)
        assert positions.dtype.kind == "i"

    def test_closing_and_holding_rules_with_previous_positions(self) -> None:
        """Close a long above -0.5 and a short below +0.75; otherwise hold."""
        scores = _scores([-1.5, -1.0, -0.4, 0.0, 0.8, 1.3, math.nan])
        previous = pd.Series([0, 1, 1, -1, -1, 0, 1], index=scores.index)
        # By hand: a opens a long (-1.5 < -1.25); b holds its long (-1.0 is not
        # above -0.5); c closes its long (-0.4 > -0.5); d closes its short (0.0 <
        # 0.75); e holds its short (0.8 is not below 0.75); f opens a short
        # (1.3 > 1.25); g has no score so it is flat.
        assert pca_signals(scores, previous_positions=previous).tolist() == [
            1,
            1,
            0,
            0,
            -1,
            -1,
            0,
        ]

    def test_thresholds_are_strict(self) -> None:
        """A score exactly on a threshold neither opens nor closes."""
        scores = _scores([-1.25, 1.25, -0.5, 0.75])
        previous = pd.Series([0, 0, 1, -1], index=scores.index)
        assert pca_signals(scores, previous_positions=previous).tolist() == [
            0,
            0,
            1,
            -1,
        ]

    def test_a_closed_position_is_not_reopened_on_the_same_date(self) -> None:
        """A long that is closed because the score jumped above +1.25 stays flat."""
        scores = _scores([1.5])
        previous = pd.Series([1], index=scores.index)
        assert pca_signals(scores, previous_positions=previous).tolist() == [0]

    def test_previous_positions_may_cover_other_assets_or_miss_some(self) -> None:
        """Assets absent from ``previous_positions`` start flat; extras are ignored."""
        scores = _scores([-1.0, -1.0])
        previous = pd.Series({"b": 1, "zzz": -1})
        assert pca_signals(scores, previous_positions=previous).tolist() == [0, 1]

    def test_custom_thresholds(self) -> None:
        """Tighter entry levels move the opening points."""
        scores = _scores([-1.0, 1.0, 0.2])
        positions = pca_signals(
            scores, buy_below=-0.9, sell_above=0.9, exit_long=-0.1, exit_short=0.1
        )
        assert positions.tolist() == [1, -1, 0]

    def test_an_empty_vector_gives_an_empty_result(self) -> None:
        """No assets, no positions."""
        assert pca_signals(pd.Series(dtype=float)).empty

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"buy_below": -0.4},
            {"buy_below": -0.5},
            {"sell_above": 0.7},
            {"sell_above": 0.75},
            {"buy_below": 2.0, "exit_long": 3.0, "exit_short": 0.0, "sell_above": 1.0},
        ],
    )
    def test_inconsistent_thresholds_raise(self, kwargs: dict[str, float]) -> None:
        """Each exit must lie between the two entry levels."""
        with pytest.raises(ConfigurationError, match="below"):
            pca_signals(_scores([0.0]), **kwargs)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "1", None])
    @pytest.mark.parametrize(
        "name", ["buy_below", "sell_above", "exit_long", "exit_short"]
    )
    def test_thresholds_must_be_finite_numbers(self, name: str, bad: object) -> None:
        """NaN would make every comparison false and silently trade nothing."""
        with pytest.raises(ConfigurationError, match=f"{name} must be a finite"):
            pca_signals(_scores([0.0]), **{name: bad})  # type: ignore[arg-type]

    def test_integer_thresholds_are_accepted(self) -> None:
        """Whole-number thresholds, Python or numpy, work like floats."""
        positions = pca_signals(
            _scores([-2.0, 2.0]),
            buy_below=-1,
            sell_above=np.int64(1),
            exit_long=0,
            exit_short=0,
        )
        assert positions.tolist() == [1, -1]

    def test_previous_positions_outside_minus_one_to_one_raise(self) -> None:
        """Only -1, 0 and +1 are positions."""
        scores = _scores([0.0, 0.0])
        with pytest.raises(DataError, match="previous_positions"):
            pca_signals(
                scores, previous_positions=pd.Series([2, 0], index=scores.index)
            )

    def test_non_series_previous_positions_raise(self) -> None:
        """Positions are matched to assets by label."""
        with pytest.raises(DataError, match="previous_positions"):
            pca_signals(_scores([0.0]), previous_positions={"a": 1})  # type: ignore[arg-type]

    def test_non_numeric_scores_raise(self) -> None:
        """Text among the scores is a data error, not a raw ValueError."""
        scores = pd.Series(["high", "low"], index=["a", "b"])
        with pytest.raises(DataError, match="s_scores must contain only numbers"):
            pca_signals(scores)

    def test_non_series_input_raises(self) -> None:
        """Scores need labels to line up with the previous positions."""
        with pytest.raises(DataError, match="Series"):
            pca_signals(np.array([0.0]))  # type: ignore[arg-type]
