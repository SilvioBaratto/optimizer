"""Tests for the geometric Brownian motion simulator and its calibration.

The golden numbers are the closed-form moments of geometric Brownian motion,
E[S_T] = S0 exp(mu T) and E[ln S_T] = ln S0 + (mu - sigma^2 / 2) T, and values
computed with ``math`` and ``statistics`` on hand-made series. None of them comes
from the code under test.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.simulation import (
    GbmConfig,
    GbmFit,
    calibrate_gbm,
    simulate_correlated_gbm,
    simulate_gbm,
)

_S0 = 100.0
_MU = 0.08
_SIGMA = 0.2
_STEPS = 10
_DT = 0.1
_HORIZON = _STEPS * _DT
_SE_BANDS = 4.0
_SEED = 20240607

_CONFIG = GbmConfig(
    initial_price=_S0,
    mu=_MU,
    sigma=_SIGMA,
    n_steps=_STEPS,
    n_paths=100_000,
    dt=_DT,
    random_state=_SEED,
)


def _centered_log_increments(
    paths: npt.NDArray[np.float64], mu: float, sigma: float, dt: float
) -> npt.NDArray[np.float64]:
    """Return the log increments with the deterministic drift removed."""
    return np.diff(np.log(paths), axis=1) - (mu - sigma**2 / 2.0) * dt


class TestSimulateGbm:
    """``simulate_gbm`` paths of one asset."""

    def test_shape_and_initial_column(self) -> None:
        """Column zero is the initial price and there is one column per step."""
        config = dataclasses.replace(_CONFIG, n_paths=7, n_steps=5)
        paths = simulate_gbm(config)
        assert paths.shape == (7, 6)
        assert (paths[:, 0] == _S0).all()
        assert (paths > 0.0).all()

    @pytest.mark.parametrize("antithetic", [True, False])
    def test_terminal_price_mean_matches_the_closed_form(
        self, antithetic: bool
    ) -> None:
        """E[S_T] = S0 exp(mu T), within four standard errors."""
        config = dataclasses.replace(_CONFIG, antithetic=antithetic)
        terminal = simulate_gbm(config)[:, -1]
        expected = _S0 * math.exp(_MU * _HORIZON)
        variance = (
            _S0**2
            * math.exp(2.0 * _MU * _HORIZON)
            * (math.exp(_SIGMA**2 * _HORIZON) - 1.0)
        )
        standard_error = math.sqrt(variance / config.n_paths)
        assert abs(terminal.mean() - expected) < _SE_BANDS * standard_error

    @pytest.mark.parametrize("antithetic", [True, False])
    def test_terminal_log_price_mean_matches_the_closed_form(
        self, antithetic: bool
    ) -> None:
        """E[ln S_T] = ln S0 + (mu - sigma^2 / 2) T, within four standard errors."""
        config = dataclasses.replace(_CONFIG, antithetic=antithetic)
        terminal = simulate_gbm(config)[:, -1]
        expected = math.log(_S0) + (_MU - _SIGMA**2 / 2.0) * _HORIZON
        standard_error = _SIGMA * math.sqrt(_HORIZON / config.n_paths)
        assert abs(np.log(terminal).mean() - expected) < _SE_BANDS * standard_error

    def test_terminal_log_price_variance_matches_the_closed_form(self) -> None:
        """Var(ln S_T) = sigma^2 T for independent paths, within four SE."""
        config = dataclasses.replace(_CONFIG, antithetic=False)
        sample = np.log(simulate_gbm(config)[:, -1])
        expected = _SIGMA**2 * _HORIZON
        standard_error = expected * math.sqrt(2.0 / (config.n_paths - 1))
        assert abs(sample.var(ddof=1) - expected) < _SE_BANDS * standard_error

    def test_zero_volatility_gives_the_deterministic_path(self) -> None:
        """With sigma = 0 every path is S0 exp(mu t)."""
        config = dataclasses.replace(_CONFIG, sigma=0.0, n_paths=3, antithetic=False)
        paths = simulate_gbm(config)
        expected = [_S0 * math.exp(_MU * step * _DT) for step in range(_STEPS + 1)]
        np.testing.assert_allclose(paths, np.tile(expected, (3, 1)), rtol=1e-12)

    def test_exact_discretisation_of_the_shocks(self) -> None:
        """S_{t+dt} = S_t exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z) step by step."""
        config = dataclasses.replace(_CONFIG, n_paths=4, n_steps=3, antithetic=False)
        shocks = np.random.default_rng(_SEED).standard_normal((4, 3))
        paths = simulate_gbm(config)
        for path in range(4):
            price = _S0
            for step in range(3):
                drift = (_MU - _SIGMA**2 / 2.0) * _DT
                price *= math.exp(drift + _SIGMA * math.sqrt(_DT) * shocks[path, step])
                assert paths[path, step + 1] == pytest.approx(price, rel=1e-12)

    def test_antithetic_pairs_mirror_each_other_in_log_space(self) -> None:
        """The second half of the paths uses the negated shocks."""
        config = dataclasses.replace(_CONFIG, n_paths=10, n_steps=5, antithetic=True)
        centered = _centered_log_increments(simulate_gbm(config), _MU, _SIGMA, _DT)
        np.testing.assert_allclose(centered[:5], -centered[5:], atol=1e-12)

    def test_odd_path_count_leaves_one_path_without_a_mirror(self) -> None:
        """Seven paths are three mirrored pairs and one unpaired path."""
        config = dataclasses.replace(_CONFIG, n_paths=7, n_steps=5, antithetic=True)
        paths = simulate_gbm(config)
        assert paths.shape == (7, 6)
        centered = _centered_log_increments(paths, _MU, _SIGMA, _DT)
        np.testing.assert_allclose(centered[4:], -centered[:3], atol=1e-12)

    def test_a_single_antithetic_path_is_allowed(self) -> None:
        """One path has no partner and is simply a plain draw."""
        config = dataclasses.replace(_CONFIG, n_paths=1, antithetic=True)
        assert simulate_gbm(config).shape == (1, _STEPS + 1)

    def test_same_seed_reproduces_and_another_seed_differs(self) -> None:
        """The seed alone decides the draw."""
        config = dataclasses.replace(_CONFIG, n_paths=20)
        first = simulate_gbm(config)
        again = simulate_gbm(config)
        other = simulate_gbm(dataclasses.replace(config, random_state=_SEED + 1))
        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, other)

    def test_no_seed_still_simulates(self) -> None:
        """``random_state=None`` draws fresh entropy."""
        config = dataclasses.replace(_CONFIG, n_paths=5, random_state=None)
        assert simulate_gbm(config).shape == (5, _STEPS + 1)

    def test_overflowing_prices_raise(self) -> None:
        """Paths that leave the float range are reported, not returned as infinity."""
        config = GbmConfig(
            initial_price=_S0, mu=1000.0, sigma=0.0, n_steps=10, n_paths=2, dt=1.0
        )
        with pytest.raises(ConfigurationError, match="overflow"):
            simulate_gbm(config)


_VOLS = np.array([0.2, 0.3])
_RHO = 0.6
_COV = np.array(
    [
        [_VOLS[0] ** 2, _RHO * _VOLS[0] * _VOLS[1]],
        [_RHO * _VOLS[0] * _VOLS[1], _VOLS[1] ** 2],
    ]
)
_MUS = np.array([0.05, 0.10])
_PRICES = np.array([100.0, 50.0])


def _correlated(
    covariance: npt.ArrayLike = _COV,
    *,
    n_paths: int = 20_000,
    n_steps: int = 10,
    seed: int | None = _SEED,
) -> npt.NDArray[np.float64]:
    """Simulate the two-asset reference market."""
    return simulate_correlated_gbm(
        _PRICES, _MUS, covariance, n_steps, n_paths, dt=_DT, random_state=seed
    )


class TestSimulateCorrelatedGbm:
    """``simulate_correlated_gbm`` paths of several assets."""

    def test_shape_and_initial_prices(self) -> None:
        """The axes are paths, time (with the start) and assets."""
        paths = _correlated(n_paths=6, n_steps=4)
        assert paths.shape == (6, 5, 2)
        assert (paths[:, 0, :] == _PRICES).all()

    def test_increments_match_the_target_moments(self) -> None:
        """Drift, volatility and correlation of the log increments hit the target."""
        increments = np.diff(np.log(_correlated()), axis=1).reshape(-1, 2)
        count = increments.shape[0]
        for asset in range(2):
            sd = _VOLS[asset] * math.sqrt(_DT)
            drift = (_MUS[asset] - _VOLS[asset] ** 2 / 2.0) * _DT
            column = increments[:, asset]
            assert abs(column.mean() - drift) < _SE_BANDS * sd / math.sqrt(count)
            assert abs(column.std(ddof=1) - sd) < _SE_BANDS * sd / math.sqrt(2 * count)
        correlation = np.corrcoef(increments.T)[0, 1]
        correlation_se = (1.0 - _RHO**2) / math.sqrt(count)
        assert abs(correlation - _RHO) < _SE_BANDS * correlation_se

    def test_perfectly_correlated_assets_move_together(self) -> None:
        """A singular covariance matrix is simulated through its eigen square root."""
        covariance = np.array([[0.04, 0.04], [0.04, 0.04]])
        paths = simulate_correlated_gbm(
            [100.0, 100.0], [0.05, 0.05], covariance, 5, 50, dt=_DT, random_state=1
        )
        np.testing.assert_allclose(paths[:, :, 0], paths[:, :, 1], rtol=1e-9)

    def test_a_rank_deficient_market_keeps_its_linear_structure(self) -> None:
        """The third asset's shock is the sum of the first two shocks' loadings."""
        loadings = np.array([[0.2, 0.0], [0.1, 0.15], [0.3, 0.15]])
        covariance = loadings @ loadings.T
        mus = np.array([0.03, 0.04, 0.05])
        paths = simulate_correlated_gbm(
            [10.0, 20.0, 30.0], mus, covariance, 6, 40, dt=_DT, random_state=2
        )
        drift = (mus - np.diag(covariance) / 2.0) * _DT
        centered = np.diff(np.log(paths), axis=1) - drift
        np.testing.assert_allclose(
            centered[:, :, 2], centered[:, :, 0] + centered[:, :, 1], atol=1e-9
        )

    def test_zero_covariance_gives_deterministic_paths(self) -> None:
        """No risk leaves only the drift: S0 exp(mu t)."""
        paths = simulate_correlated_gbm(
            _PRICES, _MUS, np.zeros((2, 2)), 4, 3, dt=_DT, random_state=3
        )
        for asset in range(2):
            expected = [
                _PRICES[asset] * math.exp(_MUS[asset] * step * _DT) for step in range(5)
            ]
            np.testing.assert_allclose(
                paths[:, :, asset], np.tile(expected, (3, 1)), rtol=1e-12
            )

    def test_same_seed_reproduces_and_another_seed_differs(self) -> None:
        """The seed alone decides the draw."""
        first = _correlated(n_paths=10)
        again = _correlated(n_paths=10)
        other = _correlated(n_paths=10, seed=_SEED + 1)
        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, other)

    def test_not_positive_semi_definite_raises(self) -> None:
        """A covariance with a negative eigenvalue (3 and -1 here) is rejected."""
        with pytest.raises(DataError, match="positive semi-definite"):
            _correlated([[1.0, 2.0], [2.0, 1.0]])

    def test_not_symmetric_raises(self) -> None:
        """An asymmetric matrix is not a covariance matrix."""
        with pytest.raises(DataError, match="symmetric"):
            _correlated([[0.04, 0.01], [0.02, 0.09]])

    @pytest.mark.parametrize(
        "covariance", [np.eye(3), np.ones((2, 3)), np.array([0.04, 0.09])]
    )
    def test_covariance_shape_must_match_the_assets(
        self, covariance: npt.NDArray[np.float64]
    ) -> None:
        """The covariance is square with one row per asset."""
        with pytest.raises(DataError, match="covariance"):
            _correlated(covariance)

    def test_non_finite_covariance_raises(self) -> None:
        """NaN is not a covariance."""
        with pytest.raises(DataError, match="finite"):
            _correlated([[0.04, math.nan], [math.nan, 0.09]])

    @pytest.mark.parametrize(
        ("prices", "mus"),
        [
            ([100.0, 0.0], [0.05, 0.1]),
            ([100.0, math.nan], [0.05, 0.1]),
            ([[100.0, 50.0]], [0.05, 0.1]),
            ([100.0, 50.0], [0.05]),
            ([100.0, 50.0], [0.05, math.inf]),
            ([100.0, "x"], [0.05, 0.1]),
            ([[100.0], [50.0, 60.0]], [0.05, 0.1]),
            ([100.0, 50.0], [0.05, "x"]),
        ],
    )
    def test_invalid_prices_or_drifts_raise(
        self, prices: npt.ArrayLike, mus: npt.ArrayLike
    ) -> None:
        """Start prices and drifts are settings: positive, finite, one per asset."""
        with pytest.raises(ConfigurationError):
            simulate_correlated_gbm(prices, mus, _COV, 3, 3)

    @pytest.mark.parametrize(
        "covariance", [[[0.04, 0.0], [0.0]], [["a", "b"], ["c", "d"]]]
    )
    def test_ragged_or_non_numeric_covariance_raises(
        self, covariance: npt.ArrayLike
    ) -> None:
        """A covariance numpy cannot read raises the documented data error."""
        with pytest.raises(DataError, match="covariance"):
            _correlated(covariance)

    @pytest.mark.parametrize(
        ("n_steps", "n_paths", "dt"),
        [
            (0, 3, _DT),
            (3, 0, _DT),
            (3, 3, 0.0),
            (2.5, 3, _DT),
            (3, True, _DT),
            (3, math.nan, _DT),
        ],
    )
    def test_invalid_horizon_settings_raise(
        self, n_steps: Any, n_paths: Any, dt: float
    ) -> None:
        """Steps, paths and the time step are validated like ``GbmConfig``."""
        with pytest.raises(ConfigurationError):
            simulate_correlated_gbm(_PRICES, _MUS, _COV, n_steps, n_paths, dt=dt)

    @pytest.mark.parametrize("seed", [-1, 1.5, True])
    def test_invalid_seed_raises(self, seed: Any) -> None:
        """A seed is a non-negative integer or ``None``."""
        with pytest.raises(ConfigurationError, match="random_state"):
            _correlated(seed=seed)

    def test_overflowing_prices_raise(self) -> None:
        """Paths that leave the float range are reported."""
        with pytest.raises(ConfigurationError, match="overflow"):
            simulate_correlated_gbm(
                _PRICES, [1000.0, 1000.0], np.zeros((2, 2)), 10, 2, dt=1.0
            )


class TestCalibrateGbm:
    """``calibrate_gbm``: moment estimates from the log returns."""

    # Prices 100, 102, 101, 105: log returns 0.0198026, -0.0098523, 0.0388398 have
    # mean 0.01626339 and sample variance 6.021255e-4 (statistics.variance).
    _SERIES = (100.0, 102.0, 101.0, 105.0)

    def test_matches_the_closed_form_on_a_hand_made_series(self) -> None:
        """sigma^2 = var / dt and mu = mean / dt + sigma^2 / 2."""
        fit = calibrate_gbm(self._SERIES)
        assert isinstance(fit, GbmFit)
        assert fit.sigma == pytest.approx(0.38953258535319674)
        assert fit.mu == pytest.approx(4.174241607758269)

    def test_time_step_rescales_the_estimates(self) -> None:
        """One unit of time per observation gives the per-period figures."""
        fit = calibrate_gbm(self._SERIES, dt=1.0)
        assert fit.sigma == pytest.approx(0.02453824639049047)
        assert fit.mu == pytest.approx(0.016564450824437575)

    def test_accepts_a_pandas_series(self) -> None:
        """A price series with a date index is a valid input."""
        series = pd.Series(self._SERIES, index=pd.date_range("2024-01-01", periods=4))
        assert calibrate_gbm(series) == calibrate_gbm(self._SERIES)

    def test_recovers_the_parameters_of_a_long_simulated_path(self) -> None:
        """The estimates land within four standard errors of the true values."""
        steps = 100_000
        dt = 1.0 / 252.0
        config = GbmConfig(
            initial_price=_S0,
            mu=_MU,
            sigma=_SIGMA,
            n_steps=steps,
            n_paths=1,
            dt=dt,
            antithetic=False,
            random_state=_SEED,
        )
        fit = calibrate_gbm(simulate_gbm(config)[0], dt=dt)
        sigma_se = _SIGMA / math.sqrt(2.0 * (steps - 1))
        mu_se = math.sqrt(
            _SIGMA**2 / (steps * dt) + (_SIGMA**2 / 2.0) ** 2 * 2.0 / (steps - 1)
        )
        assert abs(fit.sigma - _SIGMA) < _SE_BANDS * sigma_se
        assert abs(fit.mu - _MU) < _SE_BANDS * mu_se

    @pytest.mark.parametrize(
        "prices",
        [
            [100.0, 101.0],
            [100.0, 0.0, 101.0],
            [100.0, -1.0, 101.0],
            [100.0, math.nan, 101.0],
            [[100.0, 101.0, 102.0]],
            [100.0, "x", 101.0],
            [[100.0], [101.0, 102.0], [103.0]],
        ],
    )
    def test_invalid_prices_raise(self, prices: npt.ArrayLike) -> None:
        """Three positive, finite prices in one dimension are the minimum."""
        with pytest.raises(DataError):
            calibrate_gbm(prices)

    @pytest.mark.parametrize("dt", [0.0, -1.0, math.nan])
    def test_invalid_time_step_raises(self, dt: float) -> None:
        """The time step is positive and finite."""
        with pytest.raises(ConfigurationError, match="dt"):
            calibrate_gbm(self._SERIES, dt=dt)
