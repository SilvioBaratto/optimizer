"""Tests for the bootstrap wealth paths and the path summary.

The golden numbers are compounding done with ``math``, the exact moments of a
product of independent draws, and summaries worked out by hand on a few paths.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.simulation import (
    BootstrapConfig,
    PathSummary,
    simulate_bootstrap,
    summarize_paths,
)

_SEED = 11
_SE_BANDS = 4.0


def _config(**overrides: int | None) -> BootstrapConfig:
    """Return a small bootstrap config with the given fields replaced."""
    base = BootstrapConfig(n_paths=8, n_steps=6, random_state=_SEED)
    return dataclasses.replace(base, **overrides)


class TestSimulateBootstrap:
    """``simulate_bootstrap`` wealth paths from resampled returns."""

    def test_constant_returns_compound_deterministically(self) -> None:
        """Every path is initial_value * (1 + r)^t."""
        paths = simulate_bootstrap([0.01] * 5, _config(n_paths=4), initial_value=250.0)
        expected = [250.0 * 1.01**step for step in range(7)]
        assert paths.shape == (4, 7)
        np.testing.assert_allclose(paths, np.tile(expected, (4, 1)), rtol=1e-12)

    def test_default_initial_value_is_one(self) -> None:
        """Wealth starts at one unit unless told otherwise."""
        paths = simulate_bootstrap([0.01, -0.02], _config())
        assert (paths[:, 0] == 1.0).all()

    def test_total_loss_keeps_wealth_at_zero(self) -> None:
        """A return of -100% wipes the wealth out for good."""
        paths = simulate_bootstrap([-1.0], _config(n_paths=2))
        np.testing.assert_array_equal(paths[:, 1:], np.zeros((2, 6)))

    def test_mean_terminal_wealth_matches_the_exact_moment(self) -> None:
        """For i.i.d. draws E[W_T] = W_0 (1 + mean r)^T, within four standard errors."""
        returns = np.array([-0.1, 0.2, 0.05, 0.0])
        steps = 8
        config = _config(n_paths=50_000, n_steps=steps)
        terminal = simulate_bootstrap(returns, config)[:, -1]
        growth = 1.0 + returns
        mean = growth.mean() ** steps
        variance = (growth**2).mean() ** steps - mean**2
        standard_error = math.sqrt(variance / config.n_paths)
        assert abs(terminal.mean() - mean) < _SE_BANDS * standard_error

    def test_block_bootstrap_keeps_consecutive_observations_together(self) -> None:
        """Inside a block the positions in the source series run in order."""
        returns = np.arange(20) / 100.0
        config = _config(n_paths=200, n_steps=12, block_size=4)
        paths = simulate_bootstrap(returns, config)
        positions = np.rint((paths[:, 1:] / paths[:, :-1] - 1.0) * 100.0).astype(int)
        blocks = positions.reshape(200, 3, 4)
        assert (np.diff(blocks, axis=2) == 1).all()
        assert positions.min() >= 0
        assert positions.max() <= 19

    def test_blocks_start_independently(self) -> None:
        """Consecutive blocks are not forced to follow each other."""
        returns = np.arange(20) / 100.0
        paths = simulate_bootstrap(
            returns, _config(n_paths=200, n_steps=8, block_size=4)
        )
        positions = np.rint((paths[:, 1:] / paths[:, :-1] - 1.0) * 100.0).astype(int)
        assert (positions[:, 4] - positions[:, 3] != 1).any()

    def test_last_block_is_cut_to_the_horizon(self) -> None:
        """Ten steps in blocks of four use two whole blocks and half of a third."""
        returns = np.arange(20) / 100.0
        paths = simulate_bootstrap(
            returns, _config(n_paths=50, n_steps=10, block_size=4)
        )
        positions = np.rint((paths[:, 1:] / paths[:, :-1] - 1.0) * 100.0).astype(int)
        assert paths.shape == (50, 11)
        assert (positions[:, 9] - positions[:, 8] == 1).all()

    def test_a_block_as_long_as_the_series_replays_the_series(self) -> None:
        """With one possible start every block is the whole history."""
        returns = np.array([0.1, -0.05, 0.02])
        paths = simulate_bootstrap(returns, _config(n_paths=3, n_steps=6, block_size=3))
        expected = np.cumprod(np.tile(1.0 + returns, 2))
        np.testing.assert_allclose(paths[:, 1:], np.tile(expected, (3, 1)), rtol=1e-12)

    def test_same_seed_reproduces_and_another_seed_differs(self) -> None:
        """The seed alone decides the draw."""
        returns = np.linspace(-0.05, 0.05, 11)
        first = simulate_bootstrap(returns, _config())
        again = simulate_bootstrap(returns, _config())
        other = simulate_bootstrap(returns, _config(random_state=_SEED + 1))
        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, other)

    def test_accepts_a_pandas_series(self) -> None:
        """A return series with a date index is a valid input."""
        series = pd.Series(
            [0.01, 0.02, -0.01], index=pd.date_range("2024-01-01", periods=3)
        )
        assert simulate_bootstrap(series, _config()).shape == (8, 7)

    @pytest.mark.parametrize(
        "returns",
        [
            [],
            [[0.01, 0.02]],
            [0.01, math.nan],
            [0.01, math.inf],
            [0.01, -1.5],
        ],
    )
    def test_invalid_returns_raise(self, returns: list[float]) -> None:
        """Returns are a finite, non-empty vector of simple returns of -1 or more."""
        with pytest.raises(DataError):
            simulate_bootstrap(returns, _config())

    def test_block_longer_than_the_series_raises(self) -> None:
        """A block cannot be cut from a series shorter than itself."""
        with pytest.raises(DataError, match="block_size"):
            simulate_bootstrap([0.01, 0.02], _config(block_size=3))

    @pytest.mark.parametrize("initial_value", [0.0, -1.0, math.nan])
    def test_invalid_initial_value_raises(self, initial_value: float) -> None:
        """The starting wealth is positive and finite."""
        with pytest.raises(ConfigurationError, match="initial_value"):
            simulate_bootstrap([0.01], _config(), initial_value=initial_value)


# Five paths worked out by hand. Terminal over start: 1.5, 0.5, 1.2, 0.5, 1.0.
_PATHS = np.array(
    [
        [1.0, 1.1, 1.2, 1.5],
        [1.0, 0.9, 0.8, 0.5],
        [1.0, 1.2, 0.6, 1.2],
        [2.0, 3.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
)


class TestSummarizePaths:
    """``summarize_paths`` on paths with known statistics."""

    def test_summary_of_hand_made_paths(self) -> None:
        """Mean 4.7 / 5, a 2 in 5 chance of loss and the drawdowns of each path."""
        summary = summarize_paths(_PATHS)
        # Sorted terminals 0.5, 0.5, 1.0, 1.2, 1.5: with linear interpolation the
        # 5% quantile sits at 0.2 of the way from 0.5 to 0.5, the median is the
        # third value and the 95% quantile is 1.2 + 0.8 * (1.5 - 1.2).
        assert isinstance(summary, PathSummary)
        assert summary.mean_terminal == pytest.approx(0.94)
        assert summary.terminal_quantiles[0.05] == pytest.approx(0.5)
        assert summary.terminal_quantiles[0.5] == pytest.approx(1.0)
        assert summary.terminal_quantiles[0.95] == pytest.approx(1.44)
        assert summary.probability_of_loss == pytest.approx(0.4)
        # Worst drops from the running maximum: 0, 0.5, 0.5 (0.6 vs 1.2), 2/3, 0.
        assert summary.mean_max_drawdown == pytest.approx(1.0 / 3.0)
        assert summary.worst_max_drawdown == pytest.approx(2.0 / 3.0)

    def test_quantile_levels_are_configurable(self) -> None:
        """The dictionary holds exactly the requested levels."""
        summary = summarize_paths(_PATHS, quantiles=(0.0, 1.0))
        assert summary.terminal_quantiles == {
            0.0: pytest.approx(0.5),
            1.0: pytest.approx(1.5),
        }

    def test_a_path_that_only_rises_has_no_drawdown_or_loss(self) -> None:
        """Monotone growth: terminal above the start and drawdown zero."""
        summary = summarize_paths(np.array([[1.0, 1.1, 1.2], [3.0, 3.3, 3.6]]))
        assert summary.probability_of_loss == 0.0
        assert summary.mean_max_drawdown == 0.0
        assert summary.worst_max_drawdown == 0.0

    def test_summary_is_frozen(self) -> None:
        """A summary is a value and cannot be edited."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            summarize_paths(_PATHS).mean_terminal = 0.0  # type: ignore[misc]

    def test_consistent_with_a_simulation(self) -> None:
        """A simulated constant-growth bundle summarises to its deterministic values."""
        paths = simulate_bootstrap([0.02], _config(n_paths=5))
        summary = summarize_paths(paths)
        assert summary.mean_terminal == pytest.approx(1.02**6)
        assert summary.probability_of_loss == 0.0

    @pytest.mark.parametrize(
        "paths",
        [
            np.array([1.0, 1.1, 1.2]),
            np.array([[1.0], [1.0]]),
            np.empty((0, 3)),
            np.ones((2, 3, 2)),
            np.array([[1.0, math.nan, 1.0]]),
            np.array([[0.0, 1.0, 1.0]]),
            np.array([[1.0, -0.5, 1.0]]),
        ],
    )
    def test_invalid_paths_raise(self, paths: npt.NDArray[np.float64]) -> None:
        """Paths are a finite non-negative matrix with a positive start."""
        with pytest.raises(DataError):
            summarize_paths(paths)

    @pytest.mark.parametrize("quantiles", [(), (-0.1,), (1.1,), (math.nan,)])
    def test_invalid_quantiles_raise(self, quantiles: tuple[float, ...]) -> None:
        """Quantile levels are a non-empty set inside [0, 1]."""
        with pytest.raises(ConfigurationError, match="quantiles"):
            summarize_paths(_PATHS, quantiles=quantiles)
