"""Tests for the probabilistic and deflated Sharpe ratio.

The golden numbers come from the Bailey and Lopez de Prado (2014) formulas
evaluated independently of the module under test.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.validation import (
    DeflatedSharpeConfig,
    DeflatedSharpeResult,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    run_deflated_sharpe,
    run_deflated_sharpe_from_trials,
)

_DAILY_SHARPE = 0.1
_YEAR = 252


def _noise(seed: int, size: int, mean: float = 0.0, sigma: float = 0.01) -> np.ndarray:
    """Return reproducible normal returns."""
    return np.random.default_rng(seed).normal(mean, sigma, size)


def _sharpe(values: np.ndarray) -> float:
    """Return the per-period Sharpe ratio of a sample."""
    return float(values.mean() / values.std(ddof=1))


def _trial_sharpes(trials: pd.DataFrame) -> pd.Series:
    """Return the per-period Sharpe ratio of every trial column."""
    return pd.Series(
        {name: _sharpe(trials[name].to_numpy()) for name in trials.columns}
    )


class TestProbabilisticSharpeRatio:
    """``probabilistic_sharpe_ratio`` from the moments of a return sample."""

    def test_normal_returns_match_the_closed_form(self) -> None:
        """Normal moments reduce the formula to a plain normal CDF."""
        psr = probabilistic_sharpe_ratio(_DAILY_SHARPE, _YEAR, 0.0, 3.0)
        assert psr == pytest.approx(0.942987, abs=1e-6)

    def test_negative_skew_and_fat_tails_lower_the_confidence(self) -> None:
        """Skew and kurtosis widen the standard error of the Sharpe ratio."""
        psr = probabilistic_sharpe_ratio(_DAILY_SHARPE, _YEAR, -0.5, 5.0)
        assert psr == pytest.approx(0.938074, abs=1e-6)

    def test_a_benchmark_above_zero_lowers_the_confidence(self) -> None:
        """A higher hurdle is harder to beat."""
        psr = probabilistic_sharpe_ratio(
            _DAILY_SHARPE, _YEAR, 0.0, 3.0, benchmark_sharpe=0.05
        )
        assert psr == pytest.approx(0.785288, abs=1e-6)

    def test_a_sharpe_equal_to_the_benchmark_is_a_coin_flip(self) -> None:
        """Beating the benchmark by nothing gives a probability of one half."""
        psr = probabilistic_sharpe_ratio(0.05, 500, 0.3, 4.0, benchmark_sharpe=0.05)
        assert psr == pytest.approx(0.5)

    def test_a_sharpe_below_the_benchmark_is_below_a_coin_flip(self) -> None:
        """A Sharpe ratio under the hurdle is more likely than not to be worse."""
        assert probabilistic_sharpe_ratio(0.01, 500, 0.0, 3.0, 0.05) < 0.5

    def test_more_observations_raise_the_confidence(self) -> None:
        """The same Sharpe ratio is more convincing over a longer record."""
        short = probabilistic_sharpe_ratio(_DAILY_SHARPE, 60, 0.0, 3.0)
        long = probabilistic_sharpe_ratio(_DAILY_SHARPE, 600, 0.0, 3.0)
        assert short < long

    def test_a_non_positive_variance_term_raises(self) -> None:
        """Extreme skew can make the standard error undefined."""
        with pytest.raises(DataError, match="variance"):
            probabilistic_sharpe_ratio(2.0, 252, 3.0, 1.0)

    def test_fewer_than_two_observations_raise(self) -> None:
        """One observation has no standard error."""
        with pytest.raises(DataError, match="observations"):
            probabilistic_sharpe_ratio(_DAILY_SHARPE, 1, 0.0, 3.0)

    @pytest.mark.parametrize("benchmark", [math.nan, math.inf, -math.inf])
    def test_a_benchmark_that_is_not_finite_raises(self, benchmark: float) -> None:
        """A NaN benchmark would otherwise return NaN without a word."""
        with pytest.raises(ConfigurationError, match="benchmark_sharpe"):
            probabilistic_sharpe_ratio(
                _DAILY_SHARPE, _YEAR, 0.0, 3.0, benchmark_sharpe=benchmark
            )


class TestExpectedMaxSharpe:
    """``expected_max_sharpe``: the best Sharpe ratio luck alone produces."""

    @pytest.mark.parametrize(
        ("n_trials", "variance", "expected"),
        [(100, 0.01, 0.25306), (10, 0.04, 0.31492)],
    )
    def test_matches_the_extreme_value_approximation(
        self, n_trials: int, variance: float, expected: float
    ) -> None:
        """The value equals the Bailey and Lopez de Prado approximation."""
        assert expected_max_sharpe(n_trials, variance) == pytest.approx(
            expected, abs=1e-5
        )

    def test_a_single_trial_has_no_selection_effect(self) -> None:
        """Nothing was selected out of one trial."""
        assert expected_max_sharpe(1, 0.04) == 0.0

    def test_identical_trials_have_no_selection_effect(self) -> None:
        """Without dispersion across trials, picking the best gains nothing."""
        assert expected_max_sharpe(50, 0.0) == 0.0

    def test_more_trials_raise_the_bar(self) -> None:
        """The expected maximum grows strictly with the number of trials."""
        bars = [expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
        assert bars == sorted(bars)
        assert len(set(bars)) == len(bars)

    def test_the_bar_scales_with_the_dispersion_of_the_trials(self) -> None:
        """Doubling the standard deviation of the trials doubles the bar."""
        narrow = expected_max_sharpe(50, 0.01)
        wide = expected_max_sharpe(50, 0.04)
        assert wide == pytest.approx(2.0 * narrow)

    @pytest.mark.parametrize("n_trials", [0, -3])
    def test_fewer_than_one_trial_raises(self, n_trials: int) -> None:
        """A search has to have tried at least one strategy."""
        with pytest.raises(ConfigurationError, match="n_trials"):
            expected_max_sharpe(n_trials, 0.01)

    @pytest.mark.parametrize("variance", [-0.01, math.nan, math.inf])
    def test_an_invalid_variance_raises(self, variance: float) -> None:
        """A variance must be a finite non-negative number."""
        with pytest.raises(ConfigurationError, match="variance"):
            expected_max_sharpe(10, variance)


class TestMinimumTrackRecordLength:
    """``minimum_track_record_length``: observations needed for a confidence."""

    def test_matches_the_closed_form(self) -> None:
        """Normal moments give the textbook length."""
        length = minimum_track_record_length(_DAILY_SHARPE, 0.0, 3.0, 0.0, 0.95)
        assert length == pytest.approx(272.9071, rel=1e-6)

    @pytest.mark.parametrize("sharpe", [0.05, 0.0, -0.1])
    def test_a_sharpe_at_or_below_the_benchmark_never_qualifies(
        self, sharpe: float
    ) -> None:
        """No amount of data makes a strategy beat a hurdle it does not clear."""
        length = minimum_track_record_length(sharpe, 0.0, 3.0, 0.05, 0.95)
        assert math.isinf(length)

    def test_a_better_sharpe_needs_a_shorter_record(self) -> None:
        """A larger edge is established faster."""
        weak = minimum_track_record_length(0.05, 0.0, 3.0, 0.0, 0.95)
        strong = minimum_track_record_length(0.15, 0.0, 3.0, 0.0, 0.95)
        assert strong < weak

    def test_a_higher_confidence_needs_a_longer_record(self) -> None:
        """Asking for more certainty costs more data."""
        loose = minimum_track_record_length(0.1, 0.0, 3.0, 0.0, 0.90)
        tight = minimum_track_record_length(0.1, 0.0, 3.0, 0.0, 0.99)
        assert loose < tight

    def test_the_length_reaches_the_requested_confidence(self) -> None:
        """Feeding the length back into the PSR returns the requested level."""
        length = minimum_track_record_length(0.1, -0.4, 4.5, 0.02, 0.95)
        psr = probabilistic_sharpe_ratio(0.1, length, -0.4, 4.5, benchmark_sharpe=0.02)
        assert psr == pytest.approx(0.95, abs=1e-9)

    @pytest.mark.parametrize("benchmark", [math.nan, math.inf])
    def test_a_benchmark_that_is_not_finite_raises(self, benchmark: float) -> None:
        """A non-finite benchmark is rejected, as the config rejects it."""
        with pytest.raises(ConfigurationError, match="benchmark_sharpe"):
            minimum_track_record_length(_DAILY_SHARPE, 0.0, 3.0, benchmark, 0.95)


class TestDeflatedSharpeConfig:
    """``DeflatedSharpeConfig`` defaults and validation."""

    def test_defaults(self) -> None:
        """The defaults are a zero hurdle at 95% confidence on daily data."""
        config = DeflatedSharpeConfig()
        assert config.benchmark_sharpe == 0.0
        assert config.confidence_level == 0.95
        assert config.periods_per_year == _YEAR
        assert config.min_observations == 30

    def test_is_immutable(self) -> None:
        """A shared config cannot be changed behind a caller's back."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            DeflatedSharpeConfig().periods_per_year = 12  # type: ignore[misc]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"confidence_level": 0.0},
            {"confidence_level": 1.0},
            {"periods_per_year": 0},
            {"min_observations": 3},
            {"benchmark_sharpe": math.nan},
        ],
    )
    def test_invalid_values_raise(self, overrides: dict[str, float]) -> None:
        """Out-of-range settings fail when the config is built."""
        with pytest.raises(ConfigurationError):
            DeflatedSharpeConfig(**overrides)  # type: ignore[arg-type]


class TestRunDeflatedSharpe:
    """``run_deflated_sharpe`` on one strategy's return series."""

    @pytest.fixture
    def returns(self) -> np.ndarray:
        """Three years of daily returns with a small positive drift."""
        return _noise(7, 756, mean=0.001)

    def test_reports_the_moments_of_the_sample(self, returns: np.ndarray) -> None:
        """The result carries the sample Sharpe ratio, its annualisation and size."""
        result = run_deflated_sharpe(returns, n_trials=1, trials_sharpe_variance=0.0)
        assert isinstance(result, DeflatedSharpeResult)
        assert result.sharpe == pytest.approx(_sharpe(returns))
        assert result.annualized_sharpe == pytest.approx(
            _sharpe(returns) * math.sqrt(_YEAR)
        )
        assert result.n_observations == 756
        assert result.n_trials == 1
        assert result.selected_trial is None

    def test_a_single_trial_is_not_deflated(self, returns: np.ndarray) -> None:
        """With one trial the deflated ratio equals the probabilistic one."""
        result = run_deflated_sharpe(returns, n_trials=1, trials_sharpe_variance=0.0)
        assert result.expected_max_sharpe == 0.0
        assert result.deflated_sharpe == pytest.approx(result.probabilistic_sharpe)

    def test_many_trials_deflate_the_confidence(self, returns: np.ndarray) -> None:
        """A wide search lowers the confidence in its best result."""
        result = run_deflated_sharpe(
            returns, n_trials=200, trials_sharpe_variance=0.002
        )
        assert result.expected_max_sharpe > 0.0
        assert result.deflated_sharpe < result.probabilistic_sharpe

    def test_the_benchmark_only_moves_the_probabilistic_ratio(
        self, returns: np.ndarray
    ) -> None:
        """The hurdle applies to the PSR; the DSR is measured against the search."""
        plain = run_deflated_sharpe(returns, n_trials=20, trials_sharpe_variance=0.002)
        hurdle = run_deflated_sharpe(
            returns,
            n_trials=20,
            trials_sharpe_variance=0.002,
            config=DeflatedSharpeConfig(benchmark_sharpe=0.05),
        )
        assert hurdle.probabilistic_sharpe < plain.probabilistic_sharpe
        assert hurdle.deflated_sharpe == pytest.approx(plain.deflated_sharpe)

    def test_a_series_a_list_and_an_array_agree(self, returns: np.ndarray) -> None:
        """Every array-like input produces the same result."""
        from_array = run_deflated_sharpe(
            returns, n_trials=5, trials_sharpe_variance=0.001
        )
        from_series = run_deflated_sharpe(
            pd.Series(returns), n_trials=5, trials_sharpe_variance=0.001
        )
        from_list = run_deflated_sharpe(
            returns.tolist(), n_trials=5, trials_sharpe_variance=0.001
        )
        assert from_series == from_array
        assert from_list == from_array

    def test_the_track_record_length_is_reported(self, returns: np.ndarray) -> None:
        """The result carries the length needed for the configured confidence."""
        result = run_deflated_sharpe(returns, n_trials=1, trials_sharpe_variance=0.0)
        assert result.min_track_record_length == pytest.approx(
            minimum_track_record_length(
                result.sharpe, result.skewness, result.kurtosis, 0.0, 0.95
            )
        )

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            (np.array([0.01, np.nan, 0.02] * 20), "finite"),
            (np.array([0.01, np.inf, 0.02] * 20), "finite"),
            (np.full(60, 0.01), "constant"),
            (np.array([0.01, -0.01, 0.02]), "observations"),
            (np.zeros((60, 2)), "one-dimensional"),
            (["a", "b", "c"] * 20, "numeric"),
            ([[0.01, 0.02], [0.01]] * 20, "numeric"),
        ],
    )
    def test_unusable_returns_raise(self, values: npt.ArrayLike, message: str) -> None:
        """Non-finite, constant, short or 2-D input is rejected with a reason."""
        with pytest.raises(DataError, match=message):
            run_deflated_sharpe(values, n_trials=1, trials_sharpe_variance=0.0)


class TestRunDeflatedSharpeFromTrials:
    """``run_deflated_sharpe_from_trials`` on a matrix of trial returns."""

    @staticmethod
    def _noise_trials(n_trials: int, seed: int = 11, size: int = 500) -> pd.DataFrame:
        """Return ``n_trials`` columns of strategies with no edge."""
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            rng.normal(0.0, 0.01, (size, n_trials)),
            columns=[f"trial_{i}" for i in range(n_trials)],
        )

    def test_the_best_of_many_noise_trials_does_not_survive_deflation(self) -> None:
        """Picking the best of 60 coin-flip strategies is not evidence of skill."""
        trials = self._noise_trials(60)
        result = run_deflated_sharpe_from_trials(trials)
        assert result.n_trials == 60
        assert result.selected_trial == _trial_sharpes(trials).idxmax()
        assert result.probabilistic_sharpe > result.deflated_sharpe
        assert result.deflated_sharpe < 0.95

    def test_a_real_edge_survives_deflation(self) -> None:
        """A strategy with a genuine drift stays convincing among 20 noise trials."""
        trials = self._noise_trials(20)
        trials["planted"] = _noise(5, 500, mean=0.004)
        result = run_deflated_sharpe_from_trials(trials)
        assert result.selected_trial == "planted"
        assert result.deflated_sharpe > 0.99

    def test_the_dispersion_comes_from_the_trial_sharpes(self) -> None:
        """The variance fed to the deflation is the spread of the trial Sharpes."""
        trials = self._noise_trials(15)
        result = run_deflated_sharpe_from_trials(trials)
        variance = float(np.var(_trial_sharpes(trials).to_numpy(), ddof=1))
        assert result.trials_sharpe_variance == pytest.approx(variance)
        assert result.expected_max_sharpe == pytest.approx(
            expected_max_sharpe(15, variance)
        )

    def test_one_trial_is_not_enough_to_estimate_a_dispersion(self) -> None:
        """A spread needs at least two trials."""
        with pytest.raises(DataError, match="at least two"):
            run_deflated_sharpe_from_trials(self._noise_trials(1))

    def test_a_missing_value_raises(self) -> None:
        """A NaN in any trial is rejected rather than silently dropped."""
        trials = self._noise_trials(5)
        trials.iloc[3, 2] = np.nan
        with pytest.raises(DataError, match="finite"):
            run_deflated_sharpe_from_trials(trials)

    def test_a_non_numeric_column_raises_a_data_error(self) -> None:
        """A text column is reported as a data problem, not a raw ValueError."""
        trials = self._noise_trials(4)
        trials["label"] = "not a return"
        with pytest.raises(DataError, match="numeric"):
            run_deflated_sharpe_from_trials(trials)
