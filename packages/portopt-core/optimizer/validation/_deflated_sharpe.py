"""Probabilistic and deflated Sharpe ratio.

A backtested Sharpe ratio is a noisy estimate and, when it is the best of many
trials, a biased one. Bailey and Lopez de Prado (2014), "The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
give two corrections:

* the *probabilistic* Sharpe ratio (PSR) is the probability that the true Sharpe
  ratio beats a benchmark, given the sample length, skewness and kurtosis;
* the *deflated* Sharpe ratio (DSR) is the PSR measured against the best Sharpe
  ratio that luck alone would produce across the strategies that were tried.

Every Sharpe ratio here is per period (not annualised) unless a name says
``annualized``, because the formulas are written in per-period units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from optimizer.exceptions import ConfigurationError, DataError

_EULER_MASCHERONI: Final = 0.5772156649015329
# Fewer than four points give no usable skewness or kurtosis (the kurtosis of two
# points is always 1), and the standard error of the Sharpe ratio needs both.
_MIN_MOMENT_OBSERVATIONS: Final = 4
# The standard error divides by sqrt(n_observations - 1).
_MIN_OBSERVATIONS_FOR_STANDARD_ERROR: Final = 2
# A dispersion needs at least two Sharpe ratios to be measured.
_MIN_TRIALS_FOR_DISPERSION: Final = 2


@dataclass(frozen=True)
class DeflatedSharpeConfig:
    """Immutable configuration for the probabilistic and deflated Sharpe ratio.

    Attributes:
        benchmark_sharpe: Per-period Sharpe ratio the strategy has to beat for the
            probabilistic Sharpe ratio and the track record length. The deflated
            ratio ignores it and uses the expected maximum over the trials instead.
        confidence_level: Probability the minimum track record length is sized for.
        periods_per_year: Periods per year, used only to annualise the reported
            Sharpe ratio.
        min_observations: Fewest observations a return series may have.
    """

    benchmark_sharpe: float = 0.0
    confidence_level: float = 0.95
    periods_per_year: int = 252
    min_observations: int = 30

    def __post_init__(self) -> None:
        """Reject settings the formulas cannot use."""
        _require_finite_benchmark(self.benchmark_sharpe)
        _require_confidence(self.confidence_level)
        if self.periods_per_year < 1:
            msg = f"periods_per_year must be at least 1, got {self.periods_per_year}"
            raise ConfigurationError(msg)
        if self.min_observations < _MIN_MOMENT_OBSERVATIONS:
            msg = (
                f"min_observations must be at least {_MIN_MOMENT_OBSERVATIONS}, "
                f"got {self.min_observations}"
            )
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Outcome of a probabilistic and deflated Sharpe ratio evaluation.

    Attributes:
        sharpe: Per-period Sharpe ratio of the evaluated return series.
        annualized_sharpe: ``sharpe`` scaled by the square root of the periods per
            year.
        skewness: Sample skewness of the returns.
        kurtosis: Sample kurtosis of the returns (3 for a normal distribution).
        n_observations: Number of returns evaluated.
        n_trials: Number of strategies tried before this one was kept.
        trials_sharpe_variance: Variance of the per-period Sharpe ratios across
            the trials.
        expected_max_sharpe: Best per-period Sharpe ratio luck alone is expected to
            produce across ``n_trials`` trials.
        probabilistic_sharpe: Probability that the true Sharpe ratio beats the
            configured benchmark.
        deflated_sharpe: Probability that the true Sharpe ratio beats
            ``expected_max_sharpe``.
        min_track_record_length: Observations needed for the probabilistic Sharpe
            ratio to reach the configured confidence level; infinite when the
            Sharpe ratio does not beat the benchmark.
        selected_trial: Name of the best trial when the result comes from a matrix
            of trials, otherwise ``None``.
    """

    sharpe: float
    annualized_sharpe: float
    skewness: float
    kurtosis: float
    n_observations: int
    n_trials: int
    trials_sharpe_variance: float
    expected_max_sharpe: float
    probabilistic_sharpe: float
    deflated_sharpe: float
    min_track_record_length: float
    selected_trial: str | None = None


def _require_confidence(confidence_level: float) -> None:
    if not 0.0 < confidence_level < 1.0:
        msg = (
            f"confidence_level must be strictly between 0 and 1, got {confidence_level}"
        )
        raise ConfigurationError(msg)


def _require_finite_benchmark(benchmark_sharpe: float) -> None:
    """Reject a benchmark that would turn every probability into NaN."""
    if not math.isfinite(benchmark_sharpe):
        msg = f"benchmark_sharpe must be finite, got {benchmark_sharpe}"
        raise ConfigurationError(msg)


def _variance_factor(sharpe: float, skewness: float, kurtosis: float) -> float:
    """Return the factor the Sharpe ratio's variance is proportional to."""
    factor = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if not factor > 0.0:
        msg = (
            "the variance of the Sharpe ratio estimate is not positive for "
            f"sharpe={sharpe}, skewness={skewness}, kurtosis={kurtosis}"
        )
        raise DataError(msg)
    return factor


def _sharpe_ratios(
    returns: npt.NDArray[np.float64], axis: int = 0
) -> npt.NDArray[np.float64]:
    """Return the per-period Sharpe ratio of ``returns`` along ``axis``."""
    return returns.mean(axis=axis) / returns.std(axis=axis, ddof=1)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_observations: float,
    skewness: float,
    kurtosis: float,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Return the probability that the true Sharpe ratio beats a benchmark.

    Args:
        sharpe: Per-period Sharpe ratio of the sample.
        n_observations: Number of returns behind ``sharpe``; it may be fractional
            when sizing a track record length.
        skewness: Sample skewness of the returns.
        kurtosis: Sample kurtosis of the returns (3 for a normal distribution).
        benchmark_sharpe: Per-period Sharpe ratio to beat.

    Returns:
        A probability in ``[0, 1]``; one half when ``sharpe`` equals the benchmark.

    Raises:
        DataError: If fewer than two observations are given, or the moments make
            the variance of the Sharpe ratio estimate non-positive.
        ConfigurationError: If ``benchmark_sharpe`` is not finite.
    """
    _require_finite_benchmark(benchmark_sharpe)
    if n_observations < _MIN_OBSERVATIONS_FOR_STANDARD_ERROR:
        msg = f"at least two observations are needed, got {n_observations}"
        raise DataError(msg)
    factor = _variance_factor(sharpe, skewness, kurtosis)
    z_score = (sharpe - benchmark_sharpe) * math.sqrt(n_observations - 1.0)
    return float(stats.norm.cdf(z_score / math.sqrt(factor)))


def expected_max_sharpe(n_trials: int, trials_sharpe_variance: float) -> float:
    """Return the best Sharpe ratio expected from ``n_trials`` skill-less trials.

    Uses the extreme-value approximation of the expected maximum of independent
    normal draws, scaled by the dispersion of the trials' Sharpe ratios.

    Args:
        n_trials: Number of strategies tried, at least 1.
        trials_sharpe_variance: Variance of the per-period Sharpe ratios across the
            trials, finite and non-negative.

    Returns:
        The per-period Sharpe ratio a search of this size reaches by luck; zero for
        a single trial or for trials that do not differ.

    Raises:
        ConfigurationError: If ``n_trials`` is below 1 or the variance is negative
            or not finite.
    """
    if n_trials < 1:
        msg = f"n_trials must be at least 1, got {n_trials}"
        raise ConfigurationError(msg)
    if not math.isfinite(trials_sharpe_variance) or trials_sharpe_variance < 0.0:
        msg = (
            "trials_sharpe_variance must be finite and non-negative, "
            f"got {trials_sharpe_variance}"
        )
        raise ConfigurationError(msg)
    if n_trials == 1 or trials_sharpe_variance == 0.0:
        return 0.0
    gamma = _EULER_MASCHERONI
    quantile = (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials)
    quantile += gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(trials_sharpe_variance) * float(quantile)


def minimum_track_record_length(
    sharpe: float,
    skewness: float,
    kurtosis: float,
    benchmark_sharpe: float,
    confidence_level: float,
) -> float:
    """Return the observations needed for the PSR to reach a confidence level.

    Args:
        sharpe: Per-period Sharpe ratio of the sample.
        skewness: Sample skewness of the returns.
        kurtosis: Sample kurtosis of the returns (3 for a normal distribution).
        benchmark_sharpe: Per-period Sharpe ratio to beat.
        confidence_level: Required probability, strictly between 0 and 1.

    Returns:
        The track record length in observations; infinite when ``sharpe`` does not
        exceed ``benchmark_sharpe``, because no amount of data would establish it.

    Raises:
        ConfigurationError: If ``confidence_level`` is not strictly between 0 and 1
            or ``benchmark_sharpe`` is not finite.
        DataError: If the moments make the variance of the Sharpe ratio estimate
            non-positive.
    """
    _require_confidence(confidence_level)
    _require_finite_benchmark(benchmark_sharpe)
    if sharpe <= benchmark_sharpe:
        return math.inf
    factor = _variance_factor(sharpe, skewness, kurtosis)
    z_score = float(stats.norm.ppf(confidence_level))
    return 1.0 + factor * (z_score / (sharpe - benchmark_sharpe)) ** 2


def _as_sample(
    returns: npt.ArrayLike, min_observations: int
) -> npt.NDArray[np.float64]:
    """Return ``returns`` as a validated one-dimensional float array."""
    try:
        sample = np.asarray(returns, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"returns must hold only numeric values: {error}"
        raise DataError(msg) from error
    if sample.ndim != 1:
        msg = f"returns must be one-dimensional, got shape {sample.shape}"
        raise DataError(msg)
    if sample.size < min_observations:
        msg = f"at least {min_observations} observations are needed, got {sample.size}"
        raise DataError(msg)
    if not np.isfinite(sample).all():
        msg = "returns must be finite: drop or fill missing values first"
        raise DataError(msg)
    if np.ptp(sample) == 0.0:
        msg = "returns are constant, so the Sharpe ratio is undefined"
        raise DataError(msg)
    return sample


def _evaluate(
    sample: npt.NDArray[np.float64],
    *,
    n_trials: int,
    trials_sharpe_variance: float,
    config: DeflatedSharpeConfig,
    selected_trial: str | None,
) -> DeflatedSharpeResult:
    """Compute every statistic for one validated return sample."""
    sharpe = float(_sharpe_ratios(sample))
    skewness = float(stats.skew(sample, bias=True))
    kurtosis = float(stats.kurtosis(sample, fisher=False, bias=True))
    n_observations = int(sample.size)
    expected_max = expected_max_sharpe(n_trials, trials_sharpe_variance)
    return DeflatedSharpeResult(
        sharpe=sharpe,
        annualized_sharpe=sharpe * math.sqrt(config.periods_per_year),
        skewness=skewness,
        kurtosis=kurtosis,
        n_observations=n_observations,
        n_trials=n_trials,
        trials_sharpe_variance=trials_sharpe_variance,
        expected_max_sharpe=expected_max,
        probabilistic_sharpe=probabilistic_sharpe_ratio(
            sharpe, n_observations, skewness, kurtosis, config.benchmark_sharpe
        ),
        deflated_sharpe=probabilistic_sharpe_ratio(
            sharpe, n_observations, skewness, kurtosis, expected_max
        ),
        min_track_record_length=minimum_track_record_length(
            sharpe, skewness, kurtosis, config.benchmark_sharpe, config.confidence_level
        ),
        selected_trial=selected_trial,
    )


def run_deflated_sharpe(
    returns: npt.ArrayLike,
    *,
    n_trials: int,
    trials_sharpe_variance: float,
    config: DeflatedSharpeConfig | None = None,
) -> DeflatedSharpeResult:
    """Evaluate one return series that was selected out of ``n_trials`` trials.

    Args:
        returns: Per-period returns of the selected strategy, in excess of the
            risk-free rate if the Sharpe ratio should be one.
        n_trials: Number of strategies tried before this one was kept.
        trials_sharpe_variance: Variance of the per-period Sharpe ratios across
            those trials.
        config: Settings; defaults to ``DeflatedSharpeConfig()``.

    Returns:
        The probabilistic and deflated Sharpe ratio with the moments behind them.

    Raises:
        DataError: If ``returns`` is not a finite, non-constant, one-dimensional
            series of at least ``config.min_observations`` values.
        ConfigurationError: If ``n_trials`` or ``trials_sharpe_variance`` is invalid.
    """
    settings = config or DeflatedSharpeConfig()
    return _evaluate(
        _as_sample(returns, settings.min_observations),
        n_trials=n_trials,
        trials_sharpe_variance=trials_sharpe_variance,
        config=settings,
        selected_trial=None,
    )


def run_deflated_sharpe_from_trials(
    trial_returns: pd.DataFrame,
    *,
    config: DeflatedSharpeConfig | None = None,
) -> DeflatedSharpeResult:
    """Pick the best of several trials and evaluate it against the whole search.

    The number of trials is the number of columns and the dispersion is the sample
    variance of the columns' Sharpe ratios, so a grid search or a set of
    cross-validated candidates can be passed as it is.

    Args:
        trial_returns: One column of per-period returns for each trial, all on the
            same dates.
        config: Settings; defaults to ``DeflatedSharpeConfig()``.

    Returns:
        The evaluation of the trial with the highest Sharpe ratio; ``selected_trial``
        holds its column name.

    Raises:
        DataError: If there are fewer than two trials or fewer rows than
            ``config.min_observations``, or a column is not numeric, or any value is
            not finite, or a trial is constant.
    """
    settings = config or DeflatedSharpeConfig()
    try:
        trial_matrix = trial_returns.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"trial_returns must hold only numeric columns: {error}"
        raise DataError(msg) from error
    if trial_matrix.shape[1] < _MIN_TRIALS_FOR_DISPERSION:
        msg = (
            "at least two trials (columns) are needed to estimate how much their "
            f"Sharpe ratios differ, got shape {trial_matrix.shape}"
        )
        raise DataError(msg)
    for column in range(trial_matrix.shape[1]):
        _as_sample(trial_matrix[:, column], settings.min_observations)
    sharpes = _sharpe_ratios(trial_matrix)
    best = int(np.argmax(sharpes))
    return _evaluate(
        trial_matrix[:, best],
        n_trials=trial_matrix.shape[1],
        trials_sharpe_variance=float(np.var(sharpes, ddof=1)),
        config=settings,
        selected_trial=str(trial_returns.columns[best]),
    )
