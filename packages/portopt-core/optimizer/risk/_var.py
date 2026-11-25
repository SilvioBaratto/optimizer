"""Value-at-Risk and expected shortfall of a return sample.

Value-at-Risk (VaR) at confidence ``c`` is the loss that a portfolio exceeds with
probability ``1 - c`` over a holding period; expected shortfall (ES) is the average
loss on the periods that exceed it (Acerbi and Tasche, 2002). Both are reported as
**positive fractions of portfolio value**, a loss being positive, and are estimated
from simple per-period returns. A negative value means the tail itself earns money.

Four estimators are offered:

* ``HISTORICAL``: the empirical quantile of the returns, scaled to the horizon by the
  square root of time;
* ``PARAMETRIC``: a normal distribution fitted to the sample mean and deviation;
* ``CORNISH_FISHER``: the normal quantile corrected for sample skewness and excess
  kurtosis (Zangari, 1996; Maillard, 2012);
* ``MONTE_CARLO``: a bootstrap that compounds ``horizon`` resampled returns.

``rolling_var_forecasts`` turns any of them into a one-step forecast series that
only ever looks at past returns, ready for ``optimizer.risk.backtest_var``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import integrate, stats

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.risk._config import VarConfig, VarMethod, require_integer_at_least

_SQRT_TWO_PI: Final = math.sqrt(2.0 * math.pi)
# Relative to the largest absolute return; see ``_empirical_var_es``.
_TAIL_SLACK: Final = 1e-12
# The Cornish-Fisher expansion has to be monotone down to this fraction of the tail
# probability. Below it the density weight is too small to move the expected
# shortfall, and demanding monotonicity out to probability zero would reject
# ordinary near-normal samples.
_TAIL_FRACTION_CHECKED: Final = 1e-3

_Estimator = Callable[[npt.NDArray[np.float64], VarConfig], tuple[float, float]]


@dataclass(frozen=True)
class VarResult:
    """Outcome of a Value-at-Risk estimate.

    Attributes:
        var: Value-at-Risk as a positive fraction of portfolio value.
        expected_shortfall: Average loss beyond the VaR, on the same scale.
        confidence_level: Confidence level the estimate was made at.
        method: Estimator that produced it.
        horizon: Holding period in return periods.
        n_observations: Number of returns the estimate rests on.
    """

    var: float
    expected_shortfall: float
    confidence_level: float
    method: VarMethod
    horizon: int
    n_observations: int


def cornish_fisher_quantile(z: float, skewness: float, excess_kurtosis: float) -> float:
    """Return the Cornish-Fisher adjusted quantile of a standardised distribution.

    The expansion of Cornish and Fisher (1938) corrects the normal quantile ``z``
    for the third and fourth moments:
    ``z + (z^2 - 1) S / 6 + (z^3 - 3 z) K / 24 - (2 z^3 - 5 z) S^2 / 36``.

    Args:
        z: Normal quantile to adjust, for instance ``Phi^-1(1 - confidence)``.
        skewness: Skewness ``S`` of the distribution.
        excess_kurtosis: Excess kurtosis ``K`` of the distribution (0 for a normal
            distribution).

    Returns:
        The adjusted quantile of the standardised distribution; equal to ``z``
        when both moments are zero. It is only a quantile function while the
        expansion is monotone, which is not checked here.
    """
    return (
        z
        + (z**2 - 1.0) * skewness / 6.0
        + (z**3 - 3.0 * z) * excess_kurtosis / 24.0
        - (2.0 * z**3 - 5.0 * z) * skewness**2 / 36.0
    )


def as_series(values: npt.ArrayLike | pd.Series, name: str) -> pd.Series:
    """Return ``values`` as a one-dimensional float series, keeping its index.

    Args:
        values: A series, or any one-dimensional numeric array-like (which is
            indexed by position).
        name: Argument name used in the error message.

    Returns:
        A float series.

    Raises:
        DataError: If ``values`` is not one-dimensional and numeric.
    """
    try:
        return pd.Series(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be a one-dimensional numeric series"
        raise DataError(msg) from error


def _require_finite(values: npt.NDArray[np.float64]) -> None:
    """Reject missing or infinite returns."""
    if not np.isfinite(values).all():
        msg = "returns must be finite: drop or fill missing values first"
        raise DataError(msg)


def _as_sample(
    returns: npt.ArrayLike | pd.Series, min_observations: int
) -> npt.NDArray[np.float64]:
    """Return ``returns`` as a validated one-dimensional float array."""
    sample: npt.NDArray[np.float64] = as_series(returns, "returns").to_numpy(
        dtype=np.float64
    )
    if sample.size < min_observations:
        msg = f"at least {min_observations} observations are needed, got {sample.size}"
        raise DataError(msg)
    _require_finite(sample)
    return sample


def _empirical_var_es(
    sample: npt.NDArray[np.float64], tail_probability: float
) -> tuple[float, float]:
    """Return the empirical loss quantile and tail average of ``sample``."""
    quantile = float(np.quantile(sample, tail_probability))
    # np.quantile locates the order statistic through a floating-point index, so a
    # quantile that sits on a return can land a few ulps below it and a bare
    # ``<=`` would silently drop that return from the tail.
    slack = _TAIL_SLACK * float(np.max(np.abs(sample)))
    tail = sample[sample <= quantile + slack]
    return -quantile, -float(tail.mean())


def _horizon_moments(
    sample: npt.NDArray[np.float64], horizon: int
) -> tuple[float, float]:
    """Return the mean and standard deviation scaled to ``horizon`` periods."""
    return float(sample.mean()) * horizon, float(sample.std(ddof=1)) * math.sqrt(
        horizon
    )


def _standard_normal_pdf(x: float) -> float:
    """Return the standard normal density."""
    return math.exp(-0.5 * x * x) / _SQRT_TWO_PI


def _historical(
    sample: npt.NDArray[np.float64], config: VarConfig
) -> tuple[float, float]:
    """Return the empirical VaR and ES scaled by the square root of the horizon."""
    var, expected_shortfall = _empirical_var_es(sample, 1.0 - config.confidence_level)
    scale = math.sqrt(config.horizon)
    return var * scale, expected_shortfall * scale


def _parametric(
    sample: npt.NDArray[np.float64], config: VarConfig
) -> tuple[float, float]:
    """Return the VaR and ES of a normal distribution fitted to ``sample``."""
    z = float(stats.norm.isf(config.confidence_level))
    mean, std = _horizon_moments(sample, config.horizon)
    tail_probability = 1.0 - config.confidence_level
    var = -(mean + std * z)
    expected_shortfall = -mean + std * _standard_normal_pdf(z) / tail_probability
    return var, expected_shortfall


def _standardised_moments(sample: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Return the sample skewness and excess kurtosis (moment estimators)."""
    deviations = sample - sample.mean()
    variance = float(np.mean(deviations**2))
    if variance == 0.0:
        return 0.0, 0.0
    skewness = float(np.mean(deviations**3)) / variance**1.5
    excess_kurtosis = float(np.mean(deviations**4)) / variance**2 - 3.0
    return skewness, excess_kurtosis


def _tail_is_monotone(
    tail_probability: float, skewness: float, excess_kurtosis: float
) -> bool:
    """Tell whether the expansion rises with ``z`` over the tail it is averaged on."""
    z_high = float(stats.norm.ppf(tail_probability))
    z_low = float(stats.norm.ppf(tail_probability * _TAIL_FRACTION_CHECKED))
    # The derivative of the expansion is a quadratic in z, so its minimum over an
    # interval is at an end point or at the vertex.
    quadratic = excess_kurtosis / 8.0 - skewness**2 / 6.0
    linear = skewness / 3.0
    constant = 1.0 - excess_kurtosis / 8.0 + 5.0 * skewness**2 / 36.0
    candidates = [z_low, z_high]
    if quadratic != 0.0:
        vertex = -linear / (2.0 * quadratic)
        if z_low < vertex < z_high:
            candidates.append(vertex)
    return all((quadratic * z + linear) * z + constant >= 0.0 for z in candidates)


def _quantile_weighted_by_density(
    z: float, skewness: float, excess_kurtosis: float
) -> float:
    """Return the Cornish-Fisher quantile times the normal density at ``z``."""
    return cornish_fisher_quantile(z, skewness, excess_kurtosis) * _standard_normal_pdf(
        z
    )


def _cornish_fisher(
    sample: npt.NDArray[np.float64], config: VarConfig
) -> tuple[float, float]:
    """Return the Cornish-Fisher VaR and ES of ``sample``.

    The skewness and kurtosis are those of the one-period returns and are not
    rescaled to the horizon.
    """
    tail_probability = 1.0 - config.confidence_level
    skewness, excess_kurtosis = _standardised_moments(sample)
    if not _tail_is_monotone(tail_probability, skewness, excess_kurtosis):
        msg = (
            "the Cornish-Fisher expansion is not monotone over the tail for a "
            f"sample with skewness {skewness:.3f} and excess kurtosis "
            f"{excess_kurtosis:.3f}: it is outside its range of validity, use the "
            "historical or parametric method instead"
        )
        raise DataError(msg)
    z = float(stats.norm.isf(config.confidence_level))
    # Averaging the quantile function over the tail probabilities is the same as
    # integrating quantile times density over the normal quantiles up to the tail
    # edge; integrating in that variable avoids the singularity of the quantile
    # function at probability zero.
    integral, _ = integrate.quad(
        _quantile_weighted_by_density, -np.inf, z, args=(skewness, excess_kurtosis)
    )
    mean, std = _horizon_moments(sample, config.horizon)
    adjusted_quantile = cornish_fisher_quantile(z, skewness, excess_kurtosis)
    var = -(mean + std * adjusted_quantile)
    expected_shortfall = -(mean + std * integral / tail_probability)
    return var, expected_shortfall


def _monte_carlo(
    sample: npt.NDArray[np.float64], config: VarConfig
) -> tuple[float, float]:
    """Return the VaR and ES of bootstrapped ``horizon``-period compound returns."""
    rng = np.random.default_rng(config.random_state)
    growth = np.ones(config.n_simulations)
    # One draw per period keeps memory at n_simulations however long the horizon.
    for _ in range(config.horizon):
        growth *= 1.0 + rng.choice(sample, size=config.n_simulations)
    return _empirical_var_es(growth - 1.0, 1.0 - config.confidence_level)


_ESTIMATORS: Final[dict[VarMethod, _Estimator]] = {
    VarMethod.HISTORICAL: _historical,
    VarMethod.PARAMETRIC: _parametric,
    VarMethod.CORNISH_FISHER: _cornish_fisher,
    VarMethod.MONTE_CARLO: _monte_carlo,
}


def _resolve_config(config: VarConfig | None) -> VarConfig:
    """Return ``config``, or the defaults when none is given."""
    if config is None:
        return VarConfig()
    if not isinstance(config, VarConfig):
        msg = f"config must be a VarConfig, got {type(config).__name__}"
        raise ConfigurationError(msg)
    return config


def _evaluate(sample: npt.NDArray[np.float64], config: VarConfig) -> VarResult:
    """Estimate VaR and ES for one validated sample."""
    var, expected_shortfall = _ESTIMATORS[config.method](sample, config)
    return VarResult(
        var=var,
        expected_shortfall=expected_shortfall,
        confidence_level=config.confidence_level,
        method=config.method,
        horizon=config.horizon,
        n_observations=int(sample.size),
    )


def compute_var(
    returns: npt.ArrayLike | pd.Series, config: VarConfig | None = None
) -> VarResult:
    """Estimate Value-at-Risk and expected shortfall from a sample of returns.

    The historical method takes the empirical quantile of the returns at
    ``1 - confidence_level`` with ``np.quantile``'s linear interpolation between
    order statistics, and the expected shortfall is minus the mean of the returns
    at or below that quantile. Both are multiplied by the square root of the
    horizon, an approximation that assumes independent returns and a mean close to
    zero. The parametric and Cornish-Fisher methods scale the mean by the horizon
    and the standard deviation by its square root; the Cornish-Fisher expansion
    uses the one-period skewness and excess kurtosis unchanged, which overstates
    the tail thickness of a multi-period return. The Monte Carlo method resamples
    ``horizon`` returns with replacement for each of ``n_simulations`` paths,
    compounds them and reads the empirical VaR and ES of the paths, with no
    square-root scaling.

    Args:
        returns: Simple per-period returns of the portfolio (0.01 is a gain of 1%).
        config: Settings; defaults to ``VarConfig()``.

    Returns:
        The VaR and the expected shortfall as positive fractions of portfolio value.

    Raises:
        ConfigurationError: If ``config`` is not a ``VarConfig``.
        DataError: If ``returns`` is not a finite, one-dimensional series of at
            least ``config.min_observations`` values, or the Cornish-Fisher
            expansion is not monotone over the tail for the sample's moments.
    """
    settings = _resolve_config(config)
    return _evaluate(_as_sample(returns, settings.min_observations), settings)


def rolling_var_forecasts(
    returns: npt.ArrayLike | pd.Series,
    window: int,
    config: VarConfig | None = None,
) -> pd.Series:
    """Forecast the one-period VaR at each date from the returns before it.

    The forecast at date ``t`` is ``compute_var`` on the ``window`` returns that
    precede ``t``, so it never sees the return it will be compared with. A fixed
    ``random_state`` is reused for every window, which keeps a Monte Carlo
    forecast reproducible.

    Args:
        returns: Simple per-period returns, preferably a series with a date index.
        window: Number of past returns behind each forecast, a whole number (a
            float or bool is rejected).
        config: Settings with ``horizon == 1``; defaults to ``VarConfig()``.

    Returns:
        A series indexed like ``returns`` whose first ``window`` entries are NaN.

    Raises:
        ConfigurationError: If ``config`` is not a ``VarConfig``, its horizon is
            not 1, or ``window`` is not an integer of at least the config's
            ``min_observations``.
        DataError: If ``returns`` is not a finite, one-dimensional series longer
            than ``window``, or a window is outside the Cornish-Fisher range.
    """
    settings = _resolve_config(config)
    if settings.horizon != 1:
        msg = (
            f"rolling forecasts are one-step: horizon must be 1, got {settings.horizon}"
        )
        raise ConfigurationError(msg)
    require_integer_at_least("window", window, settings.min_observations)
    series = as_series(returns, "returns")
    if series.size <= window:
        msg = f"returns must be longer than the window ({window}), got {series.size}"
        raise DataError(msg)
    values = series.to_numpy(dtype=np.float64)
    _require_finite(values)
    forecasts = np.full(values.size, np.nan)
    for position in range(window, values.size):
        forecasts[position] = _evaluate(
            values[position - window : position], settings
        ).var
    return pd.Series(forecasts, index=series.index, name="var_forecast")
