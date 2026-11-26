"""Kalman-filter hedge ratio for a pair.

A fixed least-squares hedge ratio assumes the relation between two prices never
changes. Chan (2013), "Algorithmic Trading", Example 3.3, lets it drift: the state
``[beta_t, alpha_t]`` follows a random walk and ``y_t = beta_t x_t + alpha_t`` plus
noise is the observation. Each step predicts the state, measures how far ``y_t``
falls from the prediction (the innovation) and moves the state by the Kalman gain
times that error. The innovation is a ready-made spread, and dividing it by its own
standard deviation gives a z-score that needs no rolling window.

The estimate at ``t`` uses only observations up to and including ``t``.

Research tools only: nothing in ``optimizer.statarb`` models transaction costs,
borrow costs or short-sale limits, and the fund mandate is long-only and
paper-only, so none of it is wired to execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from optimizer.statarb._config import KalmanConfig
from optimizer.statarb._inputs import aligned_pair

# Columns of the filter's output, in the order ``_run_filter`` fills them; the
# innovation standard deviation is the last one and ``zscore`` is derived from it.
_OUTPUT_NAMES: Final = ("beta", "alpha", "spread", "spread_std")


@dataclass(frozen=True, eq=False)
class KalmanResult:
    """Outcome of a Kalman-filter hedge ratio, all on ``y``'s index.

    The result holds pandas Series, so it compares by identity. The first values
    reflect the prior rather than the data: discard a burn-in before trading.

    Attributes:
        beta: Filtered hedge ratio after seeing ``y_t``.
        alpha: Filtered intercept after seeing ``y_t``.
        spread: One-step-ahead prediction error ``y_t - (beta_{t|t-1} x_t +
            alpha_{t|t-1})``, made before ``y_t`` is used.
        spread_std: Standard deviation of that error, the square root of the
            innovation variance.
        zscore: ``spread / spread_std``; unit variance when the model fits.
    """

    beta: pd.Series
    alpha: pd.Series
    spread: pd.Series
    spread_std: pd.Series
    zscore: pd.Series


def _run_filter(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64], config: KalmanConfig
) -> npt.NDArray[np.float64]:
    """Return ``[beta, alpha, error, innovation std]`` for every step."""
    state = np.zeros(2)
    covariance = np.eye(2) * config.initial_state_variance
    drift = np.eye(2) * (config.delta / (1.0 - config.delta))
    rows = np.empty((len(y), len(_OUTPUT_NAMES)))
    for t in range(len(y)):
        regressors = np.array([x[t], 1.0])
        predicted = covariance + drift
        cross = predicted @ regressors
        innovation_variance = float(regressors @ cross) + config.observation_variance
        error = y[t] - float(regressors @ state)
        gain = cross / innovation_variance
        state = state + gain * error
        covariance = predicted - np.outer(gain, cross)
        rows[t] = (state[0], state[1], error, math.sqrt(innovation_variance))
    return rows


def kalman_hedge_ratio(
    y: pd.Series, x: pd.Series, config: KalmanConfig | None = None
) -> KalmanResult:
    """Track the hedge ratio of ``y`` on ``x`` with a Kalman filter.

    The state ``[beta, alpha]`` starts at zero with covariance
    ``initial_state_variance`` times the identity and follows a random walk with
    covariance ``delta / (1 - delta)`` times the identity; the observation is
    ``y_t = beta_t x_t + alpha_t + noise`` with variance ``observation_variance``.
    The state noise and the observation noise are in the units of the data, so a
    price series in the hundreds needs larger variances than the defaults.

    This is a research tool. It ignores costs and borrow limits, and nothing here
    is wired to execution.

    Args:
        y: Dependent series, for example a log price.
        x: Independent series on exactly the same index.
        config: Settings; defaults to ``KalmanConfig()``.

    Returns:
        The filtered ``beta`` and ``alpha``, and the one-step-ahead spread, its
        standard deviation and the z-score, each as a Series on ``y``'s index.

    Raises:
        DataError: If the inputs are not aligned Series, are empty, or contain a
            value that is not a finite number.
    """
    settings = config or KalmanConfig()
    y_values, x_values = aligned_pair(y, x, min_observations=1)
    rows = _run_filter(y_values, x_values, settings)
    columns = {
        name: pd.Series(rows[:, position], index=y.index, name=name)
        for position, name in enumerate(_OUTPUT_NAMES)
    }
    zscore = (columns["spread"] / columns["spread_std"]).rename("zscore")
    return KalmanResult(zscore=zscore, **columns)
