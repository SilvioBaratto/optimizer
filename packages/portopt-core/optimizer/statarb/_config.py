"""Configuration objects for the statistical-arbitrage research tools.

Research tools only: nothing in ``optimizer.statarb`` models transaction costs,
borrow costs or short-sale limits, and the fund mandate is long-only and
paper-only, so none of it is wired to execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from optimizer.exceptions import ConfigurationError
from optimizer.statarb._inputs import require_finite, require_integer

# Asymptotic MacKinnon (2010) critical values of the two-variable Engle-Granger
# test with a constant; the table has no finite-sample correction.
ENGLE_GRANGER_CRITICAL_VALUES: Final = {"1%": -3.90, "5%": -3.34, "10%": -3.04}

# The ADF regression keeps a lag per row of the search, so a shorter sample leaves
# no degrees of freedom to speak of.
_MIN_ADF_OBSERVATIONS: Final = 20
# An AR(1) fit of a cumulative residual over fewer points than this is noise.
_MIN_PCA_OBSERVATIONS: Final = 10


@dataclass(frozen=True)
class EngleGrangerConfig:
    """Immutable configuration of the Engle-Granger cointegration test.

    Attributes:
        adf_lags: Lagged differences in the Augmented Dickey-Fuller regression of
            the residuals; ``None`` selects the number by the Akaike criterion over
            ``0..max_lags``.
        max_lags: Largest lag count the Akaike search considers; ``None`` uses the
            Schwert rule ``floor(12 (T / 100) ** 0.25)``. It is capped so the
            longest regression keeps a couple of degrees of freedom. Ignored when
            ``adf_lags`` is set.
        significance: Level the decision is made at, one of ``"1%"``, ``"5%"`` or
            ``"10%"``.
        min_observations: Fewest observations a pair of series may have.
    """

    adf_lags: int | None = None
    max_lags: int | None = None
    significance: str = "5%"
    min_observations: int = 50

    def __post_init__(self) -> None:
        """Reject settings the test cannot use."""
        if self.adf_lags is not None:
            require_integer("adf_lags", self.adf_lags, 0)
        if self.max_lags is not None:
            require_integer("max_lags", self.max_lags, 0)
        if (
            not isinstance(self.significance, str)
            or self.significance not in ENGLE_GRANGER_CRITICAL_VALUES
        ):
            msg = (
                f"significance must be one of {sorted(ENGLE_GRANGER_CRITICAL_VALUES)}, "
                f"got {self.significance!r}"
            )
            raise ConfigurationError(msg)
        require_integer(
            "min_observations", self.min_observations, _MIN_ADF_OBSERVATIONS
        )


def _require_positive_finite(name: str, value: object) -> None:
    if not require_finite(name, value) > 0.0:
        msg = f"{name} must be positive and finite, got {value!r}"
        raise ConfigurationError(msg)


@dataclass(frozen=True)
class KalmanConfig:
    """Immutable configuration of the Kalman-filter hedge ratio.

    The state ``[beta, alpha]`` follows a random walk whose covariance is
    ``delta / (1 - delta)`` times the identity, as in Chan (2013), and the
    observation ``y = beta x + alpha`` carries noise of variance
    ``observation_variance``. Both variances are in the units of the data, so
    rescale ``y`` and ``x`` (or the variances) when prices are far from one.

    Attributes:
        delta: Weight of the state noise, strictly between 0 and 1; larger values
            let the hedge ratio drift faster.
        observation_variance: Variance of the measurement noise, positive.
        initial_state_variance: Variance of the zero-mean prior on each state
            component, positive.
    """

    delta: float = 1e-4
    observation_variance: float = 1e-3
    initial_state_variance: float = 1.0

    def __post_init__(self) -> None:
        """Reject settings that make the filter degenerate."""
        if not 0.0 < require_finite("delta", self.delta) < 1.0:
            msg = f"delta must be strictly between 0 and 1, got {self.delta!r}"
            raise ConfigurationError(msg)
        _require_positive_finite("observation_variance", self.observation_variance)
        _require_positive_finite("initial_state_variance", self.initial_state_variance)


@dataclass(frozen=True)
class PcaResidualConfig:
    """Immutable configuration of the PCA residual (eigenportfolio) model.

    Attributes:
        periods_per_year: Periods in a year, used only to annualise the
            mean-reversion speed ``kappa``.
        min_observations: Fewest rows the returns may have; Avellaneda and Lee
            (2010) use a window of 60 trading days.
        min_kappa: Smallest annualised reversion speed a residual needs to count
            as mean reverting. The default 0 keeps every residual whose AR(1)
            coefficient lies in ``(0, 1)``; Avellaneda and Lee trade only
            ``kappa > 8.4``, a reversion time under 30 days at 252 periods a year.
    """

    periods_per_year: int = 252
    min_observations: int = 60
    min_kappa: float = 0.0

    def __post_init__(self) -> None:
        """Reject settings the fit cannot use."""
        require_integer("periods_per_year", self.periods_per_year, 1)
        require_integer(
            "min_observations", self.min_observations, _MIN_PCA_OBSERVATIONS
        )
        if require_finite("min_kappa", self.min_kappa) < 0.0:
            msg = f"min_kappa must be non-negative, got {self.min_kappa!r}"
            raise ConfigurationError(msg)
