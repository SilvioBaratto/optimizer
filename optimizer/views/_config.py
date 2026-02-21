"""Configuration for view integration frameworks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from optimizer.moments._config import MomentEstimationConfig


class ViewUncertaintyMethod(str, Enum):
    """View uncertainty calibration method for Black-Litterman.

    Maps to the ``view_confidences`` parameter in
    :class:`skfolio.prior.BlackLitterman`.
    """

    HE_LITTERMAN = "he_litterman"
    IDZOREK = "idzorek"
    EMPIRICAL_TRACK_RECORD = "empirical_track_record"


@dataclass(frozen=True)
class BlackLittermanConfig:
    """Immutable configuration for the Black-Litterman prior.

    All parameters map 1:1 to :class:`skfolio.prior.BlackLitterman`
    constructor arguments, keeping the config serialisable and suitable
    for hyperparameter sweeps.

    Parameters
    ----------
    views : tuple[str, ...]
        View expressions (absolute or relative).
    tau : float
        Uncertainty scaling parameter.
    risk_free_rate : float
        Risk-free rate added to posterior expected returns.
    uncertainty_method : ViewUncertaintyMethod
        How to calibrate view uncertainty (omega matrix).
    view_confidences : tuple[float, ...] or None
        Per-view confidence levels in [0, 1] for the Idzorek method.
    groups : dict[str, list[str]] or None
        Asset group mapping for group-relative views.
    prior_config : MomentEstimationConfig or None
        Inner prior configuration.  Defaults to
        ``MomentEstimationConfig.for_equilibrium_ledoitwolf()``.
    use_factor_model : bool
        If ``True``, wrap the Black-Litterman prior in a
        :class:`skfolio.prior.FactorModel`.
    residual_variance : bool
        Whether to include residual variance in ``FactorModel``.
    """

    views: tuple[str, ...]
    tau: float = 0.05
    risk_free_rate: float = 0.0
    uncertainty_method: ViewUncertaintyMethod = ViewUncertaintyMethod.HE_LITTERMAN
    view_confidences: tuple[float, ...] | None = None
    groups: dict[str, list[str]] | None = None
    prior_config: MomentEstimationConfig | None = None
    use_factor_model: bool = False
    residual_variance: bool = True

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError(
                f"tau must be strictly positive, got {self.tau}"
            )

    # -- factory methods -----------------------------------------------------

    @classmethod
    def for_equilibrium(cls, views: tuple[str, ...]) -> BlackLittermanConfig:
        """Standard BL with EquilibriumMu prior, tau=0.05."""
        return cls(
            views=views,
            prior_config=MomentEstimationConfig.for_equilibrium_ledoitwolf(),
        )

    @classmethod
    def for_factor_model(cls, views: tuple[str, ...]) -> BlackLittermanConfig:
        """BL Factor Model variant."""
        return cls(
            views=views,
            prior_config=MomentEstimationConfig.for_equilibrium_ledoitwolf(),
            use_factor_model=True,
        )


@dataclass(frozen=True)
class EntropyPoolingConfig:
    """Immutable configuration for the Entropy Pooling prior.

    All parameters map 1:1 to :class:`skfolio.prior.EntropyPooling`
    constructor arguments.

    Parameters
    ----------
    mean_views : tuple[str, ...] or None
        Mean view expressions.
    variance_views : tuple[str, ...] or None
        Variance view expressions.
    correlation_views : tuple[str, ...] or None
        Correlation view expressions.
    skew_views : tuple[str, ...] or None
        Skewness view expressions.
    kurtosis_views : tuple[str, ...] or None
        Kurtosis view expressions.
    cvar_views : tuple[str, ...] or None
        CVaR view expressions.
    cvar_beta : float
        Confidence level for CVaR views.
    groups : dict[str, list[str]] or None
        Asset group mapping for group-relative views.
    solver : str
        Scipy solver for the dual optimisation.
    solver_params : dict[str, object] or None
        Additional solver parameters.
    prior_config : MomentEstimationConfig or None
        Inner prior configuration.  Defaults to ``EmpiricalPrior()``.
    """

    mean_views: tuple[str, ...] | None = None
    mean_inequality_views: tuple[str, ...] | None = None
    variance_views: tuple[str, ...] | None = None
    correlation_views: tuple[str, ...] | None = None
    skew_views: tuple[str, ...] | None = None
    kurtosis_views: tuple[str, ...] | None = None
    cvar_views: tuple[str, ...] | None = None
    cvar_beta: float = 0.95
    groups: dict[str, list[str]] | None = None
    solver: str = "TNC"
    solver_params: dict[str, object] | None = None
    prior_config: MomentEstimationConfig | None = None

    # -- factory methods -----------------------------------------------------

    @classmethod
    def for_mean_views(cls, mean_views: tuple[str, ...]) -> EntropyPoolingConfig:
        """Mean-only Entropy Pooling."""
        return cls(mean_views=mean_views)

    @classmethod
    def for_stress_test(
        cls,
        variance_views: tuple[str, ...],
        correlation_views: tuple[str, ...],
    ) -> EntropyPoolingConfig:
        """Stress-test Entropy Pooling with variance + correlation views."""
        return cls(
            variance_views=variance_views,
            correlation_views=correlation_views,
        )


@dataclass(frozen=True)
class OpinionPoolingConfig:
    """Immutable configuration for the Opinion Pooling prior.

    The ``estimators`` argument is passed directly to the factory
    function (not stored here) because estimator objects are not
    serialisable in a frozen dataclass.

    Parameters
    ----------
    opinion_probabilities : tuple[float, ...] or None
        Per-expert weight.
    is_linear_pooling : bool
        ``True`` for arithmetic (linear) pooling, ``False`` for
        geometric (logarithmic) pooling.
    divergence_penalty : float
        KL-divergence penalty for robust pooling.
    n_jobs : int or None
        Number of parallel jobs for expert fitting.
    prior_config : MomentEstimationConfig or None
        Common prior configuration.
    """

    opinion_probabilities: tuple[float, ...] | None = None
    is_linear_pooling: bool = True
    divergence_penalty: float = 0.0
    n_jobs: int | None = None
    prior_config: MomentEstimationConfig | None = None

    def __post_init__(self) -> None:
        if self.opinion_probabilities is not None:
            for p in self.opinion_probabilities:
                if not (0.0 <= p <= 1.0):
                    raise ValueError(
                        f"Each opinion probability must be in [0, 1], got {p}"
                    )
            total = sum(self.opinion_probabilities)
            if total > 1.0 + 1e-10:
                raise ValueError(
                    f"opinion_probabilities must sum to at most 1.0, "
                    f"got {total}"
                )
