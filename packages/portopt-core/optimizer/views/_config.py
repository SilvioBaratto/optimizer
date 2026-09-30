"""Configuration for view integration frameworks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from optimizer.moments._config import MomentEstimationConfig


class ViewUncertaintyMethod(str, Enum):
    """View uncertainty calibration method for Black-Litterman.

    Maps to the ``view_confidences`` parameter in
    `BlackLitterman`.
    """

    HE_LITTERMAN = "he_litterman"
    IDZOREK = "idzorek"
    EMPIRICAL_TRACK_RECORD = "empirical_track_record"


@dataclass(frozen=True)
class BlackLittermanConfig:
    """Immutable configuration for the Black-Litterman prior.

    All parameters map 1:1 to skfolio.prior.BlackLitterman constructor
    arguments, keeping the config serialisable and suitable for
    hyperparameter sweeps.

    Args:
        views: View expressions (absolute or relative).
        tau: Uncertainty scaling parameter; must be strictly positive.
        risk_free_rate: Risk-free rate added to posterior expected returns.
        uncertainty_method: How to calibrate view uncertainty (omega matrix).
        view_confidences: Per-view confidence levels in [0, 1] for the
            Idzorek method; must have one entry per view.
        groups: Asset group mapping for group-relative views.
        prior_config: Inner prior configuration. Defaults to
            ``MomentEstimationConfig.for_equilibrium_ledoitwolf()``.
        use_factor_model: When True, wraps the Black-Litterman prior in a
            skfolio.prior.TimeSeriesFactorModel. Fit with factor returns
            via the keyword ``factors=`` (skfolio 1.0).
    """

    views: tuple[str, ...]
    tau: float = 0.05
    risk_free_rate: float = 0.0
    uncertainty_method: ViewUncertaintyMethod = ViewUncertaintyMethod.HE_LITTERMAN
    view_confidences: tuple[float, ...] | None = None
    groups: dict[str, list[str]] | None = None
    prior_config: MomentEstimationConfig | None = None
    use_factor_model: bool = False

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError(f"tau must be strictly positive, got {self.tau}")
        if self.view_confidences is not None:
            for c in self.view_confidences:
                if not (0.0 <= c <= 1.0):
                    raise ValueError(f"Each view confidence must be in [0, 1], got {c}")
            if len(self.view_confidences) != len(self.views):
                raise ValueError(
                    "view_confidences must have one entry per view; got "
                    f"{len(self.view_confidences)} confidences for "
                    f"{len(self.views)} views"
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

    @classmethod
    def for_idzorek(
        cls,
        views: tuple[str, ...],
        view_confidences: tuple[float, ...],
    ) -> BlackLittermanConfig:
        """Idzorek method with per-view confidence levels."""
        return cls(
            views=views,
            uncertainty_method=ViewUncertaintyMethod.IDZOREK,
            view_confidences=view_confidences,
            prior_config=MomentEstimationConfig.for_equilibrium_ledoitwolf(),
        )


@dataclass(frozen=True)
class EntropyPoolingConfig:
    """Immutable configuration for the Entropy Pooling prior.

    All parameters map 1:1 to skfolio.prior.EntropyPooling constructor
    arguments.

    Args:
        mean_views: Mean equality view expressions.
        mean_inequality_views: Mean inequality view expressions.
        variance_views: Variance view expressions.
        relative_mean_views: Relative mean views as (asset, multiplier) pairs.
        relative_variance_views: Relative variance views as (asset, multiplier) pairs.
        correlation_views: Correlation view expressions.
        skew_views: Skewness view expressions.
        kurtosis_views: Kurtosis view expressions.
        value_at_risk_views: Value-at-Risk (VaR) view expressions (skfolio 1.0).
            Supports both inequalities and ``prior()`` references, e.g.
            ``"SPX >= 0.03"`` or ``"SX5E == 1.5 * prior(SX5E)"``.
        cvar_views: CVaR view expressions.
        value_at_risk_beta: Confidence level for VaR views; must be in (0, 1).
        cvar_beta: Confidence level for CVaR views; must be in (0, 1).
        groups: Asset group mapping for group-relative views.
        solver: Scipy solver for the dual optimisation.
        solver_params: Additional solver parameters passed through to the solver.
        prior_config: Inner prior configuration. Defaults to ``EmpiricalPrior()``.
    """

    mean_views: tuple[str, ...] | None = None
    mean_inequality_views: tuple[str, ...] | None = None
    variance_views: tuple[str, ...] | None = None
    relative_mean_views: tuple[tuple[str, float], ...] | None = None
    relative_variance_views: tuple[tuple[str, float], ...] | None = None
    correlation_views: tuple[str, ...] | None = None
    skew_views: tuple[str, ...] | None = None
    kurtosis_views: tuple[str, ...] | None = None
    value_at_risk_views: tuple[str, ...] | None = None
    cvar_views: tuple[str, ...] | None = None
    value_at_risk_beta: float = 0.95
    cvar_beta: float = 0.95
    groups: dict[str, list[str]] | None = None
    solver: str = "TNC"
    solver_params: dict[str, object] | None = None
    prior_config: MomentEstimationConfig | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.cvar_beta < 1.0):
            raise ValueError(
                f"cvar_beta must be in the open interval (0, 1), got {self.cvar_beta}"
            )
        if not (0.0 < self.value_at_risk_beta < 1.0):
            raise ValueError(
                "value_at_risk_beta must be in the open interval (0, 1), got "
                f"{self.value_at_risk_beta}"
            )

    # -- factory methods -----------------------------------------------------

    @classmethod
    def for_mean_views(cls, mean_views: tuple[str, ...]) -> EntropyPoolingConfig:
        """Mean-only Entropy Pooling."""
        return cls(mean_views=mean_views)

    @classmethod
    def for_tail_risk(
        cls,
        cvar_views: tuple[str, ...],
        value_at_risk_views: tuple[str, ...] | None = None,
        beta: float = 0.95,
    ) -> EntropyPoolingConfig:
        """Tail-risk Entropy Pooling with CVaR (and optional VaR) views."""
        return cls(
            cvar_views=cvar_views,
            value_at_risk_views=value_at_risk_views,
            cvar_beta=beta,
            value_at_risk_beta=beta,
        )

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

    @classmethod
    def for_group_views(
        cls,
        mean_views: tuple[str, ...],
        groups: dict[str, list[str]],
    ) -> EntropyPoolingConfig:
        """Entropy Pooling with group-relative mean views."""
        return cls(
            mean_views=mean_views,
            groups=groups,
        )


@dataclass(frozen=True)
class OpinionPoolingConfig:
    """Immutable configuration for the Opinion Pooling prior.

    The ``estimators`` argument is passed directly to the factory function
    (not stored here) because estimator objects are not serialisable in a
    frozen dataclass.

    Args:
        opinion_probabilities: Per-expert weight; each value must be in [0, 1]
            and the sum must not exceed 1.0.
        is_linear_pooling: When True, uses arithmetic (linear) pooling; when
            False, uses geometric (logarithmic) pooling.
        divergence_penalty: KL-divergence penalty for robust pooling; must be
            non-negative.
        n_jobs: Number of parallel jobs for expert fitting.
        prior_config: Common prior configuration shared across experts.
    """

    opinion_probabilities: tuple[float, ...] | None = None
    is_linear_pooling: bool = True
    divergence_penalty: float = 0.0
    n_jobs: int | None = None
    prior_config: MomentEstimationConfig | None = None

    def __post_init__(self) -> None:
        if self.divergence_penalty < 0.0:
            raise ValueError(
                "divergence_penalty must be non-negative, got "
                f"{self.divergence_penalty}"
            )
        if self.opinion_probabilities is not None:
            for p in self.opinion_probabilities:
                if not (0.0 <= p <= 1.0):
                    raise ValueError(
                        f"Each opinion probability must be in [0, 1], got {p}"
                    )
            total = sum(self.opinion_probabilities)
            if total > 1.0 + 1e-10:
                raise ValueError(
                    f"opinion_probabilities must sum to at most 1.0, got {total}"
                )
