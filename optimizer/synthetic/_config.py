"""Configuration for synthetic data generation and vine copula models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DependenceMethodType(str, Enum):
    """Dependence method for vine copula tree construction.

    Maps to :class:`skfolio.distribution.DependenceMethod`.
    """

    KENDALL_TAU = "kendall_tau"
    MUTUAL_INFORMATION = "mutual_information"
    WASSERSTEIN_DISTANCE = "wasserstein_distance"


class SelectionCriterionType(str, Enum):
    """Information criterion for copula family selection.

    Maps to :class:`skfolio.distribution.SelectionCriterion`.
    """

    AIC = "aic"
    BIC = "bic"


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VineCopulaConfig:
    """Immutable configuration for :class:`skfolio.distribution.VineCopula`.

    Vine copulas decompose a multivariate distribution into marginal
    distributions and bivariate copulas organised in a tree structure.

    All fields are frozen-serialisable. ``marginal_candidates`` and
    ``copula_candidates`` carry skfolio class names (resolved to
    estimator instances at factory call time); ``central_assets``
    carries asset-symbol strings.

    Parameters
    ----------
    fit_marginals : bool
        Whether to fit univariate marginals.
    max_depth : int
        Maximum depth of the vine tree.
    log_transform : bool
        Whether to apply log transformation.
    dependence_method : DependenceMethodType
        Method for measuring pairwise dependence when building
        the vine structure.
    selection_criterion : SelectionCriterionType
        Information criterion for selecting copula families.
    independence_level : float
        Significance level for independence testing.
    n_jobs : int or None
        Number of parallel jobs.
    random_state : int or None
        Random state for reproducibility.
    marginal_candidates : tuple[str, ...] or None
        Names of skfolio univariate distribution classes
        (e.g. ``("Gaussian", "StudentT")``). ``None`` defers to
        skfolio default.
    copula_candidates : tuple[str, ...] or None
        Names of skfolio bivariate copula classes
        (e.g. ``("ClaytonCopula", "GaussianCopula")``). ``None``
        defers to skfolio default.
    central_assets : tuple[str, ...] or None
        Asset symbols treated as central nodes in vine
        construction. ``None`` lets skfolio choose.
    """

    fit_marginals: bool = True
    max_depth: int = 4
    log_transform: bool = False
    dependence_method: DependenceMethodType = DependenceMethodType.KENDALL_TAU
    selection_criterion: SelectionCriterionType = SelectionCriterionType.AIC
    independence_level: float = 0.05
    n_jobs: int | None = None
    random_state: int | None = None
    marginal_candidates: tuple[str, ...] | None = None
    copula_candidates: tuple[str, ...] | None = None
    central_assets: tuple[str, ...] | None = None

    @classmethod
    def for_with_t_marginals(cls) -> VineCopulaConfig:
        """Preset including Student-t marginals alongside Gaussian."""
        return cls(marginal_candidates=("Gaussian", "StudentT"))

    @classmethod
    def for_clayton_only(cls) -> VineCopulaConfig:
        """Preset restricting copula candidates to Clayton (lower-tail)."""
        return cls(copula_candidates=("ClaytonCopula",))


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Immutable configuration for :class:`skfolio.prior.SyntheticData`.

    Generates synthetic return scenarios from a fitted distribution
    model (typically a vine copula).  Supports conditional stress
    testing via the ``sample_args`` factory parameter.

    Non-serialisable objects (``distribution_estimator``,
    ``sample_args``) are passed as keyword arguments to the factory
    function.

    Parameters
    ----------
    n_samples : int
        Number of synthetic scenarios to generate.
    vine_copula_config : VineCopulaConfig or None
        Configuration for building a ``VineCopula`` distribution
        estimator.  Ignored when ``distribution_estimator`` is
        passed to the factory directly.
    """

    n_samples: int = 1_000
    vine_copula_config: VineCopulaConfig | None = None

    # -- factory methods -----------------------------------------------------

    @classmethod
    def for_scenario_generation(
        cls,
        n_samples: int = 10_000,
    ) -> SyntheticDataConfig:
        """Large-sample scenario generation with default vine copula."""
        return cls(
            n_samples=n_samples,
            vine_copula_config=VineCopulaConfig(),
        )

    @classmethod
    def for_stress_test(
        cls,
        n_samples: int = 10_000,
    ) -> SyntheticDataConfig:
        """Stress-test configuration (conditioning dict passed to factory).

        Uses BIC for copula selection (penalises complexity more than AIC)
        and deeper vine trees (``max_depth=6``) to capture tail dependence.
        """
        return cls(
            n_samples=n_samples,
            vine_copula_config=VineCopulaConfig(
                selection_criterion=SelectionCriterionType.BIC,
                max_depth=6,
            ),
        )
