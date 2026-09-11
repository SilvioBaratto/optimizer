"""Configuration for skfolio uncertainty-set estimators.

Two parallel Configs (mu, covariance) carry the same field surface and
dispatch to the matching skfolio class via :class:`MuUncertaintySetType`
and :class:`CovarianceUncertaintySetType` enums.

Three variants are supported per side:

* ``EMPIRICAL`` — closed-form confidence ellipsoid / box around the
  empirical estimate (``EmpiricalMuUncertaintySet`` /
  ``EmpiricalCovarianceUncertaintySet``).
* ``BOOTSTRAP`` — stationary-bootstrap confidence region
  (``BootstrapMuUncertaintySet`` / ``BootstrapCovarianceUncertaintySet``).
  Calls ``arch.StationaryBootstrap`` under the hood; ``block_size=None``
  triggers the Politis-White rule of thumb.
* ``ORTHOGONAL`` — factor-model orthogonal-subspace sets
  (``OrthogonalMuUncertaintySet`` / ``OrthogonalCovarianceUncertaintySet``,
  new in skfolio 1.0). These require a factor-model return distribution
  at fit time (supplied automatically by ``MeanRisk`` when its
  ``prior_estimator`` is a factor model).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from optimizer.exceptions import ConfigurationError


class MuUncertaintySetType(str, Enum):
    """Mu uncertainty-set selection."""

    EMPIRICAL = "empirical"
    BOOTSTRAP = "bootstrap"
    ORTHOGONAL = "orthogonal"


class CovarianceUncertaintySetType(str, Enum):
    """Covariance uncertainty-set selection."""

    EMPIRICAL = "empirical"
    BOOTSTRAP = "bootstrap"
    ORTHOGONAL = "orthogonal"


class CrossSectionalWeighting(str, Enum):
    """Cross-sectional metric used to define the orthogonality of an
    orthogonal uncertainty set (mirrors skfolio ``CSWeighting``)."""

    BENCHMARK = "benchmark"
    REGRESSION = "regression"
    INVERSE_IDIO_VARIANCE = "inverse_idio_variance"
    IDENTITY = "identity"


class OrthogonalUncertaintyShape(str, Enum):
    """Shape used inside the orthogonal subspace of an
    ``OrthogonalMuUncertaintySet``."""

    IDENTITY = "identity"
    IDIO_VARIANCE = "idio_variance"


_DEFAULT_CS_WEIGHTING = CrossSectionalWeighting.INVERSE_IDIO_VARIANCE


def _validate_shared_fields(
    *,
    is_empirical: bool,
    is_bootstrap: bool,
    is_orthogonal: bool,
    n_bootstrap_samples: int,
    block_size: float | None,
    random_state: int | None,
    n_eff: float | None,
    diagonal: bool,
    cs_weighting: CrossSectionalWeighting,
) -> None:
    """Reject kind-specific fields set on the wrong variant.

    Parameters
    ----------
    is_empirical, is_bootstrap, is_orthogonal : bool
        Exactly one is ``True`` — the selected variant.
    n_bootstrap_samples : int
        Bootstrap sample count; default ``1000``. Bootstrap-only.
    block_size : float or None
        Politis-White block size; ``None`` triggers auto. Bootstrap-only.
    random_state : int or None
        Bootstrap RNG seed. Bootstrap-only.
    n_eff : float or None
        Effective number of observations. Empirical-only.
    diagonal : bool
        Whether the set is diagonal (box) rather than a full ellipsoid.
        Empirical / bootstrap only.
    cs_weighting : CrossSectionalWeighting
        Orthogonality metric. Orthogonal-only.
    """
    if not is_bootstrap:
        if n_bootstrap_samples != 1000:
            raise ConfigurationError(
                "n_bootstrap_samples is only valid for BOOTSTRAP kind"
            )
        if block_size is not None:
            raise ConfigurationError("block_size is only valid for BOOTSTRAP kind")
        if random_state is not None:
            raise ConfigurationError("random_state is only valid for BOOTSTRAP kind")
    if n_eff is not None and not is_empirical:
        raise ConfigurationError("n_eff is only valid for EMPIRICAL kind")
    if diagonal is not True and is_orthogonal:
        raise ConfigurationError(
            "diagonal is only valid for EMPIRICAL and BOOTSTRAP kinds"
        )
    if cs_weighting != _DEFAULT_CS_WEIGHTING and not is_orthogonal:
        raise ConfigurationError("cs_weighting is only valid for ORTHOGONAL kind")


@dataclass(frozen=True)
class MuUncertaintySetConfig:
    """Immutable configuration for a mu uncertainty-set estimator.

    Parameters
    ----------
    kind : MuUncertaintySetType
        Empirical, bootstrap, or orthogonal uncertainty set.
    confidence_level : float
        Confidence ball level in :math:`(0, 1)`. Default ``0.95``. Used by
        every variant (the orthogonal set converts it to a chi-squared
        radius).
    diagonal : bool
        Whether the empirical/bootstrap set is diagonal (box, ``True``,
        default) or a full ellipsoid (``False``). Empirical/bootstrap only.
    n_eff : float or None
        Effective number of observations overriding the sample count in the
        empirical confidence radius. Empirical-only.
    n_bootstrap_samples : int
        Stationary bootstrap sample count. Bootstrap-only.
    block_size : float or None
        Stationary bootstrap mean block size; ``None`` defaults to the
        Politis-White rule of thumb. Bootstrap-only.
    random_state : int or None
        RNG seed for the stationary bootstrap. Bootstrap-only. Mapped to
        ``seed`` when forwarded to skfolio.
    cs_weighting : CrossSectionalWeighting
        Cross-sectional orthogonality metric. Orthogonal-only.
    uncertainty_shape : OrthogonalUncertaintyShape
        Shape used inside the orthogonal subspace. Orthogonal-only.
    """

    kind: MuUncertaintySetType = MuUncertaintySetType.EMPIRICAL
    confidence_level: float = 0.95
    diagonal: bool = True
    n_eff: float | None = None
    n_bootstrap_samples: int = 1000
    block_size: float | None = None
    random_state: int | None = None
    cs_weighting: CrossSectionalWeighting = _DEFAULT_CS_WEIGHTING
    uncertainty_shape: OrthogonalUncertaintyShape = OrthogonalUncertaintyShape.IDENTITY

    def __post_init__(self) -> None:
        is_orthogonal = self.kind == MuUncertaintySetType.ORTHOGONAL
        _validate_shared_fields(
            is_empirical=self.kind == MuUncertaintySetType.EMPIRICAL,
            is_bootstrap=self.kind == MuUncertaintySetType.BOOTSTRAP,
            is_orthogonal=is_orthogonal,
            n_bootstrap_samples=self.n_bootstrap_samples,
            block_size=self.block_size,
            random_state=self.random_state,
            n_eff=self.n_eff,
            diagonal=self.diagonal,
            cs_weighting=self.cs_weighting,
        )
        if (
            self.uncertainty_shape != OrthogonalUncertaintyShape.IDENTITY
            and not is_orthogonal
        ):
            raise ConfigurationError(
                "uncertainty_shape is only valid for ORTHOGONAL kind"
            )

    @classmethod
    def for_empirical(
        cls,
        confidence_level: float = 0.95,
        diagonal: bool = True,
        n_eff: float | None = None,
    ) -> MuUncertaintySetConfig:
        """Empirical mu uncertainty-set preset."""
        return cls(
            kind=MuUncertaintySetType.EMPIRICAL,
            confidence_level=confidence_level,
            diagonal=diagonal,
            n_eff=n_eff,
        )

    @classmethod
    def for_bootstrap(
        cls,
        confidence_level: float = 0.95,
        n_bootstrap_samples: int = 1000,
        block_size: float | None = None,
        random_state: int | None = None,
        diagonal: bool = True,
    ) -> MuUncertaintySetConfig:
        """Stationary bootstrap mu uncertainty-set preset."""
        return cls(
            kind=MuUncertaintySetType.BOOTSTRAP,
            confidence_level=confidence_level,
            n_bootstrap_samples=n_bootstrap_samples,
            block_size=block_size,
            random_state=random_state,
            diagonal=diagonal,
        )

    @classmethod
    def for_orthogonal(
        cls,
        confidence_level: float = 0.95,
        cs_weighting: CrossSectionalWeighting = _DEFAULT_CS_WEIGHTING,
        uncertainty_shape: OrthogonalUncertaintyShape = (
            OrthogonalUncertaintyShape.IDENTITY
        ),
    ) -> MuUncertaintySetConfig:
        """Orthogonal (factor-model) mu uncertainty-set preset."""
        return cls(
            kind=MuUncertaintySetType.ORTHOGONAL,
            confidence_level=confidence_level,
            cs_weighting=cs_weighting,
            uncertainty_shape=uncertainty_shape,
        )


@dataclass(frozen=True)
class CovarianceUncertaintySetConfig:
    """Immutable configuration for a covariance uncertainty-set estimator.

    Parameters
    ----------
    kind : CovarianceUncertaintySetType
        Empirical, bootstrap, or orthogonal uncertainty set.
    confidence_level : float
        Confidence ball level in :math:`(0, 1)`. Default ``0.95``. Used by
        the empirical and bootstrap variants; the orthogonal covariance set
        is parameterised by ``radius`` instead.
    diagonal : bool
        Whether the empirical/bootstrap set is diagonal (box, ``True``,
        default) or a full ellipsoid (``False``). Empirical/bootstrap only.
    n_eff : float or None
        Effective number of observations overriding the sample count in the
        empirical confidence radius. Empirical-only.
    n_bootstrap_samples : int
        Stationary bootstrap sample count. Bootstrap-only.
    block_size : float or None
        Stationary bootstrap mean block size; ``None`` defaults to the
        Politis-White rule of thumb. Bootstrap-only.
    random_state : int or None
        RNG seed for the stationary bootstrap. Bootstrap-only. Mapped to
        ``seed`` when forwarded to skfolio.
    cs_weighting : CrossSectionalWeighting
        Cross-sectional orthogonality metric. Orthogonal-only.
    radius : float
        Radius :math:`\\kappa` of the orthogonal covariance ball.
        Orthogonal-only. Default ``1.0``.
    """

    kind: CovarianceUncertaintySetType = CovarianceUncertaintySetType.EMPIRICAL
    confidence_level: float = 0.95
    diagonal: bool = True
    n_eff: float | None = None
    n_bootstrap_samples: int = 1000
    block_size: float | None = None
    random_state: int | None = None
    cs_weighting: CrossSectionalWeighting = _DEFAULT_CS_WEIGHTING
    radius: float = 1.0

    def __post_init__(self) -> None:
        is_orthogonal = self.kind == CovarianceUncertaintySetType.ORTHOGONAL
        _validate_shared_fields(
            is_empirical=self.kind == CovarianceUncertaintySetType.EMPIRICAL,
            is_bootstrap=self.kind == CovarianceUncertaintySetType.BOOTSTRAP,
            is_orthogonal=is_orthogonal,
            n_bootstrap_samples=self.n_bootstrap_samples,
            block_size=self.block_size,
            random_state=self.random_state,
            n_eff=self.n_eff,
            diagonal=self.diagonal,
            cs_weighting=self.cs_weighting,
        )
        if self.radius != 1.0 and not is_orthogonal:
            raise ConfigurationError("radius is only valid for ORTHOGONAL kind")
        if self.confidence_level != 0.95 and is_orthogonal:
            raise ConfigurationError(
                "confidence_level is not valid for ORTHOGONAL covariance kind; "
                "use radius instead"
            )

    @classmethod
    def for_empirical(
        cls,
        confidence_level: float = 0.95,
        diagonal: bool = True,
        n_eff: float | None = None,
    ) -> CovarianceUncertaintySetConfig:
        """Empirical covariance uncertainty-set preset."""
        return cls(
            kind=CovarianceUncertaintySetType.EMPIRICAL,
            confidence_level=confidence_level,
            diagonal=diagonal,
            n_eff=n_eff,
        )

    @classmethod
    def for_bootstrap(
        cls,
        confidence_level: float = 0.95,
        n_bootstrap_samples: int = 1000,
        block_size: float | None = None,
        random_state: int | None = None,
        diagonal: bool = True,
    ) -> CovarianceUncertaintySetConfig:
        """Stationary bootstrap covariance uncertainty-set preset."""
        return cls(
            kind=CovarianceUncertaintySetType.BOOTSTRAP,
            confidence_level=confidence_level,
            n_bootstrap_samples=n_bootstrap_samples,
            block_size=block_size,
            random_state=random_state,
            diagonal=diagonal,
        )

    @classmethod
    def for_orthogonal(
        cls,
        radius: float = 1.0,
        cs_weighting: CrossSectionalWeighting = _DEFAULT_CS_WEIGHTING,
    ) -> CovarianceUncertaintySetConfig:
        """Orthogonal (factor-model) covariance uncertainty-set preset."""
        return cls(
            kind=CovarianceUncertaintySetType.ORTHOGONAL,
            radius=radius,
            cs_weighting=cs_weighting,
        )
