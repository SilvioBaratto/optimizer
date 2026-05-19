"""Configuration for skfolio uncertainty-set estimators.

Two parallel Configs (mu, covariance) carry the same field surface and
dispatch to the matching skfolio class via :class:`MuUncertaintySetType`
and :class:`CovarianceUncertaintySetType` enums.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from optimizer.exceptions import ConfigurationError


class MuUncertaintySetType(str, Enum):
    """Mu uncertainty-set selection."""

    EMPIRICAL = "empirical"
    BOOTSTRAP = "bootstrap"


class CovarianceUncertaintySetType(str, Enum):
    """Covariance uncertainty-set selection."""

    EMPIRICAL = "empirical"
    BOOTSTRAP = "bootstrap"


def _validate_bootstrap_only_fields(
    *,
    is_bootstrap: bool,
    n_bootstrap_samples: int,
    block_size: float | None,
    random_state: int | None,
) -> None:
    """Reject bootstrap-only fields when kind is EMPIRICAL.

    Parameters
    ----------
    is_bootstrap : bool
        Whether the parent config selects the bootstrap variant.
    n_bootstrap_samples : int
        Bootstrap sample count; default ``1000``.
    block_size : float or None
        Politis-White block size; ``None`` triggers auto.
    random_state : int or None
        Bootstrap RNG seed.
    """
    if is_bootstrap:
        return
    if n_bootstrap_samples != 1000:
        raise ConfigurationError("n_bootstrap_samples is only valid for BOOTSTRAP kind")
    if block_size is not None:
        raise ConfigurationError("block_size is only valid for BOOTSTRAP kind")
    if random_state is not None:
        raise ConfigurationError("random_state is only valid for BOOTSTRAP kind")


@dataclass(frozen=True)
class MuUncertaintySetConfig:
    """Immutable configuration for a mu uncertainty-set estimator.

    Parameters
    ----------
    kind : MuUncertaintySetType
        Empirical or bootstrap uncertainty set.
    confidence_level : float
        Confidence ball level in :math:`(0, 1)`. Default ``0.95``.
    n_bootstrap_samples : int
        Stationary bootstrap sample count. Bootstrap-only.
    block_size : float or None
        Stationary bootstrap mean block size; ``None`` defaults to the
        Politis-White rule of thumb. Bootstrap-only.
    random_state : int or None
        RNG seed for the stationary bootstrap. Bootstrap-only. Mapped to
        ``seed`` when forwarded to skfolio.
    """

    kind: MuUncertaintySetType = MuUncertaintySetType.EMPIRICAL
    confidence_level: float = 0.95
    n_bootstrap_samples: int = 1000
    block_size: float | None = None
    random_state: int | None = None

    def __post_init__(self) -> None:
        _validate_bootstrap_only_fields(
            is_bootstrap=self.kind == MuUncertaintySetType.BOOTSTRAP,
            n_bootstrap_samples=self.n_bootstrap_samples,
            block_size=self.block_size,
            random_state=self.random_state,
        )

    @classmethod
    def for_empirical(
        cls,
        confidence_level: float = 0.95,
    ) -> MuUncertaintySetConfig:
        """Empirical mu uncertainty-set preset."""
        return cls(
            kind=MuUncertaintySetType.EMPIRICAL,
            confidence_level=confidence_level,
        )

    @classmethod
    def for_bootstrap(
        cls,
        confidence_level: float = 0.95,
        n_bootstrap_samples: int = 1000,
        block_size: float | None = None,
        random_state: int | None = None,
    ) -> MuUncertaintySetConfig:
        """Stationary bootstrap mu uncertainty-set preset."""
        return cls(
            kind=MuUncertaintySetType.BOOTSTRAP,
            confidence_level=confidence_level,
            n_bootstrap_samples=n_bootstrap_samples,
            block_size=block_size,
            random_state=random_state,
        )


@dataclass(frozen=True)
class CovarianceUncertaintySetConfig:
    """Immutable configuration for a covariance uncertainty-set estimator.

    Parameters
    ----------
    kind : CovarianceUncertaintySetType
        Empirical or bootstrap uncertainty set.
    confidence_level : float
        Confidence ball level in :math:`(0, 1)`. Default ``0.95``.
    n_bootstrap_samples : int
        Stationary bootstrap sample count. Bootstrap-only.
    block_size : float or None
        Stationary bootstrap mean block size; ``None`` defaults to the
        Politis-White rule of thumb. Bootstrap-only.
    random_state : int or None
        RNG seed for the stationary bootstrap. Bootstrap-only. Mapped to
        ``seed`` when forwarded to skfolio.
    """

    kind: CovarianceUncertaintySetType = CovarianceUncertaintySetType.EMPIRICAL
    confidence_level: float = 0.95
    n_bootstrap_samples: int = 1000
    block_size: float | None = None
    random_state: int | None = None

    def __post_init__(self) -> None:
        _validate_bootstrap_only_fields(
            is_bootstrap=self.kind == CovarianceUncertaintySetType.BOOTSTRAP,
            n_bootstrap_samples=self.n_bootstrap_samples,
            block_size=self.block_size,
            random_state=self.random_state,
        )

    @classmethod
    def for_empirical(
        cls,
        confidence_level: float = 0.95,
    ) -> CovarianceUncertaintySetConfig:
        """Empirical covariance uncertainty-set preset."""
        return cls(
            kind=CovarianceUncertaintySetType.EMPIRICAL,
            confidence_level=confidence_level,
        )

    @classmethod
    def for_bootstrap(
        cls,
        confidence_level: float = 0.95,
        n_bootstrap_samples: int = 1000,
        block_size: float | None = None,
        random_state: int | None = None,
    ) -> CovarianceUncertaintySetConfig:
        """Stationary bootstrap covariance uncertainty-set preset."""
        return cls(
            kind=CovarianceUncertaintySetType.BOOTSTRAP,
            confidence_level=confidence_level,
            n_bootstrap_samples=n_bootstrap_samples,
            block_size=block_size,
            random_state=random_state,
        )
