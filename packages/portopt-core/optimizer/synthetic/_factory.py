"""Factory functions for building synthetic data and vine copula estimators.

Input contract (real DB data):
These factories build **data-free** skfolio estimators; the calibration data
is supplied later at ``fit(X)`` time and is validated by skfolio via
``sklearn.utils.validation.validate_data`` with ``ensure_all_finite=True``.
Both `VineCopula` and `SyntheticData` therefore **reject NaN/inf** and have no
native NaN mask. When seeding a generator from the DB's real price history,
prepare ``X`` upstream (in ``preprocessing``/``moments``, not here):

* cast ``Numeric`` columns ``Decimal`` -> ``float`` (never pass ``Decimal`` /
  SQL ``NULL``/``None`` objects);
* normalise mixed ``price_unit`` scale/currency before ``prices_to_returns``;
* pass **linear** (simple) returns only, each strictly greater than -1: the
  default ``VineCopulaConfig(log_transform=True)`` fits on ``log1p(R)``, which
  is ``-inf`` at exactly -100% (a delisting to zero) and NaN below it, so
  `VineCopula.fit` raises `DataError` -- clip such observations above -1
  upstream or build the vine with ``log_transform=False``;
* drop/align ragged listing history so no leading or interior NaN remain
  (unequal-length series across assets otherwise trip the finite check);
* keep ticker **column names** on the DataFrame -- symbol-keyed conditioning
  (see `build_conditional_synthetic_data`) resolves asset names to
  column positions, so a bare ndarray breaks conditioning by symbol.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np
from skfolio.distribution import (
    ClaytonCopula,
    DependenceMethod,
    Gaussian,
    GaussianCopula,
    GumbelCopula,
    JoeCopula,
    JohnsonSU,
    NormalInverseGaussian,
    SelectionCriterion,
    StudentT,
    StudentTCopula,
    VineCopula,
)
from skfolio.distribution._base import BaseDistribution
from skfolio.prior import SyntheticData

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.synthetic._config import (
    DependenceMethodType,
    SelectionCriterionType,
    SyntheticDataConfig,
    VineCopulaConfig,
)

_MARGINAL_MAP: dict[str, type[BaseDistribution]] = {
    "Gaussian": Gaussian,
    "StudentT": StudentT,
    "JohnsonSU": JohnsonSU,
    "NormalInverseGaussian": NormalInverseGaussian,
}

# IndependentCopula is deliberately absent: skfolio applies it internally to
# edges that pass the independence test, and as a user candidate it crashes
# select_bivariate_copula (it has no ``itau`` attribute).
_COPULA_MAP: dict[str, type[BaseDistribution]] = {
    "ClaytonCopula": ClaytonCopula,
    "GaussianCopula": GaussianCopula,
    "GumbelCopula": GumbelCopula,
    "JoeCopula": JoeCopula,
    "StudentTCopula": StudentTCopula,
}


def _resolve_distributions(
    names: tuple[str, ...] | None,
    mapping: dict[str, type[BaseDistribution]],
    kind: str,
) -> list[BaseDistribution] | None:
    """Resolve a tuple of distribution names to fresh estimator instances."""
    if names is None:
        return None
    instances: list[BaseDistribution] = []
    for name in names:
        if name not in mapping:
            valid = ", ".join(sorted(mapping))
            raise ConfigurationError(
                f"Unknown {kind} candidate {name!r}. Valid: {valid}"
            )
        instances.append(mapping[name]())
    return instances


logger = logging.getLogger(__name__)

_DEPENDENCE_MAP: dict[DependenceMethodType, DependenceMethod] = {
    DependenceMethodType.KENDALL_TAU: DependenceMethod.KENDALL_TAU,
    DependenceMethodType.MUTUAL_INFORMATION: DependenceMethod.MUTUAL_INFORMATION,
    DependenceMethodType.WASSERSTEIN_DISTANCE: DependenceMethod.WASSERSTEIN_DISTANCE,
}

_SELECTION_MAP: dict[SelectionCriterionType, SelectionCriterion] = {
    SelectionCriterionType.AIC: SelectionCriterion.AIC,
    SelectionCriterionType.BIC: SelectionCriterion.BIC,
}


class _LogReturnVineCopula(VineCopula):
    """`VineCopula` that names ``log_transform`` when a return breaks ``log1p``.

    ``log1p(R)`` is ``-inf`` at R == -1 and NaN below it, so skfolio's generic
    finite-input check would fail with a message that never mentions the option
    responsible. Raising here points the caller at the offending observations
    and at the opt-out.
    """

    def fit(self, X: Any, y: None = None) -> VineCopula:
        # A per-asset dict/array log_transform (only reachable through factory
        # kwargs) needs skfolio's fitted feature names to map onto columns, so
        # it keeps skfolio's own validation.
        if np.isscalar(self.log_transform) and self.log_transform:
            returns = np.asarray(X, dtype=float)
            n_invalid = int(np.count_nonzero(returns <= -1.0))
            if n_invalid:
                raise DataError(
                    f"log_transform=True needs every return to be > -1 (log1p "
                    f"is undefined at and below -100%), found {n_invalid} "
                    f"observation(s) <= -1 (minimum {np.nanmin(returns):.4g}). "
                    "Clip them above -1 upstream or build the vine with "
                    "VineCopulaConfig(log_transform=False)."
                )
        return super().fit(X, y)


def build_vine_copula(
    config: VineCopulaConfig | None = None,
    **kwargs: Any,
) -> VineCopula:
    """Build a skfolio `VineCopula` from `config`.

    Args:
        config: Vine copula configuration. Defaults to ``VineCopulaConfig()``.
        **kwargs: Additional keyword arguments forwarded to the
            `VineCopula` constructor (for non-serialisable
            parameters such as ``marginal_candidates``,
            ``copula_candidates``, ``central_assets``).

    Returns:
        A fitted-ready skfolio vine copula estimator. With
        ``config.log_transform`` set, ``fit`` raises `DataError` when any
        return is <= -1.
    """
    if config is None:
        config = VineCopulaConfig()

    kwargs.setdefault(
        "marginal_candidates",
        _resolve_distributions(config.marginal_candidates, _MARGINAL_MAP, "marginal"),
    )
    kwargs.setdefault(
        "copula_candidates",
        _resolve_distributions(config.copula_candidates, _COPULA_MAP, "copula"),
    )
    kwargs.setdefault(
        "central_assets",
        list(config.central_assets) if config.central_assets is not None else None,
    )

    return _LogReturnVineCopula(
        fit_marginals=config.fit_marginals,
        max_depth=config.max_depth,
        log_transform=config.log_transform,
        dependence_method=_DEPENDENCE_MAP[config.dependence_method],
        selection_criterion=_SELECTION_MAP[config.selection_criterion],
        independence_level=config.independence_level,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        **kwargs,
    )


class _SeededSyntheticData(SyntheticData):
    """`SyntheticData` whose conditional jitter honours the vine's seed.

    skfolio adds ``1e-6 * np.random.randn`` noise to the constant columns that
    conditioning produces, drawing from the *global* NumPy generator instead of
    ``random_state``. Fixed-seed stress runs would otherwise differ in the
    conditioned column from one fit to the next. When the distribution
    estimator carries an integer seed, the global generator is seeded for the
    duration of ``fit`` and its previous state restored afterwards, so
    callers' own global-RNG streams are left untouched.
    """

    def fit(self, X: Any, y: None = None, **fit_params: Any) -> SyntheticData:
        seed = getattr(self.distribution_estimator, "random_state", None)
        if not isinstance(seed, Integral):
            return super().fit(X, y, **fit_params)
        saved_state = np.random.get_state()
        try:
            np.random.seed(int(seed))
            return super().fit(X, y, **fit_params)
        finally:
            np.random.set_state(saved_state)


def build_synthetic_data(
    config: SyntheticDataConfig | None = None,
    *,
    distribution_estimator: VineCopula | None = None,
    sample_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> SyntheticData:
    """Build a skfolio `SyntheticData` prior from `config`.

    Args:
        config: Synthetic data configuration. Defaults to ``SyntheticDataConfig()``.
        distribution_estimator: Pre-built distribution estimator. When ``None``, one is
            built from ``config.vine_copula_config`` (or skfolio default).
        sample_args: Arguments passed to the distribution's ``sample`` method.
            Use ``{"conditioning": {"AAPL": -0.10}}`` for conditional stress testing.
        **kwargs: Additional keyword arguments forwarded to the
            `SyntheticData` constructor.

    Returns:
        A fitted-ready skfolio prior estimator generating synthetic return scenarios.
    """
    if config is None:
        config = SyntheticDataConfig()

    if distribution_estimator is None and config.vine_copula_config is not None:
        distribution_estimator = build_vine_copula(config.vine_copula_config)

    return _SeededSyntheticData(
        distribution_estimator=distribution_estimator,
        n_samples=config.n_samples,
        sample_args=sample_args,
        **kwargs,
    )


def build_conditional_synthetic_data(
    conditioning: Mapping[str, Any],
    config: SyntheticDataConfig | None = None,
    *,
    distribution_estimator: VineCopula | None = None,
    warn_non_central: bool = True,
    **kwargs: Any,
) -> SyntheticData:
    """Build a conditional (stressed) `SyntheticData` prior.

    Convenience wrapper over `build_synthetic_data` that wires the
    `conditioning` mapping into ``sample_args={"conditioning": ...}`` for
    CCAR-style stress scenarios (e.g. ``{"AAPL": -0.10}`` shocks AAPL by
    -10%).  Conditioning values may be a fixed float, a ``(low, high)``
    bound tuple, or a per-sample 1-D array (kept out of the serialisable
    config on purpose).

    skfolio recommends the conditioned assets be `central` in the vine.
    When ``warn_non_central`` is set, a warning is logged for any
    conditioned asset not present in the estimator's ``central_assets``;
    build the estimator with `for_conditional_stress` /
    `for_conditional_sampling` to avoid it.

    Args:
        conditioning: Per-asset conditioning specification. Must be non-empty.
        config: Synthetic data configuration. Defaults to ``SyntheticDataConfig()``.
        distribution_estimator: Pre-built distribution estimator (overrides the vine
            in ``config``).
        warn_non_central: Emit a warning when a conditioned asset is not central.
        **kwargs: Forwarded to the `SyntheticData` constructor.

    Returns:
        A conditional synthetic-data prior estimator.
    """
    if not conditioning:
        raise ConfigurationError(
            "conditioning must be a non-empty mapping of asset -> shock"
        )

    model = build_synthetic_data(
        config,
        distribution_estimator=distribution_estimator,
        sample_args={"conditioning": dict(conditioning)},
        **kwargs,
    )

    if warn_non_central:
        estimator = model.distribution_estimator
        central = getattr(estimator, "central_assets", None) or []
        non_central = [asset for asset in conditioning if asset not in central]
        if non_central:
            logger.warning(
                "Conditioning on non-central asset(s) %s; conditional "
                "sampling is faster and more accurate when these are marked "
                "central during vine construction (see "
                "SyntheticDataConfig.for_conditional_stress / "
                "VineCopulaConfig.for_conditional_sampling).",
                non_central,
            )

    return model
