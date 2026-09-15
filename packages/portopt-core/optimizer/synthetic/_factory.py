"""Factory functions for building synthetic data and vine copula estimators.

Input contract (real DB data)
-----------------------------
These factories build **data-free** skfolio estimators; the calibration data
is supplied later at ``fit(X)`` time and is validated by skfolio via
``sklearn.utils.validation.validate_data`` with ``ensure_all_finite=True``.
Both :class:`~skfolio.distribution.VineCopula` and
:class:`~skfolio.prior.SyntheticData` therefore **reject NaN/inf** and have no
native NaN mask. When seeding a generator from the DB's real price history,
prepare ``X`` upstream (in ``preprocessing``/``moments``, not here):

* cast ``Numeric`` columns ``Decimal`` -> ``float`` (never pass ``Decimal`` /
  SQL ``NULL``/``None`` objects);
* normalise mixed ``price_unit`` scale/currency before ``prices_to_returns``;
* pass **linear** (simple) returns only;
* drop/align ragged listing history so no leading or interior NaN remain
  (unequal-length series across assets otherwise trip the finite check);
* keep ticker **column names** on the DataFrame -- symbol-keyed conditioning
  (see :func:`build_conditional_synthetic_data`) resolves asset names to
  column positions, so a bare ndarray breaks conditioning by symbol.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from skfolio.distribution import (
    ClaytonCopula,
    DependenceMethod,
    Gaussian,
    GaussianCopula,
    GumbelCopula,
    IndependentCopula,
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

from optimizer.exceptions import ConfigurationError
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

_COPULA_MAP: dict[str, type[BaseDistribution]] = {
    "ClaytonCopula": ClaytonCopula,
    "GaussianCopula": GaussianCopula,
    "GumbelCopula": GumbelCopula,
    "IndependentCopula": IndependentCopula,
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

# ---------------------------------------------------------------------------
# Mapping dicts
# ---------------------------------------------------------------------------

_DEPENDENCE_MAP: dict[DependenceMethodType, DependenceMethod] = {
    DependenceMethodType.KENDALL_TAU: DependenceMethod.KENDALL_TAU,
    DependenceMethodType.MUTUAL_INFORMATION: DependenceMethod.MUTUAL_INFORMATION,
    DependenceMethodType.WASSERSTEIN_DISTANCE: DependenceMethod.WASSERSTEIN_DISTANCE,
}

_SELECTION_MAP: dict[SelectionCriterionType, SelectionCriterion] = {
    SelectionCriterionType.AIC: SelectionCriterion.AIC,
    SelectionCriterionType.BIC: SelectionCriterion.BIC,
}


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_vine_copula(
    config: VineCopulaConfig | None = None,
    **kwargs: Any,
) -> VineCopula:
    """Build a skfolio :class:`VineCopula` from *config*.

    Parameters
    ----------
    config : VineCopulaConfig or None
        Vine copula configuration.  Defaults to
        ``VineCopulaConfig()``.
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`VineCopula` constructor (for non-serialisable
        parameters such as ``marginal_candidates``,
        ``copula_candidates``, ``central_assets``).

    Returns
    -------
    VineCopula
        A fitted-ready skfolio vine copula estimator.
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

    return VineCopula(
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


def build_synthetic_data(
    config: SyntheticDataConfig | None = None,
    *,
    distribution_estimator: VineCopula | None = None,
    sample_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> SyntheticData:
    """Build a skfolio :class:`SyntheticData` prior from *config*.

    Parameters
    ----------
    config : SyntheticDataConfig or None
        Synthetic data configuration.  Defaults to
        ``SyntheticDataConfig()``.
    distribution_estimator : VineCopula or None
        Pre-built distribution estimator.  When ``None``, one is
        built from ``config.vine_copula_config`` (or skfolio default).
    sample_args : dict or None
        Arguments passed to the distribution's ``sample`` method.
        Use ``{"conditioning": {"AAPL": -0.10}}`` for conditional
        stress testing.
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`SyntheticData` constructor.

    Returns
    -------
    SyntheticData
        A fitted-ready skfolio prior estimator generating synthetic
        return scenarios.
    """
    if config is None:
        config = SyntheticDataConfig()

    if distribution_estimator is None and config.vine_copula_config is not None:
        distribution_estimator = build_vine_copula(config.vine_copula_config)

    return SyntheticData(
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
    """Build a conditional (stressed) :class:`SyntheticData` prior.

    Convenience wrapper over :func:`build_synthetic_data` that wires the
    *conditioning* mapping into ``sample_args={"conditioning": ...}`` for
    CCAR-style stress scenarios (e.g. ``{"AAPL": -0.10}`` shocks AAPL by
    -10%).  Conditioning values may be a fixed float, a ``(low, high)``
    bound tuple, or a per-sample 1-D array (kept out of the serialisable
    config on purpose).

    skfolio recommends the conditioned assets be *central* in the vine.
    When ``warn_non_central`` is set, a warning is logged for any
    conditioned asset not present in the estimator's ``central_assets``;
    build the estimator with
    :meth:`SyntheticDataConfig.for_conditional_stress` /
    :meth:`VineCopulaConfig.for_conditional_sampling` to avoid it.

    Parameters
    ----------
    conditioning : mapping of str to float, (float, float) tuple, or array-like
        Per-asset conditioning specification.  Must be non-empty.
    config : SyntheticDataConfig or None
        Synthetic data configuration.  Defaults to ``SyntheticDataConfig()``.
    distribution_estimator : VineCopula or None
        Pre-built distribution estimator (overrides ``config``'s vine).
    warn_non_central : bool
        Emit a warning when a conditioned asset is not central.
    **kwargs
        Forwarded to the :class:`SyntheticData` constructor.

    Returns
    -------
    SyntheticData
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
