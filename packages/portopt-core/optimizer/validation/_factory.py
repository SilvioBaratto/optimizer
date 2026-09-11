"""Factory functions for building skfolio cross-validators."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
    optimal_folds_number,
)
from skfolio.model_selection import (
    cross_val_predict as _skfolio_cross_val_predict,
)
from skfolio.portfolio import FailedPortfolio

from optimizer.validation._config import (
    CPCVConfig,
    MultipleRandomizedCVConfig,
    WalkForwardConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-validator factories
# ---------------------------------------------------------------------------


def build_walk_forward(
    config: WalkForwardConfig | None = None,
) -> WalkForward:
    """Build a skfolio :class:`WalkForward` cross-validator from *config*.

    Parameters
    ----------
    config : WalkForwardConfig or None
        Walk-forward configuration.  Defaults to
        ``WalkForwardConfig()`` (quarterly rolling with one-year
        training window).

    Returns
    -------
    WalkForward
        A skfolio temporal cross-validator.
    """
    if config is None:
        config = WalkForwardConfig()

    freq_offset = (
        pd.tseries.frequencies.to_offset(config.freq_offset)
        if config.freq_offset is not None
        else None
    )

    return WalkForward(
        test_size=config.test_size,
        train_size=config.train_size,
        purged_size=config.purged_size,
        expand_train=config.expend_train,
        reduce_test=config.reduce_test,
        freq=config.freq,
        freq_offset=freq_offset,
        previous=config.previous,
    )


def build_cpcv(
    config: CPCVConfig | None = None,
) -> CombinatorialPurgedCV:
    """Build a skfolio :class:`CombinatorialPurgedCV` cross-validator from *config*.

    Parameters
    ----------
    config : CPCVConfig or None
        CPCV configuration.  Defaults to ``CPCVConfig()``
        (10 folds, 8 test folds).

    Returns
    -------
    CombinatorialPurgedCV
        A skfolio combinatorial purged cross-validator.
    """
    if config is None:
        config = CPCVConfig()

    return CombinatorialPurgedCV(
        n_folds=config.n_folds,
        n_test_folds=config.n_test_folds,
        purged_size=config.purged_size,
        embargo_size=config.embargo_size,
    )


def build_multiple_randomized_cv(
    config: MultipleRandomizedCVConfig | None = None,
) -> MultipleRandomizedCV:
    """Build a :class:`MultipleRandomizedCV` cross-validator from *config*.

    Parameters
    ----------
    config : MultipleRandomizedCVConfig or None
        Multiple randomised CV configuration.  Defaults to
        ``MultipleRandomizedCVConfig()``.

    Returns
    -------
    MultipleRandomizedCV
        A skfolio multi-randomised cross-validator.
    """
    if config is None:
        config = MultipleRandomizedCVConfig()

    wf = build_walk_forward(config.walk_forward_config)

    return MultipleRandomizedCV(
        walk_forward=wf,
        n_subsamples=config.n_subsamples,
        asset_subset_size=config.asset_subset_size,
        window_size=config.window_size,
        random_state=config.random_state,
    )


# ---------------------------------------------------------------------------
# cross_val_predict convenience wrapper
# ---------------------------------------------------------------------------


def run_cross_val(
    estimator: Any,
    X: Any,
    *,
    cv: WalkForward | CombinatorialPurgedCV | MultipleRandomizedCV | None = None,
    y: Any | None = None,
    params: dict[str, Any] | None = None,
    n_jobs: int | None = None,
    portfolio_params: dict[str, Any] | None = None,
) -> Any:
    """Run cross-validated prediction with a temporal cross-validator.

    Thin wrapper around :func:`skfolio.model_selection.cross_val_predict`
    that enforces temporal splitting (no random shuffle).

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted-ready skfolio optimisation estimator or pipeline.
    X : array-like
        Return matrix (observations x assets).
    cv : temporal cross-validator or None
        Cross-validator.  Defaults to ``WalkForward`` with quarterly
        test windows.
    y : array-like or None
        Benchmark returns or factor returns (for models that
        require ``fit(X, y)``).
    params : dict or None
        Auxiliary metadata forwarded to nested estimators via sklearn
        metadata routing (e.g. ``{"implied_vol": implied_vol_df}``).
        Requires ``sklearn.set_config(enable_metadata_routing=True)``
        and the relevant ``set_fit_request`` calls on sub-estimators.
    n_jobs : int or None
        Number of parallel jobs.
    portfolio_params : dict or None
        Additional parameters forwarded to the portfolio constructor.

    Returns
    -------
    MultiPeriodPortfolio or Population
        Out-of-sample portfolio predictions.  ``WalkForward`` returns
        a ``MultiPeriodPortfolio``; ``CombinatorialPurgedCV`` and
        ``MultipleRandomizedCV`` return a ``Population``.
    """
    if cv is None:
        cv = build_walk_forward()

    return _skfolio_cross_val_predict(
        estimator=estimator,
        X=X,
        **({} if y is None else {"y": y}),
        cv=cv,
        params=params,
        n_jobs=n_jobs,
        portfolio_params=portfolio_params,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def compute_optimal_folds(
    n_observations: int,
    target_train_size: int,
    target_n_test_paths: int,
    weight_train_size: float = 1.0,
    weight_n_test_paths: float = 1.0,
) -> tuple[int, int]:
    """Compute optimal fold counts for CPCV.

    Wraps :func:`skfolio.model_selection.optimal_folds_number`.

    Parameters
    ----------
    n_observations : int
        Total number of observations.
    target_train_size : int
        Desired training window size.
    target_n_test_paths : int
        Desired number of backtest paths.
    weight_train_size : float
        Relative importance of matching train size.
    weight_n_test_paths : float
        Relative importance of matching path count.

    Returns
    -------
    tuple[int, int]
        ``(n_folds, n_test_folds)`` optimal parameters.
    """
    return optimal_folds_number(
        n_observations=n_observations,
        target_train_size=target_train_size,
        target_n_test_paths=target_n_test_paths,
        weight_train_size=weight_train_size,
        weight_n_test_paths=weight_n_test_paths,
    )


# ---------------------------------------------------------------------------
# Optimizer resilience layer (skfolio 1.0 fallback / FailedPortfolio)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossValFailureReport:
    """Summary of failed folds in a cross-validated backtest.

    skfolio 1.0 optimizers accept ``fallback=`` / ``raise_on_failure=False``.
    When an optimizer configured with ``raise_on_failure=False`` cannot solve
    a fold, ``cross_val_predict`` yields a
    :class:`skfolio.portfolio.FailedPortfolio` sentinel for that fold instead
    of raising.  This report counts those sentinels so a backtest can surface
    solver fragility rather than silently averaging over failed rebalances.

    Attributes
    ----------
    total : int
        Total number of (leaf) portfolios inspected.
    n_failed : int
        Number of :class:`FailedPortfolio` sentinels.
    failure_rate : float
        ``n_failed / total`` (``0.0`` when ``total`` is zero).
    failures : list[tuple[str, str]]
        ``(portfolio_name, optimization_error)`` for each failed fold.
    """

    total: int
    n_failed: int
    failure_rate: float
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        """``True`` when at least one fold failed to optimize."""
        return self.n_failed > 0


def _iter_leaf_portfolios(pred: Any) -> list[Any]:
    """Flatten a CV prediction into individual (leaf) portfolios.

    Handles the three shapes ``cross_val_predict`` can return:
    a single ``Portfolio``, a ``MultiPeriodPortfolio`` (has ``.portfolios``),
    or a ``Population`` (iterable of ``Portfolio`` / ``MultiPeriodPortfolio``).
    """
    leaves: list[Any] = []
    # A Population is iterable of members; a MultiPeriodPortfolio exposes
    # ``.portfolios``.  A bare Portfolio has neither, so wrap it.
    if hasattr(pred, "portfolios"):
        members: Any = pred.portfolios
    else:
        try:
            members = list(pred)
        except TypeError:
            return [pred]

    for member in members:
        if hasattr(member, "portfolios"):
            leaves.extend(member.portfolios)
        else:
            leaves.append(member)
    return leaves


def summarize_cv_failures(pred: Any) -> CrossValFailureReport:
    """Count failed folds in a cross-validated backtest prediction.

    Works on the output of :func:`run_cross_val` (a
    ``MultiPeriodPortfolio`` for single-path CV or a ``Population`` for
    ``CombinatorialPurgedCV`` / ``MultipleRandomizedCV``).  Use together
    with skfolio 1.0's optimizer resilience layer
    (``MeanRisk(..., raise_on_failure=False, fallback=...)``) so a failed
    rebalance produces a :class:`FailedPortfolio` instead of aborting the
    whole backtest.

    Parameters
    ----------
    pred : Portfolio, MultiPeriodPortfolio or Population
        A cross-validation prediction.

    Returns
    -------
    CrossValFailureReport
        Aggregate failure counts and per-fold error messages.
    """
    leaves = _iter_leaf_portfolios(pred)
    failures = [
        (
            str(getattr(p, "name", "") or ""),
            str(getattr(p, "optimization_error", "") or ""),
        )
        for p in leaves
        if isinstance(p, FailedPortfolio)
    ]
    total = len(leaves)
    n_failed = len(failures)
    failure_rate = n_failed / total if total > 0 else 0.0
    return CrossValFailureReport(
        total=total,
        n_failed=n_failed,
        failure_rate=failure_rate,
        failures=failures,
    )
