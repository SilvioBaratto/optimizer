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


def build_walk_forward(
    config: WalkForwardConfig | None = None,
) -> WalkForward:
    """Build a skfolio WalkForward cross-validator from config.

    Args:
        config: Walk-forward configuration. Defaults to WalkForwardConfig()
            (quarterly rolling with one-year training window).

    Returns:
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
    """Build a skfolio CombinatorialPurgedCV cross-validator from config.

    Args:
        config: CPCV configuration. Defaults to CPCVConfig()
            (10 folds, 8 test folds).

    Returns:
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
    """Build a MultipleRandomizedCV cross-validator from config.

    Args:
        config: Multiple randomised CV configuration. Defaults to
            MultipleRandomizedCVConfig().

    Returns:
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

    Thin wrapper around skfolio.model_selection.cross_val_predict that
    enforces temporal splitting (no random shuffle).

    Args:
        estimator: A fitted-ready skfolio optimisation estimator or pipeline.
        X: Return matrix (observations x assets).
        cv: Cross-validator. Defaults to WalkForward with quarterly test windows.
        y: Benchmark returns or factor returns (for models that require
            fit(X, y)).
        params: Auxiliary metadata forwarded to nested estimators via sklearn
            metadata routing (e.g. {"implied_vol": implied_vol_df}). Requires
            sklearn.set_config(enable_metadata_routing=True) and the relevant
            set_fit_request calls on sub-estimators. With routing off (the
            default), a Pipeline instead accepts only step-prefixed keys
            (e.g. {"optimizer__factors": factor_returns}) and rejects bare
            ones; with routing on it is the reverse.
        n_jobs: Number of parallel jobs.
        portfolio_params: Additional parameters forwarded to the portfolio
            constructor.

    Returns:
        Out-of-sample portfolio predictions. WalkForward returns a
        MultiPeriodPortfolio; CombinatorialPurgedCV and MultipleRandomizedCV
        return a Population.

    Note:
        No look-ahead: every cross-validator built by this module is temporal
        and never shuffles observations, so each training window strictly
        precedes its test window. Do not substitute a shuffling splitter
        (KFold(shuffle=True) / train_test_split(shuffle=True)) — that silently
        leaks future data. Use purged_size to also excise the autocorrelated
        boundary.

        Single-regime constraints: the estimator is fixed for the whole
        backtest, so any constraint it carries (sector bands, weight bounds,
        turnover caps) is constant across every fold. Walk-forward CV cannot
        vary constraints per rebalance; a run is therefore single-regime, not
        per-rebalance. To study a regime-dependent constraint set, run one
        backtest per regime and compare.

        Survivorship / delisted universe: X should be the delisted-inclusive
        return matrix. Include instruments that were delisted during the window,
        with their terminal delisting return applied on the delist date and NaN
        thereafter. Passing only currently-listed names reintroduces
        survivorship bias. skfolio's default Empirical moment estimators raise
        on NaN — for a matrix with delisted (NaN-tailed) columns, use
        NaN-tolerant estimators or align/zero-fill upstream.
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


def compute_optimal_folds(
    n_observations: int,
    target_train_size: int,
    target_n_test_paths: int,
    weight_train_size: float = 1.0,
    weight_n_test_paths: float = 1.0,
) -> tuple[int, int]:
    """Compute optimal fold counts for CPCV.

    Wraps skfolio.model_selection.optimal_folds_number.

    Args:
        n_observations: Total number of observations.
        target_train_size: Desired training window size.
        target_n_test_paths: Desired number of backtest paths.
        weight_train_size: Relative importance of matching train size.
        weight_n_test_paths: Relative importance of matching path count.

    Returns:
        (n_folds, n_test_folds) optimal parameters.
    """
    return optimal_folds_number(
        n_observations=n_observations,
        target_train_size=target_train_size,
        target_n_test_paths=target_n_test_paths,
        weight_train_size=weight_train_size,
        weight_n_test_paths=weight_n_test_paths,
    )


@dataclass(frozen=True)
class CrossValFailureReport:
    """Summary of failed and fallback-rescued folds in a cross-validated backtest.

    skfolio optimizers accept raise_on_failure=False. When an optimizer
    configured that way cannot solve a fold, cross_val_predict yields a
    FailedPortfolio sentinel for that fold instead of raising. When a
    ``fallback`` is configured as well and rescues the fold, the result is an
    ordinary Portfolio whose ``fallback_chain`` records the primary failure.
    This report counts both so a backtest can surface solver fragility rather
    than silently averaging over failed or substituted rebalances.

    Attributes:
        total: Total number of (leaf) portfolios inspected.
        n_failed: Number of FailedPortfolio sentinels.
        failure_rate: n_failed / total (0.0 when total is zero).
        failures: (portfolio_name, optimization_error) for each failed fold.
        failed_folds: 0-based position of each failed fold in the flattened
            leaf sequence (rebalance order for a MultiPeriodPortfolio,
            path-major for a Population), parallel to ``failures``. Portfolio
            names repeat across folds in a real run, so the position is what
            identifies a fold.
        n_fell_back: Number of folds whose primary solve failed and was
            rescued by a fallback, so their weights are not the primary
            optimum.
        fallback_rate: n_fell_back / total (0.0 when total is zero).
        fell_back: (portfolio_name, primary_error) for each rescued fold.
        fell_back_folds: Fold positions of the rescued folds, parallel to
            ``fell_back`` and indexed like ``failed_folds``.
    """

    total: int
    n_failed: int
    failure_rate: float
    failures: list[tuple[str, str]] = field(default_factory=list)
    failed_folds: list[int] = field(default_factory=list)
    n_fell_back: int = 0
    fallback_rate: float = 0.0
    fell_back: list[tuple[str, str]] = field(default_factory=list)
    fell_back_folds: list[int] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        """``True`` when at least one fold failed to optimize."""
        return self.n_failed > 0

    @property
    def has_fallbacks(self) -> bool:
        """``True`` when at least one fold was rescued by a fallback."""
        return self.n_fell_back > 0


_MAX_LOGGED_FOLDS = 10


def _format_fold_positions(positions: list[int]) -> str:
    """Render fold positions for a log line, capped so the line stays bounded.

    CombinatorialPurgedCV can yield hundreds of folds, and every fold of a real
    run shares one portfolio name, so positions are the only useful identifier.
    """
    shown = ", ".join(str(i) for i in positions[:_MAX_LOGGED_FOLDS])
    hidden = len(positions) - _MAX_LOGGED_FOLDS
    if hidden > 0:
        shown += f", ... (+{hidden} more)"
    return f"[{shown}]"


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
    """Count failed and fallback-rescued folds in a cross-validated prediction.

    Works on the output of run_cross_val (a MultiPeriodPortfolio for
    single-path CV or a Population for CombinatorialPurgedCV /
    MultipleRandomizedCV). Use together with skfolio's optimizer resilience
    layer (MeanRisk(..., raise_on_failure=False, fallback=...)) so a failed
    rebalance does not abort the whole backtest. A fold the fallback rescues is
    an ordinary Portfolio, so it is reported separately in ``fell_back``
    instead of counting as a failure; a fold the whole chain could not solve is
    a FailedPortfolio and is reported in ``failures``. Each list has a parallel
    ``*_folds`` list of fold positions. A warning naming the first fold
    positions is logged whenever any fold fell back.

    Args:
        pred: A cross-validation prediction (Portfolio, MultiPeriodPortfolio,
            or Population).

    Returns:
        Aggregate failure and fallback counts with per-fold error messages.
    """
    leaves = _iter_leaf_portfolios(pred)
    failures: list[tuple[str, str]] = []
    failed_folds: list[int] = []
    fell_back: list[tuple[str, str]] = []
    fell_back_folds: list[int] = []
    for position, p in enumerate(leaves):
        name = str(getattr(p, "name", "") or "")
        if isinstance(p, FailedPortfolio):
            failures.append((name, str(getattr(p, "optimization_error", "") or "")))
            failed_folds.append(position)
            continue
        chain = getattr(p, "fallback_chain", None)
        if chain:
            fell_back.append((name, str(chain[0][1])))
            fell_back_folds.append(position)

    total = len(leaves)
    n_failed = len(failures)
    n_fell_back = len(fell_back)
    if n_fell_back:
        logger.warning(
            "%d of %d folds fell back to a fallback optimizer after the primary "
            "solve failed; their weights are not the primary optimum and a "
            "previous-weights fallback may leave weight unallocated; fold "
            "positions: %s",
            n_fell_back,
            total,
            _format_fold_positions(fell_back_folds),
        )
    return CrossValFailureReport(
        total=total,
        n_failed=n_failed,
        failure_rate=n_failed / total if total > 0 else 0.0,
        failures=failures,
        failed_folds=failed_folds,
        n_fell_back=n_fell_back,
        fallback_rate=n_fell_back / total if total > 0 else 0.0,
        fell_back=fell_back,
        fell_back_folds=fell_back_folds,
    )
