"""Rolling block out-of-sample validation for factor predictive power."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from optimizer.factors._validation import (
    compute_ic_series,
    compute_icir,
    compute_quantile_spread,
)
from optimizer.validation._config import CPCVConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config and result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorOOSConfig:
    """Configuration for rolling block OOS validation.

    Parameters
    ----------
    train_months : int
        Length of the training window in months.  Default: 36.
    val_months : int
        Length of the validation window in months.  Default: 12.
    step_months : int
        Number of months to roll forward between folds.  Default: 6.
    """

    train_months: int = 36
    val_months: int = 12
    step_months: int = 6


@dataclass
class FactorOOSResult:
    """Results from rolling block OOS factor validation.

    Attributes
    ----------
    per_fold_ic : pd.DataFrame
        ``n_folds × factors`` matrix of mean IC per fold per factor.
    per_fold_spread : pd.DataFrame
        ``n_folds × factors`` matrix of mean quintile spread per fold.
    mean_oos_ic : pd.Series
        Mean OOS IC aggregated across folds (one value per factor).
    mean_oos_icir : pd.Series
        OOS ICIR (mean IC / std IC across folds) per factor.
    n_folds : int
        Number of folds generated.
    """

    per_fold_ic: pd.DataFrame
    per_fold_spread: pd.DataFrame
    mean_oos_ic: pd.Series
    mean_oos_icir: pd.Series
    n_folds: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_folds(
    dates: pd.Index,
    train_months: int,
    val_months: int,
    step_months: int,
) -> list[tuple[pd.Index, pd.Index]]:
    """Generate (train, val) date-index pairs for rolling block CV.

    Fold count = ``floor((len(dates) - train_months) / step_months)``.
    Each validation window starts immediately after its training window,
    guaranteeing no date overlap between train and val within the same fold.
    Val windows are truncated to the available date range when necessary.
    """
    total = len(dates)
    n = max(0, (total - train_months) // step_months)
    folds: list[tuple[pd.Index, pd.Index]] = []
    for i in range(n):
        t_start = i * step_months
        t_end = t_start + train_months
        v_start = t_end
        v_end = min(v_start + val_months, total)
        if v_start >= total:
            continue
        folds.append((dates[t_start:t_end], dates[v_start:v_end]))
    return folds


def _make_cpcv_folds(
    dates: pd.Index,
    cpcv_config: CPCVConfig,
) -> list[tuple[pd.Index, pd.Index]]:
    """Generate combinatorial purged cross-validation folds.

    Divides *dates* into ``n_folds`` contiguous blocks, then generates
    all C(n_folds, n_test_folds) combinations.  For each combination the
    selected blocks form the test set and the rest form the train set,
    with purging and embargo applied at train-test boundaries.

    Parameters
    ----------
    dates : pd.Index
        Sorted date index.
    cpcv_config : CPCVConfig
        CPCV parameters (n_folds, n_test_folds, purged_size, embargo_size).

    Returns
    -------
    list[tuple[pd.Index, pd.Index]]
        (train_dates, test_dates) pairs.
    """
    n = len(dates)
    n_folds = cpcv_config.n_folds
    n_test_folds = cpcv_config.n_test_folds
    purged_size = cpcv_config.purged_size
    embargo_size = cpcv_config.embargo_size

    # Split into n_folds contiguous blocks
    block_indices: list[tuple[int, int]] = []
    block_size = n // n_folds
    for i in range(n_folds):
        start = i * block_size
        end = (i + 1) * block_size if i < n_folds - 1 else n
        block_indices.append((start, end))

    folds: list[tuple[pd.Index, pd.Index]] = []
    for test_combo in combinations(range(n_folds), n_test_folds):
        test_set = set(test_combo)
        train_set = set(range(n_folds)) - test_set

        # Collect test indices
        test_idx: list[int] = []
        for b in test_combo:
            s, e = block_indices[b]
            test_idx.extend(range(s, e))

        # Collect train indices
        train_idx: list[int] = []
        for b in sorted(train_set):
            s, e = block_indices[b]
            train_idx.extend(range(s, e))

        # Apply purging: remove train indices within purged_size of any
        # train-test boundary
        if purged_size > 0:
            purge_exclusions: set[int] = set()
            for t_idx in test_idx:
                for offset in range(1, purged_size + 1):
                    purge_exclusions.add(t_idx - offset)
                    purge_exclusions.add(t_idx + offset)
            train_idx = [i for i in train_idx if i not in purge_exclusions]

        # Apply embargo: remove train indices in embargo_size positions
        # after each test block
        if embargo_size > 0:
            embargo_exclusions: set[int] = set()
            for b in test_combo:
                _, block_end = block_indices[b]
                for offset in range(embargo_size):
                    embargo_exclusions.add(block_end + offset)
            train_idx = [i for i in train_idx if i not in embargo_exclusions]

        train_dates = dates[train_idx]
        test_dates = dates[list(test_idx)]
        if len(train_dates) > 0 and len(test_dates) > 0:
            folds.append((train_dates, test_dates))

    return folds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_factor_oos_validation(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    config: FactorOOSConfig | None = None,
    cpcv_config: CPCVConfig | None = None,
) -> FactorOOSResult:
    """Rolling block or CPCV out-of-sample validation of factor IC and spreads.

    Parameters
    ----------
    scores : pd.DataFrame
        Panel of standardised factor scores with a two-level row MultiIndex
        ``(date, ticker)`` and one column per factor.
    returns : pd.DataFrame
        Forward returns panel with the same ``(date, ticker)`` MultiIndex
        and a single return column.
    config : FactorOOSConfig or None
        Rolling window parameters.  Defaults to ``FactorOOSConfig()``.
        Ignored when ``cpcv_config`` is provided.
    cpcv_config : CPCVConfig or None
        When provided, uses combinatorial purged cross-validation
        instead of rolling blocks.  Overrides ``config``.

    Returns
    -------
    FactorOOSResult
        Per-fold and aggregate IC and quintile spread statistics.

    Notes
    -----
    The validation window computation uses **only val-window dates**; no
    training-window data is used.  Fold count equals
    ``floor((total_months - train_months) / step_months)`` for rolling,
    or ``C(n_folds, n_test_folds)`` for CPCV.
    """
    if config is None:
        config = FactorOOSConfig()

    all_dates = scores.index.get_level_values(0).unique().sort_values()

    if cpcv_config is not None:
        folds = _make_cpcv_folds(all_dates, cpcv_config)
    else:
        folds = _make_folds(
            all_dates, config.train_months, config.val_months, config.step_months
        )

    factors = list(scores.columns)

    # Pivot to wide format: date × (factor, ticker) and date × ticker
    scores_wide = scores.unstack()  # date index, (factor, ticker) column MultiIndex
    returns_wide = returns.iloc[:, 0].unstack()  # date index, ticker columns

    per_fold_ic_rows: list[dict[str, float]] = []
    per_fold_spread_rows: list[dict[str, float]] = []

    for _train_dates, val_dates in folds:
        fold_ic: dict[str, float] = {}
        fold_spread: dict[str, float] = {}

        scores_val = scores_wide.loc[scores_wide.index.isin(val_dates)]
        returns_val = returns_wide.loc[returns_wide.index.isin(val_dates)]

        for factor in factors:
            factor_history = scores_val[factor]  # date × ticker for this factor
            ic_series = compute_ic_series(factor_history, returns_val, factor)
            fold_ic[factor] = (
                float(ic_series.mean()) if len(ic_series) > 0 else float("nan")
            )

            spreads: list[float] = []
            for date in val_dates:
                if date not in factor_history.index or date not in returns_val.index:
                    continue
                f_scores = factor_history.loc[date].dropna()
                r_returns = returns_val.loc[date].dropna()
                spread = compute_quantile_spread(f_scores, r_returns)
                if not np.isnan(spread):
                    spreads.append(spread)
            fold_spread[factor] = float(np.mean(spreads)) if spreads else float("nan")

        per_fold_ic_rows.append(fold_ic)
        per_fold_spread_rows.append(fold_spread)

    n_folds = len(folds)
    per_fold_ic = pd.DataFrame(per_fold_ic_rows, columns=factors)
    per_fold_spread = pd.DataFrame(per_fold_spread_rows, columns=factors)
    mean_oos_ic = per_fold_ic.mean(axis=0)
    mean_oos_icir = per_fold_ic.apply(compute_icir, axis=0)

    return FactorOOSResult(
        per_fold_ic=per_fold_ic,
        per_fold_spread=per_fold_spread,
        mean_oos_ic=mean_oos_ic,
        mean_oos_icir=mean_oos_icir,
        n_folds=n_folds,
    )
