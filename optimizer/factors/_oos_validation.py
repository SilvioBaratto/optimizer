"""Rolling block out-of-sample validation for factor predictive power."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimizer.factors._validation import (
    compute_ic_series,
    compute_icir,
    compute_quantile_spread,
)

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_factor_oos_validation(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    config: FactorOOSConfig | None = None,
) -> FactorOOSResult:
    """Rolling block OOS validation of factor IC and quintile spreads.

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

    Returns
    -------
    FactorOOSResult
        Per-fold and aggregate IC and quintile spread statistics.

    Notes
    -----
    The validation window computation uses **only val-window dates**; no
    training-window data is used.  Fold count equals
    ``floor((total_months - train_months) / step_months)``.
    """
    if config is None:
        config = FactorOOSConfig()

    all_dates = scores.index.get_level_values(0).unique().sort_values()
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
                if (
                    date not in factor_history.index
                    or date not in returns_val.index
                ):
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
