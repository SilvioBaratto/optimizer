"""Factor scores history building and validation utilities.

This module orchestrates point-in-time factor construction, standardization,
and validation over a rolling rebalancing schedule.  It is kept separate from
the ``optimizer`` library so that research experiments can freely import and
monkey-patch the module-level references to ``compute_all_factors`` and
``compute_vif`` without touching the library internals.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import pandas as pd
from numpy.linalg import LinAlgError
from skfolio.preprocessing import prices_to_returns

from optimizer.exceptions import FactorCoverageError
from optimizer.factors._config import (
    FactorBuildHealth,
    FactorConstructionConfig,
    PublicationLagConfig,
    StandardizationConfig,
)

# Module-level references — kept at top level so tests can patch them via
#   patch("research._factors.compute_all_factors", ...)
#   patch("research._factors.compute_vif", ...)
from optimizer.factors._construction import align_to_pit, compute_all_factors
from optimizer.factors._standardization import standardize_all_factors
from optimizer.factors._validation import FactorValidationReport, compute_vif

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Point-in-time fundamental slicing
# ---------------------------------------------------------------------------


def _slice_fundamentals_at(
    as_of_date: pd.Timestamp,
    fundamental_history: pd.DataFrame,
    snapshot_fundamentals: pd.DataFrame,
    lag_config: PublicationLagConfig,
) -> pd.DataFrame:
    """Return a point-in-time cross-section of fundamentals indexed by ticker.

    Parameters
    ----------
    as_of_date : pd.Timestamp
        Computation date.  Only records whose publication date is on or before
        this date are eligible.
    fundamental_history : pd.DataFrame
        Panel of historical fundamentals.  Must have a MultiIndex with levels
        ``(period_date, ticker)`` **or** contain ``period_date``, ``ticker``,
        and ``period_type`` columns when reset.  An empty DataFrame causes the
        function to fall back to ``snapshot_fundamentals``.
    snapshot_fundamentals : pd.DataFrame
        Current-day snapshot indexed by ticker.  Used as the fallback when
        ``fundamental_history`` is empty and as the source of snapshot-only
        columns (e.g. ``market_cap``) that are not present in the history.
    lag_config : PublicationLagConfig
        Publication lag days for annual and quarterly records.

    Returns
    -------
    pd.DataFrame
        Cross-sectional fundamentals indexed by ticker.
        - If ``fundamental_history`` is empty → returns ``snapshot_fundamentals``
          unchanged.
        - History records take precedence over the snapshot for shared columns.
        - Snapshot-only columns are merged in to fill gaps.
        - Quarterly records take precedence over annual for the same ticker.
    """
    if fundamental_history is None or fundamental_history.empty:
        return snapshot_fundamentals

    hist = fundamental_history.reset_index()

    # Split by period_type if the column is present
    has_period_type = "period_type" in hist.columns

    annual_slice: pd.DataFrame = pd.DataFrame()
    quarterly_slice: pd.DataFrame = pd.DataFrame()

    if has_period_type:
        annual_mask = hist["period_type"] == "annual"
        quarterly_mask = hist["period_type"] == "quarterly"

        annual_data = hist.loc[annual_mask]
        quarterly_data = hist.loc[quarterly_mask]

        if not annual_data.empty:
            annual_slice = align_to_pit(
                annual_data,
                period_date_col="period_date",
                as_of_date=as_of_date,
                lag_days=lag_config.annual_days,
                ticker_col="ticker",
            )

        if not quarterly_data.empty:
            quarterly_slice = align_to_pit(
                quarterly_data,
                period_date_col="period_date",
                as_of_date=as_of_date,
                lag_days=lag_config.quarterly_days,
                ticker_col="ticker",
            )
    else:
        # No period_type column — treat all records as annual
        if not hist.empty:
            annual_slice = align_to_pit(
                hist,
                period_date_col="period_date",
                as_of_date=as_of_date,
                lag_days=lag_config.annual_days,
                ticker_col="ticker",
            )

    # Merge annual and quarterly, with quarterly overriding annual per ticker
    if annual_slice.empty and quarterly_slice.empty:
        return snapshot_fundamentals

    if annual_slice.empty:
        pit_slice = quarterly_slice
    elif quarterly_slice.empty:
        pit_slice = annual_slice
    else:
        # Start from annual, update with quarterly (quarterly wins)
        pit_slice = annual_slice.copy()
        # update() aligns on index (ticker) and overwrites with non-NaN values
        # from quarterly_slice.  Tickers only in quarterly_slice are added.
        pit_slice = pit_slice.combine_first(
            quarterly_slice[
                [c for c in quarterly_slice.columns if c not in pit_slice.columns]
            ]
        )
        shared_cols = [
            c for c in quarterly_slice.columns if c in pit_slice.columns
        ]
        if shared_cols:
            pit_slice.update(quarterly_slice[shared_cols])

    # Add snapshot-only columns (e.g. market_cap) that are not in history
    snapshot_only_cols = [
        c for c in snapshot_fundamentals.columns if c not in pit_slice.columns
    ]
    if snapshot_only_cols:
        pit_slice = pit_slice.join(
            snapshot_fundamentals[snapshot_only_cols], how="left"
        )

    return pit_slice


# ---------------------------------------------------------------------------
# Factor scores history
# ---------------------------------------------------------------------------


def build_factor_scores_history(
    investable_prices: pd.DataFrame,
    investable_volumes: pd.DataFrame,
    investable_fundamentals: pd.DataFrame,
    assembly: Any,
    factor_config: FactorConstructionConfig,
    std_config: StandardizationConfig,
    sector_mapping: dict[str, str],
    rebalance_freq: int = 63,
    fundamental_history: pd.DataFrame | None = None,
    min_success_fraction: float = 0.5,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, FactorBuildHealth]:
    """Build a time-series of factor scores over rolling rebalancing periods.

    Parameters
    ----------
    investable_prices : pd.DataFrame
        Price matrix (dates × tickers).
    investable_volumes : pd.DataFrame
        Volume matrix (dates × tickers).
    investable_fundamentals : pd.DataFrame
        Snapshot fundamentals indexed by ticker.  Used as the fallback when
        ``fundamental_history`` is ``None`` or when no history record is
        available for a given rebalancing date.
    assembly : Any
        Data assembly object.  Must expose ``.analyst_data`` and
        ``.insider_data`` attributes (both ``pd.DataFrame``).
    factor_config : FactorConstructionConfig
        Factor construction parameters.
    std_config : StandardizationConfig
        Factor standardization parameters.
    sector_mapping : dict[str, str]
        Mapping of ticker → sector label used for sector neutralization.
    rebalance_freq : int
        Number of trading days between rebalancing dates.  Default: 63
        (approximately one quarter).
    fundamental_history : pd.DataFrame or None
        Panel of historical fundamentals with MultiIndex ``(period_date,
        ticker)`` for point-in-time slicing.  When ``None``, the snapshot is
        used for every date and a look-ahead-bias warning is emitted.
    min_success_fraction : float
        Minimum fraction of rebalancing dates that must succeed before
        :class:`~optimizer.exceptions.FactorCoverageError` is raised.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], pd.DataFrame, FactorBuildHealth]
        - ``factor_scores_dict`` — mapping of factor name → (dates × tickers)
          DataFrame of standardized scores.
        - ``returns_history`` — (dates × tickers) DataFrame of mean forward
          holding-period returns for each rebalancing window ``(dt, next_dt]``.
        - ``health`` — diagnostic report with success/failure counts.

    Raises
    ------
    FactorCoverageError
        When the fraction of succeeded dates falls below
        ``min_success_fraction``.
    """
    if fundamental_history is None:
        warnings.warn(
            "fundamental_history is None — using snapshot fundamentals for all "
            "rebalancing dates.  This may introduce look-ahead bias into factor "
            "scores computed before snapshot data was available.",
            UserWarning,
            stacklevel=2,
        )

    # -----------------------------------------------------------------------
    # Rebalancing schedule
    # -----------------------------------------------------------------------
    dates = investable_prices.index
    rebal_dates = dates[rebalance_freq - 1 :: rebalance_freq]

    # Pre-compute returns once
    all_returns = prices_to_returns(investable_prices)

    lag_config = factor_config.publication_lag

    # Accumulators
    factor_scores_dict: dict[str, list[pd.Series]] = {}
    factor_dates: list[pd.Timestamp] = []
    returns_rows: list[pd.Series] = []
    returns_dates: list[pd.Timestamp] = []

    succeeded = 0
    failed = 0
    failures: dict[str, str] = {}

    for i, dt in enumerate(rebal_dates):
        # Forward return window (dt, next_dt] — excludes the rebalancing-day
        # return to prevent look-ahead bias
        if i + 1 < len(rebal_dates):
            next_dt = rebal_dates[i + 1]
        else:
            # Final period: extend to the last available return date
            future_returns = all_returns.loc[all_returns.index > dt]
            if future_returns.empty:
                continue
            next_dt = future_returns.index[-1]

        fwd_mask = (all_returns.index > dt) & (all_returns.index <= next_dt)
        fwd_window = all_returns.loc[fwd_mask]
        if fwd_window.empty:
            continue

        fwd_mean = fwd_window.mean()

        # Price and volume history up to and including dt
        price_history = investable_prices.loc[investable_prices.index <= dt]
        volume_history = investable_volumes.loc[investable_volumes.index <= dt]

        # Point-in-time fundamentals
        if fundamental_history is not None:
            pit_fundamentals = _slice_fundamentals_at(
                as_of_date=dt,
                fundamental_history=fundamental_history,
                snapshot_fundamentals=investable_fundamentals,
                lag_config=lag_config,
            )
        else:
            pit_fundamentals = investable_fundamentals

        try:
            raw_factors = compute_all_factors(
                fundamentals=pit_fundamentals,
                price_history=price_history,
                volume_history=volume_history,
                analyst_data=(
                    assembly.analyst_data
                    if not assembly.analyst_data.empty
                    else None
                ),
                insider_data=(
                    assembly.insider_data
                    if not assembly.insider_data.empty
                    else None
                ),
                config=factor_config,
            )

            if raw_factors.empty:
                raise ValueError("compute_all_factors returned an empty DataFrame")

            # Build sector labels aligned to the raw_factors index (tickers)
            sector_labels: pd.Series | None = None
            if sector_mapping:
                sector_labels = pd.Series(
                    {t: sector_mapping.get(t, "Unknown") for t in raw_factors.index},
                    name="sector",
                )

            standardized, _ = standardize_all_factors(
                raw_factors,
                config=std_config,
                sector_labels=sector_labels,
            )

            # Accumulate per-factor scores
            for factor_name in standardized.columns:
                if factor_name not in factor_scores_dict:
                    factor_scores_dict[factor_name] = []
                factor_scores_dict[factor_name].append(standardized[factor_name])

            factor_dates.append(dt)
            returns_rows.append(fwd_mean)
            returns_dates.append(dt)
            succeeded += 1

        except Exception as exc:
            logger.warning(
                "skipping rebalancing date %s — factor computation failed: %s",
                dt.date(),
                exc,
            )
            failures[dt.isoformat()] = str(exc)
            failed += 1

    # -----------------------------------------------------------------------
    # Assemble outputs
    # -----------------------------------------------------------------------
    result_factor_scores: dict[str, pd.DataFrame] = {}
    for factor_name, series_list in factor_scores_dict.items():
        result_factor_scores[factor_name] = pd.DataFrame(
            series_list,
            index=pd.DatetimeIndex(factor_dates[: len(series_list)]),
        )

    if returns_rows:
        returns_history = pd.DataFrame(
            returns_rows,
            index=pd.DatetimeIndex(returns_dates),
        )
    else:
        returns_history = pd.DataFrame()

    health = FactorBuildHealth(
        total_dates=len(rebal_dates),
        succeeded_dates=succeeded,
        failed_dates=failed,
        failures=failures,
        min_success_fraction=min_success_fraction,
    )

    if not health.is_healthy:
        raise FactorCoverageError(
            f"Factor history build failed: {succeeded}/{len(rebal_dates)} "
            f"dates succeeded "
            f"(minimum required: {min_success_fraction:.0%}).  "
            f"First failures: {dict(list(failures.items())[:3])}"
        )

    return result_factor_scores, returns_history, health


# ---------------------------------------------------------------------------
# Factor validation
# ---------------------------------------------------------------------------


def validate_factors(
    factor_scores_history: dict[str, pd.DataFrame],
    returns_history: pd.DataFrame,
    standardized: pd.DataFrame,
) -> FactorValidationReport:
    """Validate factor scores and compute diagnostic statistics.

    Parameters
    ----------
    factor_scores_history : dict[str, pd.DataFrame]
        Mapping of factor name → (dates × tickers) history of scores.
    returns_history : pd.DataFrame
        (dates × tickers) forward returns aligned with ``factor_scores_history``.
    standardized : pd.DataFrame
        Cross-sectional standardized factor matrix (tickers × factors) used
        for VIF computation.

    Returns
    -------
    FactorValidationReport
        Validation report.  ``vif_scores`` is ``None`` when VIF computation
        fails due to collinearity (``LinAlgError``) or invalid input
        (``ValueError``).  ``TypeError`` is propagated to the caller.
    """
    vif_scores: pd.Series | None = None

    try:
        vif_scores = compute_vif(standardized)
    except (LinAlgError, ValueError) as exc:
        logger.warning(
            "VIF computation failed — skipping multicollinearity diagnostics: %s",
            exc,
        )
        vif_scores = None
    # TypeError is intentionally not caught — let it propagate

    return FactorValidationReport(vif_scores=vif_scores)
