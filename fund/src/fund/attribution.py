"""Quantified macro-attribution engine (SPEC Phase 06).

Turns the *qualitative* macro→portfolio story the economist agent reasons about into
numbers: regress a paper portfolio's NAV return series (Phase 05) on macro factor
returns to estimate exposures (betas), then attribute each period's return to each
factor (``β·f``) plus an idiosyncratic residual that absorbs alpha and noise. The
pieces sum to the total period return by construction.

Compute-on-demand — no table, no migration — mirroring on-demand
``portfolio_metrics``. Agent-stack-free: imports ``optimizer`` / ``skfolio`` /
``numpy`` / ``pandas`` (allowed) but never ``deepagents`` / ``langgraph`` /
``fund.audit``; ``portopt_db`` repositories and ``fund.tools.macro`` are lazy-imported
*inside* functions so a bare ``import fund.attribution`` stays light and hygiene-clean.

The regression default is robust Huber; it auto-falls-back to closed-form OLS on short
NAV history (where Huber's l-bfgs fails to converge) and degrades honestly — a factor
whose source has no data over the NAV window is dropped, short history flags
``insufficient_history``, and nothing raises on thin data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class RegressionMethod(str, Enum):
    """Inner regressor for the exposure (loading-matrix) estimate.

    ``HUBER`` is the robust default; ``OLS`` is the closed-form fallback used
    automatically on short history; ``RIDGE`` is an L2-regularised alternative.
    """

    HUBER = "huber"
    OLS = "ols"
    RIDGE = "ridge"


class FactorTransform(str, Enum):
    """How a raw FRED level series becomes a per-period factor return.

    ``DIFF`` takes the first difference of the level (yield / spread changes);
    ``PCT_CHANGE`` takes the simple return of a ratio/index level.
    """

    DIFF = "diff"
    PCT_CHANGE = "pct_change"


@dataclass(frozen=True)
class FactorSpec:
    """One macro factor: a name, its source, and the level→return transform.

    Attributes:
        name: Factor label used as the exposure/contribution key and column name.
        fred_series_id: FRED series id supplying the raw level, or ``None`` for a
            non-FRED factor (``equity``, sourced from the benchmark return series).
        transform: Level→return transform applied on the NAV grid.
    """

    name: str
    fred_series_id: str | None
    transform: FactorTransform

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FactorSpec.name must be non-empty")


_DEFAULT_FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec("rates", "DGS10", FactorTransform.DIFF),
    FactorSpec("credit", "BAA10Y", FactorTransform.DIFF),
    FactorSpec("usd", "DTWEXBGS", FactorTransform.PCT_CHANGE),
    FactorSpec("equity", None, FactorTransform.PCT_CHANGE),
)


@dataclass(frozen=True)
class MacroAttributionConfig:
    """Serialisable configuration for a macro-attribution run.

    Holds only primitives / enums / nested frozen dataclasses (the estimator
    instances are built internally from :attr:`method`). ``equity`` has
    ``fred_series_id is None`` — it is sourced from the benchmark return series, not
    FRED.

    Attributes:
        factors: Ordered factor specs to build and regress on.
        method: Inner regressor for the exposure estimate.
        min_observations: Minimum aligned, finite periods below which the exposure
            estimate degrades to an empty, ``insufficient_history`` result.
        lookback_days: Optional trailing-window cap (calendar days) on the NAV
            series the engine regresses; ``None`` uses the full history.
    """

    factors: tuple[FactorSpec, ...] = field(default_factory=lambda: _DEFAULT_FACTORS)
    method: RegressionMethod = RegressionMethod.HUBER
    min_observations: int = 12
    lookback_days: int | None = None

    def __post_init__(self) -> None:
        if not self.factors:
            raise ValueError("MacroAttributionConfig.factors must be non-empty")
        if self.min_observations < 2:
            raise ValueError("min_observations must be >= 2")
        if self.lookback_days is not None and self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive when set")

    @classmethod
    def for_default(
        cls, *, method: RegressionMethod = RegressionMethod.HUBER
    ) -> MacroAttributionConfig:
        """Default preset: rates / credit / usd (FRED) + equity (benchmark)."""
        return cls(factors=_DEFAULT_FACTORS, method=method)


@dataclass(frozen=True)
class ExposureResult:
    """Estimated macro exposures of a NAV return series.

    Attributes:
        exposures: ``{factor_name: beta}``; empty when the estimate degraded.
        alpha: Fitted intercept (per-period); ``0.0`` on a degraded estimate.
        n_obs: Aligned, finite periods available to the regression.
        r2: In-sample coefficient of determination, or ``None`` when undefined.
        method_used: The regressor actually used (``OLS`` after a fallback).
        insufficient_history: True when history was too short for the robust fit —
            either an OLS fallback fired or the estimate degraded to empty.
    """

    exposures: dict[str, float]
    alpha: float
    n_obs: int
    r2: float | None
    method_used: RegressionMethod
    insufficient_history: bool


@dataclass(frozen=True)
class AttributionResult:
    """Decomposition of a period's NAV return into factor contributions + residual.

    ``sum(contributions.values()) + residual == total_return`` holds by construction;
    the residual absorbs alpha and idiosyncratic noise.

    Attributes:
        total_return: Additive sum of simple NAV returns over the aligned window.
        contributions: ``{factor_name: β·Σf}`` contribution to the total.
        residual: ``total_return − Σ contributions`` (alpha + idiosyncratic).
        factor_names: The factors that received a contribution, in order.
        insufficient_history: Carried from the exposure estimate.
    """

    total_return: float
    contributions: dict[str, float]
    residual: float
    factor_names: tuple[str, ...]
    insufficient_history: bool


@dataclass(frozen=True)
class FactorSeries:
    """Macro factor returns aligned to the NAV grid, plus the dropped factors.

    Attributes:
        returns: Dates × factors simple-return frame on the NAV return grid; a
            factor absent over the window is omitted, never fabricated.
        missing: Factor names dropped because their source had no usable data over
            the window (FRED series absent, or the benchmark missing for ``equity``).
    """

    returns: pd.DataFrame
    missing: list[str]


def _build_regressor(method: RegressionMethod) -> BaseEstimator:
    """Construct the sklearn inner regressor for ``method`` (intercept fitted)."""
    from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge

    if method is RegressionMethod.HUBER:
        return HuberRegressor(fit_intercept=True, max_iter=500)
    if method is RegressionMethod.RIDGE:
        return Ridge(fit_intercept=True)
    return LinearRegression(fit_intercept=True)


def _aligned_finite(
    nav_returns: pd.Series, factor_returns: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    """Restrict both to their common index where NAV and every factor are finite."""
    common = nav_returns.index.intersection(factor_returns.index)
    nav = nav_returns.loc[common]
    fac = factor_returns.loc[common]
    mask = nav.notna() & fac.notna().all(axis=1)
    return nav[mask], fac[mask]


def _empty_exposure(n_obs: int, method_used: RegressionMethod) -> ExposureResult:
    """A degraded exposure estimate: no betas, zero alpha, flagged insufficient."""
    return ExposureResult(
        exposures={},
        alpha=0.0,
        n_obs=n_obs,
        r2=None,
        method_used=method_used,
        insufficient_history=True,
    )


def estimate_exposures(
    nav_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    config: MacroAttributionConfig,
) -> ExposureResult:
    """Regress NAV returns on factor returns to estimate macro exposures (betas).

    Uses :func:`optimizer.factors.fit_factor_prior` with a
    ``LoadingMatrixRegression`` override so the betas and a real intercept come from
    :attr:`config.method` (the skfolio default inner regressor is
    ``LassoCV(fit_intercept=False)``, which would zero alpha and shrink betas).

    Degrades honestly: when the aligned, finite sample is shorter than ``3·(K+1)``
    the robust Huber fit cannot converge, so it falls back to closed-form OLS and
    sets :attr:`ExposureResult.insufficient_history`. When the sample is below
    ``config.min_observations`` (or no factors are supplied) it returns an empty,
    flagged estimate rather than raising.

    Args:
        nav_returns: Date-indexed portfolio simple returns (the regression target).
        factor_returns: Dates × factors simple-return frame (the regressors).
        config: Run configuration (regressor choice + ``min_observations``).

    Returns:
        An :class:`ExposureResult`; ``exposures`` empty on a degraded estimate.
    """
    from optimizer.exceptions import DataError
    from optimizer.factors import FactorPriorConfig, fit_factor_prior
    from skfolio.prior import LoadingMatrixRegression

    nav_f, fac_f = _aligned_finite(nav_returns, factor_returns)
    n_obs = len(nav_f)
    n_factors = factor_returns.shape[1]

    if n_factors == 0 or n_obs < config.min_observations:
        return _empty_exposure(n_obs, config.method)

    insufficient = n_obs < 3 * (n_factors + 1)
    method_used = RegressionMethod.OLS if insufficient else config.method
    estimator = LoadingMatrixRegression(linear_regressor=_build_regressor(method_used))

    try:
        model = fit_factor_prior(
            nav_f.to_frame(name="nav"),
            fac_f,
            config=FactorPriorConfig(min_observations=config.min_observations),
            loading_matrix_estimator=estimator,
        )
    except DataError:
        logger.info("estimate_exposures: too few aligned periods; degrading")
        return _empty_exposure(n_obs, method_used)

    fitted = model.loading_matrix_estimator_
    betas = np.asarray(fitted.loading_matrix_[0], dtype=float)
    alpha = float(fitted.intercepts_[0])
    # The loading matrix row is positionally aligned to the fitted factor columns
    # (verified against skfolio 1.4.11), so the fit frame's column order names them.
    names = [str(col) for col in fac_f.columns]
    exposures = {name: float(beta) for name, beta in zip(names, betas, strict=True)}

    r2 = _r_squared(nav_f, fac_f, exposures, alpha)
    return ExposureResult(
        exposures=exposures,
        alpha=alpha,
        n_obs=n_obs,
        r2=r2,
        method_used=method_used,
        insufficient_history=insufficient,
    )


def _r_squared(
    nav_f: pd.Series,
    fac_f: pd.DataFrame,
    exposures: dict[str, float],
    alpha: float,
) -> float | None:
    """In-sample R² of ``alpha + Σ β·f`` against NAV; ``None`` when degenerate."""
    beta_vec = np.array([exposures.get(c, 0.0) for c in fac_f.columns], dtype=float)
    pred = alpha + fac_f.to_numpy() @ beta_vec
    actual = nav_f.to_numpy()
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    if ss_tot <= 0.0:
        return None
    ss_res = float(np.sum((actual - pred) ** 2))
    return 1.0 - ss_res / ss_tot


def attribute(
    nav_returns: pd.Series,
    factor_returns: pd.DataFrame,
    exposure: ExposureResult,
) -> AttributionResult:
    """Decompose the period's NAV return into per-factor contributions + residual.

    ``contribution_k = β_k · Σ_t f_{k,t}`` summed over the aligned window;
    ``residual = total_return − Σ contributions``. Because ``total_return`` is the
    additive sum of NAV simple returns over the **same** window the contributions use,
    the pieces reconstruct the total exactly. With no exposures the whole return is
    residual.

    Args:
        nav_returns: Date-indexed portfolio simple returns.
        factor_returns: Dates × factors simple-return frame.
        exposure: The exposure estimate from :func:`estimate_exposures`.

    Returns:
        An :class:`AttributionResult` whose contributions + residual sum to the total.
    """
    cols = [c for c in exposure.exposures if c in factor_returns.columns]
    if not cols:
        total = float(nav_returns.dropna().sum())
        return AttributionResult(
            total_return=total,
            contributions={},
            residual=total,
            factor_names=(),
            insufficient_history=exposure.insufficient_history,
        )

    nav_f, fac_f = _aligned_finite(nav_returns, factor_returns.loc[:, cols])
    total = float(nav_f.sum())
    contributions = {
        col: float(exposure.exposures[col] * fac_f[col].sum()) for col in cols
    }
    residual = total - sum(contributions.values())
    return AttributionResult(
        total_return=total,
        contributions=contributions,
        residual=residual,
        factor_names=tuple(cols),
        insufficient_history=exposure.insufficient_history,
    )


def _level_to_return(level_on_grid: pd.Series, transform: FactorTransform) -> pd.Series:
    """Turn a grid-aligned level series into per-period returns per ``transform``."""
    if transform is FactorTransform.DIFF:
        return level_on_grid.diff()
    return level_on_grid.pct_change(fill_method=None)


def build_macro_factor_returns(
    *,
    session: Any,
    dates: pd.DatetimeIndex,
    benchmark_returns: pd.Series | None,
    config: MacroAttributionConfig,
) -> FactorSeries:
    """Build macro factor returns on the NAV grid, FRED-first, degrading honestly.

    Applies the canonical alignment recipe to each FRED factor: fetch the full level
    history up to the grid's last date (via :func:`fund.tools.macro.load_macro_frame`,
    lazy-imported), reindex it onto the NAV grid filling each day from the most recent
    prior observation, transform to a return (``diff`` or ``pct_change`` per the
    factor's spec), then drop the first (NaN) row so the result sits on the NAV return
    grid (``dates[1:]``). The non-FRED ``equity`` factor is the benchmark return series
    reindexed onto that grid. A FRED series with no rows, an all-NaN transformed column,
    or a missing benchmark for ``equity`` is dropped into ``missing`` rather than
    fabricated.

    Args:
        session: Open SQLAlchemy session (reader; no commit).
        dates: The NAV date grid (ascending ``DatetimeIndex``) from the NAV series.
        benchmark_returns: The benchmark simple-return series (already returns, e.g.
            :func:`fund.performance.nav_return_series` with ``source="benchmark"``);
            ``None`` degrades every non-FRED factor to ``missing``.
        config: Run configuration whose ``factors`` drive what is built.

    Returns:
        A :class:`FactorSeries`; ``returns`` empty and every factor ``missing`` when
        the grid has fewer than two dates.
    """
    columns: dict[str, pd.Series] = {}
    missing: list[str] = []

    if len(dates) < 2:
        return FactorSeries(
            returns=pd.DataFrame(index=dates[1:]),
            missing=[spec.name for spec in config.factors],
        )

    grid = pd.DatetimeIndex(dates).sort_values()
    target = grid[1:]

    fred_specs = [s for s in config.factors if s.fred_series_id is not None]
    frame = pd.DataFrame()
    if fred_specs:
        from fund.tools.macro import load_macro_frame

        series_ids = [str(s.fred_series_id) for s in fred_specs]
        frame, _ = load_macro_frame(session, series_ids, grid[-1].date())

    for spec in config.factors:
        if spec.fred_series_id is None:
            column = _equity_factor(benchmark_returns, target)
        else:
            column = _fred_factor(frame, spec, grid, target)
        if column is None:
            missing.append(spec.name)
        else:
            columns[spec.name] = column

    returns = (
        pd.DataFrame(columns, index=target) if columns else pd.DataFrame(index=target)
    )
    return FactorSeries(returns=returns, missing=missing)


def _fred_factor(
    frame: pd.DataFrame,
    spec: FactorSpec,
    grid: pd.DatetimeIndex,
    target: pd.DatetimeIndex,
) -> pd.Series | None:
    """One FRED factor's returns on ``target``, or ``None`` when it has no data."""
    series_id = str(spec.fred_series_id)
    if series_id not in frame.columns:
        return None
    # ffill internal gaps first: load_macro_frame unions each series' own dates, so a
    # day one series lacks is an internal NaN, which reindex(method="ffill") won't fill.
    level_full = frame[series_id].ffill()
    level_on_grid = level_full.reindex(grid, method="ffill")
    column = _level_to_return(level_on_grid, spec.transform).reindex(target)
    if column.notna().sum() == 0:
        return None
    return column


def _equity_factor(
    benchmark_returns: pd.Series | None, target: pd.DatetimeIndex
) -> pd.Series | None:
    """The benchmark return series on ``target``, or ``None`` when unavailable."""
    if benchmark_returns is None or benchmark_returns.empty:
        return None
    column = benchmark_returns.reindex(target)
    if column.notna().sum() == 0:
        return None
    return column


__all__ = [
    "AttributionResult",
    "ExposureResult",
    "FactorSeries",
    "FactorSpec",
    "FactorTransform",
    "MacroAttributionConfig",
    "RegressionMethod",
    "attribute",
    "build_macro_factor_returns",
    "estimate_exposures",
]
