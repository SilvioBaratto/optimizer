"""Factor validation and statistical testing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from optimizer.exceptions import DataError
from optimizer.factors._config import FACTOR_GROUP_MAPPING, FactorValidationConfig

if TYPE_CHECKING:
    from optimizer.linear_model import CSLinearRegressionConfig

logger = logging.getLogger(__name__)

# Annualised long-short quintile spread benchmarks (low, high) per group.
# Derived from academic literature (Fama-French, AQR, Novy-Marx).
FACTOR_SPREAD_BENCHMARKS: dict[str, tuple[float, float]] = {
    "value": (0.02, 0.06),
    "profitability": (0.02, 0.05),
    "investment": (0.01, 0.04),
    "momentum": (0.04, 0.10),
    "low_risk": (0.01, 0.04),
    "liquidity": (0.01, 0.03),
    "dividend": (0.01, 0.03),
    "sentiment": (0.005, 0.02),
    "ownership": (0.005, 0.02),
}


@dataclass
class ICResult:
    """Information coefficient analysis results for a single factor."""

    factor_name: str
    mean_ic: float
    ic_std: float
    t_stat: float
    p_value: float
    significant: bool


@dataclass
class CompositeICResult:
    """IC analysis results for the composite score signal.

    Attributes:
        mean_ic: Mean IC of the composite score over the evaluation period.
        ic_std: Standard deviation of the IC series.
        t_stat: Newey-West adjusted t-statistic.
        p_value: Two-tailed p-value from the Newey-West t-statistic.
        icir: IC Information Ratio: ``mean(IC) / std(IC)``.
        significant: True when ``abs(t_stat) >= t_stat_threshold``.
        best_individual_ic: Highest mean IC among individual factors.
            ``NaN`` when no individual factors were validated alongside.
        outperforms_best_individual: True when ``mean_ic > best_individual_ic``.
    """

    mean_ic: float
    ic_std: float
    t_stat: float
    p_value: float
    icir: float
    significant: bool
    best_individual_ic: float
    outperforms_best_individual: bool


@dataclass
class GroupICResult:
    """Result of group-level IC aggregation with per-factor breakdown.

    Attributes:
        group_ic: (dates x groups) group-level IC history.  Identical in shape
            to the legacy ``build_group_ic_history`` return value.
        factor_ic: (dates x factors) per-factor IC time series.
        excluded_factors: Group name → list of factor names excluded by the
            negative-IC filter policy.  Empty when
            ``ICNegativeFilterPolicy.INCLUDE``.
    """

    group_ic: pd.DataFrame
    factor_ic: pd.DataFrame
    excluded_factors: dict[str, list[str]]


@dataclass
class QuantileSpreadResult:
    """Quantile spread analysis results for a single factor."""

    factor_name: str
    spread: float
    quantile_returns: list[float]
    within_benchmark: bool = False


@dataclass
class FactorValidationReport:
    """Complete validation report for all factors."""

    ic_results: list[ICResult] = field(default_factory=list)
    quantile_spreads: list[QuantileSpreadResult] = field(
        default_factory=list,
    )
    vif_scores: pd.Series | None = None
    significant_factors: list[str] = field(default_factory=list)
    significant_factors_holm: list[str] = field(default_factory=list)
    composite_ic_result: CompositeICResult | None = None


@dataclass
class ICStats:
    """Full IC statistics for a single factor including Newey-West inference.

    Attributes:
        mean: Mean IC over the evaluation period.
        variance_nw: Newey-West HAC variance of the IC series.
        t_stat_nw: Newey-West adjusted t-statistic:
            ``IC_mean / sqrt(Var_NW / T)``.
        p_value: Two-tailed p-value derived from the Newey-West t-statistic.
        icir: Information Coefficient Information Ratio:
            ``mean(IC) / std(IC)``.
    """

    mean: float
    variance_nw: float
    t_stat_nw: float
    p_value: float
    icir: float


@dataclass
class CorrectedPValues:
    """Multiple-testing corrected p-values.

    Attributes:
        holm: Holm-Bonferroni adjusted p-values (controls FWER).
        bh: Benjamini-Hochberg adjusted p-values (controls FDR).
    """

    holm: npt.NDArray[np.float64]
    bh: npt.NDArray[np.float64]


def compute_monthly_ic(
    factor_scores: pd.Series,
    forward_returns: pd.Series,
    min_observations: int = 3,
) -> float:
    """Compute rank information coefficient (Spearman correlation).

    Args:
        factor_scores: Cross-sectional factor scores.
        forward_returns: Forward returns for the same tickers.
        min_observations: Minimum number of common non-NaN observations
            required.  Returns NaN if fewer are available.

    Returns:
        Rank IC (Spearman correlation).
    """
    common = factor_scores.dropna().index.intersection(
        forward_returns.dropna().index,
    )
    if len(common) < min_observations:
        return float(np.nan)
    return float(
        factor_scores.loc[common].corr(forward_returns.loc[common], method="spearman")
    )


def compute_ic_series(
    factor_scores_history: pd.DataFrame,
    returns_history: pd.DataFrame,
    factor_name: str,
    min_observations: int = 3,
    *,
    use_cs_regression: bool = False,
    cs_config: CSLinearRegressionConfig | None = None,
) -> pd.Series:
    """Compute IC time series for a factor.

    Args:
        factor_scores_history: Dates x tickers matrix of factor scores.
        returns_history: Dates x tickers matrix of forward returns.
        factor_name: Used only for labeling.
        min_observations: Minimum number of common non-NaN observations per
            date.
        use_cs_regression: When ``True``, replace the per-period rank
            correlation with the slope coefficient of a per-period
            cross-sectional regression (built via
            `build_cs_linear_regression`).  The default preserves the
            original Spearman-rank IC exactly.
        cs_config: Configuration forwarded to
            `build_cs_linear_regression` when ``use_cs_regression`` is
            ``True``.  ``None`` defers to `CSLinearRegressionConfig`
            defaults.

    Returns:
        IC values indexed by date.
    """
    if use_cs_regression:
        return _compute_ic_series_cs_regression(
            factor_scores_history,
            returns_history,
            factor_name,
            min_observations=min_observations,
            cs_config=cs_config,
        )

    common_dates = factor_scores_history.index.intersection(
        returns_history.index,
    )
    ics: dict[object, float] = {}
    for date in common_dates:
        ic = compute_monthly_ic(
            factor_scores_history.loc[date],
            returns_history.loc[date],
            min_observations=min_observations,
        )
        if not np.isnan(ic):
            ics[date] = ic

    return pd.Series(ics, name=factor_name, dtype=float)


def _compute_ic_series_cs_regression(
    factor_scores_history: pd.DataFrame,
    returns_history: pd.DataFrame,
    factor_name: str,
    *,
    min_observations: int,
    cs_config: CSLinearRegressionConfig | None,
) -> pd.Series:
    """Per-period CS regression slope as IC.

    Drops periods with fewer than ``min_observations`` non-NaN pairs.
    """
    from optimizer.linear_model import (
        CSLinearRegressionConfig,
        build_cs_linear_regression,
    )

    common_dates = factor_scores_history.index.intersection(
        returns_history.index,
    )
    cfg = cs_config if cs_config is not None else CSLinearRegressionConfig()

    kept_dates: list[object] = []
    rows_x: list[npt.NDArray[np.float64]] = []
    rows_y: list[npt.NDArray[np.float64]] = []
    rows_w: list[npt.NDArray[np.float64]] = []
    for date in common_dates:
        scores = factor_scores_history.loc[date]
        rets = returns_history.loc[date]
        valid = (scores.notna() & rets.notna()).to_numpy()
        if int(valid.sum()) < min_observations:
            continue
        kept_dates.append(date)
        # skfolio CSLinearRegression rejects any (period, asset) pair with a
        # positive cs_weight whose feature/target is non-finite.  Real panels
        # have ragged per-period coverage (assets enter/exit, missing scores),
        # so we exclude invalid pairs via a 0/1 weight mask and replace their
        # NaN placeholders with a finite 0.0 (ignored because their weight is
        # 0).  Fitting without the mask crashes on any partial-coverage period.
        x_row = np.where(valid, scores.to_numpy(dtype=float), 0.0)
        y_row = np.where(valid, rets.to_numpy(dtype=float), 0.0)
        rows_x.append(np.nan_to_num(x_row, nan=0.0))
        rows_y.append(np.nan_to_num(y_row, nan=0.0))
        rows_w.append(valid.astype(np.float64))

    if not kept_dates:
        return pd.Series(dtype=float, name=factor_name)

    x_panel = np.stack(rows_x, axis=0)[:, :, None]  # (T, N, 1)
    y_panel = np.stack(rows_y, axis=0)  # (T, N)
    w_panel = np.stack(rows_w, axis=0)  # (T, N)
    estimator = build_cs_linear_regression(cfg)
    estimator.fit(x_panel, y_panel, cs_weights=w_panel)
    slopes = np.asarray(estimator.coef_)[:, 0]
    return pd.Series(slopes, index=pd.Index(kept_dates), name=factor_name)


def compute_icir(ic_series: pd.Series) -> float:
    """Compute the IC Information Ratio (mean IC / std IC).

    ICIR penalises factors with high average IC but also high IC
    volatility (inconsistent predictors).  Use this as the weighting
    signal in ICIR-weighted composite scoring.

    Args:
        ic_series: Time series of IC values (one per cross-section date).

    Returns:
        ICIR value, or 0.0 if ``std(IC) == 0`` or fewer than
        2 non-NaN observations.
    """
    clean = ic_series.dropna()
    if len(clean) < 2:
        return 0.0
    mean_ic = float(clean.mean())
    ic_std = float(cast(float, clean.std(ddof=1)))
    return mean_ic / ic_std if ic_std > 0.0 else 0.0


def compute_newey_west_tstat(
    ic_series: pd.Series,
    n_lags: int = 6,
) -> tuple[float, float]:
    """Compute Newey-West t-statistic for IC significance.

    Args:
        ic_series: Time series of IC values.
        n_lags: Number of lags for HAC standard errors.

    Returns:
        (t_statistic, p_value).
    """
    n = len(ic_series)
    if n < 3:
        return 0.0, 1.0

    mean_ic = float(ic_series.mean())
    demeaned = ic_series - mean_ic

    gamma_0 = float((demeaned**2).mean())
    nw_var = gamma_0

    for lag in range(1, min(n_lags, n - 1) + 1):
        weight = 1.0 - lag / (n_lags + 1)
        lag_vals: npt.NDArray[np.float64] = np.asarray(
            demeaned.iloc[lag:], dtype=np.float64
        )
        lead_vals: npt.NDArray[np.float64] = np.asarray(
            demeaned.iloc[:-lag], dtype=np.float64
        )
        gamma_j = float((lag_vals * lead_vals).mean())
        nw_var += 2 * weight * gamma_j

    nw_var = max(nw_var, 1e-12)
    se = float(np.sqrt(nw_var / n))

    if se == 0:
        return 0.0, 1.0

    t_stat = mean_ic / se

    from scipy import stats as sp_stats

    p_value = float(2.0 * (1.0 - sp_stats.t.cdf(abs(t_stat), df=n - 1)))
    return float(t_stat), p_value


def compute_ic_stats(
    ic_series: pd.Series,
    lags: int = 5,
) -> ICStats:
    """Compute full IC statistics including Newey-West t-stat and ICIR.

    Args:
        ic_series: Time series of IC values (one per cross-section date).
        lags: Number of lags for Newey-West HAC standard errors.

    Returns:
        Dataclass containing ``mean``, ``variance_nw``, ``t_stat_nw``,
        ``p_value``, and ``icir``.
    """
    from scipy import stats as sp_stats

    n = len(ic_series)
    if n < 3:
        return ICStats(
            mean=float(np.nan),
            variance_nw=float(np.nan),
            t_stat_nw=0.0,
            p_value=1.0,
            icir=float(np.nan),
        )

    mean_ic = float(ic_series.mean())
    ic_std = float(cast(float, ic_series.std(ddof=1)))
    icir = mean_ic / ic_std if ic_std > 0.0 else 0.0

    demeaned = ic_series - mean_ic
    gamma_0 = float((demeaned**2).mean())
    nw_var = gamma_0

    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1)
        lag_vals: npt.NDArray[np.float64] = np.asarray(
            demeaned.iloc[lag:], dtype=np.float64
        )
        lead_vals: npt.NDArray[np.float64] = np.asarray(
            demeaned.iloc[:-lag], dtype=np.float64
        )
        gamma_j = float((lag_vals * lead_vals).mean())
        nw_var += 2 * weight * gamma_j

    nw_var = max(nw_var, 1e-12)
    se = float(np.sqrt(nw_var / n))
    t_stat_nw = mean_ic / se if se > 0.0 else 0.0
    p_value = float(2.0 * (1.0 - sp_stats.t.cdf(abs(t_stat_nw), df=n - 1)))

    return ICStats(
        mean=mean_ic,
        variance_nw=nw_var,
        t_stat_nw=t_stat_nw,
        p_value=p_value,
        icir=icir,
    )


def correct_pvalues(
    p_values: npt.NDArray[np.float64],
    alpha: float = 0.05,
) -> CorrectedPValues:
    """Apply Holm-Bonferroni and Benjamini-Hochberg multiple testing corrections.

    Args:
        p_values: Raw p-values in any order, shape (m,).
        alpha: Significance level used to compute the adjustments (does not
            filter here; callers compare adjusted p-values against ``alpha``).

    Returns:
        ``holm`` — FWER-controlling Holm-Bonferroni adjusted p-values.
        ``bh``   — FDR-controlling Benjamini-Hochberg adjusted p-values.
        Both arrays are returned in the **same order** as the input.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        empty: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        return CorrectedPValues(holm=empty, bh=empty)

    sort_idx = np.argsort(p)
    sorted_p = p[sort_idx]
    ranks = np.arange(1, m + 1, dtype=np.float64)

    # Holm-Bonferroni: p_adj[k] = p[k] * (m - k + 1), then cumulative max
    holm_sorted = np.minimum(1.0, sorted_p * (m - ranks + 1))
    holm_sorted = np.maximum.accumulate(holm_sorted)

    # Benjamini-Hochberg: p_adj[k] = p[k] * m/k, then cumulative min from right
    bh_sorted = np.minimum(1.0, sorted_p * (m / ranks))
    bh_sorted = np.minimum.accumulate(bh_sorted[::-1])[::-1]

    holm_out = np.empty(m, dtype=np.float64)
    bh_out = np.empty(m, dtype=np.float64)
    holm_out[sort_idx] = holm_sorted
    bh_out[sort_idx] = bh_sorted

    return CorrectedPValues(holm=holm_out, bh=bh_out)


def validate_factor_universe(
    ic_matrix: pd.DataFrame,
    lags: int = 5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Validate all factors simultaneously with multiple testing correction.

    Args:
        ic_matrix: Dates x factors matrix of IC values (one IC per period
            per factor).
        lags: Number of Newey-West HAC lags.
        alpha: Significance level for both FWER and FDR rejection decisions.

    Returns:
        Factor x statistic summary with columns:
        ``ic_mean``, ``icir``, ``t_stat_nw``, ``p_value_raw``,
        ``p_value_holm``, ``p_value_bh``, ``significant_holm``,
        ``significant_bh``.
    """
    factors = list(ic_matrix.columns)
    records: list[dict[str, float]] = []
    raw_pvalues: list[float] = []

    for factor in factors:
        stats = compute_ic_stats(ic_matrix[factor].dropna(), lags=lags)
        records.append(
            {
                "ic_mean": stats.mean,
                "icir": stats.icir,
                "t_stat_nw": stats.t_stat_nw,
                "p_value_raw": stats.p_value,
            }
        )
        raw_pvalues.append(stats.p_value)

    pvals_arr: npt.NDArray[np.float64] = np.asarray(raw_pvalues, dtype=np.float64)
    corrected = correct_pvalues(pvals_arr, alpha=alpha)

    for i, rec in enumerate(records):
        rec["p_value_holm"] = float(corrected.holm[i])
        rec["p_value_bh"] = float(corrected.bh[i])
        rec["significant_holm"] = float(corrected.holm[i] <= alpha)
        rec["significant_bh"] = float(corrected.bh[i] <= alpha)

    return pd.DataFrame(records, index=factors)


def compute_composite_ic(
    composite_scores_history: pd.DataFrame,
    returns_history: pd.DataFrame,
    newey_west_lags: int = 6,
    t_stat_threshold: float = 2.0,
    min_observations: int = 3,
) -> CompositeICResult:
    """Compute IC statistics for the composite score signal.

    Args:
        composite_scores_history: Dates x tickers matrix of composite scores.
        returns_history: Dates x tickers matrix of forward returns.
        newey_west_lags: Number of lags for HAC standard errors.
        t_stat_threshold: Threshold for significance decision.
        min_observations: Minimum non-NaN observations per cross-section date.

    Returns:
        IC statistics for the composite score.  The
        ``best_individual_ic`` and ``outperforms_best_individual``
        fields are populated by ``run_factor_validation`` when
        individual factor results are available.
    """
    ic_series = compute_ic_series(
        composite_scores_history,
        returns_history,
        "composite",
        min_observations=min_observations,
    )

    if len(ic_series) < 2:
        return CompositeICResult(
            mean_ic=float("nan"),
            ic_std=float("nan"),
            t_stat=0.0,
            p_value=1.0,
            icir=float("nan"),
            significant=False,
            best_individual_ic=float("nan"),
            outperforms_best_individual=False,
        )

    mean_ic = float(ic_series.mean())
    ic_std = float(cast(float, ic_series.std(ddof=1)))
    icir = mean_ic / ic_std if ic_std > 0.0 else 0.0

    t_stat, p_value = compute_newey_west_tstat(ic_series, n_lags=newey_west_lags)
    significant = abs(t_stat) >= t_stat_threshold

    return CompositeICResult(
        mean_ic=mean_ic,
        ic_std=ic_std,
        t_stat=t_stat,
        p_value=p_value,
        icir=icir,
        significant=significant,
        best_individual_ic=float("nan"),
        outperforms_best_individual=False,
    )


def compute_quantile_spread(
    factor_scores: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 5,
) -> float:
    """Compute long-short quantile spread return.

    Args:
        factor_scores: Cross-sectional factor scores.
        forward_returns: Forward returns.
        n_quantiles: Number of quantile buckets.

    Returns:
        Top quantile return minus bottom quantile return.
    """
    common = factor_scores.dropna().index.intersection(
        forward_returns.dropna().index,
    )
    if len(common) < n_quantiles:
        return float(np.nan)

    scores = factor_scores.loc[common]
    returns = forward_returns.loc[common]

    pct_ranks = scores.rank(pct=True, method="average")
    labels = pd.cut(pct_ranks, bins=n_quantiles, labels=False, include_lowest=True)
    quantile_returns: pd.Series = returns.groupby(labels).mean()

    if len(quantile_returns) < 2:
        return float(np.nan)

    return float(quantile_returns.iloc[-1] - quantile_returns.iloc[0])


# Guard near-singular regressions: if 1 - R² falls below this threshold the
# residual variance is dominated by floating-point noise and VIF is meaningless.
# 1e-10 preserves all diagnostically meaningful VIF values (up to ~1e10) while
# mapping IEEE 754 near-exact collinearity artefacts to inf.  Chosen to match
# statsmodels' numerical rank tolerance for OLS singular matrices.
_VIF_R2_SINGULARITY_TOL: float = 1e-10


def compute_vif(factor_matrix: pd.DataFrame) -> pd.Series:
    """Compute variance inflation factors for multicollinearity.

    Args:
        factor_matrix: Tickers x factors matrix (no NaN).  Must contain at
            least 2 factors.

    Returns:
        VIF per factor.  Values are >= 1.0 by construction.

    Raises:
        ValueError: If fewer than 2 factor columns are provided.
    """
    if len(factor_matrix.columns) < 2:
        raise DataError(
            "compute_vif requires at least 2 factor columns, "
            f"got {len(factor_matrix.columns)}"
        )
    clean = factor_matrix.dropna()
    if len(clean) < 2:
        return pd.Series(1.0, index=factor_matrix.columns)

    vifs: dict[str, float] = {}
    X: npt.NDArray[np.float64] = clean.values
    for i, col in enumerate(clean.columns):
        mask = [j for j in range(X.shape[1]) if j != i]
        y = X[:, i]
        X_other = X[:, mask]

        X_aug = np.column_stack([np.ones(len(y)), X_other])

        try:
            coeffs = np.linalg.lstsq(X_aug, y, rcond=None)[0]
            y_hat = X_aug @ coeffs
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vifs[str(col)] = (
                1.0 / (1.0 - r_sq)
                if 1.0 - r_sq > _VIF_R2_SINGULARITY_TOL
                else float(np.inf)
            )
        except np.linalg.LinAlgError:
            vifs[str(col)] = float(np.inf)

    return pd.Series(vifs)


def benjamini_hochberg(
    p_values: pd.Series,
    alpha: float = 0.05,
) -> pd.Series:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: Raw p-values indexed by factor name.
        alpha: FDR significance level.

    Returns:
        Boolean series indicating significant factors.
    """
    sorted_pvals = p_values.sort_values()
    n = len(sorted_pvals)
    thresholds: npt.NDArray[np.float64] = alpha * (np.arange(1, n + 1) / n)
    pval_arr: npt.NDArray[np.float64] = sorted_pvals.to_numpy(
        dtype=np.float64,
    )
    significant = pval_arr <= thresholds
    # All factors up to the last significant one are significant
    if significant.any():
        last_sig = int(np.max(np.where(significant)))
        significant[: last_sig + 1] = True
    return pd.Series(significant, index=sorted_pvals.index, dtype=bool).reindex(
        p_values.index
    )


def _factor_to_group_name(factor_name: str) -> str:
    """Map a factor name to its group name for benchmark lookup."""
    from optimizer.factors._config import FactorType

    try:
        factor_type = FactorType(factor_name)
        return FACTOR_GROUP_MAPPING[factor_type].value
    except (ValueError, KeyError):
        return factor_name.lower()


def run_factor_validation(
    factor_scores_history: dict[str, pd.DataFrame],
    returns_history: pd.DataFrame,
    config: FactorValidationConfig | None = None,
    composite_scores_history: pd.DataFrame | None = None,
) -> FactorValidationReport:
    """Run complete factor validation suite.

    Args:
        factor_scores_history: Factor name -> (dates x tickers) score history.
        returns_history: Dates x tickers forward return matrix.
        config: Validation parameters.
        composite_scores_history: Dates x tickers matrix of composite scores.
            When provided, IC analysis is run on the composite signal and
            compared against the best individual factor IC.

    Returns:
        Complete validation results.
    """
    if config is None:
        config = FactorValidationConfig()

    report = FactorValidationReport()
    p_values: dict[str, float] = {}

    for factor_name, scores_df in factor_scores_history.items():
        ic_series = compute_ic_series(
            scores_df,
            returns_history,
            factor_name,
            min_observations=config.min_ic_observations,
        )
        if len(ic_series) == 0:
            continue

        t_stat, p_value = compute_newey_west_tstat(ic_series, config.newey_west_lags)
        significant = abs(t_stat) >= config.t_stat_threshold

        report.ic_results.append(
            ICResult(
                factor_name=factor_name,
                mean_ic=float(ic_series.mean()),
                ic_std=float(cast(float, ic_series.std())),
                t_stat=t_stat,
                p_value=p_value,
                significant=significant,
            )
        )
        p_values[factor_name] = p_value

        common_dates = scores_df.index.intersection(
            returns_history.index,
        )
        if len(common_dates) > 0:
            latest = common_dates[-1]
            spread = compute_quantile_spread(
                scores_df.loc[latest],
                returns_history.loc[latest],
                n_quantiles=config.n_quantiles,
            )
            if not np.isnan(spread):
                group_name = _factor_to_group_name(factor_name)
                bench = FACTOR_SPREAD_BENCHMARKS.get(group_name)
                in_bench = (
                    bench[0] <= abs(spread) <= bench[1] if bench is not None else False
                )
                report.quantile_spreads.append(
                    QuantileSpreadResult(
                        factor_name=factor_name,
                        spread=spread,
                        quantile_returns=[],
                        within_benchmark=in_bench,
                    )
                )

    if p_values:
        factor_names = list(p_values.keys())
        pval_arr = np.array([p_values[f] for f in factor_names], dtype=np.float64)
        corrected = correct_pvalues(pval_arr, config.fdr_alpha)

        report.significant_factors = [
            f
            for f, p in zip(factor_names, corrected.bh, strict=True)
            if p <= config.fdr_alpha
        ]
        report.significant_factors_holm = [
            f
            for f, p in zip(factor_names, corrected.holm, strict=True)
            if p <= config.fdr_alpha
        ]

    if composite_scores_history is not None:
        composite_result = compute_composite_ic(
            composite_scores_history,
            returns_history,
            newey_west_lags=config.newey_west_lags,
            t_stat_threshold=config.t_stat_threshold,
            min_observations=config.composite_min_observations,
        )
        if report.ic_results:
            best_ic = max(r.mean_ic for r in report.ic_results)
            composite_result.best_individual_ic = best_ic
            composite_result.outperforms_best_individual = (
                composite_result.mean_ic > best_ic
            )
        report.composite_ic_result = composite_result

    return report
