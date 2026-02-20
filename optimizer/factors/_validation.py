"""Factor validation and statistical testing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from optimizer.factors._config import FactorValidationConfig

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


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
class QuantileSpreadResult:
    """Quantile spread analysis results for a single factor."""

    factor_name: str
    spread: float
    quantile_returns: list[float]


@dataclass
class FactorValidationReport:
    """Complete validation report for all factors."""

    ic_results: list[ICResult] = field(default_factory=list)
    quantile_spreads: list[QuantileSpreadResult] = field(
        default_factory=list,
    )
    vif_scores: pd.Series | None = None
    significant_factors: list[str] = field(default_factory=list)


@dataclass
class ICStats:
    """Full IC statistics for a single factor including Newey-West inference.

    Attributes
    ----------
    mean : float
        Mean IC over the evaluation period.
    variance_nw : float
        Newey-West HAC variance of the IC series.
    t_stat_nw : float
        Newey-West adjusted t-statistic: ``IC_mean / sqrt(Var_NW / T)``.
    p_value : float
        Two-tailed p-value derived from the Newey-West t-statistic.
    icir : float
        Information Coefficient Information Ratio: ``mean(IC) / std(IC)``.
    """

    mean: float
    variance_nw: float
    t_stat_nw: float
    p_value: float
    icir: float


@dataclass
class CorrectedPValues:
    """Multiple-testing corrected p-values.

    Attributes
    ----------
    holm : ndarray
        Holm-Bonferroni adjusted p-values (controls FWER).
    bh : ndarray
        Benjamini-Hochberg adjusted p-values (controls FDR).
    """

    holm: npt.NDArray[np.float64]
    bh: npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# IC analysis
# ---------------------------------------------------------------------------


def compute_monthly_ic(
    factor_scores: pd.Series,
    forward_returns: pd.Series,
) -> float:
    """Compute rank information coefficient (Spearman correlation).

    Parameters
    ----------
    factor_scores : pd.Series
        Cross-sectional factor scores.
    forward_returns : pd.Series
        Forward returns for the same tickers.

    Returns
    -------
    float
        Rank IC (Spearman correlation).
    """
    common = factor_scores.dropna().index.intersection(
        forward_returns.dropna().index,
    )
    if len(common) < 3:
        return float(np.nan)
    return float(
        factor_scores.loc[common].corr(
            forward_returns.loc[common], method="spearman"
        )
    )


def compute_ic_series(
    factor_scores_history: pd.DataFrame,
    returns_history: pd.DataFrame,
    factor_name: str,
) -> pd.Series:
    """Compute IC time series for a factor.

    Parameters
    ----------
    factor_scores_history : pd.DataFrame
        Dates x tickers matrix of factor scores.
    returns_history : pd.DataFrame
        Dates x tickers matrix of forward returns.
    factor_name : str
        Used only for labeling.

    Returns
    -------
    pd.Series
        IC values indexed by date.
    """
    common_dates = factor_scores_history.index.intersection(
        returns_history.index,
    )
    ics: dict[object, float] = {}
    for date in common_dates:
        ic = compute_monthly_ic(
            factor_scores_history.loc[date],
            returns_history.loc[date],
        )
        if not np.isnan(ic):
            ics[date] = ic

    return pd.Series(ics, name=factor_name, dtype=float)


def compute_icir(ic_series: pd.Series) -> float:
    """Compute the IC Information Ratio (mean IC / std IC).

    ICIR penalises factors with high average IC but also high IC
    volatility (inconsistent predictors).  Use this as the weighting
    signal in ICIR-weighted composite scoring.

    Parameters
    ----------
    ic_series : pd.Series
        Time series of IC values (one per cross-section date).

    Returns
    -------
    float
        ICIR value, or 0.0 if ``std(IC) == 0`` or fewer than
        2 non-NaN observations.
    """
    clean = ic_series.dropna()
    if len(clean) < 2:
        return 0.0
    mean_ic = float(clean.mean())
    ic_std = float(clean.std(ddof=1))
    return mean_ic / ic_std if ic_std > 0.0 else 0.0


def compute_newey_west_tstat(
    ic_series: pd.Series,
    n_lags: int = 6,
) -> tuple[float, float]:
    """Compute Newey-West t-statistic for IC significance.

    Parameters
    ----------
    ic_series : pd.Series
        Time series of IC values.
    n_lags : int
        Number of lags for HAC standard errors.

    Returns
    -------
    tuple[float, float]
        (t_statistic, p_value).
    """
    n = len(ic_series)
    if n < 3:
        return 0.0, 1.0

    mean_ic = float(ic_series.mean())
    demeaned = ic_series - mean_ic

    # Newey-West variance estimator
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

    p_value = float(
        2.0 * (1.0 - sp_stats.t.cdf(abs(t_stat), df=n - 1))
    )
    return float(t_stat), p_value


def compute_ic_stats(
    ic_series: pd.Series,
    lags: int = 5,
) -> ICStats:
    """Compute full IC statistics including Newey-West t-stat and ICIR.

    Parameters
    ----------
    ic_series : pd.Series
        Time series of IC values (one per cross-section date).
    lags : int
        Number of lags for Newey-West HAC standard errors.

    Returns
    -------
    ICStats
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
    ic_std = float(ic_series.std(ddof=1))
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

    Parameters
    ----------
    p_values : ndarray, shape (m,)
        Raw p-values in any order.
    alpha : float
        Significance level used to compute the adjustments (does not filter
        here; callers compare adjusted p-values against ``alpha``).

    Returns
    -------
    CorrectedPValues
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

    # Restore original order
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

    Parameters
    ----------
    ic_matrix : pd.DataFrame
        Dates × factors matrix of IC values (one IC per period per factor).
    lags : int
        Number of Newey-West HAC lags.
    alpha : float
        Significance level for both FWER and FDR rejection decisions.

    Returns
    -------
    pd.DataFrame
        Factor × statistic summary with columns:
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


# ---------------------------------------------------------------------------
# Quantile spread
# ---------------------------------------------------------------------------


def compute_quantile_spread(
    factor_scores: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 5,
) -> float:
    """Compute long-short quantile spread return.

    Parameters
    ----------
    factor_scores : pd.Series
        Cross-sectional factor scores.
    forward_returns : pd.Series
        Forward returns.
    n_quantiles : int
        Number of quantile buckets.

    Returns
    -------
    float
        Top quantile return minus bottom quantile return.
    """
    common = factor_scores.dropna().index.intersection(
        forward_returns.dropna().index,
    )
    if len(common) < n_quantiles:
        return float(np.nan)

    scores = factor_scores.loc[common]
    returns = forward_returns.loc[common]

    quantile_labels = pd.qcut(
        scores, n_quantiles, labels=False, duplicates="drop"
    )
    quantile_returns: pd.Series = returns.groupby(quantile_labels).mean()

    if len(quantile_returns) < 2:
        return float(np.nan)

    return float(quantile_returns.iloc[-1] - quantile_returns.iloc[0])


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------


def compute_vif(factor_matrix: pd.DataFrame) -> pd.Series:
    """Compute variance inflation factors for multicollinearity.

    Parameters
    ----------
    factor_matrix : pd.DataFrame
        Tickers x factors matrix (no NaN).  Must contain at least 2 factors.

    Returns
    -------
    pd.Series
        VIF per factor.  Values are ≥ 1.0 by construction.

    Raises
    ------
    ValueError
        If fewer than 2 factor columns are provided.
    """
    if len(factor_matrix.columns) < 2:
        raise ValueError(
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

        # Add intercept
        X_aug = np.column_stack([np.ones(len(y)), X_other])

        # OLS: R^2 = 1 - RSS/TSS
        try:
            coeffs = np.linalg.lstsq(X_aug, y, rcond=None)[0]
            y_hat = X_aug @ coeffs
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vifs[str(col)] = (
                1.0 / (1.0 - r_sq) if r_sq < 1.0 else float(np.inf)
            )
        except np.linalg.LinAlgError:
            vifs[str(col)] = float(np.inf)

    return pd.Series(vifs)


# ---------------------------------------------------------------------------
# FDR correction
# ---------------------------------------------------------------------------


def benjamini_hochberg(
    p_values: pd.Series,
    alpha: float = 0.05,
) -> pd.Series:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : pd.Series
        Raw p-values indexed by factor name.
    alpha : float
        FDR significance level.

    Returns
    -------
    pd.Series
        Boolean series indicating significant factors.
    """
    sorted_pvals = p_values.sort_values()
    n = len(sorted_pvals)
    thresholds: npt.NDArray[np.float64] = alpha * (
        np.arange(1, n + 1) / n
    )
    pval_arr: npt.NDArray[np.float64] = sorted_pvals.to_numpy(
        dtype=np.float64,
    )
    significant = pval_arr <= thresholds
    # All factors up to the last significant one are significant
    if significant.any():
        last_sig = int(np.max(np.where(significant)))
        significant[: last_sig + 1] = True
    return pd.Series(
        significant, index=sorted_pvals.index, dtype=bool
    ).reindex(p_values.index)


# ---------------------------------------------------------------------------
# Full validation
# ---------------------------------------------------------------------------


def run_factor_validation(
    factor_scores_history: dict[str, pd.DataFrame],
    returns_history: pd.DataFrame,
    config: FactorValidationConfig | None = None,
) -> FactorValidationReport:
    """Run complete factor validation suite.

    Parameters
    ----------
    factor_scores_history : dict[str, pd.DataFrame]
        Factor name -> (dates x tickers) score history.
    returns_history : pd.DataFrame
        Dates x tickers forward return matrix.
    config : FactorValidationConfig or None
        Validation parameters.

    Returns
    -------
    FactorValidationReport
        Complete validation results.
    """
    if config is None:
        config = FactorValidationConfig()

    report = FactorValidationReport()
    p_values: dict[str, float] = {}

    for factor_name, scores_df in factor_scores_history.items():
        # IC analysis
        ic_series = compute_ic_series(
            scores_df, returns_history, factor_name
        )
        if len(ic_series) == 0:
            continue

        t_stat, p_value = compute_newey_west_tstat(
            ic_series, config.newey_west_lags
        )
        significant = abs(t_stat) >= config.t_stat_threshold

        report.ic_results.append(
            ICResult(
                factor_name=factor_name,
                mean_ic=float(ic_series.mean()),
                ic_std=float(ic_series.std()),
                t_stat=t_stat,
                p_value=p_value,
                significant=significant,
            )
        )
        p_values[factor_name] = p_value

        # Quantile spread (use latest cross-section)
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
                report.quantile_spreads.append(
                    QuantileSpreadResult(
                        factor_name=factor_name,
                        spread=spread,
                        quantile_returns=[],
                    )
                )

    # FDR correction
    if p_values:
        pval_series = pd.Series(p_values)
        fdr_sig = benjamini_hochberg(pval_series, config.fdr_alpha)
        report.significant_factors = list(fdr_sig.index[fdr_sig])

    return report
