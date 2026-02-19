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
        lag_vals: npt.NDArray[np.float64] = demeaned.iloc[lag:].values
        lead_vals: npt.NDArray[np.float64] = demeaned.iloc[:-lag].values
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
        Tickers x factors matrix (no NaN).

    Returns
    -------
    pd.Series
        VIF per factor.
    """
    clean = factor_matrix.dropna()
    if len(clean) < 2 or len(clean.columns) < 2:
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
