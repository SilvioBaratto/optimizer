"""Cross-sectional factor standardization."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.factors._config import (
    FACTOR_DIRECTION,
    StandardizationConfig,
    StandardizationMethod,
    WinsorizeMethod,
)

logger = logging.getLogger(__name__)


def winsorize_cross_section(
    scores: pd.Series,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.Series:
    """Clip scores at percentile boundaries.

    Args:
        scores: Raw factor scores.
        lower_pct: Lower percentile boundary, in [0, 1].
        upper_pct: Upper percentile boundary, in [0, 1].

    Returns:
        Winsorized scores with values outside [lower_pct, upper_pct] clipped.
    """
    valid = scores.dropna()
    if len(valid) == 0:
        return scores
    lower = valid.quantile(lower_pct)
    upper = valid.quantile(upper_pct)
    return scores.clip(lower=lower, upper=upper)


def winsorize_cross_section_mad(
    scores: pd.Series,
    mad_multiplier: float = 3.0,
) -> pd.Series:
    """Clip scores using Median Absolute Deviation (MAD).

    Uses the normal-consistent scale factor ``1.4826 * MAD`` to set clip
    boundaries at ``median +/- mad_multiplier * scale``, following the
    MSCI Barra USE4 convention (+/-3 MAD).

    Args:
        scores: Raw factor scores (may contain NaN).
        mad_multiplier: Number of scaled-MAD units for clip boundaries.

    Returns:
        Winsorized scores.
    """
    valid = scores.dropna()
    if len(valid) == 0:
        return scores
    med = valid.median()
    mad = (valid - med).abs().median()
    if mad == 0.0:
        return scores
    scale = 1.4826 * mad
    lower = med - mad_multiplier * scale
    upper = med + mad_multiplier * scale
    return scores.clip(lower=lower, upper=upper)


def z_score_standardize(scores: pd.Series) -> pd.Series:
    """Standardize scores as (x - mean) / std.

    Args:
        scores: Factor scores (may contain NaN).

    Returns:
        Scores with mean 0 and std 1; a zero-filled series when std is 0 or NaN.
    """
    mean = scores.mean()
    std = scores.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=scores.index)
    return (scores - mean) / std


def rank_normal_standardize(scores: pd.Series) -> pd.Series:
    """Standardize scores via rank-normal (inverse normal) transform.

    Uses ``Phi^-1((rank - 0.5) / N)`` to map ranks to a normal
    distribution, robust to heavy-tailed distributions.

    Args:
        scores: Factor scores (may contain NaN).

    Returns:
        Rank-normalized scores; NaN positions in the input remain NaN.
    """
    valid = scores.dropna()
    if len(valid) == 0:
        return scores
    ranks = valid.rank()
    n = len(valid)
    uniform = (ranks - 0.5) / n
    normal_scores = pd.Series(
        sp_stats.norm.ppf(uniform),
        index=valid.index,
    )
    return normal_scores.reindex(scores.index)


def neutralize_sector(
    scores: pd.Series,
    sector_labels: pd.Series,
    country_labels: pd.Series | None = None,
) -> pd.Series:
    """Demean scores within each sector (and optionally country).

    Args:
        scores: Standardized factor scores.
        sector_labels: Sector label per ticker.
        country_labels: Country label per ticker for country neutralization.
            When provided, neutralization groups by (sector, country) pairs.

    Returns:
        Sector-neutralized scores.
    """
    if country_labels is not None:
        group_key = sector_labels.astype(str) + "_" + country_labels.astype(str)
    else:
        group_key = sector_labels

    aligned = scores.reindex(group_key.index)
    group_means = aligned.groupby(group_key).transform("mean")
    return aligned - group_means


def _resolve_method(
    factor_name: str,
    config: StandardizationConfig,
) -> StandardizationMethod:
    """Resolve standardization method for a factor, respecting overrides."""
    if config.factor_method_overrides and factor_name:
        overrides = dict(config.factor_method_overrides)
        if factor_name in overrides:
            return StandardizationMethod(overrides[factor_name])
    return config.method


def standardize_factor(
    raw_scores: pd.Series,
    config: StandardizationConfig | None = None,
    sector_labels: pd.Series | None = None,
    country_labels: pd.Series | None = None,
    *,
    factor_name: str = "",
) -> pd.Series:
    """Apply the full standardization pipeline to a single factor.

    Args:
        raw_scores: Raw factor values.
        config: Standardization parameters; defaults to ``StandardizationConfig()``
            when ``None``.
        sector_labels: Sector labels for neutralization.
        country_labels: Country labels for neutralization.
        factor_name: Column name of the factor, used to look up per-factor method
            overrides in ``config.factor_method_overrides`` and the
            ``FACTOR_DIRECTION`` sign convention.

    Returns:
        Standardized factor scores.
    """
    if config is None:
        config = StandardizationConfig()

    if config.winsorize_method == WinsorizeMethod.MAD:
        scores = winsorize_cross_section_mad(raw_scores)
    else:
        scores = winsorize_cross_section(
            raw_scores,
            lower_pct=config.winsorize_lower,
            upper_pct=config.winsorize_upper,
        )

    # Invert "lower is better" factors so all downstream scores share the
    # same convention (higher = more desirable). Direction is +1 for factors
    # not listed in FACTOR_DIRECTION.
    direction = FACTOR_DIRECTION.get(factor_name, 1)
    if direction == -1:
        scores = scores * -1

    method = _resolve_method(factor_name, config)
    if method == StandardizationMethod.Z_SCORE:
        scores = z_score_standardize(scores)
    else:
        scores = rank_normal_standardize(scores)

    neutralized = False
    if config.neutralize_sector and sector_labels is not None:
        country = country_labels if config.neutralize_country else None
        scores = neutralize_sector(scores, sector_labels, country)
        neutralized = True

    if config.re_standardize_after_neutralization and neutralized:
        scores = z_score_standardize(scores)

    return scores


def standardize_all_factors(
    raw_factors: pd.DataFrame,
    config: StandardizationConfig | None = None,
    sector_labels: pd.Series | None = None,
    country_labels: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize all factors and compute per-ticker coverage.

    Args:
        raw_factors: Tickers x factors matrix of raw values.
        config: Standardization parameters; defaults to ``StandardizationConfig()``
            when ``None``.
        sector_labels: Sector labels for neutralization.
        country_labels: Country labels for neutralization.

    Returns:
        A tuple ``(standardized_scores, coverage)`` where ``coverage`` is a
        boolean DataFrame indicating non-NaN values in the standardized output.
    """
    if config is None:
        config = StandardizationConfig()

    standardized: dict[str, pd.Series] = {}
    for col in raw_factors.columns:
        standardized[col] = standardize_factor(
            raw_factors[col],
            config=config,
            sector_labels=sector_labels,
            country_labels=country_labels,
            factor_name=col,
        )

    scores = pd.DataFrame(standardized, index=raw_factors.index)
    coverage = scores.notna()
    return scores, coverage


def orthogonalize_factors(
    factor_scores: pd.DataFrame,
    method: str = "pca",
    min_variance_explained: float = 0.95,
) -> pd.DataFrame:
    """Project factor scores onto orthogonal principal components.

    Eliminates multicollinearity among factor scores by projecting
    them into a lower-dimensional PCA space.  Retains the minimum
    number of components that explain at least ``min_variance_explained``
    of the total variance.

    Args:
        factor_scores: Tickers x factors matrix of factor scores.
        method: Projection method.  Only ``"pca"`` is supported.
        min_variance_explained: Minimum cumulative explained variance ratio
            for retained components.  Must be in ``(0, 1]``.

    Returns:
        Tickers x PCs matrix with columns named ``PC1``, ``PC2``, ....
        Rows with NaN in the input are filled with NaN in the output
        but otherwise preserve the original index.

    Raises:
        ConfigurationError: If *method* is not ``"pca"``.
        DataError: If fewer than 2 factors or fewer than 2 non-NaN observations.
    """
    if method != "pca":
        raise ConfigurationError(
            f"Unsupported orthogonalization method {method!r}; only 'pca' is supported"
        )

    if factor_scores.shape[1] < 2:
        raise DataError(
            "orthogonalize_factors requires at least 2 factors, "
            f"got {factor_scores.shape[1]}"
        )

    clean = factor_scores.dropna()
    if len(clean) < 2:
        raise DataError(
            "orthogonalize_factors requires at least 2 non-NaN observations, "
            f"got {len(clean)}"
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(clean.to_numpy(dtype=np.float64))

    pca = PCA()
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, min_variance_explained)) + 1
    n_keep = min(n_keep, len(cumvar))

    projected = X @ pca.components_[:n_keep].T
    col_names = [f"PC{i + 1}" for i in range(n_keep)]

    result = pd.DataFrame(
        projected,
        index=clean.index,
        columns=col_names,
        dtype=float,
    )

    return result.reindex(factor_scores.index)
