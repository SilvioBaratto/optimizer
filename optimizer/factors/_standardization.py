"""Cross-sectional factor standardization."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.factors._config import StandardizationConfig, StandardizationMethod

logger = logging.getLogger(__name__)


def winsorize_cross_section(
    scores: pd.Series,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.Series:
    """Clip scores at percentile boundaries.

    Parameters
    ----------
    scores : pd.Series
        Raw factor scores.
    lower_pct : float
        Lower percentile (0-1).
    upper_pct : float
        Upper percentile (0-1).

    Returns
    -------
    pd.Series
        Winsorized scores.
    """
    valid = scores.dropna()
    if len(valid) == 0:
        return scores
    lower = valid.quantile(lower_pct)
    upper = valid.quantile(upper_pct)
    return scores.clip(lower=lower, upper=upper)


def z_score_standardize(scores: pd.Series) -> pd.Series:
    """Z-score standardization: (x - mean) / std.

    Parameters
    ----------
    scores : pd.Series
        Factor scores (may contain NaN).

    Returns
    -------
    pd.Series
        Standardized scores with mean 0 and std 1.
    """
    mean = scores.mean()
    std = scores.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=scores.index)
    return (scores - mean) / std


def rank_normal_standardize(scores: pd.Series) -> pd.Series:
    """Rank-normal (inverse normal) standardization.

    Uses ``Phi^-1((rank - 0.5) / N)`` to map ranks to a normal
    distribution, robust to heavy-tailed distributions.

    Parameters
    ----------
    scores : pd.Series
        Factor scores (may contain NaN).

    Returns
    -------
    pd.Series
        Rank-normalized scores.
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

    Parameters
    ----------
    scores : pd.Series
        Standardized factor scores.
    sector_labels : pd.Series
        Sector label per ticker.
    country_labels : pd.Series or None
        Country label per ticker for country neutralization.

    Returns
    -------
    pd.Series
        Sector-neutralized scores.
    """
    if country_labels is not None:
        group_key = sector_labels.astype(str) + "_" + country_labels.astype(str)
    else:
        group_key = sector_labels

    aligned = scores.reindex(group_key.index)
    group_means = aligned.groupby(group_key).transform("mean")
    return aligned - group_means


def standardize_factor(
    raw_scores: pd.Series,
    config: StandardizationConfig | None = None,
    sector_labels: pd.Series | None = None,
    country_labels: pd.Series | None = None,
) -> pd.Series:
    """Full standardization pipeline for a single factor.

    Parameters
    ----------
    raw_scores : pd.Series
        Raw factor values.
    config : StandardizationConfig or None
        Standardization parameters.
    sector_labels : pd.Series or None
        Sector labels for neutralization.
    country_labels : pd.Series or None
        Country labels for neutralization.

    Returns
    -------
    pd.Series
        Standardized factor scores.
    """
    if config is None:
        config = StandardizationConfig()

    # 1. Winsorize
    scores = winsorize_cross_section(
        raw_scores,
        lower_pct=config.winsorize_lower,
        upper_pct=config.winsorize_upper,
    )

    # 2. Standardize
    if config.method == StandardizationMethod.Z_SCORE:
        scores = z_score_standardize(scores)
    else:
        scores = rank_normal_standardize(scores)

    # 3. Sector/country neutralize
    neutralized = False
    if config.neutralize_sector and sector_labels is not None:
        country = country_labels if config.neutralize_country else None
        scores = neutralize_sector(scores, sector_labels, country)
        neutralized = True

    # 4. Re-standardize after neutralization (optional)
    if config.re_standardize_after_neutralization and neutralized:
        scores = z_score_standardize(scores)

    return scores


def standardize_all_factors(
    raw_factors: pd.DataFrame,
    config: StandardizationConfig | None = None,
    sector_labels: pd.Series | None = None,
    country_labels: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize all factors and compute coverage.

    Parameters
    ----------
    raw_factors : pd.DataFrame
        Tickers x factors matrix of raw values.
    config : StandardizationConfig or None
        Standardization parameters.
    sector_labels : pd.Series or None
        Sector labels for neutralization.
    country_labels : pd.Series or None
        Country labels for neutralization.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (standardized_scores, coverage) where coverage is a
        boolean DataFrame indicating non-NaN values.
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

    Parameters
    ----------
    factor_scores : pd.DataFrame
        Tickers × factors matrix of factor scores.
    method : str
        Projection method.  Only ``"pca"`` is supported.
    min_variance_explained : float
        Minimum cumulative explained variance ratio for retained
        components.  Must be in ``(0, 1]``.

    Returns
    -------
    pd.DataFrame
        Tickers × PCs matrix with columns named ``PC1``, ``PC2``, ....
        Rows with NaN in the input are filled with NaN in the output
        but otherwise preserve the original index.

    Raises
    ------
    ConfigurationError
        If *method* is not ``"pca"``.
    DataError
        If fewer than 2 factors or fewer than 2 non-NaN observations.
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
