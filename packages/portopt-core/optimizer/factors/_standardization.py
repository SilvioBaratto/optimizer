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

# Relative to the largest |value|, so tiny-scale factors (e.g. Amihud) are not
# mistaken for constants.  A constant series' std is only a few ulps of its
# magnitude (~1e-16 relative), far below this.
_CONSTANT_RTOL = 1e-12


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
        Scores with mean 0 and std 1.  When std is NaN or zero up to
        floating-point noise (constant values, or fewer than two valid
        observations) every valid value maps to 0.0.
        NaN positions in the input always remain NaN, matching
        `rank_normal_standardize`, so missing data is never reported as covered.
    """
    mean = scores.mean()
    std = scores.std()
    # pandas derives the variance from a rounded mean, so a constant non-dyadic
    # series (0.1, 1/3, ...) has std of a few ulps rather than exactly 0, and
    # dividing by it would inflate rounding noise into unit-variance scores.
    if np.isnan(std) or std <= _CONSTANT_RTOL * scores.abs().max():
        return pd.Series(
            np.where(scores.isna(), np.nan, 0.0), index=scores.index, dtype=float
        )
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


def _peer_group_key(
    index: pd.Index,
    sector_labels: pd.Series,
    country_labels: pd.Series | None,
) -> pd.Series:
    """Build the peer-group key per ticker; NaN when any label is missing."""
    sector = sector_labels.reindex(index)
    if country_labels is None:
        return sector
    country = country_labels.reindex(index)
    group_key = sector.astype(str) + "_" + country.astype(str)
    return group_key.where(sector.notna() & country.notna())


def _label_kind(country_labels: pd.Series | None) -> str:
    """Name the label(s) a ticker can be missing, for log wording."""
    return "sector" if country_labels is None else "sector/country"


def _count_unlabeled(group_key: pd.Series, scored: pd.Series) -> tuple[int, int]:
    """Return ``(scored tickers without a peer group, scored tickers)``."""
    return int((group_key.isna() & scored).sum()), int(scored.sum())


def _warn_unlabeled(
    scored: pd.Series,
    config: StandardizationConfig,
    sector_labels: pd.Series | None,
    country_labels: pd.Series | None,
) -> None:
    """Warn once about scored tickers that neutralization will leave untouched.

    Reported by the public ``standardize_*`` entry points rather than by
    ``neutralize_sector`` so a multi-factor run logs the (per-universe, not
    per-factor) count a single time.
    """
    if not config.neutralize_sector or sector_labels is None:
        return
    country = country_labels if config.neutralize_country else None
    group_key = _peer_group_key(scored.index, sector_labels, country)
    n_unlabeled, n_scored = _count_unlabeled(group_key, scored)
    if n_unlabeled:
        logger.warning(
            "%d of %d scored tickers have no %s label; "
            "their scores are left un-neutralized",
            n_unlabeled,
            n_scored,
            _label_kind(country),
        )


def neutralize_sector(
    scores: pd.Series,
    sector_labels: pd.Series,
    country_labels: pd.Series | None = None,
) -> pd.Series:
    """Demean scores within each sector (and optionally country).

    Tickers without a usable label (absent from ``sector_labels``, or a NaN
    sector, or a NaN country when ``country_labels`` is given) cannot be
    assigned a peer group.  They keep their standardized score un-demeaned
    rather than being turned into NaN, which would silently drop them from
    selection.  The count is logged at debug level; ``standardize_factor`` and
    ``standardize_all_factors`` raise it to a single warning per call.

    Args:
        scores: Standardized factor scores.
        sector_labels: Sector label per ticker.
        country_labels: Country label per ticker for country neutralization.
            When provided, neutralization groups by (sector, country) pairs.

    Returns:
        Sector-neutralized scores indexed exactly like ``scores``.
    """
    group_key = _peer_group_key(scores.index, sector_labels, country_labels)
    labeled = group_key.notna()

    n_unlabeled, n_scored = _count_unlabeled(group_key, scores.notna())
    if n_unlabeled:
        logger.debug(
            "neutralize_sector: %d of %d scored tickers have no %s label; "
            "their scores are left un-neutralized",
            n_unlabeled,
            n_scored,
            _label_kind(country_labels),
        )

    group_means = scores.groupby(group_key).transform("mean")
    residual = scores - group_means
    # A group of identical scores demeans to rounding noise rather than 0.0;
    # left as is, a later re-standardization would inflate it to unit variance.
    residual = residual.mask(residual.abs() <= _CONSTANT_RTOL * scores.abs().max(), 0.0)
    return residual.where(labeled, scores)


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
        Standardized factor scores.  Tickers without a sector (or country)
        label keep their un-neutralized score and are reported in one warning.
    """
    if config is None:
        config = StandardizationConfig()

    _warn_unlabeled(raw_scores.notna(), config, sector_labels, country_labels)
    return _standardize_scores(
        raw_scores, config, sector_labels, country_labels, factor_name
    )


def _standardize_scores(
    raw_scores: pd.Series,
    config: StandardizationConfig,
    sector_labels: pd.Series | None,
    country_labels: pd.Series | None,
    factor_name: str,
) -> pd.Series:
    """Run winsorize, direction, standardize and neutralize for one factor."""
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
    elif config.neutralize_country and country_labels is not None:
        # neutralize_sector only needs one group key; with no sector axis the
        # country labels are that key.
        scores = neutralize_sector(scores, country_labels)
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
        Tickers without a sector (or country) label are reported in a single
        warning for the whole call, not once per factor.
    """
    if config is None:
        config = StandardizationConfig()

    _warn_unlabeled(
        raw_factors.notna().any(axis=1), config, sector_labels, country_labels
    )

    standardized: dict[str, pd.Series] = {}
    for col in raw_factors.columns:
        standardized[col] = _standardize_scores(
            raw_factors[col], config, sector_labels, country_labels, col
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
