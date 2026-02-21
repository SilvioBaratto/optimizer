"""Multicollinearity diagnostics for factor score matrices."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from optimizer.exceptions import DataError
from optimizer.factors._validation import compute_vif

logger = logging.getLogger(__name__)


@dataclass
class FactorPCAResult:
    """Principal component analysis result for a factor score matrix.

    Attributes
    ----------
    explained_variance_ratio : ndarray, shape (n_components,)
        Fraction of variance explained by each principal component,
        sorted in descending order.
    loadings : pd.DataFrame, shape (n_factors, n_components)
        PCA loading matrix.  Rows are factor names; columns are
        ``PC1``, ``PC2``, ... .  Each column is a unit eigenvector of
        the correlation matrix of the factor scores.
    n_components_95pct : int
        Smallest number of components whose cumulative explained
        variance ratio is ≥ 0.95.
    """

    explained_variance_ratio: npt.NDArray[np.float64]
    loadings: pd.DataFrame
    n_components_95pct: int


def compute_factor_pca(
    scores: pd.DataFrame,
    n_components: int | None = None,
) -> FactorPCAResult:
    """Compute PCA on a cross-sectional factor score matrix.

    Rows with any NaN are dropped before fitting.  Scores are
    standardised (zero mean, unit variance per factor) so that PCA
    operates on the correlation structure rather than the covariance
    structure.

    Parameters
    ----------
    scores : pd.DataFrame
        Tickers × factors matrix of factor scores.  Columns are factor
        names; rows are asset observations.
    n_components : int or None, default None
        Number of principal components to retain.  ``None`` keeps all
        components (min(n_samples, n_features)).

    Returns
    -------
    FactorPCAResult
        See :class:`FactorPCAResult` for field descriptions.

    Raises
    ------
    ValueError
        If fewer than 2 factors or fewer than 2 observations are
        available after dropping NaN rows.
    """
    if scores.shape[1] < 2:
        raise DataError(
            "compute_factor_pca requires at least 2 factor columns, "
            f"got {scores.shape[1]}"
        )

    clean = scores.dropna()
    if len(clean) < 2:
        raise DataError(
            "compute_factor_pca requires at least 2 observations after "
            f"dropping NaN rows, got {len(clean)}"
        )

    scaler = StandardScaler()
    X: npt.NDArray[np.float64] = scaler.fit_transform(clean.to_numpy(dtype=np.float64))

    pca = PCA(n_components=n_components)
    pca.fit(X)

    evr: npt.NDArray[np.float64] = pca.explained_variance_ratio_
    n_comps = int(pca.n_components_)
    col_names = [f"PC{i + 1}" for i in range(n_comps)]

    # loadings: shape (n_factors, n_components) — transpose of components_
    loadings = pd.DataFrame(
        pca.components_.T,
        index=clean.columns,
        columns=col_names,
        dtype=float,
    )

    # Smallest n s.t. cumulative explained variance ≥ 0.95
    cumsum: npt.NDArray[np.float64] = np.cumsum(evr)
    idx = int(np.searchsorted(cumsum, 0.95))
    n_components_95pct = min(idx + 1, n_comps)

    return FactorPCAResult(
        explained_variance_ratio=evr,
        loadings=loadings,
        n_components_95pct=n_components_95pct,
    )


def flag_redundant_factors(
    scores: pd.DataFrame,
    vif_threshold: float = 10.0,
) -> list[str]:
    """Return factor names whose VIF exceeds *vif_threshold*.

    A VIF above the threshold indicates that the factor's variance is
    largely explained by the remaining factors, making it a candidate
    for merging or removal from the composite score.

    Parameters
    ----------
    scores : pd.DataFrame
        Tickers × factors matrix of factor scores.  Must contain at
        least 2 factor columns.
    vif_threshold : float, default 10.0
        VIF cutoff above which a factor is considered redundant.
        Commonly used values: 5 (conservative) or 10 (standard).

    Returns
    -------
    list[str]
        Factor names with ``VIF > vif_threshold``, in the order they
        appear in ``scores.columns``.  Empty list if none exceed the
        threshold.

    Raises
    ------
    ValueError
        Propagated from :func:`compute_vif` if fewer than 2 factors
        are provided.
    """
    vif = compute_vif(scores)
    return [str(name) for name, val in vif.items() if val > vif_threshold]


def check_survivorship_bias(
    returns: pd.DataFrame,
    final_periods: int = 12,
    zero_threshold: float = 1e-10,
) -> bool:
    """Check for potential survivorship bias in a return panel.

    Survivorship bias occurs when delisted or failed assets are excluded
    from the sample.  A simple heuristic: if **no** asset has near-zero
    returns in the final ``final_periods`` rows (i.e., no asset appears
    to have stopped trading), the panel may suffer from survivorship
    bias.

    Parameters
    ----------
    returns : pd.DataFrame
        Dates × assets return matrix.
    final_periods : int
        Number of trailing periods to inspect.
    zero_threshold : float
        Absolute threshold below which a return is considered "zero".

    Returns
    -------
    bool
        ``True`` if survivorship bias is suspected, ``False`` otherwise.
    """
    if len(returns) < final_periods:
        return False

    tail = returns.iloc[-final_periods:]
    has_zeros = (tail.abs() <= zero_threshold).any(axis=0)

    if not has_zeros.any():
        warnings.warn(
            "No assets have near-zero returns in the final "
            f"{final_periods} periods.  The dataset may suffer from "
            "survivorship bias (delisted assets excluded).",
            UserWarning,
            stacklevel=2,
        )
        return True

    return False
