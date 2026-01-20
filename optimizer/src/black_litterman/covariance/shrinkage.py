"""
Shrinkage Utilities - Helper functions for covariance shrinkage.
"""

import logging
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ShrinkageTarget(Enum):
    """Available shrinkage targets."""
    CONSTANT_VARIANCE = "constant_variance"  # Diagonal with equal variances
    IDENTITY = "identity"  # Identity matrix (all variances = 1)
    SINGLE_FACTOR = "single_factor"  # Market factor model
    DIAGONAL = "diagonal"  # Diagonal with sample variances


def compute_shrinkage_intensity(
    returns: pd.DataFrame,
    target: ShrinkageTarget = ShrinkageTarget.CONSTANT_VARIANCE,
) -> float:
    """
    Compute optimal shrinkage intensity using the Ledoit-Wolf formula.

    This implements the analytical formula from Ledoit & Wolf (2004):
    "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices"

    Args:
        returns: DataFrame of returns
        target: Shrinkage target type

    Returns:
        Optimal shrinkage intensity (0 to 1)
    """
    X = returns.values
    n, p = X.shape

    # Sample covariance
    S = np.cov(X, rowvar=False)

    # Get target matrix
    F = _get_target_matrix(S, target)

    # Compute shrinkage intensity using Ledoit-Wolf formula
    # (Simplified version - full implementation in sklearn)
    delta = F - S
    delta_sq = np.sum(delta ** 2)

    if delta_sq == 0:
        return 0.0

    # Estimate optimal shrinkage
    shrinkage = min(1.0, max(0.0, _estimate_shrinkage(X, S, F)))

    logger.debug(f"Computed shrinkage intensity: {shrinkage:.4f} for target {target.value}")

    return shrinkage


def _get_target_matrix(
    sample_cov: np.ndarray,
    target: ShrinkageTarget,
) -> np.ndarray:
    """
    Get the shrinkage target matrix.

    Args:
        sample_cov: Sample covariance matrix
        target: Target type

    Returns:
        Target matrix for shrinkage
    """
    p = sample_cov.shape[0]

    if target == ShrinkageTarget.IDENTITY:
        return np.eye(p)

    elif target == ShrinkageTarget.CONSTANT_VARIANCE:
        # Diagonal with average variance
        avg_var = np.mean(np.diag(sample_cov))
        return np.eye(p) * avg_var

    elif target == ShrinkageTarget.DIAGONAL:
        # Diagonal with sample variances
        return np.diag(np.diag(sample_cov))

    elif target == ShrinkageTarget.SINGLE_FACTOR:
        # Single factor model (market factor)
        variances = np.diag(sample_cov)
        avg_corr = (np.sum(sample_cov) - np.sum(variances)) / (p * (p - 1))
        avg_var = np.mean(variances)
        F = np.eye(p) * avg_var
        off_diag = avg_corr * np.sqrt(np.outer(variances, variances))
        F = F + off_diag - np.diag(np.diag(off_diag))
        return F

    else:
        raise ValueError(f"Unknown shrinkage target: {target}")


def _estimate_shrinkage(
    X: np.ndarray,
    S: np.ndarray,
    F: np.ndarray,
) -> float:
    """
    Estimate optimal shrinkage using Oracle Approximating Shrinkage.

    This is a simplified version - the full formula is complex.
    For production, use sklearn.covariance.LedoitWolf.

    Args:
        X: Data matrix (n x p)
        S: Sample covariance (p x p)
        F: Target matrix (p x p)

    Returns:
        Shrinkage intensity estimate
    """
    n, p = X.shape

    # Simplified Ledoit-Wolf estimator
    # Based on asymptotic formula: shrinkage ~ (sum of squared bias) / (sum of squared error)

    delta = F - S
    delta_norm = np.sum(delta ** 2)

    if delta_norm < 1e-10:
        return 1.0

    # Estimate variance of sample covariance elements
    X_centered = X - X.mean(axis=0)
    sigma_sq = 0.0

    for i in range(p):
        for j in range(p):
            if i <= j:
                elem_sq = (X_centered[:, i] * X_centered[:, j]) ** 2
                sigma_sq += np.var(elem_sq) / n

    # Shrinkage formula
    shrinkage = sigma_sq / delta_norm

    return shrinkage


def apply_shrinkage(
    sample_cov: np.ndarray,
    shrinkage: float,
    target: ShrinkageTarget = ShrinkageTarget.CONSTANT_VARIANCE,
) -> np.ndarray:
    """
    Apply shrinkage to sample covariance matrix.

    Args:
        sample_cov: Sample covariance matrix
        shrinkage: Shrinkage intensity (0 = sample, 1 = target)
        target: Shrinkage target type

    Returns:
        Shrunk covariance matrix
    """
    F = _get_target_matrix(sample_cov, target)
    return (1 - shrinkage) * sample_cov + shrinkage * F


def ensure_positive_definite(
    cov_matrix: np.ndarray,
    min_eigenvalue: float = 1e-6,
) -> np.ndarray:
    """
    Ensure covariance matrix is positive definite.

    Adjusts eigenvalues if necessary to ensure numerical stability.

    Args:
        cov_matrix: Covariance matrix to check/adjust
        min_eigenvalue: Minimum allowed eigenvalue

    Returns:
        Adjusted positive definite matrix
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Check if already PD
    if np.all(eigenvalues > min_eigenvalue):
        return cov_matrix

    # Adjust eigenvalues
    adjusted_eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

    # Reconstruct matrix
    adjusted_cov = eigenvectors @ np.diag(adjusted_eigenvalues) @ eigenvectors.T

    # Ensure symmetry
    adjusted_cov = (adjusted_cov + adjusted_cov.T) / 2

    logger.warning(
        f"Covariance matrix was not positive definite. "
        f"Adjusted {np.sum(eigenvalues < min_eigenvalue)} eigenvalues."
    )

    return adjusted_cov
