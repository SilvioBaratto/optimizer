"""ML-based composite scoring: ridge regression and gradient-boosted trees."""

from __future__ import annotations

import logging
from typing import TypeAlias

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Type alias for the fitted model returned by fit_ridge_composite or fit_gbt_composite
FittedMLModel: TypeAlias = RidgeCV | GradientBoostingRegressor


def fit_ridge_composite(
    scores: pd.DataFrame,
    forward_returns: pd.Series,
    alpha: float = 1.0,
) -> RidgeCV:
    """Fit a ridge regression model mapping factor scores to forward returns.

    Parameters
    ----------
    scores : pd.DataFrame
        Historical tickers x factors matrix (training observations).
        Must be aligned with ``forward_returns`` on the index.
    forward_returns : pd.Series
        Forward return per ticker for the training period.
    alpha : float
        L2 regularisation strength.  A single-element array is passed to
        ``RidgeCV`` so cross-validation still runs internally if multiple
        alphas are desired; here we keep one alpha for determinism.

    Returns
    -------
    RidgeCV
        Fitted ridge model.  Call ``predict(scores)`` for new data.
    """
    common = scores.index.intersection(forward_returns.index)
    X = scores.loc[common].values.astype(float)
    y = forward_returns.loc[common].values.astype(float)

    # Drop rows with NaN in either X or y
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]

    model = RidgeCV(alphas=[alpha], fit_intercept=True)
    model.fit(X, y)
    return model


def fit_gbt_composite(
    scores: pd.DataFrame,
    forward_returns: pd.Series,
    max_depth: int = 3,
    n_estimators: int = 50,
) -> GradientBoostingRegressor:
    """Fit a gradient-boosted tree model mapping factor scores to forward returns.

    Parameters
    ----------
    scores : pd.DataFrame
        Historical tickers x factors matrix (training observations).
    forward_returns : pd.Series
        Forward return per ticker for the training period.
    max_depth : int
        Maximum depth of individual regression trees (3–5 recommended to
        limit extrapolation and retain interpretability).
    n_estimators : int
        Number of boosting rounds.

    Returns
    -------
    GradientBoostingRegressor
        Fitted GBT model.
    """
    common = scores.index.intersection(forward_returns.index)
    X = scores.loc[common].values.astype(float)
    y = forward_returns.loc[common].values.astype(float)

    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]

    model = GradientBoostingRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=0,
    )
    model.fit(X, y)
    return model


def predict_composite_scores(
    model: FittedMLModel,
    scores: pd.DataFrame,
) -> pd.Series:
    """Apply a fitted ridge or GBT model to produce normalised composite scores.

    The raw predictions are standardised to zero mean and unit variance so
    the output is on the same scale as z-score factor inputs.

    Parameters
    ----------
    model : RidgeCV or GradientBoostingRegressor
        A model returned by :func:`fit_ridge_composite` or
        :func:`fit_gbt_composite`.
    scores : pd.DataFrame
        Current-period tickers x factors matrix.

    Returns
    -------
    pd.Series
        Normalised composite score per ticker (zero mean, unit variance).
        Tickers with all-NaN factor rows receive ``NaN``.
    """
    X = scores.values.astype(float)
    row_has_nan = np.isnan(X).any(axis=1)

    # Fill NaN with column means for prediction; mask afterwards
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X_filled = np.where(nan_mask, col_means, X)

    raw: np.ndarray = model.predict(X_filled)

    # Standardise to zero mean, unit variance
    scaler = StandardScaler()
    raw_2d = raw.reshape(-1, 1)
    if raw_2d.shape[0] > 1:
        normalised: np.ndarray = scaler.fit_transform(raw_2d).ravel()
    else:
        normalised = np.zeros_like(raw)

    result = pd.Series(normalised, index=scores.index, dtype=float)
    result[row_has_nan] = np.nan
    return result
