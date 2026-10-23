"""ML-based composite scoring: ridge regression and gradient-boosted trees."""

from __future__ import annotations

import logging
from typing import TypeAlias

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from optimizer.exceptions import DataError

logger = logging.getLogger(__name__)

FittedMLModel: TypeAlias = RidgeCV | GradientBoostingRegressor


def _training_matrix(
    scores: pd.DataFrame,
    forward_returns: pd.Series,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return complete-case training rows as a named-column frame and targets.

    The frame keeps factor names so sklearn records ``feature_names_in_``,
    which ``predict_composite_scores`` uses to align prediction columns.
    """
    common = scores.index.intersection(forward_returns.index)
    X = scores.loc[common].astype(float)
    X.columns = X.columns.map(str)
    y = forward_returns.loc[common].to_numpy(dtype=float)

    valid = ~(np.isnan(X.to_numpy()).any(axis=1) | np.isnan(y))
    return X.loc[valid], y[valid]


def fit_ridge_composite(
    scores: pd.DataFrame,
    forward_returns: pd.Series,
    alpha: float = 1.0,
) -> RidgeCV:
    """Fit a ridge regression model mapping factor scores to forward returns.

    Args:
        scores: Historical tickers x factors matrix (training observations),
            aligned with ``forward_returns`` on the index.
        forward_returns: Forward return per ticker for the training period.
        alpha: L2 regularisation strength. A single-element array is passed to
            ``RidgeCV`` so cross-validation still runs internally if multiple
            alphas are desired; here we keep one alpha for determinism.

    Returns:
        Fitted ridge model. Call ``predict(scores)`` for new data.
    """
    X, y = _training_matrix(scores, forward_returns)

    model = RidgeCV(alphas=[alpha], fit_intercept=True)
    model.fit(X, y)
    return model


def fit_gbt_composite(
    scores: pd.DataFrame,
    forward_returns: pd.Series,
    max_depth: int = 3,
    n_estimators: int = 50,
    random_state: int = 0,
) -> GradientBoostingRegressor:
    """Fit a gradient-boosted tree model mapping factor scores to forward returns.

    Args:
        scores: Historical tickers x factors matrix (training observations).
        forward_returns: Forward return per ticker for the training period.
        max_depth: Maximum depth of individual regression trees (3-5 recommended
            to limit extrapolation and retain interpretability).
        n_estimators: Number of boosting rounds.
        random_state: Seed for reproducibility.

    Returns:
        Fitted GBT model.
    """
    X, y = _training_matrix(scores, forward_returns)

    model = GradientBoostingRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def _align_to_training_columns(
    model: FittedMLModel,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Reorder prediction columns to the model's training columns.

    Models fitted without factor names (e.g. on a bare array) keep the
    positional contract, so ``scores`` is returned unchanged.

    Raises:
        DataError: If ``scores`` lacks a training factor or carries an extra one.
    """
    trained = getattr(model, "feature_names_in_", None)
    if trained is None:
        return scores

    columns = [str(c) for c in trained]
    frame = scores.copy()
    frame.columns = frame.columns.map(str)
    missing = [c for c in columns if c not in frame.columns]
    unexpected = [c for c in frame.columns if c not in columns]
    if missing or unexpected:
        raise DataError(
            "Prediction factors do not match the training factors: "
            f"missing {missing!r}, unexpected {unexpected!r}"
        )
    return frame.loc[:, columns]


def predict_composite_scores(
    model: FittedMLModel,
    scores: pd.DataFrame,
) -> pd.Series:
    """Apply a fitted ridge or GBT model to produce normalised composite scores.

    The raw predictions are standardised to zero mean and unit variance so
    the output is on the same scale as z-score factor inputs.

    Args:
        model: A model returned by `fit_ridge_composite` or
            `fit_gbt_composite`.
        scores: Current-period tickers x factors matrix.  Columns are matched
            to the training factors by name, so their order may differ from
            the training frame.

    Returns:
        Normalised composite score per ticker (zero mean, unit variance over
        the scored tickers).  Tickers missing only some factors are scored with
        the cross-sectional mean standing in for the missing values; tickers
        with all-NaN factor rows receive ``NaN``.

    Raises:
        DataError: If ``scores`` lacks a factor the model was trained on or
            carries one it was not.
    """
    frame = _align_to_training_columns(model, scores)
    X = frame.to_numpy(dtype=float)

    result = pd.Series(np.nan, index=scores.index, dtype=float)
    scorable = ~np.isnan(X).all(axis=1)
    if not scorable.any():
        return result
    X = X[scorable]

    # sklearn predictors reject NaN, so a ticker covering only some factors
    # takes the cross-sectional column mean (neutral for standardised inputs)
    # for the rest.  Masked rows are excluded before this and before scaling
    # so they cannot shift the statistics of the tickers that are scored.
    col_means = np.nanmean(X, axis=0)
    X_filled = np.where(np.isnan(X), col_means, X)

    # Feeding a frame to a model fitted on a bare array (or vice versa) makes
    # sklearn warn about mismatched feature names.
    X_in = (
        pd.DataFrame(X_filled, columns=frame.columns)
        if hasattr(model, "feature_names_in_")
        else X_filled
    )

    raw: np.ndarray = model.predict(X_in)

    scaler = StandardScaler()
    raw_2d = raw.reshape(-1, 1)
    if raw_2d.shape[0] > 1:
        normalised: np.ndarray = scaler.fit_transform(raw_2d).ravel()
    else:
        normalised = np.zeros_like(raw)

    result[scorable] = normalised
    return result
