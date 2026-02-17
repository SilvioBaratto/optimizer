"""Three-group outlier treatment transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class OutlierTreater(BaseEstimator, TransformerMixin):
    """Three-group outlier methodology on per-column z-scores.

    During ``fit``, compute per-column mean (``mu_``) and standard deviation
    (``sigma_``) from the training data.

    During ``transform``, classify each observation into one of three groups
    based on its z-score ``z = (x - mu) / sigma``:

    1. **Data errors** — ``|z| > remove_threshold`` → replaced with ``NaN``.
    2. **Outliers** — ``winsorize_threshold <= |z| <= remove_threshold`` →
       winsorised to ``mu ± winsorize_threshold * sigma``.
    3. **Normal** — ``|z| < winsorize_threshold`` → kept as-is.

    Parameters
    ----------
    winsorize_threshold : float, default=3.0
        Z-score boundary between normal observations and outliers.
    remove_threshold : float, default=10.0
        Z-score boundary between outliers and data errors.
    """

    winsorize_threshold: float
    remove_threshold: float

    def __init__(
        self,
        winsorize_threshold: float = 3.0,
        remove_threshold: float = 10.0,
    ) -> None:
        self.winsorize_threshold = winsorize_threshold
        self.remove_threshold = remove_threshold

    def fit(
        self, X: pd.DataFrame, y: object = None
    ) -> OutlierTreater:
        """Compute per-column mean and std from training data."""
        X = self._validate_input(X)
        self.n_features_in_: int = X.shape[1]
        self.feature_names_in_: np.ndarray = np.asarray(X.columns)
        self.mu_: pd.Series = X.mean()
        self.sigma_: pd.Series = X.std()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply three-group treatment based on z-scores."""
        check_is_fitted(self)
        X = self._validate_input(X)

        out = X.copy()
        # Guard against zero-sigma columns (constant series).
        # Treat their z-score as 0 and let DropZeroVariance handle them later.
        safe_sigma = self.sigma_.replace(0, np.nan)

        z = (out - self.mu_) / safe_sigma

        # Group 1: data errors → NaN
        out[z.abs() > self.remove_threshold] = np.nan

        # Group 2: outliers → winsorise to μ ± threshold * σ
        upper = self.mu_ + self.winsorize_threshold * self.sigma_
        lower = self.mu_ - self.winsorize_threshold * self.sigma_
        out = out.clip(lower=lower, upper=upper, axis=1)

        return out

    def get_feature_names_out(
        self, input_features: object = None
    ) -> np.ndarray:
        """Return feature names (pass-through)."""
        check_is_fitted(self)
        return self.feature_names_in_

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"OutlierTreater requires a pandas DataFrame, got {type(X).__name__}"
            )
        return X
