"""Three-group outlier treatment transformer."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from optimizer.exceptions import DataError
from optimizer.preprocessing._coerce import _coerce_numeric

logger = logging.getLogger(__name__)


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
    protected_mask : pd.DataFrame or None, default=None
        Optional boolean matrix (dates x tickers) flagging cells that are
        genuine economic events, not data errors, and so must be exempt from
        outlier treatment.  Protected cells are (a) excluded from the ``mu_`` /
        ``sigma_`` estimate during ``fit`` so a single extreme value cannot
        inflate the scale and mask real outliers elsewhere, and (b) never
        removed or winsorised during ``transform``.  The mask is realigned to
        each ``X`` by label (``reindex(..., fill_value=False)``), so it works
        unchanged across CV folds and any column subset.  The canonical
        producer is ``delisting_protection_mask`` (in
        :mod:`optimizer.preprocessing._delisting`), which flags each delisted
        asset's terminal-return cell.  ``None`` (default) protects nothing —
        identical to the pre-existing behaviour.
    """

    winsorize_threshold: float
    remove_threshold: float

    def __init__(
        self,
        winsorize_threshold: float = 3.0,
        remove_threshold: float = 10.0,
        protected_mask: pd.DataFrame | None = None,
    ) -> None:
        self.winsorize_threshold = winsorize_threshold
        self.remove_threshold = remove_threshold
        self.protected_mask = protected_mask

    def fit(self, X: pd.DataFrame, y: object = None) -> OutlierTreater:
        """Compute per-column mean and std from training data."""
        X = self._validate_input(X)
        self.n_features_in_: int = X.shape[1]
        self.feature_names_in_: np.ndarray = np.asarray(X.columns)
        # Exclude protected cells from the moment estimate so an exempt extreme
        # (e.g. a -0.30 delisting return) does not inflate sigma_ and mask
        # genuine outliers in the same column.
        X_fit = X.mask(self._aligned_protected_mask(X))
        self.mu_: pd.Series = X_fit.mean()
        self.sigma_: pd.Series = X_fit.std()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply three-group treatment based on z-scores."""
        check_is_fitted(self)
        X = self._validate_input(X)

        out = X.copy()
        prot = self._aligned_protected_mask(X)
        # Guard against zero-sigma columns (constant series).
        # Treat their z-score as 0 and let DropZeroVariance handle them later.
        safe_sigma = self.sigma_.replace(0, np.nan)

        z = (out - self.mu_) / safe_sigma

        # Group 1: data errors → NaN (values at exactly the threshold are
        # errors).  Protected cells are exempt — they carry real events.
        err_mask = (z.abs() >= self.remove_threshold) & ~prot
        out[err_mask] = np.nan

        # Group 2: outliers → winsorise to μ ± threshold * σ
        # Only clip cells in the winsorize band, not already-NaN or protected.
        win_mask = (z.abs() >= self.winsorize_threshold) & ~err_mask & ~prot
        upper = self.mu_ + self.winsorize_threshold * self.sigma_
        lower = self.mu_ - self.winsorize_threshold * self.sigma_
        out = out.where(~win_mask, out.clip(lower=lower, upper=upper, axis=1))

        return out

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return feature names (pass-through)."""
        check_is_fitted(self)
        return self.feature_names_in_

    # -- internals -----------------------------------------------------------

    def _aligned_protected_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        """Boolean mask aligned to *X* (``False`` when no mask was supplied)."""
        if self.protected_mask is None:
            return pd.DataFrame(False, index=X.index, columns=X.columns)
        return self.protected_mask.reindex(
            index=X.index, columns=X.columns, fill_value=False
        ).astype(bool)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise DataError(
                f"OutlierTreater requires a pandas DataFrame, got {type(X).__name__}"
            )
        return _coerce_numeric(X, "OutlierTreater")
