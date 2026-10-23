"""Three-group outlier treatment transformer."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from optimizer.exceptions import DataError
from optimizer.preprocessing._alignment import (
    _align_to_fit_columns,
    _fit_feature_names_out,
)
from optimizer.preprocessing._coerce import _coerce_numeric

logger = logging.getLogger(__name__)

# Makes the MAD a consistent estimator of the standard deviation for normal data.
_MAD_TO_SIGMA = 1.4826

# The fit-time screen only has to catch errors so large that they hide
# themselves, so it sits well beyond ``remove_threshold``.  A MAD reflects the
# calm days of a column; at the removal bar itself it would also drop the crisis
# days of a column whose volatility switches between regimes and shrink
# ``sigma_`` until ``transform`` deletes more of them.
_SCREEN_THRESHOLD_MULTIPLE = 2.0


class OutlierTreater(BaseEstimator, TransformerMixin):
    """Three-group outlier methodology on per-column z-scores.

    During ``fit``, compute per-column mean (``mu_``) and standard deviation
    (``sigma_``) from the training data, leaving out cells that sit at least
    twice ``remove_threshold`` robust standard deviations (median / scaled MAD)
    from the column median.  A sample z-score cannot exceed ``(n - 1) / sqrt(n)``,
    so a gross error left in the sample would inflate ``sigma_`` until it can no
    longer reach ``remove_threshold`` on short windows; the screen is looser than
    the removal bar so that crisis days of a regime-switching column still count
    towards the scale.  The MAD is taken over the non-zero cells only, because
    the exact-zero returns that forward-filled prices produce would otherwise
    shrink it far below the real volatility.  Columns whose non-zero cells all
    coincide have no scale and are not screened.

    During ``transform``, classify each observation into one of three groups
    based on its z-score ``z = (x - mu) / sigma``:

    1. **Data errors** — ``|z| > remove_threshold`` → replaced with ``NaN``.
    2. **Outliers** — ``winsorize_threshold <= |z| <= remove_threshold`` →
       winsorised to ``mu ± winsorize_threshold * sigma``.
    3. **Normal** — ``|z| < winsorize_threshold`` → kept as-is.

    Args:
        winsorize_threshold: Z-score boundary between normal observations and
            outliers.
        remove_threshold: Z-score boundary between outliers and data errors.
        protected_mask: Boolean matrix (dates × tickers) flagging cells that
            are genuine economic events rather than data errors, exempt from
            outlier treatment. Protected cells are (a) excluded from the
            ``mu_`` / ``sigma_`` estimate during ``fit`` so a single extreme
            value cannot inflate the scale and mask real outliers elsewhere,
            and (b) never removed or winsorised during ``transform``. The mask
            is realigned to each ``X`` by label
            (``reindex(..., fill_value=False)``), so it works unchanged across
            CV folds and any column subset; ``fit`` raises ``DataError`` when a
            non-empty mask shares no index or column label with ``X``. The
            canonical producer is ``delisting_protection_mask`` in
            ``optimizer.preprocessing._delisting``, which flags each delisted
            asset's terminal-return cell. Pass ``None`` (default) to protect
            nothing. The run of exact ``0.0`` (or ``NaN``) cells that follows a
            column's protected cell through the end of ``X`` is the synthetic
            cash padding written by ``apply_delisting_returns``; it is neither
            observed data nor treated, and is left out of ``mu_`` / ``sigma_``
            so a long dead tail cannot shrink the scale and clip the asset's
            genuine history.
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
        """Fit mean and standard deviation from training data.

        Args:
            X: Returns DataFrame, shape (n_dates, n_assets).
            y: Ignored; present for sklearn API compatibility.

        Returns:
            The fitted estimator.

        Raises:
            DataError: If X is not a DataFrame, or ``protected_mask`` is
                non-empty but shares no index or column label with X.
        """
        X = self._validate_input(X)
        self._check_mask_labels_overlap(X)
        self.n_features_in_: int = X.shape[1]
        self.feature_names_in_: np.ndarray = np.asarray(X.columns)
        # Exclude protected cells from the moment estimate so an exempt extreme
        # (e.g. a -0.30 delisting return) does not inflate sigma_ and mask
        # genuine outliers in the same column.  The padding after it is dropped
        # for the opposite reason: its exact zeros would deflate sigma_.
        X_fit = X.mask(self._exempt_mask(X))
        X_fit = X_fit.mask(self._gross_error_mask(X_fit))
        self.mu_: pd.Series = X_fit.mean()
        self.sigma_: pd.Series = X_fit.std()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply three-group outlier treatment to the input returns.

        Args:
            X: Returns DataFrame with the same columns as the training data,
                in any order.

        Returns:
            DataFrame with data errors replaced by ``NaN`` and outliers
            winsorised to ``mu ± winsorize_threshold * sigma``, columns in fit
            order. Protected cells pass through unmodified.

        Raises:
            DataError: If X lacks a fit column or has one unseen in fit.
        """
        check_is_fitted(self)
        X = _align_to_fit_columns(
            self._validate_input(X), self.feature_names_in_, "OutlierTreater"
        )

        out = X.copy()
        prot = self._exempt_mask(X)
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
        """Return the feature names seen during fit.

        Args:
            input_features: Optional names to check against the fit columns,
                as sklearn passes through ``Pipeline.get_feature_names_out``.

        Returns:
            Array of column names from the training DataFrame.

        Raises:
            ValueError: If *input_features* differs from the fit columns.
        """
        check_is_fitted(self)
        return _fit_feature_names_out(self.feature_names_in_, input_features)

    # -- internals -----------------------------------------------------------

    def _check_mask_labels_overlap(self, X: pd.DataFrame) -> None:
        """Reject a mask that shares no row or column label with the fit data.

        Partial overlap is expected (CV folds, column subsets), but none at all
        means a tz, dtype or naming mismatch that would silently protect nothing.
        """
        mask = self.protected_mask
        if mask is None or mask.empty:
            return
        if not mask.index.isin(X.index).any() or not mask.columns.isin(X.columns).any():
            raise DataError(
                "OutlierTreater protected_mask shares no index or column labels "
                "with X; check for a timezone, dtype or ticker-naming mismatch"
            )

    def _aligned_protected_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        """Boolean mask aligned to *X* (``False`` when no mask was supplied)."""
        if self.protected_mask is None:
            return pd.DataFrame(False, index=X.index, columns=X.columns)
        return self.protected_mask.reindex(
            index=X.index, columns=X.columns, fill_value=False
        ).astype(bool)

    def _gross_error_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cells far beyond ``remove_threshold`` robust sigmas of the column median.

        The median and MAD are unaffected by a handful of errors, unlike the
        mean and standard deviation that ``fit`` estimates afterwards.
        """
        median = X.median()
        deviation = (X - median).abs()
        # Forward-filled prices give illiquid tickers many exact-zero returns.
        # Just below a 50% share they pull the MAD of the whole column towards
        # zero without making it zero, so ten "robust sigmas" would shrink to
        # a sliver of the real volatility and genuine returns would be
        # screened as errors.  The scale comes from the cells that moved.
        moved_deviation = deviation.where(X != 0.0)
        robust_sigma = (moved_deviation.median() * _MAD_TO_SIGMA).replace(0, np.nan)
        return (deviation / robust_sigma) >= (
            _SCREEN_THRESHOLD_MULTIPLE * self.remove_threshold
        )

    def _exempt_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        """Protected cells plus the post-delisting padding that follows them."""
        prot = self._aligned_protected_mask(X)
        if not prot.to_numpy().any():
            return prot
        # Trailing run: every row from here to the end is 0.0 or NaN.  Only a
        # run that starts after a protected cell is padding, so zeros in an
        # unprotected column or before the event stay ordinary observations.
        flat = ((X == 0.0) | X.isna()).astype("int8")
        trailing_run = flat.iloc[::-1].cummin().iloc[::-1].astype(bool)
        after_event = prot.astype("int8").cummax().astype(bool) & ~prot
        return prot | (trailing_run & after_event)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise DataError(
                f"OutlierTreater requires a pandas DataFrame, got {type(X).__name__}"
            )
        return _coerce_numeric(X, "OutlierTreater")
