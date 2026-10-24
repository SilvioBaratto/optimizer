"""Data validation transformer for return DataFrames."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.preprocessing._alignment import (
    _align_to_fit_columns,
    _fit_feature_names_out,
)
from optimizer.preprocessing._coerce import _coerce_numeric

logger = logging.getLogger(__name__)


class DataValidator(BaseEstimator, TransformerMixin):
    """Replace infinities and extreme values with NaN.

    Operates on a return DataFrame. Designed as the first step in a
    pre-selection pipeline so that downstream transformers receive
    well-formed numeric data.

    Args:
        max_abs_return: Threshold above which absolute return values are
            replaced with NaN. The default (10.0, i.e. 1 000 %) is
            deliberately generous — it catches data errors while preserving
            legitimate large moves.
    """

    max_abs_return: float

    def __init__(self, max_abs_return: float = 10.0) -> None:
        self.max_abs_return = max_abs_return

    def fit(self, X: pd.DataFrame, y: object = None) -> DataValidator:
        """Record input column count and names for downstream compatibility checks.

        Args:
            X: Return DataFrame with assets as columns.
            y: Ignored; present for sklearn API compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ConfigurationError: If ``max_abs_return`` is not positive.
        """
        # Checked at fit rather than in __init__ so set_params / grid-search
        # sweeps, which bypass the config, are validated too (sklearn convention).
        if not self.max_abs_return > 0:
            raise ConfigurationError(
                f"max_abs_return must be positive, got {self.max_abs_return}"
            )
        X = self._validate_input(X)
        self.n_features_in_: int = X.shape[1]
        self.feature_names_in_: np.ndarray = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace inf/-inf and extreme returns with NaN.

        Args:
            X: Return DataFrame to clean; must carry the columns seen in fit, in
                any order.

        Returns:
            Copy of X with infinities and returns beyond max_abs_return set to
            NaN, columns in fit order.

        Raises:
            DataError: If X lacks a fit column or has one unseen in fit.
        """
        check_is_fitted(self)
        X = _align_to_fit_columns(
            self._validate_input(X), self.feature_names_in_, "DataValidator"
        )

        out = X.copy()
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out[out.abs() > self.max_abs_return] = np.nan
        return out

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return feature names recorded during fit.

        Args:
            input_features: Optional names to check against the fit columns,
                as sklearn passes through ``Pipeline.get_feature_names_out``.

        Returns:
            Array of column names recorded during fit.

        Raises:
            ValueError: If *input_features* differs from the fit columns.
        """
        check_is_fitted(self)
        return _fit_feature_names_out(self.feature_names_in_, input_features)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise DataError(
                f"DataValidator requires a pandas DataFrame, got {type(X).__name__}"
            )
        return _coerce_numeric(X, "DataValidator")
