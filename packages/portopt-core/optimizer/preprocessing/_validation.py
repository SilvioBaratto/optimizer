"""Data validation transformer for return DataFrames."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from optimizer.exceptions import DataError
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
        """
        X = self._validate_input(X)
        self.n_features_in_: int = X.shape[1]
        self.feature_names_in_: np.ndarray = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace inf/-inf and extreme returns with NaN.

        Args:
            X: Return DataFrame to clean.

        Returns:
            Copy of X with infinities and returns beyond max_abs_return set to NaN.
        """
        check_is_fitted(self)
        X = self._validate_input(X)

        out = X.copy()
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out[out.abs() > self.max_abs_return] = np.nan
        return out

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return feature names recorded during fit.

        Args:
            input_features: Ignored; present for sklearn API compatibility.

        Returns:
            Array of column names recorded during fit.
        """
        check_is_fitted(self)
        return self.feature_names_in_

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise DataError(
                f"DataValidator requires a pandas DataFrame, got {type(X).__name__}"
            )
        return _coerce_numeric(X, "DataValidator")
