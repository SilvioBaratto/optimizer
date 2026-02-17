"""Data validation transformer for return DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DataValidator(BaseEstimator, TransformerMixin):
    """Replace infinities and extreme values with NaN.

    Operates on a return DataFrame. Designed as the first step in a
    pre-selection pipeline so that downstream transformers receive
    well-formed numeric data.

    Parameters
    ----------
    max_abs_return : float, default=10.0
        Any return whose absolute value exceeds this threshold is replaced
        with ``NaN``.  The default of 10.0 (i.e. 1 000 %) is deliberately
        generous — it catches data errors while preserving legitimate
        large moves.
    """

    max_abs_return: float

    def __init__(self, max_abs_return: float = 10.0) -> None:
        self.max_abs_return = max_abs_return

    def fit(
        self, X: pd.DataFrame, y: object = None
    ) -> DataValidator:
        """Store metadata.  This transformer is stateless."""
        X = self._validate_input(X)
        self.n_features_in_: int = X.shape[1]
        self.feature_names_in_: np.ndarray = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace ``inf`` / ``-inf`` and extreme returns with ``NaN``."""
        check_is_fitted(self)
        X = self._validate_input(X)

        out = X.copy()
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out[out.abs() > self.max_abs_return] = np.nan
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
                f"DataValidator requires a pandas DataFrame, got {type(X).__name__}"
            )
        return X
