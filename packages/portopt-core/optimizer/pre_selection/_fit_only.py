"""Cleaning transformers that rewrite values only while fitting.

``sklearn.pipeline.Pipeline.fit`` drives intermediate steps through
``fit_transform`` whereas ``predict`` / ``score`` go through ``transform``.  The
classes here exploit that split: ``fit_transform`` applies the full cleaning
(so the optimiser is estimated on clean data) while ``transform`` leaves the
held-out values untouched, so the predicted ``Portfolio`` is scored on the
returns that were actually realised.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from optimizer.preprocessing._alignment import _align_to_fit_columns
from optimizer.preprocessing._imputation import SectorImputer
from optimizer.preprocessing._outliers import OutlierTreater
from optimizer.preprocessing._validation import DataValidator


class FitOnlyDataValidator(DataValidator):
    """`DataValidator` that cleans in ``fit_transform`` and passes through in
    ``transform``.

    Realised held-out returns are never replaced by ``NaN`` here, so a genuine
    extreme move still reaches the scored portfolio.
    """

    def fit_transform(
        self, X: pd.DataFrame, y: object = None, **fit_params: Any
    ) -> pd.DataFrame:
        """Fit, then return the cleaned training frame.

        Args:
            X: Return DataFrame with assets as columns.
            y: Ignored; present for sklearn API compatibility.
            **fit_params: Ignored; present for sklearn API compatibility.

        Returns:
            ``X`` with infinities and returns beyond ``max_abs_return`` set to NaN.
        """
        self.fit(X, y)
        return DataValidator.transform(self, X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the numeric-coerced frame without altering any value.

        Args:
            X: Return DataFrame with the columns seen in fit, in any order.

        Returns:
            ``X`` unchanged apart from numeric coercion and column order.

        Raises:
            DataError: If ``X`` lacks a fit column or has one unseen in fit.
        """
        check_is_fitted(self)
        return _align_to_fit_columns(
            self._validate_input(X), self.feature_names_in_, "DataValidator"
        )


class FitOnlyOutlierTreater(OutlierTreater):
    """`OutlierTreater` that treats outliers in ``fit_transform`` only.

    At ``transform`` time winsorising or removing a held-out observation would
    hide a real crash from the evaluation, so the frame passes through.
    """

    def fit_transform(
        self, X: pd.DataFrame, y: object = None, **fit_params: Any
    ) -> pd.DataFrame:
        """Fit, then return the outlier-treated training frame.

        Args:
            X: Returns DataFrame, shape (n_dates, n_assets).
            y: Ignored; present for sklearn API compatibility.
            **fit_params: Ignored; present for sklearn API compatibility.

        Returns:
            ``X`` with data errors removed and outliers winsorised.
        """
        self.fit(X, y)
        return OutlierTreater.transform(self, X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the numeric-coerced frame without altering any value.

        Args:
            X: Returns DataFrame with the columns seen in fit.

        Returns:
            ``X`` unchanged apart from numeric coercion.
        """
        check_is_fitted(self)
        return _align_to_fit_columns(
            self._validate_input(X), self.feature_names_in_, "OutlierTreater"
        )


class FitOnlySectorImputer(SectorImputer):
    """`SectorImputer` that fabricates peer returns in ``fit_transform`` only.

    At ``transform`` time a missing observation cannot be scored, so it is
    valued at zero (no P&L attributed) rather than at a peer average that was
    never earned.  Non-finite values are treated as missing because the
    selectors downstream reject them.
    """

    def fit_transform(
        self, X: pd.DataFrame, y: object = None, **fit_params: Any
    ) -> pd.DataFrame:
        """Fit, then return the sector-imputed training frame.

        Args:
            X: Asset return DataFrame where columns are tickers.
            y: Ignored; present for sklearn API compatibility.
            **fit_params: Ignored; present for sklearn API compatibility.

        Returns:
            ``X`` with missing values filled by leave-one-out sector averages.
        """
        self.fit(X, y)
        return SectorImputer.transform(self, X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Value missing or non-finite observations at zero.

        Args:
            X: Asset return DataFrame with the columns seen in fit.

        Returns:
            ``X`` with ``NaN`` and infinities replaced by ``0.0``.
        """
        check_is_fitted(self)
        X = _align_to_fit_columns(
            self._validate_input(X), self.feature_names_in_, "SectorImputer"
        )
        return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
