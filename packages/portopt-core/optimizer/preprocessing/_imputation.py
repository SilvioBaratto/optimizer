"""Sector-average NaN imputation transformer."""

from __future__ import annotations

import logging
from collections import defaultdict

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
from optimizer.preprocessing._gaps import edge_gap_mask

logger = logging.getLogger(__name__)


class SectorImputer(BaseEstimator, TransformerMixin):
    """Fill NaN values using sector cross-sectional averages.

    For each timestep (row), missing values in a given asset column are
    replaced with the mean of all *other* assets in the same sector for
    that timestep (leave-one-out sector average).  If the entire sector is
    ``NaN`` for a timestep, the global cross-sectional mean is used as a
    fallback.

    When ``sector_mapping`` is ``None``, all assets are treated as
    belonging to a single sector — effectively a global cross-sectional
    mean imputation.

    Args:
        sector_mapping: Maps column name (ticker) to sector label. Columns
            absent from the mapping are assigned to a ``"__unmapped__"``
            catch-all sector.
        fallback_strategy: What to do when the sector has no data for a
            timestep. Currently only ``"global_mean"`` is supported.
        fill_edges: Whether to fill the NaN run before an asset's first and
            after its last observation. Set ``False`` for panels that keep
            pre-listing and post-delisting NaN on purpose: filling them invents
            a return history the asset never had. Interior gaps are filled
            either way.
    """

    sector_mapping: dict[str, str] | None
    fallback_strategy: str
    fill_edges: bool

    def __init__(
        self,
        sector_mapping: dict[str, str] | None = None,
        fallback_strategy: str = "global_mean",
        fill_edges: bool = True,
    ) -> None:
        self.sector_mapping = sector_mapping
        self.fallback_strategy = fallback_strategy
        self.fill_edges = fill_edges

    def fit(self, X: pd.DataFrame, y: object = None) -> SectorImputer:
        """Compute sector groupings from the training data.

        Args:
            X: Asset return DataFrame where columns are tickers.
            y: Ignored; present for sklearn compatibility.

        Returns:
            The fitted imputer (self).

        Raises:
            ConfigurationError: If ``fallback_strategy`` is not ``"global_mean"``.
        """
        if self.fallback_strategy != "global_mean":
            raise ConfigurationError(
                f"Unsupported fallback_strategy: '{self.fallback_strategy}'. "
                "Only 'global_mean' is supported."
            )
        X = self._validate_input(X)
        self.n_features_in_: int = X.shape[1]
        self.feature_names_in_: np.ndarray = np.asarray(X.columns)

        self.sector_groups_: dict[str, list[str]] = defaultdict(list)
        mapping = self.sector_mapping or {}
        for col in X.columns:
            sector = mapping.get(col, "__unmapped__")
            self.sector_groups_[sector].append(col)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN with leave-one-out sector averages.

        Args:
            X: Asset return DataFrame; must have the same columns seen in fit,
                in any order.

        Returns:
            DataFrame with missing values filled, columns in fit order. A row in
            which every asset is missing has no peer observation to impute from
            and is returned as ``NaN``, as are edge gaps when ``fill_edges`` is
            ``False``.

        Raises:
            DataError: If X lacks a fit column or has one unseen in fit.
        """
        check_is_fitted(self)
        X = _align_to_fit_columns(
            self._validate_input(X), self.feature_names_in_, "SectorImputer"
        )

        out = X.copy()

        # Pre-compute global cross-sectional mean per row from the ORIGINAL
        # (unimputed) observations.
        global_mean = X.mean(axis=1)

        # All statistics are read from the original ``X`` snapshot — never from
        # ``out`` — so imputation is order-independent and strictly leave-one-out:
        # a value imputed into an earlier column must not feed the sector mean of
        # a later column (which would leak filled values and make the result
        # depend on column ordering).
        for _sector, cols in self.sector_groups_.items():
            sector_cols = [c for c in cols if c in X.columns]
            if not sector_cols:
                continue

            for col in sector_cols:
                mask = X[col].isna()
                if not mask.any():
                    continue

                # Leave-one-out: sector mean excluding the current column
                others = [c for c in sector_cols if c != col]
                sector_mean = X[others].mean(axis=1) if others else global_mean

                fill = sector_mean.where(sector_mean.notna(), global_mean)
                out.loc[mask, col] = fill.loc[mask]

        if not self.fill_edges:
            out = out.mask(edge_gap_mask(X))
        return out

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the feature names seen during fit.

        Args:
            input_features: Optional names to check against the fit columns,
                as sklearn passes through ``Pipeline.get_feature_names_out``.

        Returns:
            Array of column names from the fitted DataFrame.

        Raises:
            ValueError: If *input_features* differs from the fit columns.
        """
        check_is_fitted(self)
        return _fit_feature_names_out(self.feature_names_in_, input_features)

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise DataError(
                f"SectorImputer requires a pandas DataFrame, got {type(X).__name__}"
            )
        return _coerce_numeric(X, "SectorImputer")
