"""Sector-average NaN imputation transformer."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from optimizer.exceptions import DataError

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

    Parameters
    ----------
    sector_mapping : dict[str, str] or None, default=None
        Maps column name (ticker) → sector label.  Columns absent from
        the mapping are assigned to a ``"__unmapped__"`` catch-all sector.
    fallback_strategy : str, default="global_mean"
        What to do when the sector has no data for a timestep.  Currently
        only ``"global_mean"`` is supported.
    """

    sector_mapping: dict[str, str] | None
    fallback_strategy: str

    def __init__(
        self,
        sector_mapping: dict[str, str] | None = None,
        fallback_strategy: str = "global_mean",
    ) -> None:
        self.sector_mapping = sector_mapping
        self.fallback_strategy = fallback_strategy

    def fit(self, X: pd.DataFrame, y: object = None) -> SectorImputer:
        """Build the internal sector → columns index."""
        X = self._validate_input(X)
        self.n_features_in_: int = X.shape[1]
        self.feature_names_in_: np.ndarray = np.asarray(X.columns)

        # Build sector groups
        self.sector_groups_: dict[str, list[str]] = defaultdict(list)
        mapping = self.sector_mapping or {}
        for col in X.columns:
            sector = mapping.get(col, "__unmapped__")
            self.sector_groups_[sector].append(col)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN with leave-one-out sector averages."""
        check_is_fitted(self)
        X = self._validate_input(X)

        out = X.copy()

        # Pre-compute global cross-sectional mean per row
        global_mean = out.mean(axis=1)

        for _sector, cols in self.sector_groups_.items():
            sector_cols = [c for c in cols if c in out.columns]
            if not sector_cols:
                continue

            sector_df = out[sector_cols]

            for col in sector_cols:
                mask = sector_df[col].isna()
                if not mask.any():
                    continue

                # Leave-one-out: sector mean excluding the current column
                others = [c for c in sector_cols if c != col]
                sector_mean = out[others].mean(axis=1) if others else global_mean

                # Fill with sector mean; fall back to global mean where
                # the sector itself is entirely NaN
                fill = sector_mean.where(sector_mean.notna(), global_mean)
                out.loc[mask, col] = fill.loc[mask]

        return out

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return feature names (pass-through)."""
        check_is_fitted(self)
        return self.feature_names_in_

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise DataError(
                f"SectorImputer requires a pandas DataFrame, got {type(X).__name__}"
            )
        return X
