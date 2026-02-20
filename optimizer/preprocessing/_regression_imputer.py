"""Regression-based NaN imputation from correlated neighbor assets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from optimizer.preprocessing._imputation import SectorImputer


class RegressionImputer(BaseEstimator, TransformerMixin):
    """Fill NaN values using OLS regression from top-K correlated assets.

    For each asset with missing data, fits a linear regression over the
    training window using the ``n_neighbors`` most correlated assets as
    predictors::

        r_{i,t} = α + Σ_j β_j · r_{j,t} + ε_{i,t}

    This preserves the covariance structure of imputed values better than
    sector-mean imputation.

    **Cold-start handling**: if fewer than ``min_train_periods`` complete
    observations exist for an asset in training, the asset falls back to
    the ``fallback`` strategy at transform time.  The same fallback applies
    per-row when any neighbor is itself ``NaN`` at the imputation timestep.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of most-correlated assets used as regression predictors.
    min_train_periods : int, default=60
        Minimum complete-row count required to fit the OLS regression for
        an asset.  Assets below this threshold use the fallback strategy.
    fallback : str, default="sector_mean"
        Imputation strategy when regression is unavailable.  Only
        ``"sector_mean"`` is currently supported (delegates to
        :class:`SectorImputer`).
    sector_mapping : dict[str, str] or None, default=None
        Maps ticker → sector label.  Passed to the internal
        :class:`SectorImputer` used for fallback imputation.  When
        ``None``, the fallback uses a global cross-sectional mean.
    """

    n_neighbors: int
    min_train_periods: int
    fallback: str
    sector_mapping: dict[str, str] | None

    def __init__(
        self,
        n_neighbors: int = 5,
        min_train_periods: int = 60,
        fallback: str = "sector_mean",
        sector_mapping: dict[str, str] | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.min_train_periods = min_train_periods
        self.fallback = fallback
        self.sector_mapping = sector_mapping

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: object = None) -> RegressionImputer:
        """Compute neighbor rankings and fit per-asset OLS regressions.

        Parameters
        ----------
        X : pd.DataFrame
            Asset return DataFrame (dates × assets).  May contain NaN.
        y : ignored

        Returns
        -------
        self
        """
        X = self._validate_input(X)
        self.n_features_in_: int = X.shape[1]
        self.feature_names_in_: np.ndarray = np.asarray(X.columns)

        if self.fallback != "sector_mean":
            raise ValueError(
                f"Unsupported fallback strategy: '{self.fallback}'. "
                "Only 'sector_mean' is supported."
            )

        # 1. Pairwise absolute correlations (pairwise complete obs).
        corr = X.corr().abs()

        # 2. Top-K neighbors per asset (excluding self).
        self.neighbors_: dict[str, list[str]] = {}
        for col in X.columns:
            others = corr[col].drop(col)
            k = min(self.n_neighbors, len(others))
            self.neighbors_[col] = others.nlargest(k).index.tolist()

        # 3. OLS regression per asset.
        self.coefs_: dict[str, np.ndarray | None] = {}
        for col in X.columns:
            nbrs = self.neighbors_[col]
            if not nbrs:
                self.coefs_[col] = None
                continue

            cols_needed = [col, *nbrs]
            complete = X[cols_needed].dropna()

            if len(complete) < self.min_train_periods:
                self.coefs_[col] = None
                continue

            y_train = complete[col].to_numpy()
            X_train = complete[nbrs].to_numpy()
            # Design matrix: [1, r_j1, ..., r_jK]
            X_design = np.column_stack([np.ones(len(y_train)), X_train])
            coefs, *_ = np.linalg.lstsq(X_design, y_train, rcond=None)
            self.coefs_[col] = coefs  # shape (K+1,)

        # 4. Fallback SectorImputer (fitted on training data).
        self._fallback_imputer_: SectorImputer = SectorImputer(
            sector_mapping=self.sector_mapping
        ).fit(X)

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute NaN values using fitted regressions and/or fallback.

        Parameters
        ----------
        X : pd.DataFrame
            Asset return DataFrame (dates × assets).  May contain NaN.

        Returns
        -------
        pd.DataFrame
            Copy of ``X`` with NaN values filled.
        """
        check_is_fitted(self)
        X = self._validate_input(X)

        # Pre-compute full fallback values (sector / global mean).
        fallback_df = self._fallback_imputer_.transform(X)
        out = X.copy()

        for col in out.columns:
            nan_mask = out[col].isna()
            if not nan_mask.any():
                continue

            nan_idx = out.index[nan_mask]
            coefs = self.coefs_.get(col)
            nbrs = self.neighbors_.get(col, [])

            # --- regression path -------------------------------------------
            if coefs is not None and nbrs:
                nbr_data = out.loc[nan_idx, nbrs]
                # Rows where every neighbor is available.
                rows_complete = ~nbr_data.isna().any(axis=1)

                if rows_complete.any():
                    complete_idx = nan_idx[rows_complete]
                    X_pred = nbr_data.loc[complete_idx].to_numpy()
                    # intercept + dot(betas, neighbor_returns)
                    preds = coefs[0] + X_pred @ coefs[1:]
                    out.loc[complete_idx, col] = preds

            # --- fallback path (remaining NaN) -----------------------------
            still_nan = out[col].isna()
            if still_nan.any():
                out.loc[still_nan, col] = fallback_df.loc[still_nan, col]

        return out

    # ------------------------------------------------------------------
    # sklearn protocol
    # ------------------------------------------------------------------

    def get_feature_names_out(
        self, input_features: object = None
    ) -> np.ndarray:
        """Return feature names (pass-through)."""
        check_is_fitted(self)
        return self.feature_names_in_

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "RegressionImputer requires a pandas DataFrame, "
                f"got {type(X).__name__}"
            )
        return X
