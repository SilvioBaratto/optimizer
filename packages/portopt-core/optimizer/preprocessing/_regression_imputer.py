"""Regression-based NaN imputation from correlated neighbor assets."""

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
from optimizer.preprocessing._imputation import SectorImputer

logger = logging.getLogger(__name__)


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

    Args:
        n_neighbors: Number of most-correlated assets used as regression
            predictors. Candidates are taken in descending ``|corr|`` order and
            kept only while the target and all kept neighbours stay jointly
            observed on at least ``min_train_periods`` rows, so an asset may
            end up with fewer neighbours (or none).
        min_train_periods: Minimum complete-row count required to fit the OLS
            regression for an asset. Assets below this threshold use the
            fallback strategy.
        fallback: Imputation strategy when regression is unavailable. Only
            ``"sector_mean"`` is currently supported (delegates to
            SectorImputer).
        sector_mapping: Maps ticker to sector label. Passed to the internal
            SectorImputer used for fallback imputation. When None, the
            fallback uses a global cross-sectional mean.
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

    def fit(self, X: pd.DataFrame, y: object = None) -> RegressionImputer:
        """Compute neighbor rankings and fit per-asset OLS regressions.

        Args:
            X: Asset return DataFrame (dates x assets). May contain NaN.
            y: Ignored; present for sklearn pipeline compatibility.

        Returns:
            Fitted estimator.
        """
        X = self._validate_input(X)
        self.n_features_in_: int = X.shape[1]
        self.feature_names_in_: np.ndarray = np.asarray(X.columns)

        if self.fallback != "sector_mean":
            raise ConfigurationError(
                f"Unsupported fallback strategy: '{self.fallback}'. "
                "Only 'sector_mean' is supported."
            )

        # pandas' default min_periods=1 scores a 2-6 row overlap as |corr|~1, so a
        # recent listing would outrank every full-history asset. Pairs without
        # enough overlap become NaN and drop out of the ranking.
        corr = X.corr(min_periods=self.min_train_periods).abs()
        observed = X.notna()

        self.neighbors_: dict[str, list[str]] = {}
        self.coefs_: dict[str, np.ndarray | None] = {}
        for col in X.columns:
            ranked = corr[col].drop(col).nlargest(len(X.columns)).index
            # Pairwise overlap does not bound the joint overlap of K neighbours:
            # two assets with disjoint histories each overlap the target but
            # leave no complete row together. Keeping a candidate only while the
            # running joint count stays sufficient guarantees the OLS below has
            # at least min_train_periods rows.
            joint = observed[col]
            nbrs: list[str] = []
            for cand in ranked:
                if len(nbrs) >= self.n_neighbors:
                    break
                cand_joint = joint & observed[cand]
                if cand_joint.sum() >= self.min_train_periods:
                    nbrs.append(cand)
                    joint = cand_joint
            self.neighbors_[col] = nbrs

            if not nbrs:
                self.coefs_[col] = None
                continue

            complete = X.loc[joint, [col, *nbrs]]
            y_train = complete[col].to_numpy()
            X_train = complete[nbrs].to_numpy()
            # Design matrix: [1, r_j1, ..., r_jK]
            X_design = np.column_stack([np.ones(len(y_train)), X_train])
            coefs, *_ = np.linalg.lstsq(X_design, y_train, rcond=None)
            self.coefs_[col] = coefs  # shape (K+1,)

        self._fallback_imputer_: SectorImputer = SectorImputer(
            sector_mapping=self.sector_mapping
        ).fit(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute NaN values using fitted regressions and/or fallback.

        Args:
            X: Asset return DataFrame (dates x assets). May contain NaN. Must
                carry the columns seen in fit, in any order.

        Returns:
            Copy of X with NaN values filled, columns in fit order.

        Raises:
            DataError: If X lacks a fit column or has one unseen in fit.
        """
        check_is_fitted(self)
        X = _align_to_fit_columns(
            self._validate_input(X), self.feature_names_in_, "RegressionImputer"
        )

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
                # Read neighbor predictors from the ORIGINAL ``X`` snapshot, not
                # from ``out``. Reading from ``out`` would feed values imputed
                # into earlier columns back into the regression for later
                # columns, compounding imputation error and making the result
                # depend on column ordering. Only genuinely observed neighbor
                # returns should drive the OLS prediction.
                nbr_data = X.loc[nan_idx, nbrs]
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

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise DataError(
                f"RegressionImputer requires a pandas DataFrame, got {type(X).__name__}"
            )
        return _coerce_numeric(X, "RegressionImputer")
