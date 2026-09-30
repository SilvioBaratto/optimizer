"""sklearn-composable investability-screen selector.

Bridges fundamental investability screening (`universe`) into an
sklearn / skfolio ``Pipeline``.  Most pre-selection transformers in this library
operate on the return matrix alone; this selector additionally consults injected
fundamentals / price / volume data to decide which assets are *investable* before
any statistical pre-selection or optimisation runs.

Follows the same ``SelectorMixin + BaseEstimator`` contract as skfolio's own
``skfolio.pre_selection`` transformers, so it exposes ``get_support``,
``get_feature_names_out``, pandas output via ``set_output(transform="pandas")``,
and composes with ``sklearn.pipeline.Pipeline``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from optimizer.exceptions import DataError
from optimizer.universe._config import InvestabilityScreenConfig
from optimizer.universe._screener import apply_investability_screens

logger = logging.getLogger(__name__)


class InvestabilityScreenSelector(SelectorMixin, BaseEstimator):
    """Select investable assets from a return matrix via fundamental screens.

    ``fit`` runs `apply_investability_screens` over the injected
    fundamentals / price / volume data and records which columns of ``X``
    (linear-return series, one column per ticker) survive.  ``transform`` returns
    the surviving columns, preserving ``X``'s column order.

    The frozen `InvestabilityScreenConfig` carries the *serialisable*
    thresholds; the (non-serialisable) cross-sectional DataFrames and the
    current-membership index are passed as constructor arguments, mirroring the
    library convention that estimator instances / arrays / frames are estimator
    kwargs rather than config fields.

    Args:
        fundamentals: Cross-sectional data, one row per ticker (see
            `apply_investability_screens`).  Required at ``fit`` time.
        price_history: Price matrix (dates x tickers).  Required at ``fit`` time.
        volume_history: Volume matrix (dates x tickers).  Required at ``fit`` time.
        financial_statements: Optional statement-level data for the
            data-availability screen.
        config: Screening configuration.  ``None`` uses developed-market defaults.
        current_members: Tickers currently in the universe, for hysteresis.

    Attributes:
        to_keep_: Boolean mask over the columns of ``X`` seen during ``fit``.
        investable_universe_: Tickers that passed every screen (intersected with
            ``X``'s columns, preserving ``X`` order).
        n_features_in_: Number of assets (columns) seen during ``fit``.
        feature_names_in_: Column names (tickers) seen during ``fit``.
    """

    to_keep_: NDArray[np.bool_]
    investable_universe_: pd.Index

    def __init__(
        self,
        fundamentals: pd.DataFrame | None = None,
        price_history: pd.DataFrame | None = None,
        volume_history: pd.DataFrame | None = None,
        financial_statements: pd.DataFrame | None = None,
        config: InvestabilityScreenConfig | None = None,
        current_members: pd.Index | None = None,
    ) -> None:
        self.fundamentals = fundamentals
        self.price_history = price_history
        self.volume_history = volume_history
        self.financial_statements = financial_statements
        self.config = config
        self.current_members = current_members

    def fit(self, X: pd.DataFrame, y: object = None) -> InvestabilityScreenSelector:
        """Screen the universe and record which columns of ``X`` survive.

        Args:
            X: Linear returns with tickers as columns.  Feature names are
                required to map screen results onto the columns.
            y: Ignored; present for sklearn API consistency.

        Returns:
            Fitted selector.
        """
        if (
            self.fundamentals is None
            or self.price_history is None
            or self.volume_history is None
        ):
            msg = (
                "InvestabilityScreenSelector requires fundamentals, price_history "
                "and volume_history to be set before fit()."
            )
            raise DataError(msg)

        # Returns legitimately contain gaps, so NaN must be permitted. sklearn's
        # validate_data stub mistypes X as str; it accepts array-like at runtime.
        validate_data(self, X, ensure_all_finite="allow-nan")  # pyright: ignore
        if not hasattr(self, "feature_names_in_"):
            msg = (
                "InvestabilityScreenSelector requires X to be a DataFrame with "
                "ticker column names; got an array without feature names."
            )
            raise DataError(msg)

        config = self.config if self.config is not None else InvestabilityScreenConfig()

        members = apply_investability_screens(
            fundamentals=self.fundamentals,
            price_history=self.price_history,
            volume_history=self.volume_history,
            financial_statements=self.financial_statements,
            config=config,
            current_members=self.current_members,
        )

        member_set = set(members)
        columns = pd.Index(self.feature_names_in_)
        self.to_keep_ = np.array(
            [ticker in member_set for ticker in columns], dtype=bool
        )
        self.investable_universe_ = columns[self.to_keep_]
        return self

    def _get_support_mask(self) -> NDArray[np.bool_]:
        check_is_fitted(self)
        return self.to_keep_

    def __sklearn_tags__(self):  # type: ignore[no-untyped-def]
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
