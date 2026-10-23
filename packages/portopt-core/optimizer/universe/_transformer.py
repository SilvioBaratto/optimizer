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
import warnings
from typing import cast

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


def _naive_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Drop the timezone so frames and cutoffs compare on wall-clock dates."""
    return index.tz_localize(None) if index.tz is not None else index


def _window_end(index: pd.Index) -> pd.Timestamp | None:
    """Last date of a datetime-indexed fit window, or ``None`` without one."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return None
    end = _naive_index(index).max()
    return None if end is pd.NaT else cast(pd.Timestamp, end)


def _naive_datetimes(values: pd.Series) -> pd.Series:
    """Parse ``values`` as dates, with unparseable entries as ``NaT``."""
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.tz_localize(None) if parsed.dt.tz is not None else parsed


def _dated_index(index: pd.Index) -> pd.DatetimeIndex | None:
    """Read a history frame's index as wall-clock dates, or ``None`` if it has none.

    ``datetime.date`` objects and ISO 8601 strings are parsed.  Other string
    layouts are refused because a day-first or month-first guess would cut the
    frame on the wrong date without any signal, and integer positions are
    refused because ``pd.to_datetime`` would read them as nanoseconds since 1970.
    """
    if isinstance(index, pd.DatetimeIndex):
        return _naive_index(index)
    kind = pd.api.types.infer_dtype(index, skipna=True)
    if kind not in {"string", "date", "datetime", "datetime64"}:
        return None
    try:
        parsed = pd.to_datetime(index, format="ISO8601" if kind == "string" else None)
    except (ValueError, TypeError):
        return None
    return _naive_index(parsed) if isinstance(parsed, pd.DatetimeIndex) else None


def _truncate_history(
    frame: pd.DataFrame, cutoff: pd.Timestamp, name: str
) -> pd.DataFrame:
    """Keep only the rows dated on or before ``cutoff``.

    A frame whose index cannot be read as dates has no calendar to cut and is
    returned unchanged, with a warning because the screens then see data from
    after the fit window.
    """
    if frame.empty:
        return frame
    dates = _dated_index(frame.index)
    if dates is None:
        warnings.warn(
            f"{name} has no date index, so it cannot be cut at the fit window end "
            f"({cutoff.date()}); screening on its full history, which can include "
            "data from after the fit window.",
            UserWarning,
            stacklevel=2,
        )
        return frame
    return frame.loc[dates <= cutoff]


def _known_at(
    cutoff: pd.Timestamp,
    fundamentals: pd.DataFrame,
    price_history: pd.DataFrame,
    volume_history: pd.DataFrame,
    financial_statements: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Restrict the screening inputs to what was observable at ``cutoff``.

    Prices, volumes and statement period dates after ``cutoff`` are dropped, and
    a ``delisted_at`` later than ``cutoff`` is cleared so a name that was still
    trading is not excluded for dying afterwards.  ``market_cap`` and
    ``current_price`` are snapshots and are left to the caller.
    """
    price_history = _truncate_history(price_history, cutoff, "price_history")
    volume_history = _truncate_history(volume_history, cutoff, "volume_history")

    if "delisted_at" in fundamentals.columns:
        delisted = _naive_datetimes(fundamentals["delisted_at"])
        fundamentals = fundamentals.copy()
        fundamentals["delisted_at"] = fundamentals["delisted_at"].where(
            ~(delisted > cutoff)
        )

    if financial_statements is not None and "period_date" in (
        financial_statements.columns
    ):
        period = _naive_datetimes(financial_statements["period_date"])
        financial_statements = financial_statements.loc[
            period.isna() | (period <= cutoff)
        ]

    return fundamentals, price_history, volume_history, financial_statements


class InvestabilityScreenSelector(SelectorMixin, BaseEstimator):
    """Select investable assets from a return matrix via fundamental screens.

    ``fit`` runs `apply_investability_screens` over the injected
    fundamentals / price / volume data and records which columns of ``X``
    (linear-return series, one column per ticker) survive.  ``transform`` returns
    the surviving columns, preserving ``X``'s column order.

    When ``X`` has a ``DatetimeIndex`` the screens are evaluated as of its last
    date: price / volume rows and statement ``period_date`` values after it are
    ignored, and a ``delisted_at`` after it does not exclude a name.  This keeps
    a walk-forward fold from screening on data that arrives after its training
    window.  Price / volume frames indexed by ``datetime.date`` objects or date
    strings are cut the same way; a frame whose index holds no dates is screened
    in full with a ``UserWarning``, because that lets later data leak into the
    fold.  ``market_cap`` and ``current_price`` are snapshots that the caller
    must supply as of the same date, and an ``X`` without dates is screened on
    the full history.

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

        Raises:
            DataError: If ``fundamentals``, ``price_history`` or
                ``volume_history`` was not set, if ``X`` has no ticker column
                names, if ``fundamentals`` lacks ``market_cap`` or
                ``current_price``, or if ``price_history`` or ``volume_history``
                is empty, including when the fit window cuts away all of it.
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

        fundamentals = self.fundamentals
        price_history = self.price_history
        volume_history = self.volume_history
        financial_statements = self.financial_statements
        cutoff = _window_end(X.index)
        if cutoff is not None:
            fundamentals, price_history, volume_history, financial_statements = (
                _known_at(
                    cutoff,
                    fundamentals,
                    price_history,
                    volume_history,
                    financial_statements,
                )
            )

        members = apply_investability_screens(
            fundamentals=fundamentals,
            price_history=price_history,
            volume_history=volume_history,
            financial_statements=financial_statements,
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
