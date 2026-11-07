"""Serialisable wrapper around ``skfolio.preprocessing.prices_to_returns``.

``prices_to_returns`` is the single most load-bearing preprocessing step in the
library: every optimizer and estimator consumes **linear** returns (see
SKILL.md, "the one rule that matters"). This module exposes the skfolio 1.0.6
conversion surface through the project convention — a frozen, serialisable
`ReturnsConfig` plus a `to_returns` factory — so the conversion
parameters can be round-tripped, hashed, and grid-searched like any other config.

The conversion changes data *semantics* (prices -> returns) and therefore runs
**outside** the sklearn ``Pipeline`` — the pipeline operates on return
DataFrames only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import cast

import pandas as pd
from skfolio.preprocessing import prices_to_returns

from optimizer.exceptions import DataError

__all__ = [
    "JoinMethod",
    "ReturnsConfig",
    "to_returns",
]


def _coerce_price_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Cast DB-sourced ``Decimal`` / object price columns to ``float64``.

    ``price_history`` OHLCV columns are SQL ``Numeric(20, 6)`` and read back
    as Python ``Decimal`` (pandas ``object`` dtype).  ``prices_to_returns``
    does **not** coerce them — it silently emits an *object-dtype* return
    frame of ``Decimal`` values, which then breaks (or silently corrupts)
    every numpy / skfolio computation downstream (mixing ``float`` and
    ``Decimal`` raises ``TypeError``).  This casts at the price boundary,
    preserving ``NaN`` gaps; it is a no-op for already-float frames.
    """
    if all(pd.api.types.is_float_dtype(dt) for dt in frame.dtypes):
        return frame
    try:
        return frame.astype("float64")
    except (TypeError, ValueError) as exc:
        raise DataError(
            f"to_returns could not cast {name!r} price columns to float; "
            "expected numeric prices (DB Numeric/Decimal), got non-numeric data"
        ) from exc


def _check_columns_kept(
    prices: pd.DataFrame, returns: pd.DataFrame, *, name: str
) -> None:
    """Report price columns that `prices_to_returns` removed from the output.

    skfolio discards a column with no price at all, before and after the row
    filter, without a word.  An asset that disappears that way would later look
    like an asset that was never requested, so the loss is surfaced here.

    Raises:
        DataError: If a non-empty ``prices`` frame lost every column.
    """
    if prices.shape[1] == 0:
        return
    kept = set(returns.columns)
    dropped = [c for c in prices.columns if c not in kept]
    if not dropped:
        return
    if not kept:
        raise DataError(
            f"to_returns dropped every column of {name!r}: none has a price "
            "from which a return can be computed"
        )
    warnings.warn(
        f"to_returns dropped columns of {name!r} with no usable prices: {dropped}",
        UserWarning,
        stacklevel=3,
    )


class JoinMethod(str, Enum):
    """How to align the price and (optional) benchmark price frames.

    Maps to the ``join`` argument of `prices_to_returns`.  skfolio's ``"cross"``
    is deliberately absent: a Cartesian product pairs every price date with every
    benchmark date and discards the date index, so its output is not a return
    series.
    """

    LEFT = "left"
    RIGHT = "right"
    INNER = "inner"
    OUTER = "outer"


@dataclass(frozen=True)
class ReturnsConfig:
    """Frozen, serialisable configuration for price -> return conversion.

    Mirrors the keyword surface of `prices_to_returns`.  ``join`` also accepts
    its string value (``"inner"``) and is coerced to `JoinMethod`; an unknown
    value raises ``ValueError``.

    Attributes:
        log_returns: When ``False`` (the default and the **required** setting
            for every skfolio optimizer/estimator) produce *linear* (simple)
            returns. Log returns aggregate across time but not across assets,
            silently breaking every weighted-sum portfolio return. Only set
            ``True`` for a log-normal prior with an explicit investment horizon.
        nan_threshold: Drop observations (dates) on which the fraction of
            assets with a missing price exceeds this threshold. It filters
            rows, not assets: a partly sparse asset stays in the panel while
            the dates it is missing are removed for every asset. ``1.0`` keeps
            every observation. An asset with no price at all is the exception:
            skfolio removes that column outright, and `to_returns` reports it.
        join: Index-alignment strategy between the asset frame and the optional
            benchmark frame.
        drop_inceptions_nan: Drop every leading row in which any asset still
            has no price, so the panel starts at the latest first quote of any
            asset and one late listing shortens the history of all of them.
            ``False`` keeps the NaNs for a downstream pre-selection step.
        fill_nan: Forward-fill NaN prices before differencing.  This also
            fills the *trailing* gap after an asset's last quote, so a delisted
            asset ends in a run of exact ``0.0`` returns instead of ``NaN``;
            pass ``delisted_at`` to `apply_delisting_returns` in that case.
            Setting it to ``False`` keeps the NaN tail but, with the default
            ``drop_inceptions_nan=True``, drops every row after the earliest
            delisting for all assets.
    """

    log_returns: bool = False
    nan_threshold: float = 1.0
    join: JoinMethod = JoinMethod.OUTER
    drop_inceptions_nan: bool = True
    fill_nan: bool = True

    def __post_init__(self) -> None:
        # A JSON round-trip hands ``join`` back as a plain string; normalising it
        # here keeps equality, hashing and ``to_returns`` independent of how the
        # config was built.
        object.__setattr__(self, "join", JoinMethod(self.join))


def to_returns(
    prices: pd.DataFrame,
    y_prices: pd.DataFrame | None = None,
    *,
    config: ReturnsConfig | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a price panel to returns using a `ReturnsConfig`.

    Thin, convention-compliant wrapper over `prices_to_returns`. DB-sourced ``Decimal``
    (object-dtype) price columns are cast to ``float64`` at this boundary so
    the returned frame is always float — the dtype every downstream skfolio
    estimator requires.

    Args:
        prices: Wide price frame — ``DatetimeIndex`` rows, tickers as columns.
        y_prices: Optional benchmark price frame. When provided the skfolio
            function returns a ``(X, y)`` tuple of aligned return frames.
        config: Conversion configuration. ``None`` uses the linear-return
            default.

    Returns:
        Return frame, or ``(X, y)`` tuple when ``y_prices`` is supplied.

    Warns:
        UserWarning: If some columns of ``prices`` or ``y_prices`` have no
            price at all and were dropped by the conversion; the message names
            them. A large universe may tolerate a few dead tickers, but they
            must not vanish unnoticed.

    Raises:
        TypeError: If ``prices`` (or ``y_prices``) is not a pandas DataFrame.
        DataError: If a price frame carries object-dtype columns that cannot
            be cast to float (non-numeric data), or if the conversion leaves
            ``prices`` or the benchmark with no column at all.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(
            f"to_returns requires a pandas DataFrame for 'prices', "
            f"got {type(prices).__name__}"
        )
    if y_prices is not None and not isinstance(y_prices, pd.DataFrame):
        raise TypeError(
            f"to_returns requires a pandas DataFrame for 'y_prices', "
            f"got {type(y_prices).__name__}"
        )

    # DB prices arrive as Decimal (object dtype); float-cast at the boundary so
    # skfolio returns a float64 frame every downstream estimator can consume.
    prices = _coerce_price_frame(prices, name="prices")
    if y_prices is not None:
        y_prices = _coerce_price_frame(y_prices, name="y_prices")

    cfg = config or ReturnsConfig()
    kwargs = {
        "log_returns": cfg.log_returns,
        "nan_threshold": cfg.nan_threshold,
        "join": cfg.join.value,
        "drop_inceptions_nan": cfg.drop_inceptions_nan,
        "fill_nan": cfg.fill_nan,
    }
    if y_prices is not None:
        x_ret, y_ret = prices_to_returns(prices, y_prices, **kwargs)
        _check_columns_kept(prices, x_ret, name="prices")
        _check_columns_kept(y_prices, y_ret, name="y_prices")
        return x_ret, y_ret
    returns = cast(pd.DataFrame, prices_to_returns(prices, **kwargs))
    _check_columns_kept(prices, returns, name="prices")
    return returns
