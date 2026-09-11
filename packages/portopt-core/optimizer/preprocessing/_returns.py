"""Serialisable wrapper around ``skfolio.preprocessing.prices_to_returns``.

``prices_to_returns`` is the single most load-bearing preprocessing step in the
library: every optimizer and estimator consumes **linear** returns (see
SKILL.md, "the one rule that matters"). This module exposes the skfolio 1.0.6
conversion surface through the project convention — a frozen, serialisable
:class:`ReturnsConfig` plus a :func:`to_returns` factory — so the conversion
parameters can be round-tripped, hashed, and grid-searched like any other config.

The conversion changes data *semantics* (prices -> returns) and therefore runs
**outside** the sklearn ``Pipeline`` — the pipeline operates on return
DataFrames only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
from skfolio.preprocessing import prices_to_returns

__all__ = [
    "JoinMethod",
    "ReturnsConfig",
    "to_returns",
]


class JoinMethod(str, Enum):
    """How to align the price and (optional) benchmark price frames.

    Maps directly to the ``join`` argument of
    :func:`skfolio.preprocessing.prices_to_returns`.
    """

    LEFT = "left"
    RIGHT = "right"
    INNER = "inner"
    OUTER = "outer"
    CROSS = "cross"


@dataclass(frozen=True)
class ReturnsConfig:
    """Frozen, serialisable configuration for price -> return conversion.

    Mirrors the keyword surface of
    :func:`skfolio.preprocessing.prices_to_returns` (skfolio 1.0.6).

    Parameters
    ----------
    log_returns : bool, default=False
        When ``False`` (the default and the **required** setting for every
        skfolio optimizer/estimator) produce *linear* (simple) returns.  Log
        returns aggregate across time but not across assets, silently breaking
        every weighted-sum portfolio return.  Only set ``True`` for a
        log-normal prior with an explicit investment horizon.
    nan_threshold : float, default=1.0
        Drop assets whose fraction of missing prices exceeds this threshold.
        ``1.0`` keeps every asset.
    join : JoinMethod, default=JoinMethod.OUTER
        Index-alignment strategy between the asset frame and the optional
        benchmark frame.
    drop_inceptions_nan : bool, default=True
        Drop the leading NaN rows created before an asset's first quote.
    fill_nan : bool, default=True
        Forward-fill interior NaN prices before differencing.
    """

    log_returns: bool = False
    nan_threshold: float = 1.0
    join: JoinMethod = JoinMethod.OUTER
    drop_inceptions_nan: bool = True
    fill_nan: bool = True


def to_returns(
    prices: pd.DataFrame,
    y_prices: pd.DataFrame | None = None,
    *,
    config: ReturnsConfig | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a price panel to returns using a :class:`ReturnsConfig`.

    Thin, convention-compliant wrapper over
    :func:`skfolio.preprocessing.prices_to_returns`.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide price frame — ``DatetimeIndex`` rows, tickers as columns.
    y_prices : pd.DataFrame or None, default=None
        Optional benchmark price frame.  When provided the skfolio function
        returns a ``(X, y)`` tuple of aligned return frames.
    config : ReturnsConfig or None, default=None
        Conversion configuration.  ``None`` uses the linear-return default.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]
        Return frame, or ``(X, y)`` tuple when ``y_prices`` is supplied.

    Raises
    ------
    TypeError
        If ``prices`` (or ``y_prices``) is not a pandas DataFrame.
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

    cfg = config or ReturnsConfig()
    kwargs = {
        "log_returns": cfg.log_returns,
        "nan_threshold": cfg.nan_threshold,
        "join": cfg.join.value,
        "drop_inceptions_nan": cfg.drop_inceptions_nan,
        "fill_nan": cfg.fill_nan,
    }
    if y_prices is not None:
        return prices_to_returns(prices, y_prices, **kwargs)
    return prices_to_returns(prices, **kwargs)
