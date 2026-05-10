"""Lock raise-vs-warn contract for required regime columns (#571).

No DB. No network. Synthetic in-memory DataFrames only.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("typer")

from research.data_assembly import assemble_regime_data


def _macro(
    *,
    index: pd.DatetimeIndex,
    gdp_growth: float | pd.Series | None,
    yield_spread: float | pd.Series | None = 0.5,
) -> pd.DataFrame:
    """Build a macro DataFrame with optional gdp_growth/yield_spread columns."""
    cols: dict[str, pd.Series | float] = {}
    if gdp_growth is not None:
        cols["gdp_growth"] = gdp_growth
    if yield_spread is not None:
        cols["yield_spread"] = yield_spread
    return pd.DataFrame(cols, index=index)


def _empty() -> pd.DataFrame:
    return pd.DataFrame()


def test_raises_when_required_column_never_observed() -> None:
    idx = pd.date_range("2024-01-01", "2024-02-10", freq="D")
    macro = _macro(index=idx, gdp_growth=np.nan)
    with pytest.raises(ValueError, match="gdp_growth"):
        assemble_regime_data(macro, _empty(), _empty(), fill_limit=45)


def test_warns_when_required_column_has_stale_tail() -> None:
    idx = pd.date_range("2024-01-01", "2024-04-01", freq="D")
    gdp = pd.Series(np.nan, index=idx, dtype="float64")
    gdp.loc[:"2024-01-10"] = 1.0
    macro = _macro(index=idx, gdp_growth=gdp)

    with pytest.warns(UserWarning, match="gdp_growth"):
        assemble_regime_data(macro, _empty(), _empty(), fill_limit=45)


def test_no_warn_when_required_column_fresh() -> None:
    idx = pd.date_range("2024-01-01", "2024-02-10", freq="D")
    macro = _macro(index=idx, gdp_growth=2.5)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assemble_regime_data(macro, _empty(), _empty(), fill_limit=45)


def test_non_required_column_still_warns_only() -> None:
    idx = pd.date_range("2024-01-01", "2024-02-10", freq="D")
    macro = _macro(index=idx, gdp_growth=2.5, yield_spread=np.nan)

    with pytest.warns(UserWarning, match="yield_spread"):
        assemble_regime_data(macro, _empty(), _empty(), fill_limit=45)
