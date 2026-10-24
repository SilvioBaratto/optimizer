"""Locate the NaN cells that bound an asset's observed history."""

from __future__ import annotations

import pandas as pd


def edge_gap_mask(X: pd.DataFrame) -> pd.DataFrame:
    """Flag the NaN cells before each column's first and after its last value.

    Such a run marks the time before an asset was listed or after it was
    delisted, not a missed observation, so there is nothing to impute. A column
    with no observation at all is flagged in full.

    Args:
        X: Dates x assets frame, rows in time order.

    Returns:
        Boolean frame shaped like ``X``; ``True`` where ``X`` is NaN and no
        value precedes it, or none follows it, in its column.
    """
    seen = X.notna().astype(int)
    before_first = seen.cummax().eq(0)
    after_last = seen.iloc[::-1].cummax().iloc[::-1].eq(0)
    return before_first | after_last
