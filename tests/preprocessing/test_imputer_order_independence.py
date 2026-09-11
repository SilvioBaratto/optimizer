"""Order-independence / no-leakage tests for the NaN imputers.

Both SectorImputer and RegressionImputer must read their per-row statistics
from the *original* observations, never from values imputed into earlier
columns. If they leaked imputed values, the result would depend on the column
ordering of the input frame. These tests pin that invariance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimizer.preprocessing import RegressionImputer, SectorImputer


def _correlated_frame(n: int = 150, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n)
    return pd.DataFrame(
        {
            "A": 0.9 * common + 0.1 * rng.standard_normal(n),
            "B": 0.8 * common + 0.2 * rng.standard_normal(n),
            "C": 0.6 * common + 0.4 * rng.standard_normal(n),
            "D": 0.4 * common + 0.6 * rng.standard_normal(n),
        },
        index=pd.date_range("2020-01-01", periods=n),
    )


def test_sector_imputer_column_permutation_invariant() -> None:
    """Imputed values are identical regardless of input column order."""
    df = _correlated_frame()
    # Two same-sector assets missing at the same row so column order could
    # otherwise matter.
    row = df.index[100]
    df.loc[row, "A"] = np.nan
    df.loc[row, "B"] = np.nan
    mapping = {"A": "S", "B": "S", "C": "S", "D": "S"}

    out = SectorImputer(sector_mapping=mapping).fit_transform(df)
    reordered = df[["D", "C", "B", "A"]]
    out_perm = SectorImputer(sector_mapping=mapping).fit_transform(reordered)

    pd.testing.assert_frame_equal(out[["A", "B"]], out_perm[["A", "B"]])


def test_regression_imputer_column_permutation_invariant() -> None:
    """RegressionImputer output does not depend on column ordering."""
    df = _correlated_frame()
    # Target A and one of its strong neighbors B both missing at the same row.
    row = df.index[120]
    df.loc[row, "A"] = np.nan
    df.loc[row, "B"] = np.nan

    imp = RegressionImputer(n_neighbors=3, min_train_periods=20)
    out = imp.fit_transform(df)

    reordered = df[["D", "C", "B", "A"]]
    imp2 = RegressionImputer(n_neighbors=3, min_train_periods=20)
    out_perm = imp2.fit_transform(reordered)

    pd.testing.assert_frame_equal(
        out.loc[[row]].sort_index(axis=1),
        out_perm.loc[[row]].sort_index(axis=1),
    )


def test_regression_imputer_does_not_use_imputed_neighbor() -> None:
    """A neighbor that is NaN in the input must not be replaced by an imputed
    value when predicting the target — the target falls back instead."""
    df = _correlated_frame()
    imp = RegressionImputer(n_neighbors=2, min_train_periods=20).fit(df)

    row = df.index[120]
    test = df.copy()
    test.loc[row, "A"] = np.nan
    # Make A's top neighbor NaN too at the same row.
    top_nbr = imp.neighbors_["A"][0]
    test.loc[row, top_nbr] = np.nan

    out = imp.transform(test)
    # The fallback (sector/global mean of observed cells) is used for A;
    # it must equal the fitted fallback imputer's value, proving no imputed
    # neighbor leaked into the regression.
    fb = imp._fallback_imputer_.transform(test)
    assert out.loc[row, "A"] == float(fb.loc[row, "A"])
