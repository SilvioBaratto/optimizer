"""Tests for StrictSectorImputer, the imputer used by the pre-selection pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import DataError
from optimizer.pre_selection._strict_imputer import StrictSectorImputer
from optimizer.preprocessing import SectorImputer


def _frame(n_rows: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        rng.normal(0.0, 0.01, (n_rows, 3)),
        columns=list("ABC"),
        index=pd.date_range("2024-01-01", periods=n_rows),
    )


class TestStrictSectorImputer:
    def test_keeps_the_sector_imputer_interface(self) -> None:
        assert issubclass(StrictSectorImputer, SectorImputer)
        assert StrictSectorImputer().get_params() == SectorImputer().get_params()

    def test_row_missing_for_every_asset_raises_naming_the_dates(self) -> None:
        df = _frame()
        df.iloc[3, :] = np.nan

        with pytest.raises(DataError, match="2024-01-04") as exc:
            StrictSectorImputer().fit_transform(df)

        assert "1 row(s)" in str(exc.value)

    def test_many_unfillable_rows_are_counted_and_truncated(self) -> None:
        df = _frame()
        df.iloc[:7, :] = np.nan

        with pytest.raises(DataError) as exc:
            StrictSectorImputer().fit_transform(df)

        message = str(exc.value)
        assert "7 row(s)" in message
        assert "2024-01-05" in message
        assert "2024-01-06" not in message
        assert "(+2 more)" in message

    def test_frame_without_columns_is_returned_unchanged(self) -> None:
        df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=3))

        assert StrictSectorImputer().fit_transform(df).shape == (3, 0)

    def test_fillable_frame_matches_sector_imputer(self) -> None:
        df = _frame()
        df.iloc[3, 1:] = np.nan
        df.iloc[5, 0] = np.nan

        strict = StrictSectorImputer().fit_transform(df)
        plain = SectorImputer().fit_transform(df)

        pd.testing.assert_frame_equal(strict, plain)
        assert strict.notna().all().all()
