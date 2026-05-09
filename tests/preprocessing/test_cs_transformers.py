"""Tests for cross-sectional preprocessing transformers (re-exports)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.preprocessing import (
    CSGaussianRankScaler,
    CSPercentileRankScaler,
    CSStandardScaler,
    CSTanhShrinker,
    CSWinsorizer,
)

CS_CLASSES = [
    CSGaussianRankScaler,
    CSPercentileRankScaler,
    CSStandardScaler,
    CSTanhShrinker,
    CSWinsorizer,
]


@pytest.fixture
def panel() -> pd.DataFrame:
    """50 periods × 12 assets — wider than ``min_group_size=8`` default."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(size=(50, 12)),
        columns=[f"A{i:02d}" for i in range(12)],
    )


@pytest.fixture
def panel_with_nan(panel: pd.DataFrame) -> pd.DataFrame:
    """Same panel with one NaN per few rows (still ≥ 8 non-NaN per row)."""
    out = panel.copy()
    out.iloc[5, 0] = np.nan
    out.iloc[10, 1] = np.nan
    return out


@pytest.mark.parametrize("cls", CS_CLASSES, ids=lambda c: c.__name__)
def test_when_fit_transform_then_shape_preserved(
    cls: type, panel: pd.DataFrame
) -> None:
    transformer = cls()
    out = transformer.fit_transform(panel)
    assert out.shape == panel.shape


@pytest.mark.parametrize("cls", CS_CLASSES, ids=lambda c: c.__name__)
def test_when_input_has_nan_then_nan_preserved_per_row(
    cls: type, panel_with_nan: pd.DataFrame
) -> None:
    transformer = cls()
    out = transformer.fit_transform(panel_with_nan)
    out_arr = np.asarray(out)
    assert np.isnan(out_arr[5, 0])
    assert np.isnan(out_arr[10, 1])
    # Other rows remain finite (no NaN injected by the transformer).
    finite_mask = np.isnan(out_arr)
    assert finite_mask.sum() == 2


@pytest.mark.parametrize("cls", CS_CLASSES, ids=lambda c: c.__name__)
def test_when_fit_transform_then_returns_two_dim_array(
    cls: type, panel: pd.DataFrame
) -> None:
    transformer = cls()
    out = transformer.fit_transform(panel)
    assert np.asarray(out).ndim == 2


@pytest.mark.parametrize("cls", CS_CLASSES, ids=lambda c: c.__name__)
def test_when_get_params_then_round_trip(cls: type) -> None:
    transformer = cls()
    params = transformer.get_params()
    other = cls(**params)
    assert other.get_params() == params


def test_when_cs_standard_scaler_then_per_row_zero_mean(
    panel: pd.DataFrame,
) -> None:
    out = CSStandardScaler().fit_transform(panel)
    out_arr = np.asarray(out)
    # Each row should have ~zero mean after standardisation.
    np.testing.assert_allclose(np.nanmean(out_arr, axis=1), 0.0, atol=1e-9)


def test_when_cs_winsorizer_then_extremes_clipped(panel: pd.DataFrame) -> None:
    extreme = panel.copy()
    extreme.iloc[0, 0] = 1e6  # extreme outlier
    out = CSWinsorizer(low=0.05, high=0.95).fit_transform(extreme)
    out_arr = np.asarray(out)
    assert out_arr[0, 0] < 1e6
