"""Tests for rolling block out-of-sample factor validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    FactorOOSConfig,
    FactorOOSResult,
    run_factor_oos_validation,
)
from optimizer.factors._oos_validation import _make_folds

_FACTORS = ["value", "momentum"]
_TICKERS = [f"T{i:02d}" for i in range(20)]


def _make_panel(
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    factors: list[str] = _FACTORS,
    tickers: list[str] = _TICKERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic (date, ticker) MultiIndex panel fixtures."""
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    scores = pd.DataFrame(
        rng.normal(0, 1, (len(dates) * len(tickers), len(factors))),
        index=idx,
        columns=factors,
    )
    returns = pd.DataFrame(
        {"fwd_return": rng.normal(0.002, 0.04, len(dates) * len(tickers))},
        index=idx,
    )
    return scores, returns


@pytest.fixture()
def dates_48() -> pd.DatetimeIndex:
    return pd.date_range("2019-01-31", periods=48, freq="ME")


@pytest.fixture()
def panel_48(dates_48: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _make_panel(dates_48, np.random.default_rng(42))


# ---------------------------------------------------------------------------
# FactorOOSConfig
# ---------------------------------------------------------------------------


class TestFactorOOSConfig:
    def test_defaults(self) -> None:
        cfg = FactorOOSConfig()
        assert cfg.train_months == 36
        assert cfg.val_months == 12
        assert cfg.step_months == 6

    def test_frozen(self) -> None:
        cfg = FactorOOSConfig()
        with pytest.raises(AttributeError):
            cfg.train_months = 24  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = FactorOOSConfig(train_months=24, val_months=6, step_months=3)
        assert cfg.train_months == 24
        assert cfg.val_months == 6
        assert cfg.step_months == 3


# ---------------------------------------------------------------------------
# _make_folds (internal, but directly testable)
# ---------------------------------------------------------------------------


class TestMakeFolds:
    def test_fold_count_matches_formula(self) -> None:
        """n_folds = floor((total - train) / step)."""
        dates = pd.date_range("2019-01-31", periods=48, freq="ME")
        folds = _make_folds(dates, train_months=36, val_months=12, step_months=6)
        assert len(folds) == (48 - 36) // 6  # = 2

    def test_train_val_within_fold_are_disjoint(self) -> None:
        """Within each fold, train and val windows share no dates."""
        dates = pd.date_range("2019-01-31", periods=48, freq="ME")
        folds = _make_folds(dates, train_months=36, val_months=12, step_months=6)
        for train_dates, val_dates in folds:
            overlap = set(train_dates) & set(val_dates)
            assert len(overlap) == 0

    def test_val_starts_immediately_after_train(self) -> None:
        """Val window starts right after training ends."""
        dates = pd.date_range("2019-01-31", periods=60, freq="ME")
        folds = _make_folds(dates, train_months=24, val_months=12, step_months=12)
        for train_dates, val_dates in folds:
            assert train_dates[-1] < val_dates[0]

    def test_zero_folds_when_not_enough_data(self) -> None:
        dates = pd.date_range("2019-01-31", periods=24, freq="ME")
        folds = _make_folds(dates, train_months=36, val_months=12, step_months=6)
        assert len(folds) == 0

    def test_fold_count_with_custom_config(self) -> None:
        """floor((60 - 24) / 12) = 3."""
        dates = pd.date_range("2019-01-31", periods=60, freq="ME")
        folds = _make_folds(dates, train_months=24, val_months=12, step_months=12)
        assert len(folds) == (60 - 24) // 12  # = 3


# ---------------------------------------------------------------------------
# run_factor_oos_validation
# ---------------------------------------------------------------------------


class TestRunFactorOOSValidation:
    def test_returns_factor_oos_result(
        self, panel_48: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        scores, returns = panel_48
        result = run_factor_oos_validation(scores, returns)
        assert isinstance(result, FactorOOSResult)

    def test_fold_count_matches_formula(
        self, panel_48: tuple[pd.DataFrame, pd.DataFrame], dates_48: pd.DatetimeIndex
    ) -> None:
        scores, returns = panel_48
        config = FactorOOSConfig(train_months=36, val_months=12, step_months=6)
        result = run_factor_oos_validation(scores, returns, config)
        expected = (len(dates_48) - config.train_months) // config.step_months
        assert result.n_folds == expected

    def test_per_fold_ic_shape(
        self, panel_48: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        scores, returns = panel_48
        config = FactorOOSConfig(train_months=36, val_months=12, step_months=6)
        result = run_factor_oos_validation(scores, returns, config)
        assert result.per_fold_ic.shape == (result.n_folds, len(_FACTORS))
        assert list(result.per_fold_ic.columns) == _FACTORS

    def test_per_fold_spread_shape(
        self, panel_48: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        scores, returns = panel_48
        config = FactorOOSConfig(train_months=36, val_months=12, step_months=6)
        result = run_factor_oos_validation(scores, returns, config)
        assert result.per_fold_spread.shape == (result.n_folds, len(_FACTORS))
        assert list(result.per_fold_spread.columns) == _FACTORS

    def test_mean_oos_ic_equals_fold_mean(
        self, panel_48: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """mean_oos_ic is the mean of per-fold IC values."""
        scores, returns = panel_48
        result = run_factor_oos_validation(scores, returns)
        pd.testing.assert_series_equal(
            result.mean_oos_ic,
            result.per_fold_ic.mean(axis=0),
            check_names=False,
        )

    def test_mean_oos_icir_is_series_with_factors(
        self, panel_48: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        scores, returns = panel_48
        result = run_factor_oos_validation(scores, returns)
        assert isinstance(result.mean_oos_icir, pd.Series)
        assert list(result.mean_oos_icir.index) == _FACTORS

    def test_oos_ic_unaffected_by_train_data(
        self,
        panel_48: tuple[pd.DataFrame, pd.DataFrame],
        dates_48: pd.DatetimeIndex,
    ) -> None:
        """Corrupting training-window scores must not change OOS IC."""
        scores, returns = panel_48
        config = FactorOOSConfig(train_months=36, val_months=12, step_months=6)
        result_orig = run_factor_oos_validation(scores, returns, config)

        # Overwrite the first 36 months (training window of fold 0) with noise
        train_dates = dates_48[:36]
        scores_corrupted = scores.copy()
        train_mask = scores_corrupted.index.get_level_values(0).isin(train_dates)
        scores_corrupted.loc[train_mask] = 999.0

        result_corrupted = run_factor_oos_validation(scores_corrupted, returns, config)
        pd.testing.assert_frame_equal(
            result_orig.per_fold_ic, result_corrupted.per_fold_ic
        )

    def test_zero_folds_when_insufficient_data(self) -> None:
        dates = pd.date_range("2019-01-31", periods=24, freq="ME")
        rng = np.random.default_rng(0)
        scores, returns = _make_panel(dates, rng)
        config = FactorOOSConfig(train_months=36, val_months=12, step_months=6)
        result = run_factor_oos_validation(scores, returns, config)
        assert result.n_folds == 0
        assert result.per_fold_ic.empty
        assert result.per_fold_spread.empty

    def test_non_overlapping_val_windows_with_step_equals_val(self) -> None:
        """Non-overlapping val windows when step == val_months."""
        dates = pd.date_range("2019-01-31", periods=60, freq="ME")
        rng = np.random.default_rng(7)
        scores, returns = _make_panel(dates, rng)
        config = FactorOOSConfig(train_months=24, val_months=12, step_months=12)
        result = run_factor_oos_validation(scores, returns, config)
        # floor((60-24)/12) = 3
        assert result.n_folds == 3
        assert result.per_fold_ic.shape == (3, len(_FACTORS))
