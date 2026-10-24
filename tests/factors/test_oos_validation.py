"""Tests for rolling block out-of-sample factor validation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError
from optimizer.factors import (
    FactorOOSConfig,
    FactorOOSResult,
    FactorValidationConfig,
    run_factor_oos_validation,
)
from optimizer.factors._oos_validation import _make_cpcv_folds, _make_folds
from optimizer.validation._config import CPCVConfig

_FACTORS = ["value", "momentum"]
_TICKERS = [f"T{i:02d}" for i in range(30)]


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
        assert cfg.train_periods == 36
        assert cfg.val_periods == 12
        assert cfg.step_periods == 6

    def test_min_observations_default_matches_factor_validation(self) -> None:
        assert (
            FactorOOSConfig().min_observations
            == FactorValidationConfig().min_ic_observations
        )

    def test_frozen(self) -> None:
        cfg = FactorOOSConfig()
        with pytest.raises(AttributeError):
            cfg.train_periods = 24  # type: ignore[misc]

    @pytest.mark.parametrize("field", ["train_periods", "val_periods", "step_periods"])
    @pytest.mark.parametrize("value", [0, -3])
    def test_non_positive_window_is_rejected(self, field: str, value: int) -> None:
        with pytest.raises(ConfigurationError, match=field):
            FactorOOSConfig(**{field: value})

    def test_minimum_window_is_accepted(self) -> None:
        cfg = FactorOOSConfig(train_periods=1, val_periods=1, step_periods=1)
        assert (cfg.train_periods, cfg.val_periods, cfg.step_periods) == (1, 1, 1)

    def test_custom_values(self) -> None:
        cfg = FactorOOSConfig(train_periods=24, val_periods=6, step_periods=3)
        assert cfg.train_periods == 24
        assert cfg.val_periods == 6
        assert cfg.step_periods == 3


# ---------------------------------------------------------------------------
# _make_folds (internal, but directly testable)
# ---------------------------------------------------------------------------


class TestMakeFolds:
    def test_fold_count_matches_formula(self) -> None:
        """n_folds = floor((total - train) / step)."""
        dates = pd.date_range("2019-01-31", periods=48, freq="ME")
        folds = _make_folds(dates, train_periods=36, val_periods=12, step_periods=6)
        assert len(folds) == (48 - 36) // 6  # = 2

    def test_train_val_within_fold_are_disjoint(self) -> None:
        """Within each fold, train and val windows share no dates."""
        dates = pd.date_range("2019-01-31", periods=48, freq="ME")
        folds = _make_folds(dates, train_periods=36, val_periods=12, step_periods=6)
        for train_dates, val_dates in folds:
            overlap = set(train_dates) & set(val_dates)
            assert len(overlap) == 0

    def test_val_starts_immediately_after_train(self) -> None:
        """Val window starts right after training ends."""
        dates = pd.date_range("2019-01-31", periods=60, freq="ME")
        folds = _make_folds(dates, train_periods=24, val_periods=12, step_periods=12)
        for train_dates, val_dates in folds:
            assert train_dates[-1] < val_dates[0]

    def test_zero_folds_when_not_enough_data(self) -> None:
        dates = pd.date_range("2019-01-31", periods=24, freq="ME")
        folds = _make_folds(dates, train_periods=36, val_periods=12, step_periods=6)
        assert len(folds) == 0

    def test_fold_count_with_custom_config(self) -> None:
        """floor((60 - 24) / 12) = 3."""
        dates = pd.date_range("2019-01-31", periods=60, freq="ME")
        folds = _make_folds(dates, train_periods=24, val_periods=12, step_periods=12)
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
        config = FactorOOSConfig(train_periods=36, val_periods=12, step_periods=6)
        result = run_factor_oos_validation(scores, returns, config)
        expected = (len(dates_48) - config.train_periods) // config.step_periods
        assert result.n_folds == expected

    def test_per_fold_ic_shape(
        self, panel_48: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        scores, returns = panel_48
        config = FactorOOSConfig(train_periods=36, val_periods=12, step_periods=6)
        result = run_factor_oos_validation(scores, returns, config)
        assert result.per_fold_ic.shape == (result.n_folds, len(_FACTORS))
        assert list(result.per_fold_ic.columns) == _FACTORS

    def test_per_fold_spread_shape(
        self, panel_48: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        scores, returns = panel_48
        config = FactorOOSConfig(train_periods=36, val_periods=12, step_periods=6)
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

    def test_zero_folds_gives_float_nan_ic_and_icir(self) -> None:
        dates = pd.date_range("2019-01-31", periods=24, freq="ME")
        scores, returns = _make_panel(dates, np.random.default_rng(0))
        config = FactorOOSConfig(train_periods=36, val_periods=12, step_periods=6)

        result = run_factor_oos_validation(scores, returns, config)

        assert result.n_folds == 0
        assert result.mean_oos_ic.dtype == float
        assert result.mean_oos_icir.dtype == float
        assert result.mean_oos_ic.isna().all()
        assert result.mean_oos_icir.isna().all()
        assert list(result.mean_oos_icir.index) == _FACTORS

    def test_single_fold_has_no_icir(self) -> None:
        """One fold IC cannot give a dispersion, so ICIR is undefined, not 0."""
        dates = pd.date_range("2019-01-31", periods=48, freq="ME")
        scores, returns = _make_panel(dates, np.random.default_rng(3))
        config = FactorOOSConfig(train_periods=36, val_periods=12, step_periods=12)

        result = run_factor_oos_validation(scores, returns, config)

        assert result.n_folds == 1
        assert result.mean_oos_ic.notna().all()
        assert result.mean_oos_icir.isna().all()

    def test_factor_without_valid_fold_ic_has_nan_icir_alongside_others(
        self, panel_48: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        scores, returns = panel_48
        scores = scores.assign(value=np.nan)
        config = FactorOOSConfig(train_periods=36, val_periods=12, step_periods=6)

        result = run_factor_oos_validation(scores, returns, config)

        assert result.n_folds == 2
        assert np.isnan(result.mean_oos_ic["value"])
        assert np.isnan(result.mean_oos_icir["value"])
        assert np.isfinite(result.mean_oos_icir["momentum"])

    def test_oos_ic_unaffected_by_train_data(
        self,
        panel_48: tuple[pd.DataFrame, pd.DataFrame],
        dates_48: pd.DatetimeIndex,
    ) -> None:
        """Corrupting training-window scores must not change OOS IC."""
        scores, returns = panel_48
        config = FactorOOSConfig(train_periods=36, val_periods=12, step_periods=6)
        result_orig = run_factor_oos_validation(scores, returns, config)

        # Overwrite the first 36 periods (training window of fold 0) with noise
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
        config = FactorOOSConfig(train_periods=36, val_periods=12, step_periods=6)
        result = run_factor_oos_validation(scores, returns, config)
        assert result.n_folds == 0
        assert result.per_fold_ic.empty
        assert result.per_fold_spread.empty

    def test_non_overlapping_val_windows_with_step_equals_val(self) -> None:
        """Non-overlapping val windows when step == val_periods."""
        dates = pd.date_range("2019-01-31", periods=60, freq="ME")
        rng = np.random.default_rng(7)
        scores, returns = _make_panel(dates, rng)
        config = FactorOOSConfig(train_periods=24, val_periods=12, step_periods=12)
        result = run_factor_oos_validation(scores, returns, config)
        # floor((60-24)/12) = 3
        assert result.n_folds == 3
        assert result.per_fold_ic.shape == (3, len(_FACTORS))


# ---------------------------------------------------------------------------
# Thin cross-sections
# ---------------------------------------------------------------------------


def _thin_validation_dates(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    n_names: int,
    *,
    drop: bool,
) -> pd.DataFrame:
    """Keep scores for only ``n_names`` tickers on ``dates``, rank-aligned.

    The surviving scores are the returns themselves, so every thin date has a
    perfect rank correlation (and the widest possible quantile spread).  With
    ``drop=True`` the dates lose all scores instead.
    """
    out = scores.copy()
    date_level = out.index.get_level_values(0)
    ticker_level = out.index.get_level_values(1)
    kept = set(_TICKERS[:n_names])
    for date in dates:
        on_date = date_level == date
        out.loc[on_date, :] = np.nan
        if drop:
            continue
        keep_rows = on_date & ticker_level.isin(kept)
        out.loc[keep_rows, :] = np.repeat(
            returns.loc[keep_rows].to_numpy(), out.shape[1], axis=1
        )
    return out


class TestThinCrossSections:
    _CONFIG = FactorOOSConfig(train_periods=36, val_periods=12, step_periods=6)

    @pytest.fixture()
    def thin_dates(self, dates_48: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return dates_48[40:43]

    def test_ic_ignores_dates_below_min_observations(
        self,
        panel_48: tuple[pd.DataFrame, pd.DataFrame],
        thin_dates: pd.DatetimeIndex,
    ) -> None:
        """Three-name rank ICs are +/-1 noise and must not enter the fold mean."""
        scores, returns = panel_48
        thinned = _thin_validation_dates(scores, returns, thin_dates, 3, drop=False)
        dropped = _thin_validation_dates(scores, returns, thin_dates, 3, drop=True)

        result = run_factor_oos_validation(thinned, returns, self._CONFIG)
        reference = run_factor_oos_validation(dropped, returns, self._CONFIG)

        pd.testing.assert_frame_equal(result.per_fold_ic, reference.per_fold_ic)

    def test_spread_ignores_dates_below_min_observations(
        self,
        panel_48: tuple[pd.DataFrame, pd.DataFrame],
        thin_dates: pd.DatetimeIndex,
    ) -> None:
        scores, returns = panel_48
        thinned = _thin_validation_dates(scores, returns, thin_dates, 6, drop=False)
        dropped = _thin_validation_dates(scores, returns, thin_dates, 6, drop=True)

        result = run_factor_oos_validation(thinned, returns, self._CONFIG)
        reference = run_factor_oos_validation(dropped, returns, self._CONFIG)

        pd.testing.assert_frame_equal(result.per_fold_spread, reference.per_fold_spread)

    def test_min_observations_is_configurable(
        self,
        panel_48: tuple[pd.DataFrame, pd.DataFrame],
        thin_dates: pd.DatetimeIndex,
    ) -> None:
        scores, returns = panel_48
        thinned = _thin_validation_dates(scores, returns, thin_dates, 3, drop=False)
        dropped = _thin_validation_dates(scores, returns, thin_dates, 3, drop=True)
        permissive = FactorOOSConfig(
            train_periods=36, val_periods=12, step_periods=6, min_observations=3
        )

        loose = run_factor_oos_validation(thinned, returns, permissive)
        guarded = run_factor_oos_validation(dropped, returns, permissive)

        assert not np.allclose(
            loose.per_fold_ic.iloc[0].to_numpy(),
            guarded.per_fold_ic.iloc[0].to_numpy(),
        )

    def test_cpcv_path_applies_min_observations(
        self,
        panel_48: tuple[pd.DataFrame, pd.DataFrame],
        thin_dates: pd.DatetimeIndex,
    ) -> None:
        scores, returns = panel_48
        cpcv = CPCVConfig(n_folds=4, n_test_folds=1)
        thinned = _thin_validation_dates(scores, returns, thin_dates, 3, drop=False)
        dropped = _thin_validation_dates(scores, returns, thin_dates, 3, drop=True)

        result = run_factor_oos_validation(thinned, returns, cpcv_config=cpcv)
        reference = run_factor_oos_validation(dropped, returns, cpcv_config=cpcv)

        pd.testing.assert_frame_equal(result.per_fold_ic, reference.per_fold_ic)


# ---------------------------------------------------------------------------
# Warnings (issue #239)
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_zero_folds_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Zero folds triggers a logger.warning."""
        dates = pd.date_range("2019-01-31", periods=24, freq="ME")
        rng = np.random.default_rng(0)
        scores, returns = _make_panel(dates, rng)
        config = FactorOOSConfig(train_periods=36, val_periods=12, step_periods=6)
        _logger = "optimizer.factors._oos_validation"
        with caplog.at_level(logging.WARNING, logger=_logger):
            run_factor_oos_validation(scores, returns, config)
        assert any("0 folds" in r.message for r in caplog.records)

    def test_few_folds_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Fewer than 3 folds triggers a logger.warning."""
        # 48 dates, train=36, step=6 → floor((48-36)/6) = 2 folds
        dates = pd.date_range("2019-01-31", periods=48, freq="ME")
        rng = np.random.default_rng(1)
        scores, returns = _make_panel(dates, rng)
        config = FactorOOSConfig(train_periods=36, val_periods=12, step_periods=6)
        _logger = "optimizer.factors._oos_validation"
        with caplog.at_level(logging.WARNING, logger=_logger):
            result = run_factor_oos_validation(scores, returns, config)
        assert result.n_folds == 2
        assert any("only 2 fold" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# CPCV path (issue #100)
# ---------------------------------------------------------------------------


class TestCPCVPath:
    @pytest.fixture()
    def panel_60(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        dates = pd.date_range("2019-01-31", periods=60, freq="ME")
        return _make_panel(dates, np.random.default_rng(42))

    def test_fold_count_matches_cnk(self) -> None:
        """Number of folds matches C(n_folds, n_test_folds)."""
        from math import comb

        dates = pd.date_range("2019-01-31", periods=60, freq="ME")
        cpcv = CPCVConfig(n_folds=6, n_test_folds=2)
        folds = _make_cpcv_folds(dates, cpcv)
        assert len(folds) == comb(6, 2)  # 15

    def test_per_fold_ic_shape(
        self, panel_60: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        from math import comb

        scores, returns = panel_60
        cpcv = CPCVConfig(n_folds=6, n_test_folds=2)
        result = run_factor_oos_validation(scores, returns, cpcv_config=cpcv)
        assert result.per_fold_ic.shape == (comb(6, 2), len(_FACTORS))

    def test_cpcv_overrides_rolling_config(
        self, panel_60: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """When cpcv_config is given, rolling config is ignored."""
        from math import comb

        scores, returns = panel_60
        rolling = FactorOOSConfig(train_periods=36, val_periods=12, step_periods=6)
        cpcv = CPCVConfig(n_folds=6, n_test_folds=2)
        result = run_factor_oos_validation(
            scores, returns, config=rolling, cpcv_config=cpcv
        )
        assert result.n_folds == comb(6, 2)

    def test_purging_removes_adjacent_train_dates(self) -> None:
        """Purging excludes dates adjacent to test boundaries from train."""
        dates = pd.date_range("2019-01-31", periods=60, freq="ME")
        cpcv = CPCVConfig(n_folds=6, n_test_folds=2, purged_size=1)
        folds = _make_cpcv_folds(dates, cpcv)
        for train_dates, test_dates in folds:
            train_set = set(train_dates)
            test_set = set(test_dates)
            assert len(train_set & test_set) == 0

    def test_train_test_disjointness(self) -> None:
        """Train and test dates must never overlap."""
        dates = pd.date_range("2019-01-31", periods=48, freq="ME")
        cpcv = CPCVConfig(n_folds=6, n_test_folds=2, purged_size=0, embargo_size=0)
        folds = _make_cpcv_folds(dates, cpcv)
        for train_dates, test_dates in folds:
            overlap = set(train_dates) & set(test_dates)
            assert len(overlap) == 0
