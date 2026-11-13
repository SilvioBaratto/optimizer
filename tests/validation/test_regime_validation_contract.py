"""Pin the exact output of ``run_regime_validation``, branch by branch.

The expected frames are rebuilt here from the inputs, one sub-period and one
regime at a time, with ``_compute_period_metrics`` for the per-slice statistics.
That keeps the comparison exact on every platform (no float literal depends on
the local ``pow`` or reduction order) while still failing on a wrong slice, a
lost NaN row, a column out of order, a changed dtype, a different alpha share
or a missing log line.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import DataError
from optimizer.validation import (
    RegimeValidationConfig,
    RegimeValidationResult,
    run_regime_validation,
)
from optimizer.validation._regime import _compute_period_metrics

_LOGGER = "optimizer.validation._regime"

_MACRO_ROWS = {
    "expansion": {"pmi": 55.0, "spread_2s10s": 1.5, "hy_oas": 300.0},
    "slowdown": {"pmi": 50.0, "spread_2s10s": 0.5, "hy_oas": 400.0},
    "recession": {"pmi": 46.0, "spread_2s10s": -0.5, "hy_oas": 600.0},
    "recovery": {"pmi": 46.0, "spread_2s10s": 0.5, "hy_oas": 400.0},
}

# A short repeating pattern plus a per-regime drift: the same on every numpy
# version, and varied enough that every statistic is non-trivial.
_PATTERN = np.array([0.004, -0.003, 0.002, -0.001, 0.0035, -0.0025, 0.001])
_EDGE_DRIFT = {
    "expansion": 0.001,
    "slowdown": 0.0,
    "recession": -0.002,
    "recovery": 0.0005,
}
_LOSS_DRIFT = dict.fromkeys(_EDGE_DRIFT, -0.003)

# Adjacent blocks differ, so each block is exactly one sub-period.
_MIXED_LAYOUT = (
    ("expansion", 40),
    ("slowdown", 5),
    ("recession", 30),
    ("slowdown", 4),
    ("expansion", 25),
    ("recovery", 10),
    ("expansion", 36),
)
_SHARED_LAYOUT = (("expansion", 40), ("recovery", 40), ("expansion", 30))

_METRIC_COLUMNS = ["ann_return", "ann_vol", "sharpe", "max_drawdown"]
_REGIME_COLUMNS = ["obs", "coverage_pct", "obs_sufficient", *_METRIC_COLUMNS]
_SUBPERIOD_COLUMNS = ["start", "end", "regime", "obs", *_METRIC_COLUMNS]
_EMPTY_REGIME_COLUMNS = [
    "obs",
    "coverage_pct",
    "ann_return",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "obs_sufficient",
]

_STALE_LIMIT_DAYS = 10
_FIRST_MACRO_DAY = 5
_LAST_MACRO_DAY = 59


def _scenario(
    layout: tuple[tuple[str, int], ...], drift: dict[str, float]
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """Daily returns, one macro row per day, and the regime label of each day."""
    labels = [name for name, length in layout for _ in range(length)]
    days = pd.bdate_range("2024-01-02", periods=len(labels))
    macro = pd.DataFrame([_MACRO_ROWS[name] for name in labels], index=days)
    values = np.concatenate(
        [np.resize(_PATTERN, length) + drift[name] for name, length in layout]
    )
    return pd.Series(values, index=days), macro, labels


def _slice_metrics(
    returns: pd.Series, config: RegimeValidationConfig
) -> dict[str, float]:
    if len(returns) < config.min_regime_obs:
        return dict.fromkeys(_METRIC_COLUMNS, np.nan)
    rf_daily = config.risk_free_rate / config.trading_days_per_year
    return _compute_period_metrics(returns, config.trading_days_per_year, rf_daily)


def _expected_subperiods(
    returns: pd.Series, labels: list[str], config: RegimeValidationConfig
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    first = 0
    for last in range(len(labels)):
        if last + 1 < len(labels) and labels[last + 1] == labels[first]:
            continue
        block = returns.iloc[first : last + 1]
        rows.append(
            {
                "start": returns.index[first],
                "end": returns.index[last],
                "regime": labels[first],
                "obs": len(block),
                **_slice_metrics(block, config),
            }
        )
        first = last + 1
    return pd.DataFrame(rows)


def _expected_regimes(
    returns: pd.Series,
    labels: list[str],
    config: RegimeValidationConfig,
    total_obs: int,
) -> tuple[pd.DataFrame, pd.Series]:
    rf_daily = config.risk_free_rate / config.trading_days_per_year
    rows: list[dict[str, object]] = []
    alpha: dict[str, float] = {}
    label_array = np.array(labels)
    for name in sorted(set(labels)):
        regime_returns = returns[label_array == name]
        sufficient = len(regime_returns) >= config.min_regime_obs
        excess = float((regime_returns - rf_daily).mean()) if sufficient else 0.0
        alpha[name] = excess if excess > 0 else 0.0
        rows.append(
            {
                "regime": name,
                "obs": len(regime_returns),
                "coverage_pct": len(regime_returns) / total_obs,
                "obs_sufficient": sufficient,
                **_slice_metrics(regime_returns, config),
            }
        )
    total = sum(alpha.values())
    shares = (
        {name: value / total for name, value in alpha.items()}
        if total > 0
        else dict.fromkeys(alpha, 0.0)
    )
    frame = pd.DataFrame(rows).set_index("regime")
    return frame, pd.Series(shares, name="alpha_concentration")


def _assert_matches_slices(
    result: RegimeValidationResult,
    returns: pd.Series,
    labels: list[str],
    config: RegimeValidationConfig,
    total_obs: int,
) -> None:
    """Compare every field of ``result`` with the slice-by-slice expectation."""
    regimes, shares = _expected_regimes(returns, labels, config, total_obs)
    pd.testing.assert_frame_equal(result.per_regime_metrics, regimes, check_exact=True)
    pd.testing.assert_frame_equal(
        result.per_subperiod_metrics,
        _expected_subperiods(returns, labels, config),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        result.regime_alpha_concentration, shares, check_exact=True
    )
    pd.testing.assert_series_equal(
        result.regime_timeline,
        pd.Series(labels, index=returns.index, name="regime"),
        check_exact=True,
    )
    threshold = config.single_regime_alpha_threshold
    flagged = [name for name, share in shares.items() if share > threshold]
    assert result.concentrated_regimes == flagged
    assert result.total_obs == total_obs
    assert result.n_regimes_observed == len(set(labels))


def _assert_empty(result: RegimeValidationResult, total_obs: int) -> None:
    pd.testing.assert_frame_equal(
        result.per_regime_metrics, pd.DataFrame(columns=_EMPTY_REGIME_COLUMNS)
    )
    pd.testing.assert_frame_equal(
        result.per_subperiod_metrics, pd.DataFrame(columns=_SUBPERIOD_COLUMNS)
    )
    pd.testing.assert_series_equal(
        result.regime_alpha_concentration, pd.Series(dtype=float)
    )
    pd.testing.assert_series_equal(result.regime_timeline, pd.Series(dtype=str))
    assert result.concentrated_regimes == []
    assert result.total_obs == total_obs
    assert result.n_regimes_observed == 0


def _records(caplog: pytest.LogCaptureFixture) -> list[tuple[str, int, str]]:
    return [
        (record.name, record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == _LOGGER
    ]


class TestMixedLayout:
    """Sufficient and short sub-periods and regimes side by side."""

    def test_every_field_matches_the_slice_by_slice_expectation(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every field equals its per-slice value, and nothing is logged."""
        returns, macro, labels = _scenario(_MIXED_LAYOUT, _EDGE_DRIFT)
        config = RegimeValidationConfig()

        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            result = run_regime_validation(returns, macro, config)

        _assert_matches_slices(result, returns, labels, config, len(returns))
        assert _records(caplog) == []

    def test_columns_flags_and_nan_rows_are_fixed(self) -> None:
        """Column order, counts, coverage and the NaN rows are literal."""
        returns, macro, _ = _scenario(_MIXED_LAYOUT, _EDGE_DRIFT)

        result = run_regime_validation(returns, macro)

        regimes = result.per_regime_metrics
        subperiods = result.per_subperiod_metrics
        assert list(regimes.columns) == _REGIME_COLUMNS
        assert list(subperiods.columns) == _SUBPERIOD_COLUMNS
        assert regimes.index.name == "regime"
        assert list(regimes.index) == ["expansion", "recession", "recovery", "slowdown"]
        assert list(regimes["obs"]) == [101, 30, 10, 9]
        assert list(regimes["coverage_pct"]) == [101 / 150, 30 / 150, 10 / 150, 9 / 150]
        assert list(regimes["obs_sufficient"]) == [True, True, False, False]
        short_regimes = regimes[_METRIC_COLUMNS].isna().all(axis=1)
        assert list(short_regimes) == [False, False, True, True]
        assert list(subperiods["obs"]) == [40, 5, 30, 4, 25, 10, 36]
        short_blocks = subperiods[_METRIC_COLUMNS].isna().all(axis=1)
        assert list(short_blocks) == [False, True, False, True, False, True, False]
        assert result.concentrated_regimes == ["expansion"]
        assert result.regime_alpha_concentration.to_dict() == {
            "expansion": 1.0,
            "recession": 0.0,
            "recovery": 0.0,
            "slowdown": 0.0,
        }
        assert result.n_regimes_observed == 4
        assert result.total_obs == 150

    def test_no_config_means_the_standard_config(self) -> None:
        """Omitting the config gives the result of the default config."""
        returns, macro, labels = _scenario(_MIXED_LAYOUT, _EDGE_DRIFT)

        result = run_regime_validation(returns, macro, config=None)

        _assert_matches_slices(
            result, returns, labels, RegimeValidationConfig(), len(returns)
        )

    def test_risk_free_rate_enters_alpha_and_sharpe(self) -> None:
        """A positive risk-free rate shifts the alpha shares and the Sharpe ratios."""
        returns, macro, labels = _scenario(_MIXED_LAYOUT, _EDGE_DRIFT)
        config = RegimeValidationConfig(risk_free_rate=0.05, min_regime_obs=5)

        result = run_regime_validation(returns, macro, config)

        _assert_matches_slices(result, returns, labels, config, len(returns))

    def test_lower_minimum_turns_short_blocks_into_metrics(self) -> None:
        """Only blocks under the minimum keep NaN metrics."""
        returns, macro, labels = _scenario(_MIXED_LAYOUT, _EDGE_DRIFT)
        config = RegimeValidationConfig(min_regime_obs=5)

        result = run_regime_validation(returns, macro, config)

        _assert_matches_slices(result, returns, labels, config, len(returns))
        short_blocks = result.per_subperiod_metrics["sharpe"].isna()
        assert list(short_blocks) == [False, False, False, True, False, False, False]


class TestAlphaConcentrationBranches:
    """Positive, absent and shared alpha, and the concentration threshold."""

    def test_no_positive_alpha_leaves_every_share_at_zero(self) -> None:
        """Without positive alpha every share is 0.0 and nothing is flagged."""
        returns, macro, labels = _scenario(_MIXED_LAYOUT, _LOSS_DRIFT)
        config = RegimeValidationConfig()

        result = run_regime_validation(returns, macro, config)

        _assert_matches_slices(result, returns, labels, config, len(returns))
        assert result.regime_alpha_concentration.to_dict() == {
            "expansion": 0.0,
            "recession": 0.0,
            "recovery": 0.0,
            "slowdown": 0.0,
        }
        assert result.concentrated_regimes == []

    def test_shared_alpha_flags_no_regime(self) -> None:
        """Alpha split between two regimes stays under the default threshold."""
        returns, macro, labels = _scenario(_SHARED_LAYOUT, _EDGE_DRIFT)
        config = RegimeValidationConfig()

        result = run_regime_validation(returns, macro, config)

        _assert_matches_slices(result, returns, labels, config, len(returns))
        assert result.concentrated_regimes == []
        assert 0.0 < result.regime_alpha_concentration["recovery"] < 0.5

    def test_threshold_below_the_larger_share_flags_it(self) -> None:
        """A threshold under the larger share flags that regime only."""
        returns, macro, labels = _scenario(_SHARED_LAYOUT, _EDGE_DRIFT)
        config = RegimeValidationConfig(single_regime_alpha_threshold=0.5)

        result = run_regime_validation(returns, macro, config)

        _assert_matches_slices(result, returns, labels, config, len(returns))
        assert result.concentrated_regimes == ["expansion"]


class TestUnknownDays:
    """Days before the first macro row and past the staleness limit."""

    @staticmethod
    def _late_and_stale() -> tuple[pd.Series, pd.DataFrame, list[str], int]:
        """Macro rows for days 5..59 only, and the end of the fresh window."""
        returns, macro, labels = _scenario(
            (("expansion", 50), ("recession", 40)), _EDGE_DRIFT
        )
        trimmed = macro.iloc[_FIRST_MACRO_DAY : _LAST_MACRO_DAY + 1]
        horizon = trimmed.index[-1] + pd.Timedelta(days=_STALE_LIMIT_DAYS)
        fresh_end = int((returns.index <= horizon).sum())
        return returns, trimmed, labels, fresh_end

    def test_unknown_days_are_dropped_and_the_stale_ones_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Pre-macro and stale days leave the sample; the stale count is logged."""
        returns, macro, labels, fresh_end = self._late_and_stale()
        config = RegimeValidationConfig(max_macro_staleness_days=_STALE_LIMIT_DAYS)
        stale_days = len(returns) - fresh_end
        assert 0 < stale_days < len(returns) - _LAST_MACRO_DAY

        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            result = run_regime_validation(returns, macro, config)

        _assert_matches_slices(
            result,
            returns.iloc[_FIRST_MACRO_DAY:fresh_end],
            labels[_FIRST_MACRO_DAY:fresh_end],
            config,
            len(returns),
        )
        assert _records(caplog) == [
            (
                _LOGGER,
                logging.WARNING,
                f"run_regime_validation: {stale_days} OOS observations are more "
                f"than {_STALE_LIMIT_DAYS} days past the latest macro observation "
                "(stale macro data); classified as UNKNOWN.",
            )
        ]

    def test_included_unknown_days_form_their_own_regime(self) -> None:
        """With UNKNOWN included, both ends of the sample become an unknown regime."""
        returns, macro, labels, fresh_end = self._late_and_stale()
        config = RegimeValidationConfig(
            max_macro_staleness_days=_STALE_LIMIT_DAYS, include_unknown_regime=True
        )
        expected_labels = (
            ["unknown"] * _FIRST_MACRO_DAY
            + labels[_FIRST_MACRO_DAY:fresh_end]
            + ["unknown"] * (len(returns) - fresh_end)
        )

        result = run_regime_validation(returns, macro, config)

        _assert_matches_slices(result, returns, expected_labels, config, len(returns))

    def test_unbounded_staleness_carries_the_last_regime_forward(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No staleness limit carries the last macro regime to the end, silently."""
        returns, macro, labels, _ = self._late_and_stale()
        config = RegimeValidationConfig(max_macro_staleness_days=None)

        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            result = run_regime_validation(returns, macro, config)

        _assert_matches_slices(
            result,
            returns.iloc[_FIRST_MACRO_DAY:],
            labels[_FIRST_MACRO_DAY:],
            config,
            len(returns),
        )
        assert _records(caplog) == []


class TestEmptyResults:
    """The two early returns of a well-formed empty result."""

    def test_empty_macro_data_logs_and_returns_the_empty_result(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Empty macro data logs one warning and returns the empty result."""
        returns, _, _ = _scenario((("expansion", 30),), _EDGE_DRIFT)
        macro = pd.DataFrame({"pmi": []}, index=pd.DatetimeIndex([]))

        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            result = run_regime_validation(returns, macro)

        _assert_empty(result, 30)
        assert _records(caplog) == [
            (
                _LOGGER,
                logging.WARNING,
                "run_regime_validation: macro_data is empty; returning NaN metrics.",
            )
        ]

    def test_every_day_unknown_logs_and_returns_the_empty_result(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Macro data after the sample leaves every day UNKNOWN and dropped."""
        returns, macro, _ = _scenario((("expansion", 30),), _EDGE_DRIFT)
        later = macro.set_axis(macro.index + pd.Timedelta(days=400))

        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            result = run_regime_validation(returns, later)

        _assert_empty(result, 30)
        assert _records(caplog) == [
            (
                _LOGGER,
                logging.WARNING,
                "run_regime_validation: all observations classified as UNKNOWN "
                "and excluded; returning empty metrics.",
            )
        ]

    def test_stale_days_that_empty_the_sample_log_both_warnings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Stale macro data that empties the sample logs both warnings in order."""
        returns, macro, _ = _scenario((("expansion", 30),), _EDGE_DRIFT)
        month_before = returns.index[0] - pd.Timedelta(days=30)
        earlier = macro.iloc[:1].set_axis(pd.DatetimeIndex([month_before]))
        config = RegimeValidationConfig(max_macro_staleness_days=5)

        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            result = run_regime_validation(returns, earlier, config)

        _assert_empty(result, 30)
        assert [message for _, _, message in _records(caplog)] == [
            "run_regime_validation: 30 OOS observations are more than 5 days "
            "past the latest macro observation (stale macro data); classified "
            "as UNKNOWN.",
            "run_regime_validation: all observations classified as UNKNOWN "
            "and excluded; returning empty metrics.",
        ]


class TestRejectedInputs:
    """Inputs that raise, and the order of the input checks."""

    def test_empty_returns_are_rejected_before_the_macro_data_is_read(self) -> None:
        """Empty returns raise before the macro frame is looked at."""
        with pytest.raises(DataError, match=r"^oos_returns is empty$"):
            run_regime_validation(pd.Series(dtype=float), pd.DataFrame())

    def test_timezone_mismatch_is_rejected_with_the_exact_message(self) -> None:
        """Naive returns against aware macro data raise with the exact message."""
        returns, macro, _ = _scenario((("expansion", 30),), _EDGE_DRIFT)

        with pytest.raises(DataError) as raised:
            run_regime_validation(returns, macro.tz_localize("UTC"))

        assert str(raised.value) == (
            "macro_data and oos_returns must both have timezone-aware or both "
            "timezone-naive indexes (timezone mismatch)"
        )

    def test_empty_macro_data_wins_over_a_timezone_mismatch(self) -> None:
        """The empty-macro check runs before the timezone check."""
        returns, _, _ = _scenario((("expansion", 30),), _EDGE_DRIFT)
        macro = pd.DataFrame({"pmi": []}, index=pd.DatetimeIndex([], tz="UTC"))

        result = run_regime_validation(returns, macro)

        _assert_empty(result, 30)

    def test_duplicate_return_dates_fail_once_unknown_days_are_dropped(self) -> None:
        """Duplicate return dates cannot be realigned after UNKNOWN days drop."""
        returns, macro, _ = _scenario((("expansion", 30),), _EDGE_DRIFT)
        doubled = pd.concat([returns.iloc[:1], returns])

        with pytest.raises(ValueError, match="duplicate labels"):
            run_regime_validation(doubled, macro.iloc[10:])

    def test_duplicate_return_dates_fail_even_when_every_day_is_unknown(
        self,
    ) -> None:
        """The returns are realigned before the all-UNKNOWN check runs."""
        returns, macro, _ = _scenario((("expansion", 30),), _EDGE_DRIFT)
        doubled = pd.concat([returns.iloc[:1], returns])
        later = macro.set_axis(macro.index + pd.Timedelta(days=400))

        with pytest.raises(ValueError, match="duplicate labels"):
            run_regime_validation(doubled, later)


class TestNonFiniteMinimum:
    """``min_regime_obs`` values that ``__post_init__`` lets through, or not."""

    @staticmethod
    def _run(min_regime_obs: float) -> RegimeValidationResult:
        returns, macro, _ = _scenario(_MIXED_LAYOUT, _EDGE_DRIFT)
        config = RegimeValidationConfig(min_regime_obs=min_regime_obs)  # type: ignore[arg-type]
        return run_regime_validation(returns, macro, config)

    @pytest.mark.parametrize("minimum", [math.nan, math.inf], ids=["nan", "inf"])
    def test_a_minimum_no_count_reaches_leaves_every_row_unmeasured(
        self, minimum: float
    ) -> None:
        """NaN and +inf act as a minimum above the sample: no row is measured."""
        result = self._run(minimum)
        above_the_sample = self._run(sum(length for _, length in _MIXED_LAYOUT) + 1)

        regimes = result.per_regime_metrics
        assert regimes[_METRIC_COLUMNS].isna().all(axis=None)
        assert result.per_subperiod_metrics[_METRIC_COLUMNS].isna().all(axis=None)
        assert list(regimes["obs_sufficient"]) == [False, False, False, False]
        assert list(regimes["obs"]) == [101, 30, 10, 9]
        assert result.regime_alpha_concentration.to_dict() == {
            "expansion": 0.0,
            "recession": 0.0,
            "recovery": 0.0,
            "slowdown": 0.0,
        }
        assert result.concentrated_regimes == []
        pd.testing.assert_frame_equal(
            regimes, above_the_sample.per_regime_metrics, check_exact=True
        )
        pd.testing.assert_frame_equal(
            result.per_subperiod_metrics,
            above_the_sample.per_subperiod_metrics,
            check_exact=True,
        )

    def test_a_negative_infinite_minimum_is_rejected(self) -> None:
        """-inf fails the ``>= 1`` check, so it never reaches the metrics."""
        with pytest.raises(
            ValueError, match=r"^min_regime_obs must be >= 1, got -inf$"
        ):
            RegimeValidationConfig(min_regime_obs=-math.inf)  # type: ignore[arg-type]
