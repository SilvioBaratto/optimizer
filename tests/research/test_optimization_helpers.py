"""Cycle-3 §8 hockey-stick warn helper tests (issue #538)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from research.optimization._rebalance import _hockey_stick_warn

_LOGGER_NAME = "research.optimization._rebalance"


def _series_from_segments(*segments: np.ndarray) -> pd.Series:
    arr = np.concatenate(segments)
    idx = pd.bdate_range("2020-01-01", periods=len(arr))
    return pd.Series(arr, index=idx)


class TestHockeyStickWarn:
    def test_when_stable_sharpe_then_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        rng = np.random.default_rng(0)
        # 3 sub-periods, all moderate positive returns
        seg = rng.normal(loc=0.001, scale=0.01, size=63)
        returns = _series_from_segments(seg, seg.copy(), seg.copy())
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            _hockey_stick_warn(returns)
        assert "hockey-stick" not in caplog.text.lower()

    def test_when_one_period_concentrates_then_warning_emitted(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        rng = np.random.default_rng(1)
        # Sub-period 1: negative Sharpe (loss-trending)
        loss = rng.normal(loc=-0.005, scale=0.01, size=63)
        # Sub-period 2: extreme positive Sharpe (>>1.5)
        gain = rng.normal(loc=0.020, scale=0.005, size=63)
        # Sub-period 3: moderate
        moderate = rng.normal(loc=0.0005, scale=0.01, size=63)
        returns = _series_from_segments(loss, gain, moderate)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            _hockey_stick_warn(returns)
        assert "hockey-stick" in caplog.text.lower()

    def test_when_warning_emitted_then_message_names_concentration_period(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        rng = np.random.default_rng(2)
        loss = rng.normal(loc=-0.005, scale=0.01, size=63)
        gain = rng.normal(loc=0.020, scale=0.005, size=63)
        moderate = rng.normal(loc=0.0005, scale=0.01, size=63)
        returns = _series_from_segments(loss, gain, moderate)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            _hockey_stick_warn(returns)
        # Sub-period 2 is the concentration period — message should reference it
        assert any(
            "2" in record.message or "period 2" in record.message.lower()
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_when_series_shorter_than_n_subperiods_then_no_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Only 2 days of returns; n_subperiods=3 → can't split
        returns = pd.Series([0.01, 0.02], index=pd.bdate_range("2020-01-01", periods=2))
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            _hockey_stick_warn(returns, n_subperiods=3)
        assert "hockey-stick" not in caplog.text.lower()

    def test_when_returns_none_or_empty_then_no_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            _hockey_stick_warn(None)
            _hockey_stick_warn(pd.Series(dtype=float))
        assert "hockey-stick" not in caplog.text.lower()

    def test_when_only_negative_period_no_extreme_positive_then_no_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        rng = np.random.default_rng(3)
        # All sub-periods slightly negative — min<0 but max not >1.5
        seg = rng.normal(loc=-0.001, scale=0.01, size=63)
        returns = _series_from_segments(seg, seg.copy(), seg.copy())
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            _hockey_stick_warn(returns)
        assert "hockey-stick" not in caplog.text.lower()

    def test_when_uses_logger_warning_not_print(
        self,
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rng = np.random.default_rng(4)
        loss = rng.normal(loc=-0.005, scale=0.01, size=63)
        gain = rng.normal(loc=0.020, scale=0.005, size=63)
        moderate = rng.normal(loc=0.0005, scale=0.01, size=63)
        returns = _series_from_segments(loss, gain, moderate)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            _hockey_stick_warn(returns)
        captured = capsys.readouterr()
        assert "hockey-stick" not in captured.out.lower()
        assert "hockey-stick" in caplog.text.lower()
