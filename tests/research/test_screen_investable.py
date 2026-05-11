"""Tests for screen_investable() floor and band enforcement (issue #520)."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

pytest.importorskip("typer")

from research import stock_selection_pipeline as ssp
from research.pipeline import _screen as _screen_module
from research.stock_selection_pipeline import _assert_universe_size


def _make_passing(n: int) -> pd.Index:
    return pd.Index([f"T{i:05d}" for i in range(n)])


class TestAssertUniverseSize:
    def test_when_count_below_200_then_runtime_error_raised(self) -> None:
        passing = _make_passing(100)
        with pytest.raises(RuntimeError) as excinfo:
            _assert_universe_size(passing)
        msg = str(excinfo.value)
        assert "100" in msg
        assert "InvestabilityScreenConfig" in msg or "config" in msg.lower()

    def test_when_count_in_band_then_no_warning_no_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        passing = _make_passing(800)
        with caplog.at_level(logging.WARNING):
            _assert_universe_size(passing)
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_when_count_above_band_then_warning_logged_and_no_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        passing = _make_passing(2000)
        with caplog.at_level(logging.WARNING):
            _assert_universe_size(passing)
        warnings_ = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings_) == 1
        assert "2000" in warnings_[0].getMessage()
        assert "300" in warnings_[0].getMessage()
        assert "1500" in warnings_[0].getMessage()

    def test_when_count_below_band_above_floor_then_warning_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        passing = _make_passing(250)
        with caplog.at_level(logging.WARNING):
            _assert_universe_size(passing)
        warnings_ = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings_) == 1
        assert "250" in warnings_[0].getMessage()

    def test_when_count_at_band_lower_edge_then_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        passing = _make_passing(300)
        with caplog.at_level(logging.WARNING):
            _assert_universe_size(passing)
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_when_count_at_band_upper_edge_then_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        passing = _make_passing(1500)
        with caplog.at_level(logging.WARNING):
            _assert_universe_size(passing)
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_when_count_at_floor_edge_then_no_raise_warning_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        passing = _make_passing(200)
        with caplog.at_level(logging.WARNING):
            _assert_universe_size(passing)
        warnings_ = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings_) == 1


class TestScreenInvestableIntegration:
    """Verify screen_investable() applies the floor/band after screen_universe."""

    @pytest.fixture
    def stub_assembly(self) -> object:
        class _Stub:
            fundamentals = pd.DataFrame()
            prices = pd.DataFrame()
            volumes = pd.DataFrame()
            financial_statements = pd.DataFrame()

        return _Stub()

    def test_when_screen_universe_returns_too_few_then_raises(
        self,
        stub_assembly: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            _screen_module, "screen_universe", lambda **_: _make_passing(50)
        )
        with pytest.raises(RuntimeError, match="50"):
            ssp.screen_investable(stub_assembly)  # type: ignore[arg-type]

    def test_when_screen_universe_returns_band_size_then_returns_index(
        self,
        stub_assembly: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected = _make_passing(800)
        monkeypatch.setattr(_screen_module, "screen_universe", lambda **_: expected)
        result = ssp.screen_investable(stub_assembly)  # type: ignore[arg-type]
        assert list(result) == list(expected)

    def test_when_screen_universe_returns_above_band_then_warning_and_index(
        self,
        stub_assembly: object,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        expected = _make_passing(2500)
        monkeypatch.setattr(_screen_module, "screen_universe", lambda **_: expected)
        with caplog.at_level(logging.WARNING):
            result = ssp.screen_investable(stub_assembly)  # type: ignore[arg-type]
        assert list(result) == list(expected)
        assert any(
            "2500" in r.getMessage() and r.levelno == logging.WARNING
            for r in caplog.records
        )
