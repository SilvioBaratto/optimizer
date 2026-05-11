"""Tests for research.pipeline._screen — Step 2 investability screening."""

from __future__ import annotations

import logging

import pandas as pd
import pytest


def _make_passing(n: int) -> pd.Index:
    return pd.Index([f"TICKER_{i}" for i in range(n)])


class _StubAssembly:
    fundamentals = pd.DataFrame()
    prices = pd.DataFrame()
    volumes = pd.DataFrame()
    financial_statements = pd.DataFrame()


class TestAssertUniverseSize:
    def test_when_below_floor_then_raises_runtime_error(self) -> None:
        from research.pipeline._screen import _assert_universe_size

        with pytest.raises(RuntimeError, match="50"):
            _assert_universe_size(_make_passing(50))

    def test_when_in_band_then_no_error(self) -> None:
        from research.pipeline._screen import _assert_universe_size

        _assert_universe_size(_make_passing(800))

    def test_when_above_band_then_warns(self, caplog) -> None:
        from research.pipeline._screen import _assert_universe_size

        with caplog.at_level(logging.WARNING):
            _assert_universe_size(_make_passing(2500))
        assert any(
            "2500" in r.getMessage() and r.levelno == logging.WARNING
            for r in caplog.records
        )


class TestScreenInvestable:
    def test_when_screen_universe_returns_too_few_then_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research.pipeline._screen import screen_investable

        stub = _StubAssembly()
        monkeypatch.setattr(
            "research.pipeline._screen.screen_universe",
            lambda **_: _make_passing(50),
        )
        with pytest.raises(RuntimeError, match="50"):
            screen_investable(stub)  # type: ignore[arg-type]

    def test_when_screen_universe_returns_band_size_then_returns_index(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research.pipeline._screen import screen_investable

        expected = _make_passing(800)
        monkeypatch.setattr(
            "research.pipeline._screen.screen_universe",
            lambda **_: expected,
        )
        result = screen_investable(_StubAssembly())  # type: ignore[arg-type]
        assert list(result) == list(expected)

    def test_when_screen_universe_returns_above_band_then_warns_and_returns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from research.pipeline._screen import screen_investable

        expected = _make_passing(2500)
        monkeypatch.setattr(
            "research.pipeline._screen.screen_universe",
            lambda **_: expected,
        )
        with caplog.at_level(logging.WARNING):
            result = screen_investable(_StubAssembly())  # type: ignore[arg-type]
        assert list(result) == list(expected)
