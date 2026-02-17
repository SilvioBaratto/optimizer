"""Tests for scoring configs."""

from __future__ import annotations

import pytest

from optimizer.optimization import RatioMeasureType
from optimizer.scoring import ScorerConfig


class TestScorerConfig:
    def test_defaults(self) -> None:
        cfg = ScorerConfig()
        assert cfg.ratio_measure == RatioMeasureType.SHARPE_RATIO
        assert cfg.greater_is_better is None

    def test_frozen(self) -> None:
        cfg = ScorerConfig()
        with pytest.raises(AttributeError):
            cfg.ratio_measure = RatioMeasureType.SORTINO_RATIO  # type: ignore[misc]

    def test_for_sharpe(self) -> None:
        cfg = ScorerConfig.for_sharpe()
        assert cfg.ratio_measure == RatioMeasureType.SHARPE_RATIO

    def test_for_sortino(self) -> None:
        cfg = ScorerConfig.for_sortino()
        assert cfg.ratio_measure == RatioMeasureType.SORTINO_RATIO

    def test_for_calmar(self) -> None:
        cfg = ScorerConfig.for_calmar()
        assert cfg.ratio_measure == RatioMeasureType.CALMAR_RATIO

    def test_for_cvar_ratio(self) -> None:
        cfg = ScorerConfig.for_cvar_ratio()
        assert cfg.ratio_measure == RatioMeasureType.CVAR_RATIO

    def test_for_custom(self) -> None:
        cfg = ScorerConfig.for_custom()
        assert cfg.ratio_measure is None
