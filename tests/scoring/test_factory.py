"""Tests for scoring factory functions."""

from __future__ import annotations

import pytest

from optimizer.optimization import RatioMeasureType
from optimizer.scoring import ScorerConfig, build_scorer


class TestBuildScorer:
    def test_default_sharpe(self) -> None:
        scorer = build_scorer()
        assert callable(scorer)

    def test_sortino_scorer(self) -> None:
        cfg = ScorerConfig.for_sortino()
        scorer = build_scorer(cfg)
        assert callable(scorer)

    def test_all_ratio_measures(self) -> None:
        for measure in RatioMeasureType:
            cfg = ScorerConfig(ratio_measure=measure)
            scorer = build_scorer(cfg)
            assert callable(scorer)

    def test_custom_scorer(self) -> None:
        cfg = ScorerConfig.for_custom()

        def my_score(pred: object) -> float:
            return float(getattr(pred, "mean", 0.0))

        scorer = build_scorer(cfg, score_func=my_score)
        assert callable(scorer)

    def test_custom_scorer_without_func_raises(self) -> None:
        cfg = ScorerConfig.for_custom()
        with pytest.raises(ValueError, match="score_func is required"):
            build_scorer(cfg)
