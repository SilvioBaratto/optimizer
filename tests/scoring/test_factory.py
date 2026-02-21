"""Tests for scoring factory functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError
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

    def test_all_ratio_measures_except_ir(self) -> None:
        """All RatioMeasureType members except IR build without benchmark_returns."""
        for measure in RatioMeasureType:
            if measure == RatioMeasureType.INFORMATION_RATIO:
                continue
            cfg = ScorerConfig(ratio_measure=measure)
            scorer = build_scorer(cfg)
            assert callable(scorer)

    def test_custom_scorer(self) -> None:
        cfg = ScorerConfig.for_custom()

        def my_score(pred: object) -> float:
            return float(getattr(pred, "mean", 0.0))

        scorer = build_scorer(cfg, score_func=my_score)
        assert callable(scorer)

    def test_custom_scorer_greater_is_better_false(self) -> None:
        def neg_return(pred: object) -> float:
            return -float(getattr(pred, "mean", 0.0))

        cfg = ScorerConfig(ratio_measure=None, greater_is_better=False)
        scorer = build_scorer(cfg, score_func=neg_return)
        assert callable(scorer)

    def test_custom_scorer_works_in_grid_search_context(self) -> None:
        from skfolio.model_selection import WalkForward

        from optimizer.optimization import MeanRiskConfig, build_mean_risk

        # Synthetic 5-asset data
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            rng.normal(0.001, 0.02, (200, 5)),
            columns=[f"A{i}" for i in range(5)],
            index=pd.bdate_range("2023-01-01", periods=200, freq="B"),
        )

        def portfolio_variance(pred: object) -> float:
            return float(np.var(getattr(pred, "returns", [0.0])))

        cfg = ScorerConfig(ratio_measure=None, greater_is_better=False)
        scorer = build_scorer(cfg, score_func=portfolio_variance)

        from sklearn.model_selection import GridSearchCV

        model = build_mean_risk(MeanRiskConfig.for_min_variance())
        cv = WalkForward(test_size=50, train_size=100)
        gs = GridSearchCV(
            estimator=model,
            cv=cv,
            param_grid={"l2_coef": [0.0, 0.01]},
            scoring=scorer,
        )
        gs.fit(returns)
        assert gs.best_params_ is not None

    def test_custom_scorer_without_func_raises(self) -> None:
        cfg = ScorerConfig.for_custom()
        with pytest.raises(ConfigurationError, match="score_func is required"):
            build_scorer(cfg)
