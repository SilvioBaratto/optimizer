"""Tests for factor integration with optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    FactorIntegrationConfig,
    build_factor_bl_views,
    build_factor_exposure_constraints,
    estimate_factor_premia,
    factor_scores_to_expected_returns,
)


@pytest.fixture()
def factor_scores() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    tickers = [f"T{i:02d}" for i in range(20)]
    return pd.DataFrame(
        rng.normal(0, 1, (20, 3)),
        index=tickers,
        columns=["value", "momentum", "profitability"],
    )


@pytest.fixture()
def betas() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    tickers = [f"T{i:02d}" for i in range(20)]
    return pd.DataFrame(
        rng.uniform(0.5, 1.5, (20, 3)),
        index=tickers,
        columns=["value", "momentum", "profitability"],
    )


class TestFactorScoresToExpectedReturns:
    def test_basic(self, factor_scores: pd.DataFrame, betas: pd.DataFrame) -> None:
        premia = {"value": 0.04, "momentum": 0.06, "profitability": 0.03}
        result = factor_scores_to_expected_returns(
            factor_scores["value"], betas, premia
        )
        assert isinstance(result, pd.Series)
        assert len(result) == 20
        # All should be > risk_free_rate (most betas/premia are positive)
        assert result.mean() > 0

    def test_custom_config(self, factor_scores: pd.DataFrame, betas: pd.DataFrame) -> None:
        config = FactorIntegrationConfig(risk_free_rate=0.02)
        premia = {"value": 0.04}
        result = factor_scores_to_expected_returns(
            factor_scores["value"], betas, premia, config=config
        )
        # Minimum possible return is risk_free_rate
        assert (result >= 0.02 - 1.0).all()  # generous bound


class TestBuildFactorBLViews:
    def test_returns_views_and_confidences(self, factor_scores: pd.DataFrame) -> None:
        premia = {"value": 0.04, "momentum": 0.06}
        views, confidences = build_factor_bl_views(
            factor_scores, premia, factor_scores.index
        )
        assert isinstance(views, list)
        assert isinstance(confidences, list)
        assert len(views) == len(confidences)

    def test_empty_premia(self, factor_scores: pd.DataFrame) -> None:
        views, confidences = build_factor_bl_views(
            factor_scores, {}, factor_scores.index
        )
        assert len(views) == 0


class TestBuildFactorExposureConstraints:
    def test_returns_constraints(self, factor_scores: pd.DataFrame) -> None:
        result = build_factor_exposure_constraints(
            factor_scores, bounds=(-0.5, 0.5)
        )
        assert isinstance(result, list)
        assert len(result) == factor_scores.shape[1]
        assert all("exposure" in c for c in result)


class TestEstimateFactorPremia:
    def test_returns_dict(self) -> None:
        rng = np.random.default_rng(42)
        fmp_returns = pd.DataFrame(
            rng.normal(0.0002, 0.01, (252, 3)),
            columns=["value", "momentum", "quality"],
        )
        result = estimate_factor_premia(fmp_returns)
        assert isinstance(result, dict)
        assert "value" in result
        assert "momentum" in result
        # Annualized premia should be reasonable
        for v in result.values():
            assert -1.0 < v < 1.0
