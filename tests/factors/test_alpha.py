"""Tests for factor_scores_to_expected_returns (factor alpha bridge)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import factor_scores_to_expected_returns

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
GROUPS = ["value", "momentum", "quality"]


@pytest.fixture()
def scores() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.standard_normal((len(TICKERS), len(GROUPS))),
        index=TICKERS,
        columns=GROUPS,
    )


@pytest.fixture()
def betas() -> pd.Series:
    return pd.Series(
        [1.2, 0.9, 1.1, 1.3, 1.0],
        index=TICKERS,
    )


@pytest.fixture()
def factor_premiums() -> dict[str, float]:
    return {"market": 0.05, "value": 0.03, "momentum": 0.04, "quality": 0.02}


class TestReturnType:
    def test_returns_series(
        self,
        scores: pd.DataFrame,
        betas: pd.Series,
        factor_premiums: dict[str, float],
    ) -> None:
        result = factor_scores_to_expected_returns(scores, betas, factor_premiums)
        assert isinstance(result, pd.Series)

    def test_index_matches_scores(
        self,
        scores: pd.DataFrame,
        betas: pd.Series,
        factor_premiums: dict[str, float],
    ) -> None:
        result = factor_scores_to_expected_returns(scores, betas, factor_premiums)
        assert list(result.index) == list(scores.index)

    def test_dtype_is_float(
        self,
        scores: pd.DataFrame,
        betas: pd.Series,
        factor_premiums: dict[str, float],
    ) -> None:
        result = factor_scores_to_expected_returns(scores, betas, factor_premiums)
        assert result.dtype == float


class TestZeroScoresPureCAPM:
    """Zero factor scores must collapse to r_f + λ_mkt · β_i."""

    def test_zero_scores_equal_capm(self, betas: pd.Series) -> None:
        zero_scores = pd.DataFrame(0.0, index=TICKERS, columns=GROUPS)
        premiums = {"market": 0.05, "value": 0.03, "momentum": 0.04}
        rf = 0.02

        result = factor_scores_to_expected_returns(
            zero_scores, betas, premiums, risk_free_rate=rf
        )

        expected = rf + 0.05 * betas.reindex(TICKERS)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_zero_scores_no_market_premium_equals_rf(self, betas: pd.Series) -> None:
        zero_scores = pd.DataFrame(0.0, index=TICKERS, columns=GROUPS)
        premiums: dict[str, float] = {}
        rf = 0.03

        result = factor_scores_to_expected_returns(
            zero_scores, betas, premiums, risk_free_rate=rf
        )

        assert (result == rf).all()


class TestNegativeScoresBelowMarket:
    """Negative factor scores should drag expected returns below the CAPM baseline."""

    def test_negative_scores_reduce_expected_return(self, betas: pd.Series) -> None:
        # All factor scores deeply negative
        neg_scores = pd.DataFrame(-3.0, index=TICKERS, columns=GROUPS)
        pos_scores = pd.DataFrame(3.0, index=TICKERS, columns=GROUPS)
        premiums = {"market": 0.05, "value": 0.03, "momentum": 0.04, "quality": 0.02}
        rf = 0.02

        neg_result = factor_scores_to_expected_returns(
            neg_scores, betas, premiums, risk_free_rate=rf
        )
        pos_result = factor_scores_to_expected_returns(
            pos_scores, betas, premiums, risk_free_rate=rf
        )

        assert (neg_result < pos_result).all()

    def test_deeply_negative_scores_below_capm_baseline(self, betas: pd.Series) -> None:
        neg_scores = pd.DataFrame(-5.0, index=TICKERS, columns=GROUPS)
        premiums = {"market": 0.05, "value": 0.03, "momentum": 0.04, "quality": 0.02}
        rf = 0.02

        result = factor_scores_to_expected_returns(
            neg_scores, betas, premiums, risk_free_rate=rf
        )
        capm_baseline = rf + 0.05 * betas.reindex(TICKERS)

        assert (result < capm_baseline).all()


class TestMissingBetasHandledGracefully:
    """Assets absent from betas should default to β=1.0."""

    def test_missing_betas_default_to_one(self) -> None:
        scores = pd.DataFrame(0.0, index=TICKERS, columns=GROUPS)
        # Only supply betas for a subset of tickers
        partial_betas = pd.Series({"AAPL": 1.5, "MSFT": 0.8})
        premiums = {"market": 0.05}
        rf = 0.0

        result = factor_scores_to_expected_returns(
            scores, partial_betas, premiums, risk_free_rate=rf
        )

        # AAPL: 0 + 0.05 * 1.5 = 0.075
        assert pytest.approx(result["AAPL"], abs=1e-9) == 0.075
        # MSFT: 0 + 0.05 * 0.8 = 0.04
        assert pytest.approx(result["MSFT"], abs=1e-9) == 0.04
        # GOOG, AMZN, META: missing → beta=1.0 → 0.05 * 1.0 = 0.05
        for ticker in ["GOOG", "AMZN", "META"]:
            assert pytest.approx(result[ticker], abs=1e-9) == 0.05

    def test_empty_betas_all_default_to_one(
        self, scores: pd.DataFrame, factor_premiums: dict[str, float]
    ) -> None:
        empty_betas: pd.Series = pd.Series(dtype=float)
        result = factor_scores_to_expected_returns(scores, empty_betas, factor_premiums)
        assert isinstance(result, pd.Series)
        assert list(result.index) == TICKERS


class TestUnknownFactorsIgnored:
    """Keys in factor_premiums not present in scores columns are silently skipped."""

    def test_extra_premium_keys_skipped(
        self, scores: pd.DataFrame, betas: pd.Series
    ) -> None:
        premiums = {
            "market": 0.05,
            "value": 0.03,
            "unknown_factor": 99.0,  # not in scores.columns
        }
        # Should not raise
        result = factor_scores_to_expected_returns(scores, betas, premiums)
        assert isinstance(result, pd.Series)


class TestRiskFreeRateShiftsAllReturns:
    def test_rf_offset(
        self, scores: pd.DataFrame, betas: pd.Series, factor_premiums: dict[str, float]
    ) -> None:
        result_rf0 = factor_scores_to_expected_returns(
            scores, betas, factor_premiums, risk_free_rate=0.0
        )
        result_rf2 = factor_scores_to_expected_returns(
            scores, betas, factor_premiums, risk_free_rate=0.02
        )
        pd.testing.assert_series_equal(
            result_rf2 - result_rf0,
            pd.Series(0.02, index=scores.index),
            check_names=False,
        )


class TestLinearityInScores:
    """Doubling factor scores should double the factor tilt contribution."""

    def test_doubling_scores_doubles_factor_contribution(
        self, betas: pd.Series
    ) -> None:
        base_scores = pd.DataFrame(1.0, index=TICKERS, columns=GROUPS)
        double_scores = base_scores * 2
        premiums = {"value": 0.03, "momentum": 0.04, "quality": 0.02}
        rf = 0.0

        result_base = factor_scores_to_expected_returns(
            base_scores, betas, premiums, risk_free_rate=rf
        )
        result_double = factor_scores_to_expected_returns(
            double_scores, betas, premiums, risk_free_rate=rf
        )

        # factor contribution in base = total (no market premium, rf=0)
        # double should be 2x base
        pd.testing.assert_series_equal(
            result_double,
            result_base * 2,
            check_names=False,
        )
