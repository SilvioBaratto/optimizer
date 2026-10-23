"""Tests for the build_portfolio_pipeline factory (pre-selection → optimiser)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import (
    EqualWeighted,
    HierarchicalRiskParity,
    InverseVolatility,
    MeanRisk,
)
from sklearn.pipeline import Pipeline

from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pre_selection import PreSelectionConfig, build_portfolio_pipeline


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Synthetic return DataFrame with 15 assets and 300 observations."""
    rng = np.random.default_rng(42)
    n_obs, n_assets = 300, 15
    data = rng.normal(loc=0.001, scale=0.02, size=(n_obs, n_assets))
    tickers = [f"TICK_{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(
        data,
        columns=tickers,
        index=pd.date_range("2022-01-01", periods=n_obs, freq="B"),
    )


class TestBuildPortfolioPipeline:
    def test_returns_pipeline(self) -> None:
        pipe = build_portfolio_pipeline(EqualWeighted())
        assert isinstance(pipe, Pipeline)

    def test_has_optimizer_step(self) -> None:
        pipe = build_portfolio_pipeline(MeanRisk())
        step_names = [name for name, _ in pipe.steps]
        assert "optimizer" in step_names

    def test_has_preselection_steps(self) -> None:
        pipe = build_portfolio_pipeline(EqualWeighted())
        step_names = [name for name, _ in pipe.steps]
        assert "validate" in step_names
        assert "select_complete" in step_names

    def test_custom_preselection(self) -> None:
        cfg = PreSelectionConfig(correlation_threshold=0.80, top_k=10)
        pipe = build_portfolio_pipeline(EqualWeighted(), pre_selection_config=cfg)
        step_names = [name for name, _ in pipe.steps]
        assert "select_k" in step_names

    def test_get_params_exposes_nested(self) -> None:
        optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
        pipe = build_portfolio_pipeline(optimizer)
        params = pipe.get_params()
        assert "optimizer__risk_aversion" in params
        assert "validate__max_abs_return" in params
        assert "drop_correlated__threshold" in params

    def test_fit_predict(self, returns_df: pd.DataFrame) -> None:
        optimizer = build_mean_risk(MeanRiskConfig.for_min_variance())
        pipe = build_portfolio_pipeline(optimizer)
        pipe.fit(returns_df)
        portfolio = pipe.predict(returns_df)
        assert hasattr(portfolio, "weights")
        assert hasattr(portfolio, "sharpe_ratio")
        assert len(portfolio.weights) > 0

    def test_with_hrp(self, returns_df: pd.DataFrame) -> None:
        pipe = build_portfolio_pipeline(HierarchicalRiskParity())
        pipe.fit(returns_df)
        portfolio = pipe.predict(returns_df)
        assert portfolio.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_sector_mapping(self, returns_df: pd.DataFrame) -> None:
        mapping = dict.fromkeys(returns_df.columns, "Tech")
        pipe = build_portfolio_pipeline(
            EqualWeighted(),
            sector_mapping=mapping,
        )
        pipe.fit(returns_df)
        portfolio = pipe.predict(returns_df)
        assert len(portfolio.weights) > 0

    def test_expiration_dates_forwarded(self, returns_df: pd.DataFrame) -> None:
        """expiration_dates reach the pre-selection SelectNonExpiring step."""
        import datetime as dt

        exp = {t: dt.datetime(2025, 1, 1) for t in returns_df.columns}
        cfg = PreSelectionConfig(use_non_expiring=True, expiration_lookahead=30)
        pipe = build_portfolio_pipeline(
            EqualWeighted(),
            pre_selection_config=cfg,
            expiration_dates=exp,
        )
        steps = dict(pipe.steps)
        assert "select_non_expiring" in steps
        assert steps["select_non_expiring"].expiration_dates == exp

    def test_expiration_dates_noop_by_default(self, returns_df: pd.DataFrame) -> None:
        """Without use_non_expiring, no SelectNonExpiring step is added."""
        import datetime as dt

        exp = {t: dt.datetime(2025, 1, 1) for t in returns_df.columns}
        pipe = build_portfolio_pipeline(EqualWeighted(), expiration_dates=exp)
        step_names = [name for name, _ in pipe.steps]
        assert "select_non_expiring" not in step_names


class TestPredictColumnOrder:
    def test_predict_on_reordered_columns_matches_original_order(self) -> None:
        # Daily vols differ per asset so a positional relabel of a reordered
        # frame feeds one asset's series to another asset's fitted weight.
        rng = np.random.default_rng(0)
        vols = {"A": 0.005, "B": 0.01, "C": 0.02, "D": 0.03}
        X = pd.DataFrame(
            {k: rng.normal(0.0005, v, 300) for k, v in vols.items()},
            index=pd.date_range("2022-01-01", periods=300, freq="B"),
        )
        pipe = build_portfolio_pipeline(InverseVolatility())
        pipe.fit(X)

        reference = pipe.predict(X)
        reordered = pipe.predict(X[["D", "C", "B", "A"]])

        np.testing.assert_allclose(reordered.returns, reference.returns)


class TestPredictScoresRealisedReturns:
    """Held-out P&L must come from realised returns, not from cleaned ones.

    Cleaning (winsorise / remove / impute) belongs to fitting.  Applying it to
    the test window would clip real crashes and fabricate peer returns, which
    understates tails and overstates Sharpe for any out-of-sample evaluation.
    """

    @staticmethod
    def _train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(11)
        n_train, n_test, n_assets = 400, 200, 5
        cols = [f"S{i}" for i in range(n_assets)]
        idx = pd.date_range("2022-01-03", periods=n_train + n_test, freq="B")
        data = rng.normal(0.0005, 0.01, size=(n_train + n_test, n_assets))
        frame = pd.DataFrame(data, columns=cols, index=idx)
        return frame.iloc[:n_train].copy(), frame.iloc[n_train:].copy()

    def test_predict_keeps_extreme_held_out_returns(self) -> None:
        train, test = self._train_test()
        test.iloc[50, 0] = -0.30
        test.iloc[100, 1] = -0.12

        pipe = build_portfolio_pipeline(EqualWeighted())
        pipe.fit(train)
        portfolio = pipe.predict(test)

        np.testing.assert_allclose(
            np.asarray(portfolio.returns), test.mean(axis=1).to_numpy(), atol=1e-12
        )

    def test_predict_does_not_fabricate_missing_held_out_returns(self) -> None:
        train, test = self._train_test()
        test.iloc[20, 2] = np.nan

        pipe = build_portfolio_pipeline(EqualWeighted())
        pipe.fit(train)
        portfolio = pipe.predict(test)

        # A missing observation contributes no P&L; it is not a peer average.
        expected = test.fillna(0.0).mean(axis=1).to_numpy()
        np.testing.assert_allclose(np.asarray(portfolio.returns), expected, atol=1e-12)

    def test_predict_applies_column_selection_learned_at_fit(self) -> None:
        train, test = self._train_test()
        train["DEAD"] = np.nan
        test["DEAD"] = 0.0

        pipe = build_portfolio_pipeline(EqualWeighted())
        pipe.fit(train)
        portfolio = pipe.predict(test)

        assert "DEAD" not in list(portfolio.assets)
        assert len(portfolio.weights) == 5

    def test_fit_still_cleans_training_returns(self) -> None:
        train, _ = self._train_test()
        train.iloc[10, 0] = -0.30

        pipe = build_portfolio_pipeline(EqualWeighted())
        cleaned = pipe[:-1].fit_transform(train)

        assert cleaned.iloc[10, 0] > -0.30
