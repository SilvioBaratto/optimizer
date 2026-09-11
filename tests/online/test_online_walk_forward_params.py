"""Walk-forward calendar params flow through online_predict / online_score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.measures import RatioMeasure
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization import MeanRisk
from skfolio.prior import EmpiricalPrior

from optimizer.online import (
    OnlinePredictConfig,
    run_online_predict,
    run_online_score,
)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    return pd.DataFrame(
        rng.normal(0.0005, 0.012, size=(400, 4)),
        columns=list("WXYZ"),
        index=idx,
    )


def _estimator() -> MeanRisk:
    return MeanRisk(
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(half_life=40.0),
            covariance_estimator=EWCovariance(half_life=40.0),
        )
    )


def test_when_test_size_set_then_step_count_reflects_it(
    returns: pd.DataFrame,
) -> None:
    cfg = OnlinePredictConfig(warmup_size=252, test_size=21)
    result = run_online_predict(_estimator(), returns, None, config=cfg)
    # 400 - 252 = 148 out-of-sample obs; with reduce_test=False the trailing
    # partial 21-obs window is dropped -> floor(148 / 21) * 21 = 147.
    remaining = len(returns) - cfg.warmup_size
    assert len(result.returns) == (remaining // cfg.test_size) * cfg.test_size


def test_when_purged_size_set_then_predict_runs(returns: pd.DataFrame) -> None:
    cfg = OnlinePredictConfig(warmup_size=252, test_size=21, purged_size=5)
    result = run_online_predict(_estimator(), returns, None, config=cfg)
    assert hasattr(result, "returns")
    assert len(result.returns) > 0


def test_when_per_step_score_on_optimizer_then_raises(
    returns: pd.DataFrame,
) -> None:
    # skfolio 1.0.6 rejects per_step aggregation for portfolio optimizers:
    # the full MultiPeriodPortfolio is scored, not per-rebalance slices.
    cfg = OnlinePredictConfig(warmup_size=252, test_size=21)
    with pytest.raises(ValueError, match="per_step"):
        run_online_score(
            _estimator(),
            returns,
            None,
            scorer=RatioMeasure.SHARPE_RATIO,
            config=cfg,
            per_step=True,
        )


def test_when_aggregate_score_then_returns_scalar(returns: pd.DataFrame) -> None:
    cfg = OnlinePredictConfig(warmup_size=252, test_size=21)
    score = run_online_score(
        _estimator(),
        returns,
        None,
        scorer=RatioMeasure.SHARPE_RATIO,
        config=cfg,
    )
    assert isinstance(score, float)


def test_when_calendar_freq_then_predict_returns_portfolio(
    returns: pd.DataFrame,
) -> None:
    cfg = OnlinePredictConfig(warmup_size=6, test_size=1, freq="MS")
    result = run_online_predict(_estimator(), returns, None, config=cfg)
    assert hasattr(result, "returns")
    assert len(result.returns) > 0


def test_when_reduce_test_then_predict_runs(returns: pd.DataFrame) -> None:
    cfg = OnlinePredictConfig(warmup_size=252, test_size=30, reduce_test=True)
    result = run_online_predict(_estimator(), returns, None, config=cfg)
    # reduce_test keeps the trailing partial window instead of dropping it.
    assert len(result.returns) == len(returns) - cfg.warmup_size
