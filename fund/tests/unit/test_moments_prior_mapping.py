"""The profile's ``moments_estimator`` reaches the optimizer's prior.

``estimate_moments`` showed the PM one covariance while ``optimize_portfolio`` fitted
skfolio's default empirical prior, because ``ConstraintSet.moments_estimator`` was
never mapped into ``MeanRiskConfig.prior_config``. These tests pin that the
covariance the PM reads and the covariance the optimizer solves under come from one
mapping of the constraint set.
"""

from __future__ import annotations

import datetime as dt
import uuid

import numpy as np
import pandas as pd
import pytest
from langgraph.store.memory import InMemoryStore
from optimizer.moments import CovEstimatorType, MuEstimatorType
from optimizer.optimization import build_mean_risk
from skfolio import RiskMeasure
from skfolio.moments import EmpiricalCovariance, LedoitWolf
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.prior import EmpiricalPrior

from fund.agents.toolsets import RunContext, _optimizer_constraints, bind_toolset
from fund.audit import AgentRunRepository, put_constraint_set
from fund.config import settings
from fund.schemas import ConstraintSet
from fund.schemas.enums import (
    Horizon,
    MomentsEstimator,
    ObjectiveChoice,
    RiskMeasureChoice,
)
from fund.tools import moments as moments_mod
from fund.tools import optimize as optimize_mod

_PORTFOLIO_ID = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")
_ASOF = dt.date(2025, 6, 30)
_BARS = 120
_ASSETS = [f"T{i}" for i in range(10)]


def _cs(estimator: MomentsEstimator, **overrides: object) -> ConstraintSet:
    """A minimum-variance profile carrying ``estimator``."""
    kwargs: dict[str, object] = {
        "portfolio_id": str(_PORTFOLIO_ID),
        "base_currency": "EUR",
        "a_gamma": 5.0,
        "objective": ObjectiveChoice.PROTECTION,
        "risk_measure": RiskMeasureChoice.VARIANCE,
        "nu1": 0.1,
        "nu2": 0.2,
        "nu3": 0.3,
        "horizon": Horizon.LONG,
        "moments_estimator": estimator,
    }
    kwargs.update(overrides)
    return ConstraintSet(**kwargs)  # type: ignore[arg-type]


@pytest.fixture
def panel() -> pd.DataFrame:
    """Ten assets over a short window, where shrinkage visibly moves the covariance."""
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2025-01-02", periods=_BARS + 1)
    common = rng.normal(0.0, 0.008, (_BARS, 1))
    loadings = rng.uniform(0.4, 1.4, (1, len(_ASSETS)))
    noise = rng.normal(0.0, 0.01, (_BARS, len(_ASSETS))) * rng.uniform(
        0.5, 2.0, len(_ASSETS)
    )
    steps = np.vstack([np.zeros((1, len(_ASSETS))), common @ loadings + noise])
    return pd.DataFrame(
        100.0 * np.cumprod(1.0 + steps, axis=0), index=index, columns=_ASSETS
    )


def _serve(monkeypatch, frame: pd.DataFrame) -> None:
    """Serve ``frame`` from both tools' ``load_price_frame``."""

    def load(session, asof, universe, *, base_currency=None, **kwargs):
        """Hand back the panel the way ``load_price_frame`` serves the DB."""
        columns = [t for t in universe if t in frame.columns]
        return frame[columns], [t for t in universe if t not in frame.columns]

    monkeypatch.setattr(moments_mod, "load_price_frame", load)
    monkeypatch.setattr(optimize_mod, "load_price_frame", load)


class TestConstraintSetMapsItsEstimator:
    """``moments_estimator`` becomes the optimizer's moment configuration."""

    @pytest.mark.parametrize(
        ("estimator", "mu", "cov"),
        [
            (
                MomentsEstimator.LEDOIT_WOLF,
                MuEstimatorType.EMPIRICAL,
                CovEstimatorType.LEDOIT_WOLF,
            ),
            (
                MomentsEstimator.EMPIRICAL,
                MuEstimatorType.EMPIRICAL,
                CovEstimatorType.EMPIRICAL,
            ),
            (MomentsEstimator.EW, MuEstimatorType.EW, CovEstimatorType.EW),
        ],
    )
    def test_every_member_maps_to_its_estimators(self, estimator, mu, cov) -> None:
        """The map is total and each member selects its own mean and covariance."""
        config = _cs(estimator).to_moment_estimation_config()

        assert config.mu_estimator is mu
        assert config.cov_estimator is cov

    def test_the_mean_risk_config_carries_the_same_prior(self) -> None:
        """What the PM is shown and what the solver is built from cannot diverge."""
        cs = _cs(MomentsEstimator.EMPIRICAL)

        assert cs.to_mean_risk_config().prior_config == cs.to_moment_estimation_config()

    def test_the_optimizer_constraints_hand_the_prior_to_the_tool(self) -> None:
        """The dict the bound ``optimize_portfolio`` receives includes the prior."""
        cs = _cs(MomentsEstimator.EMPIRICAL)

        constraints = _optimizer_constraints(cs)

        assert constraints["prior_config"] == cs.to_moment_estimation_config()


class TestOptimizerPriorFollowsTheConstraintSet:
    """The solver's prior is the one the constraint set selects."""

    def _tool_weights(self, cs: ConstraintSet) -> dict[str, float]:
        """Weights ``optimize_portfolio`` returns under ``cs``'s mapped constraints."""
        result = optimize_mod.optimize_portfolio(
            None, _ASOF, _ASSETS, constraints=_optimizer_constraints(cs)
        )
        assert result["ok"] is True, result
        return result["data"]["weights"]

    @staticmethod
    def _reference_weights(frame: pd.DataFrame, covariance) -> dict[str, float]:
        """Minimum-variance weights from a plain skfolio fit with ``covariance``."""
        model = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.VARIANCE,
            min_weights=0.0,
            max_weights=1.0,
            budget=1.0,
            prior_estimator=EmpiricalPrior(covariance_estimator=covariance),
        )
        returns = frame.pct_change().dropna()
        model.fit(returns)
        assert model.weights_ is not None
        return dict(zip(returns.columns, model.weights_, strict=True))

    def test_ledoit_wolf_and_empirical_profiles_solve_different_problems(
        self, monkeypatch, panel
    ) -> None:
        """The two estimators disagree on this panel, so the mapping is observable."""
        _serve(monkeypatch, panel)
        shrunk = self._tool_weights(_cs(MomentsEstimator.LEDOIT_WOLF))
        sample = self._tool_weights(_cs(MomentsEstimator.EMPIRICAL))

        gap = max(abs(shrunk[a] - sample[a]) for a in _ASSETS)
        assert gap > 1e-3

    @pytest.mark.parametrize(
        ("estimator", "covariance"),
        [
            (MomentsEstimator.LEDOIT_WOLF, LedoitWolf()),
            (MomentsEstimator.EMPIRICAL, EmpiricalCovariance()),
        ],
        ids=["ledoit_wolf", "empirical"],
    )
    def test_the_tool_matches_a_plain_fit_under_the_selected_covariance(
        self, monkeypatch, panel, estimator, covariance
    ) -> None:
        """The weights are those of skfolio under the profile's own covariance."""
        _serve(monkeypatch, panel)

        weights = self._tool_weights(_cs(estimator))

        expected = self._reference_weights(panel, covariance)
        for asset in _ASSETS:
            assert weights[asset] == pytest.approx(expected[asset], abs=1e-5)

    def test_the_covariance_the_pm_reads_is_the_one_the_prior_fits(
        self, monkeypatch, panel
    ) -> None:
        """``estimate_moments`` under the profile shows the prior's covariance."""
        _serve(monkeypatch, panel)
        cs = _cs(MomentsEstimator.EMPIRICAL)

        shown = moments_mod.estimate_moments(
            None, _ASOF, _ASSETS, config=cs.to_moment_estimation_config()
        )
        model = build_mean_risk(cs.to_mean_risk_config())
        model.fit(panel.pct_change().dropna())

        assert shown["data"]["cov_estimator"] == "empirical"
        fitted = model.prior_estimator_.return_distribution_.covariance
        np.testing.assert_allclose(np.array(shown["data"]["cov"]), fitted, atol=1e-12)


class TestBoundMomentsToolUsesTheProfile:
    """The agent's ``estimate_moments`` is shown the profile's estimator."""

    def _context(self, db_session, cs: ConstraintSet) -> RunContext:
        """A run context whose store holds ``cs`` as the active constraint set."""
        store = InMemoryStore()
        put_constraint_set(store, cs, store_key=settings.constraint_set_store_key)
        run = AgentRunRepository(db_session).create_run(
            portfolio_id=_PORTFOLIO_ID,
            asof=_ASOF,
            seed=0,
            universe=_ASSETS,
            optimizer_config={"step": "toolset"},
        )
        return RunContext(
            session=db_session,
            asof=_ASOF,
            store=store,
            config=settings,
            run_id=run.id,
            portfolio_id=_PORTFOLIO_ID,
        )

    @pytest.mark.parametrize(
        ("estimator", "expected"),
        [
            (MomentsEstimator.EMPIRICAL, "empirical"),
            (MomentsEstimator.LEDOIT_WOLF, "ledoit_wolf"),
        ],
    )
    def test_the_estimator_shown_is_the_profiles(
        self, db_session, monkeypatch, panel, estimator, expected
    ) -> None:
        """The reply names the covariance estimator the profile selected."""
        _serve(monkeypatch, panel)
        ctx = self._context(db_session, _cs(estimator))
        (tool,) = [
            t for t in bind_toolset("allocator", ctx) if t.name == "estimate_moments"
        ]

        result = tool.invoke({"universe": _ASSETS})

        assert result["ok"] is True, result
        assert result["data"]["cov_estimator"] == expected


class TestAuditLogsWhatTheSolverRanWith:
    """The allocator decision records the prior and the compounded risk-free rate."""

    def test_the_decision_carries_the_prior_and_the_per_bar_rate(
        self, db_session, monkeypatch, panel
    ) -> None:
        """The audited config is the applied one, not the pre-tool mapping."""
        import json

        from portopt_db.models import AgentDecision
        from sqlalchemy import select

        _serve(monkeypatch, panel)
        cs = _cs(MomentsEstimator.EMPIRICAL, risk_free_rate=0.04)
        helper = TestBoundMomentsToolUsesTheProfile()
        ctx = helper._context(db_session, cs)
        (tool,) = [
            t for t in bind_toolset("allocator", ctx) if t.name == "optimize_portfolio"
        ]

        result = tool.invoke({"universe": _ASSETS})

        assert result["ok"] is True, result
        (decision,) = (
            db_session.execute(
                select(AgentDecision).where(AgentDecision.agent == "allocator")
            )
            .scalars()
            .all()
        )
        logged = json.loads(decision.llm_response)["optimizer_config"]
        assert logged["prior_config"]["cov_estimator"] == "empirical"
        assert logged["risk_free_rate"] == pytest.approx(
            1.04 ** (1 / result["data"]["periods_per_year"]) - 1
        )
