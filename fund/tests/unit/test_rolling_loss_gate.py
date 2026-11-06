"""The ``nu`` loss ceilings are judged on the worst loss inside a year.

``nu2`` is the client's stated ONE-YEAR loss tolerance, but the gate compared it with
the worst peak-to-trough decline over the whole history, which over-blocked any book
that bled slowly over several years. The gate now compares the ceilings with the
worst compounded decline inside any one-year window of the panel's own calendar, keeps
reporting the full-sample drawdown for information, fails closed when the history is
shorter than a year, and names a zero loss tolerance instead of reporting an opaque
"limit 0.0" breach.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import uuid

import numpy as np
import pandas as pd
import pytest
from langgraph.store.memory import InMemoryStore
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument
from skfolio.measures import get_drawdowns

from fund.agents.toolsets import RunContext, bind_toolset
from fund.audit import AgentRunRepository, put_constraint_set
from fund.config import settings
from fund.schemas import ConstraintSet
from fund.schemas.enums import Horizon, ObjectiveChoice, RiskMeasureChoice
from fund.tools import risk as risk_mod
from fund.tools.risk import risk_check, worst_window_drawdown

_PORTFOLIO_ID = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")
_ASOF = dt.date(2024, 6, 28)
_WEIGHTS = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
_PHASES = {"AAA": 0.0, "BBB": 1.3, "CCC": 2.6}
_YEAR = 365


def _full_drawdown(bars: np.ndarray) -> float:
    """skfolio's compounded max drawdown of ``bars``, as a positive fraction."""
    return float(-get_drawdowns(bars, compounded=True).min())


class TestWorstWindowDrawdown:
    """``worst_window_drawdown`` is the worst decline inside any window of bars."""

    def test_a_window_spanning_the_whole_history_is_the_full_drawdown(self) -> None:
        """With one window covering every bar it equals skfolio's max drawdown."""
        rng = np.random.default_rng(3)
        bars = rng.normal(0.0002, 0.01, 400)

        worst = worst_window_drawdown(bars, 400)

        assert worst == pytest.approx(_full_drawdown(bars), abs=1e-12)

    def test_a_slow_multi_year_bleed_is_judged_by_its_worst_year(self) -> None:
        """Three years at -15% a year is a 38% drawdown but a 15% one-year loss."""
        daily = (1.0 - 0.15) ** (1.0 / _YEAR) - 1.0
        bars = np.full(3 * _YEAR, daily)

        worst = worst_window_drawdown(bars, _YEAR)

        assert worst == pytest.approx(0.15, abs=1e-3)
        assert _full_drawdown(bars) > 0.38

    def test_a_trough_inside_the_year_counts_even_after_a_recovery(self) -> None:
        """A 25% fall that mostly recovers within the year still counts as 25%."""
        fall = np.full(100, (0.75) ** (1.0 / 100) - 1.0)
        rebound = np.full(165, (1.10) ** (1.0 / 165) - 1.0)
        bars = np.concatenate([fall, rebound])

        worst = worst_window_drawdown(bars, _YEAR - 100)

        assert worst == pytest.approx(0.25, abs=1e-9)

    def test_a_flat_path_has_no_drawdown(self) -> None:
        """No loss anywhere is exactly zero, never negative."""
        assert worst_window_drawdown(np.zeros(500), _YEAR) == 0.0

    def test_a_window_longer_than_the_history_cannot_be_measured(self) -> None:
        """Fewer bars than the window gives ``None``, not a partial figure."""
        assert worst_window_drawdown(np.full(200, -0.001), _YEAR) is None

    def test_a_non_finite_bar_cannot_be_measured(self) -> None:
        """NaN returns give ``None`` instead of a figure that compares False."""
        bars = np.full(400, 0.001)
        bars[10] = np.nan

        assert worst_window_drawdown(bars, 100) is None

    def test_a_total_wipe_out_is_a_full_loss(self) -> None:
        """A -100% bar inside the window is a 100% drawdown, with no NaN."""
        bars = np.zeros(300)
        bars[150] = -1.0

        assert worst_window_drawdown(bars, 200) == pytest.approx(1.0)

    def test_the_result_is_bounded_by_the_full_drawdown(self) -> None:
        """A shorter window can only see less of the decline than the whole path."""
        rng = np.random.default_rng(5)
        bars = rng.normal(0.0, 0.012, 1500)

        worst = worst_window_drawdown(bars, _YEAR)

        assert worst is not None
        assert 0.0 <= worst <= _full_drawdown(bars) + 1e-12


class TestZeroLossTolerance:
    """A client who tolerates no loss is named, not reported as a limit-0 breach."""

    def test_zero_nu2_with_a_drawdown_is_a_distinct_violation(self) -> None:
        """The record names the profile as the cause and replaces the tier breach."""
        result = risk_check(
            {"A": 1.0},
            constraints={"nu1": 0.0, "nu2": 0.0, "nu3": 0.0, "observed_drawdown": 0.04},
        )

        data = result["data"]
        assert data["passed"] is False
        (violation,) = data["violations"]
        assert violation["type"] == "zero_loss_tolerance"
        assert violation["value"] == pytest.approx(0.04)
        assert "max_1yr_loss_pct" in violation["reason"]

    def test_zero_nu2_with_no_drawdown_at_all_passes(self) -> None:
        """A path that never fell satisfies even a zero tolerance."""
        result = risk_check(
            {"A": 1.0},
            constraints={"nu1": 0.0, "nu2": 0.0, "nu3": 0.0, "observed_drawdown": 0.0},
        )

        assert result["data"]["passed"] is True

    def test_a_positive_nu2_keeps_the_tier_breach(self) -> None:
        """Only a zero ceiling gets the dedicated record."""
        result = risk_check(
            {"A": 1.0},
            constraints={
                "nu1": 0.05,
                "nu2": 0.10,
                "nu3": 0.20,
                "observed_drawdown": 0.15,
            },
        )

        (violation,) = result["data"]["violations"]
        assert violation["type"] == "drawdown_nu2"

    def test_an_unmeasurable_drawdown_is_still_unverified(self) -> None:
        """A NaN measurement is reported as such, even under a zero tolerance."""
        result = risk_check(
            {"A": 1.0},
            constraints={
                "nu1": 0.0,
                "nu2": 0.0,
                "nu3": 0.0,
                "observed_drawdown": math.nan,
            },
        )

        (violation,) = result["data"]["violations"]
        assert violation["type"] == "drawdown_unverified"


def _seed_paths(db_session, paths: dict[str, list[float]]) -> None:
    """Seed one close per calendar day for each ticker, the last on ``_ASOF``."""
    for ticker, closes in paths.items():
        exchange = Exchange(name=f"EX-{ticker}")
        db_session.add(exchange)
        db_session.flush()
        instrument = Instrument(
            ticker=ticker,
            short_name=ticker,
            exchange_id=exchange.id,
            instrument_type="EQUITY",
            asset_class="equity",
            yfinance_ticker=ticker,
        )
        db_session.add(instrument)
        db_session.flush()
        db_session.add_all(
            PriceHistory(
                instrument_id=instrument.id,
                date=_ASOF - dt.timedelta(days=len(closes) - 1 - i),
                close=round(close, 6),
                volume=1000,
            )
            for i, close in enumerate(closes)
        )
    db_session.flush()


def _bleed(n: int, yearly: float) -> dict[str, list[float]]:
    """Each ticker loses ``yearly`` a year, with a small wiggle so risk is non-zero."""
    return {
        ticker: [
            100.0
            * (1.0 - yearly) ** (i / _YEAR)
            * (1.0 + 0.004 * math.sin(0.5 * i + phase))
            for i in range(n)
        ]
        for ticker, phase in _PHASES.items()
    }


def _crash(n: int, fall: float) -> dict[str, list[float]]:
    """Flat for a year, then a ``fall`` decline over 120 bars, then flat."""
    paths = {}
    for ticker, phase in _PHASES.items():
        closes = []
        for i in range(n):
            progress = min(max((i - 500) / 120.0, 0.0), 1.0)
            closes.append(
                100.0
                * (1.0 - fall * progress)
                * (1.0 + 0.004 * math.sin(0.5 * i + phase))
            )
        paths[ticker] = closes
    return paths


def _cs(**overrides: object) -> ConstraintSet:
    """A growth profile whose ceilings default to 15 / 25 / 35% of capital."""
    kwargs: dict[str, object] = {
        "portfolio_id": str(_PORTFOLIO_ID),
        "base_currency": "EUR",
        "a_gamma": 2.5,
        "objective": ObjectiveChoice.GROWTH,
        "risk_measure": RiskMeasureChoice.VARIANCE,
        "beta": 0.95,
        "nu1": 0.15,
        "nu2": 0.25,
        "nu3": 0.35,
        "horizon": Horizon.LONG,
    }
    kwargs.update(overrides)
    return ConstraintSet(**kwargs)  # type: ignore[arg-type]


def _context(db_session, cs: ConstraintSet) -> RunContext:
    """A run context over a fresh audit run, holding ``cs`` and the proposal."""
    store = InMemoryStore()
    put_constraint_set(store, cs, store_key=settings.constraint_set_store_key)
    repo = AgentRunRepository(db_session)
    run = repo.create_run(
        portfolio_id=_PORTFOLIO_ID,
        asof=_ASOF,
        seed=0,
        universe=list(_WEIGHTS),
        optimizer_config={"step": "toolset"},
    )
    repo.append_decision(
        run.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": _WEIGHTS}),
    )
    return RunContext(
        session=db_session,
        asof=_ASOF,
        store=store,
        config=settings,
        run_id=run.id,
        portfolio_id=_PORTFOLIO_ID,
    )


def _tool(ctx: RunContext, role: str, name: str):
    """The bound tool ``name`` of ``role``."""
    (match,) = [t for t in bind_toolset(role, ctx) if t.name == name]
    return match


class TestGateJudgesTheWorstYear:
    """The bound ``risk_check`` / ``backtest`` compare the tiers with the worst year."""

    def test_a_slow_multi_year_bleed_passes_a_one_year_tolerance(
        self, db_session
    ) -> None:
        """A 38% multi-year drawdown does not breach a 25% one-year ceiling."""
        _seed_paths(db_session, _bleed(3 * _YEAR, 0.15))
        ctx = _context(db_session, _cs())

        verdict = _tool(ctx, "risk", "risk_check").invoke({})["data"]
        tiers = _tool(ctx, "risk", "backtest").invoke({})["data"]["drawdown_tiers"]

        assert verdict["passed"] is True, verdict
        assert tiers["verified"] is True
        assert tiers["max_drawdown"] > 0.35
        assert 0.14 < tiers["worst_rolling_loss"] < 0.25
        assert tiers["breached"] == ["nu1"]

    def test_the_full_sample_drawdown_is_still_reported(self, db_session) -> None:
        """The multi-year figure stays in the reply, as information."""
        _seed_paths(db_session, _bleed(3 * _YEAR, 0.15))
        ctx = _context(db_session, _cs())

        data = _tool(ctx, "risk", "backtest").invoke({})["data"]

        assert data["full_sample_max_drawdown"] > data["worst_rolling_loss"]
        assert data["rolling_loss_window"] == _YEAR

    def test_a_crash_inside_a_year_still_blocks(self, db_session) -> None:
        """A 40% fall within one year breaches the 25% ceiling."""
        _seed_paths(db_session, _crash(3 * _YEAR, 0.40))
        ctx = _context(db_session, _cs(nu3=0.6))

        verdict = _tool(ctx, "risk", "risk_check").invoke({})["data"]

        assert verdict["passed"] is False
        assert [v["type"] for v in verdict["violations"]] == ["drawdown_nu2"]

    def test_a_history_shorter_than_a_year_fails_closed(self, db_session) -> None:
        """Eighty days cannot show a year's loss, so the ceilings stay unverified."""
        _seed_paths(db_session, _bleed(80, 0.15))
        ctx = _context(db_session, _cs())

        verdict = _tool(ctx, "risk", "risk_check").invoke({})["data"]
        tiers = _tool(ctx, "risk", "backtest").invoke({})["data"]["drawdown_tiers"]

        assert verdict["passed"] is False
        (violation,) = verdict["violations"]
        assert violation["type"] == "drawdown_unverified"
        assert "79 bars" in violation["reason"]
        assert "shorter than the 365 bars" in violation["reason"]
        assert tiers["verified"] is False
        assert tiers["worst_rolling_loss"] is None

    def test_a_short_history_that_already_breaches_names_the_breach(
        self, db_session
    ) -> None:
        """The short sample's own drawdown is a lower bound that can prove a breach."""
        _seed_paths(db_session, _short_crash())
        ctx = _context(db_session, _cs(nu1=0.05, nu2=0.1, nu3=0.2))

        verdict = _tool(ctx, "risk", "risk_check").invoke({})["data"]

        kinds = [v["type"] for v in verdict["violations"]]
        assert verdict["passed"] is False
        assert "drawdown_nu3" in kinds
        assert "drawdown_unverified" in kinds

    def test_a_zero_loss_tolerance_is_named_at_the_gate(self, db_session) -> None:
        """Every book fails a zero tolerance, and the verdict says why."""
        _seed_paths(db_session, _bleed(3 * _YEAR, 0.15))
        ctx = _context(db_session, _cs(nu1=0.0, nu2=0.0, nu3=0.0))

        verdict = _tool(ctx, "risk", "risk_check").invoke({})["data"]

        assert verdict["passed"] is False
        (violation,) = verdict["violations"]
        assert violation["type"] == "zero_loss_tolerance"

    def test_the_mandate_message_names_the_zero_tolerance(self, db_session) -> None:
        """``place_orders`` explains the refusal in the profile's terms."""
        _seed_paths(db_session, _bleed(3 * _YEAR, 0.15))
        ctx = _context(db_session, _cs(nu1=0.0, nu2=0.0, nu3=0.0))
        _tool(ctx, "risk", "risk_check").invoke({})

        result = _tool(ctx, "orchestrator", "place_orders").invoke({})

        assert result["ok"] is False


def _short_crash() -> dict[str, list[float]]:
    """Eighty days with a 30% fall over the first twenty, then flat."""
    return {
        ticker: [
            100.0 * (1.0 - 0.30 * min(i / 20.0, 1.0)) * (1.0 + 0.002 * math.sin(i + p))
            for i in range(80)
        ]
        for ticker, p in _PHASES.items()
    }


class TestBacktestReportsBothFigures:
    """``backtest`` carries the one-year loss next to the full-sample drawdown."""

    def test_a_panel_shorter_than_the_window_reports_none(self, monkeypatch) -> None:
        """No rolling figure is invented for a history under a year."""
        index = pd.bdate_range("2024-01-02", periods=150)
        rng = np.random.default_rng(1)
        frame = pd.DataFrame(
            {
                "A": 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, 150)),
                "B": 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, 150)),
            },
            index=index,
        )
        monkeypatch.setattr(risk_mod, "load_price_frame", lambda *a, **k: (frame, []))

        result = risk_mod.backtest(None, "2024-08-01", {"A": 0.5, "B": 0.5})

        assert result["ok"] is True
        assert result["data"]["worst_rolling_loss"] is None
        assert result["data"]["full_sample_max_drawdown"] > 0.0

    def test_the_window_follows_the_panel_calendar(self, monkeypatch) -> None:
        """A weekday panel measures a year as 261 bars, not 252."""
        index = pd.bdate_range("2021-11-05", periods=1250)
        rng = np.random.default_rng(2)
        frame = pd.DataFrame(
            {
                name: 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, 1250))
                for name in ("A", "B")
            },
            index=index,
        )
        monkeypatch.setattr(risk_mod, "load_price_frame", lambda *a, **k: (frame, []))

        result = risk_mod.backtest(None, "2026-08-21", {"A": 0.5, "B": 0.5})

        assert result["ok"] is True
        assert result["data"]["rolling_loss_window"] == 261
        assert result["data"]["worst_rolling_loss"] is not None
        assert (
            result["data"]["worst_rolling_loss"]
            <= result["data"]["full_sample_max_drawdown"] + 1e-12
        )
