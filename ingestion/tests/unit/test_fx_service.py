"""FX-rate feed (SPEC Phase 05 T0/T3).

Currency derivation runs against real SQLite (seeded instruments + price
history); the bulk fetch is driven with a mocked yfinance client over the real
``FxRatesRepository`` + session; the scheduler step is verified to compose
``_run_step``. The feed must import no ``optimizer`` (hygiene guard).
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from portopt_db.models.market_data.fx_rates import FxRate
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument

from app.services.market_data.fx_service import (
    _distinct_currencies,
    _to_major,
    run_fx_fetch,
)
from tests._fixtures.clock import FakeClock


def _close_frame(dates, closes):
    """Build a yfinance-style history frame with a Close column by date."""
    return pd.DataFrame({"Close": closes}, index=pd.to_datetime(dates))


@pytest.fixture
def fx_session(monkeypatch):
    """A function-scoped committing session pointed at by the session factory.

    ``run_fx_fetch`` commits, which would leak across the shared session-scoped
    SAVEPOINT fixture; a private in-memory engine keeps each test isolated.
    """
    from contextlib import contextmanager

    from portopt_db.models import Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app import database as db_module

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    @contextmanager
    def _cm():
        yield session

    monkeypatch.setattr(db_module.database_manager, "get_session", _cm)
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


class TestToMajor:
    """Currency-code normalization to ISO-4217 major units."""

    def test_passes_through_valid_major_code(self) -> None:
        """A three-letter ISO code is returned upper-cased."""
        assert _to_major("usd") == "USD"
        assert _to_major("EUR") == "EUR"

    def test_maps_minor_units_to_major(self) -> None:
        """Sub-unit codes collapse to their major currency."""
        assert _to_major("GBp") == "GBP"
        assert _to_major("GBX") == "GBP"
        assert _to_major("ZAc") == "ZAR"
        assert _to_major("ILA") == "ILS"

    def test_rejects_blank_or_malformed(self) -> None:
        """Empty, None, or non-ISO codes yield None."""
        assert _to_major(None) is None
        assert _to_major("") is None
        assert _to_major("  ") is None
        assert _to_major("WXYZ") is None


class TestDistinctCurrencies:
    """Deriving the quote-currency set from the stored universe."""

    def test_unions_instruments_and_price_units_excluding_base(
        self, db_session
    ) -> None:
        """Currencies come from instruments + price_unit, minus the base."""
        ex = Exchange(name="LSE")
        db_session.add(ex)
        db_session.flush()
        us = Instrument(ticker="AAA", short_name="A", exchange=ex, currency_code="USD")
        gb = Instrument(ticker="BBB", short_name="B", exchange=ex, currency_code="GBP")
        eu = Instrument(ticker="CCC", short_name="C", exchange=ex, currency_code="EUR")
        # NULL currency_code — major derived from the price_unit sub-unit.
        pence = Instrument(
            ticker="DDD", short_name="D", exchange=ex, currency_code=None
        )
        db_session.add_all([us, gb, eu, pence])
        db_session.flush()
        db_session.add(
            PriceHistory(
                instrument_id=pence.id,
                date=dt.date(2026, 10, 1),
                close=100,
                price_unit="GBp",
            )
        )
        db_session.flush()

        got = _distinct_currencies(db_session, "EUR")
        assert got == {"USD", "GBP"}


def _seed_eur_usd_gbp(db_session):
    """Seed a universe with USD + GBP (base EUR excluded) instruments."""
    ex = Exchange(name="X")
    db_session.add(ex)
    db_session.flush()
    db_session.add_all(
        [
            Instrument(
                ticker="US1", short_name="US1", exchange=ex, currency_code="USD"
            ),
            Instrument(
                ticker="GB1", short_name="GB1", exchange=ex, currency_code="GBP"
            ),
            Instrument(
                ticker="EU1", short_name="EU1", exchange=ex, currency_code="EUR"
            ),
        ]
    )
    db_session.flush()


class TestRunFxFetch:
    """End-to-end FX fetch: cross-via-USD math, isolation, graceful degrade."""

    _D1 = dt.date(2026, 10, 1)
    _D2 = dt.date(2026, 10, 2)

    def _yf(self):
        """A mock client whose fetch_history returns USD-leg frames."""
        legs = {
            "EURUSD=X": _close_frame([self._D1, self._D2], [1.10, 1.08]),
            "GBPUSD=X": _close_frame([self._D1, self._D2], [1.30, 1.28]),
        }
        yf = MagicMock()
        yf.fetch_history.side_effect = lambda symbol, **kw: legs.get(symbol)
        return yf

    def test_writes_cross_via_usd_rates(self, fx_session) -> None:
        """EUR-base rates are computed as q_usd / eur_usd per aligned date."""
        _seed_eur_usd_gbp(fx_session)

        result = run_fx_fetch(self._yf(), base="EUR")

        rows = {
            (r.quote_currency, r.date): float(r.rate)
            for r in fx_session.query(FxRate).all()
        }
        assert rows[("USD", self._D1)] == pytest.approx(1.0 / 1.10)
        assert rows[("USD", self._D2)] == pytest.approx(1.0 / 1.08)
        assert rows[("GBP", self._D1)] == pytest.approx(1.30 / 1.10)
        assert rows[("GBP", self._D2)] == pytest.approx(1.28 / 1.08)
        assert result["error_count"] == 0

    def test_never_fetches_base_against_itself_as_quote(self, fx_session) -> None:
        """No row stores the base currency as its own quote."""
        _seed_eur_usd_gbp(fx_session)
        run_fx_fetch(self._yf(), base="EUR")
        assert fx_session.query(FxRate).filter_by(quote_currency="EUR").count() == 0

    def test_per_pair_failure_is_isolated(self, fx_session) -> None:
        """A failing quote leg is skipped; other pairs still persist."""
        _seed_eur_usd_gbp(fx_session)

        def _fetch(symbol, **kw):
            if symbol == "GBPUSD=X":
                raise RuntimeError("boom")
            return _close_frame([self._D1], [1.10])

        yf = MagicMock()
        yf.fetch_history.side_effect = _fetch

        result = run_fx_fetch(yf, base="EUR")

        quotes = {r.quote_currency for r in fx_session.query(FxRate).all()}
        assert "USD" in quotes
        assert "GBP" not in quotes
        assert result["error_count"] >= 1

    def test_missing_base_leg_degrades_without_rows(self, fx_session) -> None:
        """Without the base USD leg nothing can be converted; no rows, no raise."""
        _seed_eur_usd_gbp(fx_session)

        yf = MagicMock()
        yf.fetch_history.side_effect = lambda symbol, **kw: (
            None if symbol == "EURUSD=X" else _close_frame([self._D1], [1.30])
        )

        result = run_fx_fetch(yf, base="EUR")
        assert fx_session.query(FxRate).count() == 0
        assert result["rows_total"] == 0

    def test_empty_universe_completes_without_rows(self, fx_session) -> None:
        """No instruments → no quotes → zero rows, no fetch, no raise."""
        yf = MagicMock()
        result = run_fx_fetch(yf, base="EUR")
        assert fx_session.query(FxRate).count() == 0
        assert result["rows_total"] == 0
        yf.fetch_history.assert_not_called()

    def test_usd_base_stores_each_legs_usd_price_directly(self, fx_session) -> None:
        """A USD-base mandate stores each foreign leg's USD price (rate = q_usd, base=1).

        base='USD' takes the no-base-leg branch (base_usd == 1.0, dates are the quote
        leg's own), which no base='EUR' test exercises — a direction inversion there
        would otherwise pass the whole suite.
        """
        _seed_eur_usd_gbp(fx_session)

        result = run_fx_fetch(self._yf(), base="USD")

        rows = {
            (r.quote_currency, r.date): float(r.rate)
            for r in fx_session.query(FxRate).all()
        }
        assert rows[("EUR", self._D1)] == pytest.approx(1.10)
        assert rows[("EUR", self._D2)] == pytest.approx(1.08)
        assert rows[("GBP", self._D1)] == pytest.approx(1.30)
        assert rows[("GBP", self._D2)] == pytest.approx(1.28)
        assert fx_session.query(FxRate).filter_by(quote_currency="USD").count() == 0
        assert result["error_count"] == 0


class TestFinishedAt:
    """Every completion path stamps finished_at when the run ends, not starts."""

    @staticmethod
    def _run(yf, clock, on_progress):
        patched_dt = SimpleNamespace(**{**vars(dt), "datetime": clock.datetime_cls})
        with patch("app.services.market_data.fx_service.dt", patched_dt):
            run_fx_fetch(yf, base="EUR", on_progress=on_progress)

    def test_full_run(self, fx_session) -> None:
        _seed_eur_usd_gbp(fx_session)
        clock = FakeClock()
        yf = MagicMock()
        yf.fetch_history.side_effect = clock.advancing(
            _close_frame([dt.date(2026, 10, 1)], [1.10])
        )
        on_progress = MagicMock()

        self._run(yf, clock, on_progress)

        clock.assert_finished_at_marks_run_end(on_progress)

    def test_empty_universe(self, fx_session) -> None:
        clock = FakeClock()
        on_progress = MagicMock()

        with patch(
            "app.services.market_data.fx_service._distinct_currencies",
            side_effect=clock.advancing(set()),
        ):
            self._run(MagicMock(), clock, on_progress)

        clock.assert_finished_at_marks_run_end(on_progress)

    def test_missing_base_leg(self, fx_session) -> None:
        _seed_eur_usd_gbp(fx_session)
        clock = FakeClock()
        yf = MagicMock()
        yf.fetch_history.side_effect = clock.advancing(
            lambda symbol, **kw: (
                None
                if symbol == "EURUSD=X"
                else _close_frame([dt.date(2026, 10, 1)], [1.30])
            )
        )
        on_progress = MagicMock()

        self._run(yf, clock, on_progress)

        clock.assert_finished_at_marks_run_end(on_progress)


class TestSchedulerStep:
    """The fx scheduler step and its reclaim registration."""

    def test_composes_run_step(self) -> None:
        """run_fx_step dispatches _run_step with the 'fx' label + its job svc."""
        mod = "app.services.jobs.scheduler"
        with (
            patch(f"{mod}._run_step", return_value=True) as run_step,
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            ),
        ):
            from app.services.jobs.scheduler import _fx_jobs, run_fx_step

            assert run_fx_step() is True

        assert run_step.call_args.args[0] == "fx"
        assert run_step.call_args.args[1] is _fx_jobs
        # The third positional is the service fn _run_step calls: a copy-paste swapping
        # it would silently run the wrong logic on every daily pipeline, with "fx" still
        # the label — so pin the callable identity, not just the label and job svc.
        assert run_step.call_args.args[2] is run_fx_fetch

    def test_fx_job_type_is_reclaim_dispatchable(self) -> None:
        """The orphan reaper can re-dispatch a wedged fx_fetch run."""
        from app.services.jobs.scheduler import _reclaim_dispatchable

        assert _reclaim_dispatchable("fx_fetch") is True


class TestDailyPipelineWiring:
    """The fx step is wired into the daily pipeline before NAV marking."""

    def test_daily_pipeline_runs_fx_step(self) -> None:
        """The daily pipeline invokes the fx step so NAV marks see fresh rates."""
        mod = "app.services.jobs.scheduler"
        with (
            patch(f"{mod}._run_step") as mock_step,
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            ),
            patch("app.schemas.macro.macro_regime.MacroFetchRequest", MagicMock()),
            patch("app.schemas.macro.macro_regime.MacroNewsFetchRequest", MagicMock()),
            patch(
                "app.schemas.market_data.yfinance_data.YFinanceFetchRequest",
                MagicMock(),
            ),
        ):
            mock_step.side_effect = lambda label, *a, **k: True
            from app.services.jobs.scheduler import run_daily_pipeline

            run_daily_pipeline()

        labels = {c.args[0] for c in mock_step.call_args_list}
        assert "fx" in labels
