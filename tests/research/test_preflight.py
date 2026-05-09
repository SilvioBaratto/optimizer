"""Tests for research/_preflight.py — DB pre-flight health checks (issue #519)."""

from __future__ import annotations

import datetime
from collections.abc import Generator
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

pytest.importorskip("typer")

from research._preflight import (
    _MIN_INSTRUMENTS,
    _check_country_coverage,
    _check_fred_freshness,
    _check_price_staleness,
    _check_universe_coverage,
    run_db_preflight,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_schema(engine: Engine) -> None:
    """Build the minimal schema the pre-flight queries depend on."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE instruments ("
                "id TEXT PRIMARY KEY, "
                "delisted_at TEXT, "
                "yfinance_ticker TEXT)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE price_history (instrument_id TEXT, date TEXT, close REAL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE fred_observations (series_id TEXT, date TEXT, value REAL)"
            )
        )
        conn.execute(
            text("CREATE TABLE ticker_profiles (instrument_id TEXT, country TEXT)")
        )


def _seed_instruments(engine: Engine, n_active: int, n_delisted: int = 0) -> None:
    rows = []
    for i in range(n_active):
        rows.append({"id": f"a{i}", "delisted_at": None, "yfinance_ticker": f"AAA{i}"})
    for i in range(n_delisted):
        rows.append(
            {"id": f"d{i}", "delisted_at": "2020-01-01", "yfinance_ticker": f"DDD{i}"}
        )
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO instruments(id, delisted_at, yfinance_ticker) "
                "VALUES (:id, :delisted_at, :yfinance_ticker)"
            ),
            rows,
        )


def _seed_price_history(engine: Engine, max_date: datetime.date) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO price_history(instrument_id, date, close) "
                "VALUES (:i, :d, :c)"
            ),
            [{"i": "a0", "d": max_date.isoformat(), "c": 1.0}],
        )


def _seed_fred(
    engine: Engine,
    series: dict[str, datetime.date],
) -> None:
    rows = [{"sid": sid, "d": d.isoformat(), "v": 1.0} for sid, d in series.items()]
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO fred_observations(series_id, date, value) "
                "VALUES (:sid, :d, :v)"
            ),
            rows,
        )


def _seed_country(
    engine: Engine,
    instrument_ids: list[str],
    country: str | None = "United States",
) -> None:
    rows = [{"i": iid, "c": country} for iid in instrument_ids]
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO ticker_profiles(instrument_id, country) VALUES (:i, :c)"),
            rows,
        )


class _FakeDbManager:
    """Minimal stand-in exposing the same `get_session()` context-manager API."""

    def __init__(self, engine: Engine) -> None:
        self._maker = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        session = self._maker()
        try:
            yield session
        finally:
            session.close()


@pytest.fixture
def engine() -> Engine:
    eng = create_engine("sqlite:///:memory:")
    _create_schema(eng)
    return eng


@pytest.fixture
def db_manager(engine: Engine) -> _FakeDbManager:
    return _FakeDbManager(engine)


@pytest.fixture
def today() -> datetime.date:
    return datetime.date(2024, 6, 30)


# ---------------------------------------------------------------------------
# Check 1 — universe coverage
# ---------------------------------------------------------------------------


class TestUniverseCoverage:
    def test_when_count_meets_floor_then_none_returned(
        self, engine: Engine, db_manager: _FakeDbManager
    ) -> None:
        _seed_instruments(engine, n_active=_MIN_INSTRUMENTS)
        with db_manager.get_session() as session:
            assert _check_universe_coverage(session) is None

    def test_when_below_floor_then_message_names_count(
        self, engine: Engine, db_manager: _FakeDbManager
    ) -> None:
        _seed_instruments(engine, n_active=10)
        with db_manager.get_session() as session:
            msg = _check_universe_coverage(session)
        assert msg is not None
        assert "10" in msg
        assert str(_MIN_INSTRUMENTS) in msg

    def test_when_table_empty_then_failure_message_returned(
        self, db_manager: _FakeDbManager
    ) -> None:
        with db_manager.get_session() as session:
            msg = _check_universe_coverage(session)
        assert msg is not None


# ---------------------------------------------------------------------------
# Check 2 — price staleness
# ---------------------------------------------------------------------------


class TestPriceStaleness:
    def test_when_max_date_within_7_days_then_none_returned(
        self, engine: Engine, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        _seed_price_history(engine, max_date=today - datetime.timedelta(days=3))
        with db_manager.get_session() as session:
            assert _check_price_staleness(session, today=today) is None

    def test_when_max_date_stale_then_message_names_max_and_gap(
        self, engine: Engine, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        max_date = today - datetime.timedelta(days=20)
        _seed_price_history(engine, max_date=max_date)
        with db_manager.get_session() as session:
            msg = _check_price_staleness(session, today=today)
        assert msg is not None
        assert max_date.isoformat() in msg
        assert "20" in msg

    def test_when_table_empty_then_failure_message_returned(
        self, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        with db_manager.get_session() as session:
            msg = _check_price_staleness(session, today=today)
        assert msg is not None
        assert "empty" in msg.lower() or "no" in msg.lower()


# ---------------------------------------------------------------------------
# Check 3 — FRED freshness
# ---------------------------------------------------------------------------


_REQUIRED = ("DGS3MO", "DGS2", "DGS10", "BAMLH0A0HYM2", "VIXCLS", "USRECDM")


class TestFredFreshness:
    def test_when_all_series_fresh_then_none_returned(
        self, engine: Engine, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        fresh = today - datetime.timedelta(days=5)
        _seed_fred(engine, dict.fromkeys(_REQUIRED, fresh))
        with db_manager.get_session() as session:
            assert _check_fred_freshness(session, today=today) is None

    def test_when_one_series_missing_then_failure_lists_it(
        self, engine: Engine, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        fresh = today - datetime.timedelta(days=5)
        partial = {sid: fresh for sid in _REQUIRED if sid != "DGS10"}
        _seed_fred(engine, partial)
        with db_manager.get_session() as session:
            msg = _check_fred_freshness(session, today=today)
        assert msg is not None
        assert "DGS10" in msg

    def test_when_one_series_stale_then_failure_names_max_date(
        self, engine: Engine, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        fresh = today - datetime.timedelta(days=5)
        stale = today - datetime.timedelta(days=60)
        seeded = dict.fromkeys(_REQUIRED, fresh)
        seeded["VIXCLS"] = stale
        _seed_fred(engine, seeded)
        with db_manager.get_session() as session:
            msg = _check_fred_freshness(session, today=today)
        assert msg is not None
        assert "VIXCLS" in msg
        assert stale.isoformat() in msg


# ---------------------------------------------------------------------------
# Check 4 — country coverage
# ---------------------------------------------------------------------------


class TestCountryCoverage:
    def test_when_full_coverage_then_none_returned(
        self, engine: Engine, db_manager: _FakeDbManager
    ) -> None:
        _seed_instruments(engine, n_active=20)
        _seed_country(engine, [f"a{i}" for i in range(20)])
        with db_manager.get_session() as session:
            assert _check_country_coverage(session) is None

    def test_when_ratio_below_threshold_then_message_names_ratio(
        self, engine: Engine, db_manager: _FakeDbManager
    ) -> None:
        _seed_instruments(engine, n_active=20)
        _seed_country(engine, [f"a{i}" for i in range(10)])  # only 50%
        with db_manager.get_session() as session:
            msg = _check_country_coverage(session)
        assert msg is not None
        assert "0.50" in msg or "50" in msg

    def test_when_delisted_excluded_from_denominator(
        self, engine: Engine, db_manager: _FakeDbManager
    ) -> None:
        _seed_instruments(engine, n_active=20, n_delisted=200)
        # Only active ones get profiles → 100% coverage of active
        _seed_country(engine, [f"a{i}" for i in range(20)])
        with db_manager.get_session() as session:
            assert _check_country_coverage(session) is None

    def test_when_no_active_instruments_then_failure_message_returned(
        self, engine: Engine, db_manager: _FakeDbManager
    ) -> None:
        _seed_instruments(engine, n_active=0, n_delisted=5)
        with db_manager.get_session() as session:
            msg = _check_country_coverage(session)
        assert msg is not None


# ---------------------------------------------------------------------------
# Orchestrator — run_db_preflight
# ---------------------------------------------------------------------------


def _seed_healthy(engine: Engine, today: datetime.date) -> None:
    _seed_instruments(engine, n_active=_MIN_INSTRUMENTS)
    _seed_price_history(engine, max_date=today - datetime.timedelta(days=1))
    fresh = today - datetime.timedelta(days=2)
    _seed_fred(engine, dict.fromkeys(_REQUIRED, fresh))
    _seed_country(engine, [f"a{i}" for i in range(_MIN_INSTRUMENTS)])


class TestRunDbPreflight:
    def test_when_all_checks_pass_then_returns_none(
        self, engine: Engine, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        _seed_healthy(engine, today)
        run_db_preflight(db_manager, today=today)  # must not raise

    def test_when_universe_below_floor_then_runtime_error_raised(
        self, engine: Engine, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        _seed_instruments(engine, n_active=10)
        _seed_price_history(engine, max_date=today - datetime.timedelta(days=1))
        fresh = today - datetime.timedelta(days=2)
        _seed_fred(engine, dict.fromkeys(_REQUIRED, fresh))
        _seed_country(engine, [f"a{i}" for i in range(10)])
        with pytest.raises(RuntimeError, match="instruments"):
            run_db_preflight(db_manager, today=today)

    def test_when_multiple_failures_then_all_reported(
        self, engine: Engine, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        # Universe too small + price stale + FRED missing + low country coverage
        _seed_instruments(engine, n_active=5)
        _seed_price_history(engine, max_date=today - datetime.timedelta(days=30))
        # FRED entirely empty → all required series missing
        # Country: only 1 of 5 active gets profile → 20%
        _seed_country(engine, ["a0"])
        with pytest.raises(RuntimeError) as excinfo:
            run_db_preflight(db_manager, today=today)
        msg = str(excinfo.value)
        # All 4 checks must be present in the aggregated message
        assert "instruments" in msg.lower()
        assert "price" in msg.lower() or "stale" in msg.lower()
        assert "DGS3MO" in msg or "fred" in msg.lower()
        assert "country" in msg.lower() or "0.2" in msg

    def test_when_failures_aggregated_then_joined_on_newlines(
        self, engine: Engine, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        _seed_instruments(engine, n_active=5)
        _seed_price_history(engine, max_date=today - datetime.timedelta(days=30))
        with pytest.raises(RuntimeError) as excinfo:
            run_db_preflight(db_manager, today=today)
        assert "\n" in str(excinfo.value)


class TestLoadDataIntegration:
    """Verify load_data() raises before assemble_all when DB is unhealthy."""

    def test_when_preflight_fails_then_assemble_all_not_called(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research import stock_selection_pipeline as ssp

        sentinel = {"called": False}

        def _fake_assemble_all(*_args: object, **_kwargs: object) -> None:
            sentinel["called"] = True
            raise AssertionError("assemble_all must not run when preflight fails")

        def _fake_preflight(_db: object, today: datetime.date | None = None) -> None:
            raise RuntimeError("Preflight: synthetic failure")

        monkeypatch.setattr(ssp, "assemble_all", _fake_assemble_all)
        monkeypatch.setattr(ssp, "_run_db_preflight", _fake_preflight)

        class _StubMgr:
            def initialize(self) -> None:
                pass

        monkeypatch.setattr(ssp, "DatabaseManager", lambda: _StubMgr())

        with pytest.raises(RuntimeError, match="synthetic failure"):
            ssp.load_data()
        assert sentinel["called"] is False
