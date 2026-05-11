"""Tests for research/_preflight.py — DB pre-flight health checks (issue #519)."""

from __future__ import annotations

import datetime
from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

pytest.importorskip("typer")

from research._preflight import (
    _KNOWN_MAJOR_CURRENCIES,
    _MIN_INSTRUMENTS,
    _MIN_PRICE_TICKERS,
    _PRICE_COVERAGE_WINDOW_DAYS,
    _check_country_coverage,
    _check_fred_freshness,
    _check_fx_coverage,
    _check_price_coverage,
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
                "yfinance_ticker TEXT, "
                "currency_code TEXT)"
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


def _seed_instruments(
    engine: Engine,
    n_active: int,
    n_delisted: int = 0,
    currency_code: str | None = "USD",
) -> None:
    rows = []
    for i in range(n_active):
        rows.append(
            {
                "id": f"a{i}",
                "delisted_at": None,
                "yfinance_ticker": f"AAA{i}",
                "currency_code": currency_code,
            }
        )
    for i in range(n_delisted):
        rows.append(
            {
                "id": f"d{i}",
                "delisted_at": "2020-01-01",
                "yfinance_ticker": f"DDD{i}",
                "currency_code": currency_code,
            }
        )
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO instruments"
                "(id, delisted_at, yfinance_ticker, currency_code) "
                "VALUES (:id, :delisted_at, :yfinance_ticker, :currency_code)"
            ),
            rows,
        )


def _seed_instrument_with_currency(
    engine: Engine,
    instrument_id: str,
    yfinance_ticker: str,
    currency_code: str | None,
    delisted: bool = False,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO instruments"
                "(id, delisted_at, yfinance_ticker, currency_code) "
                "VALUES (:id, :delisted_at, :yfinance_ticker, :currency_code)"
            ),
            [
                {
                    "id": instrument_id,
                    "delisted_at": "2020-01-01" if delisted else None,
                    "yfinance_ticker": yfinance_ticker,
                    "currency_code": currency_code,
                }
            ],
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


def _seed_price_history_for_tickers(
    engine: Engine,
    instrument_ids: list[str],
    date: datetime.date,
) -> None:
    rows = [{"i": iid, "d": date.isoformat(), "c": 1.0} for iid in instrument_ids]
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO price_history(instrument_id, date, close) "
                "VALUES (:i, :d, :c)"
            ),
            rows,
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
# Check 2b — price coverage (distinct instrument count within window)
# ---------------------------------------------------------------------------


class TestPriceCoverage:
    def test_when_distinct_count_meets_floor_then_none_returned(
        self, engine: Engine, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        ids = [f"a{i}" for i in range(_MIN_PRICE_TICKERS)]
        _seed_price_history_for_tickers(
            engine, ids, date=today - datetime.timedelta(days=2)
        )
        with db_manager.get_session() as session:
            assert _check_price_coverage(session, today=today) is None

    def test_when_distinct_count_below_floor_then_message_names_count_and_threshold(
        self, engine: Engine, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        ids = [f"a{i}" for i in range(50)]
        _seed_price_history_for_tickers(
            engine, ids, date=today - datetime.timedelta(days=1)
        )
        with db_manager.get_session() as session:
            msg = _check_price_coverage(session, today=today)
        assert msg is not None
        assert "50" in msg
        assert str(_MIN_PRICE_TICKERS) in msg
        assert str(_PRICE_COVERAGE_WINDOW_DAYS) in msg

    def test_when_only_old_rows_then_failure_returned(
        self, engine: Engine, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        ids = [f"a{i}" for i in range(_MIN_PRICE_TICKERS)]
        old_date = today - datetime.timedelta(days=_PRICE_COVERAGE_WINDOW_DAYS + 1)
        _seed_price_history_for_tickers(engine, ids, date=old_date)
        with db_manager.get_session() as session:
            msg = _check_price_coverage(session, today=today)
        assert msg is not None
        assert "0" in msg

    def test_when_table_empty_then_failure_returned(
        self, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        with db_manager.get_session() as session:
            msg = _check_price_coverage(session, today=today)
        assert msg is not None

    def test_when_duplicate_rows_per_instrument_then_counted_once(
        self, engine: Engine, db_manager: _FakeDbManager, today: datetime.date
    ) -> None:
        # Single instrument with many rows must not satisfy the distinct floor.
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO price_history(instrument_id, date, close) "
                    "VALUES (:i, :d, :c)"
                ),
                [
                    {
                        "i": "a0",
                        "d": (today - datetime.timedelta(days=k)).isoformat(),
                        "c": 1.0,
                    }
                    for k in range(_PRICE_COVERAGE_WINDOW_DAYS)
                ],
            )
        with db_manager.get_session() as session:
            msg = _check_price_coverage(session, today=today)
        assert msg is not None
        assert "1 " in msg or msg.find(" 1 ") >= 0


# ---------------------------------------------------------------------------
# Check 2b — price coverage (mocked-scalar boundary tests + wiring)
# ---------------------------------------------------------------------------


def _mock_session_returning(scalar_value: int) -> MagicMock:
    """MagicMock session whose ``execute(...).scalar()`` returns *scalar_value*."""
    session = MagicMock(spec=Session)
    session.execute.return_value.scalar.return_value = scalar_value
    return session


class TestCheckPriceCoverageMocked:
    def test_when_mocked_scalar_meets_floor_then_none_returned(
        self, today: datetime.date
    ) -> None:
        session = _mock_session_returning(_MIN_PRICE_TICKERS)
        assert _check_price_coverage(session, today=today) is None

    def test_when_mocked_scalar_one_below_floor_then_failure_returned(
        self, today: datetime.date
    ) -> None:
        session = _mock_session_returning(_MIN_PRICE_TICKERS - 1)
        msg = _check_price_coverage(session, today=today)
        assert msg is not None

    def test_when_failure_then_message_contains_count_and_threshold(
        self, today: datetime.date
    ) -> None:
        observed = _MIN_PRICE_TICKERS - 1
        session = _mock_session_returning(observed)
        msg = _check_price_coverage(session, today=today)
        assert msg is not None
        assert str(observed) in msg
        assert str(_MIN_PRICE_TICKERS) in msg

    def test_when_called_then_query_uses_today_minus_window_cutoff(
        self, today: datetime.date
    ) -> None:
        session = _mock_session_returning(_MIN_PRICE_TICKERS)
        _check_price_coverage(session, today=today)
        session.execute.assert_called_once()
        _, kwargs_or_params = session.execute.call_args
        # Positional: (stmt, params_dict). Pull positional [1] if present.
        params = (
            session.execute.call_args.args[1]
            if len(session.execute.call_args.args) > 1
            else kwargs_or_params
        )
        expected_cutoff = (
            today - datetime.timedelta(days=_PRICE_COVERAGE_WINDOW_DAYS)
        ).isoformat()
        assert params == {"cutoff": expected_cutoff}

    def test_when_iter_results_runs_then_price_coverage_precedes_fred(
        self,
        db_manager: _FakeDbManager,
        today: datetime.date,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from research import _preflight as pf

        call_order: list[str] = []

        def _spy(name: str, retval: object = None):
            def _fn(*_args: object, **_kwargs: object) -> object:
                call_order.append(name)
                return retval

            return _fn

        monkeypatch.setattr(pf, "_check_universe_coverage", _spy("universe"))
        monkeypatch.setattr(pf, "_check_price_staleness", _spy("staleness"))
        monkeypatch.setattr(pf, "_check_price_coverage", _spy("coverage"))
        monkeypatch.setattr(pf, "_check_fred_freshness", _spy("fred"))
        monkeypatch.setattr(pf, "_check_country_coverage", _spy("country"))

        with db_manager.get_session() as session:
            list(pf._iter_check_results(session, today))

        assert "coverage" in call_order
        assert "fred" in call_order
        assert call_order.index("coverage") < call_order.index("fred")


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
# Check 5 — FX coverage (warning-only, no failure)
# ---------------------------------------------------------------------------


class TestCheckFxCoverage:
    def test_when_all_currencies_major_then_none_returned(
        self, engine: Engine, db_manager: _FakeDbManager
    ) -> None:
        for major in list(_KNOWN_MAJOR_CURRENCIES)[:5]:
            _seed_instrument_with_currency(engine, f"id_{major}", f"YF_{major}", major)
        with db_manager.get_session() as session:
            assert _check_fx_coverage(session) is None

    def test_when_non_major_currency_present_then_warning_returned(
        self, engine: Engine, db_manager: _FakeDbManager
    ) -> None:
        _seed_instrument_with_currency(engine, "id_brl", "VALE3.SA", "BRL")
        with db_manager.get_session() as session:
            msg = _check_fx_coverage(session)
        assert msg is not None
        assert "VALE3.SA" in msg
        assert "BRL" in msg
        assert "warning" in msg.lower()

    def test_when_more_than_20_offenders_then_message_capped_with_overflow(
        self, engine: Engine, db_manager: _FakeDbManager
    ) -> None:
        for i in range(25):
            _seed_instrument_with_currency(engine, f"id_{i}", f"TKR{i}.XX", "BRL")
        with db_manager.get_session() as session:
            msg = _check_fx_coverage(session)
        assert msg is not None
        assert "25" in msg
        # Overflow indicator: should mention more / additional
        assert (
            "more" in msg.lower()
            or "..." in msg
            or "additional" in msg.lower()
            or "5" in msg
        )

    def test_when_delisted_excluded_from_check(
        self, engine: Engine, db_manager: _FakeDbManager
    ) -> None:
        _seed_instrument_with_currency(
            engine, "id_brl_dead", "VALE3.SA", "BRL", delisted=True
        )
        with db_manager.get_session() as session:
            assert _check_fx_coverage(session) is None

    def test_when_currency_code_null_excluded_from_check(
        self, engine: Engine, db_manager: _FakeDbManager
    ) -> None:
        _seed_instrument_with_currency(engine, "id_null", "NULL.X", None)
        with db_manager.get_session() as session:
            assert _check_fx_coverage(session) is None


# ---------------------------------------------------------------------------
# Check 5b — FX coverage mixed-currency seed (GBX / ILA / ZAC) + orchestrator
# ---------------------------------------------------------------------------


def _seed_mixed_currencies(engine: Engine) -> None:
    """Seed two major-currency instruments and three minor-unit listings."""
    _seed_instrument_with_currency(engine, "id_usd", "AAPL", "USD")
    _seed_instrument_with_currency(engine, "id_gbp", "BARC.L", "GBP")
    _seed_instrument_with_currency(engine, "id_gbx", "VOD.L", "GBX")
    _seed_instrument_with_currency(engine, "id_ila", "TEVA.TA", "ILA")
    _seed_instrument_with_currency(engine, "id_zac", "NPN.JO", "ZAC")


class TestCheckFxCoverageMixedCurrencies:
    def test_when_mixed_seed_then_only_minor_codes_listed(
        self, engine: Engine, db_manager: _FakeDbManager
    ) -> None:
        _seed_mixed_currencies(engine)
        with db_manager.get_session() as session:
            msg = _check_fx_coverage(session)
        assert msg is not None
        # Each minor unit must appear with its ticker
        assert "VOD.L" in msg
        assert "GBX" in msg
        assert "TEVA.TA" in msg
        assert "ILA" in msg
        assert "NPN.JO" in msg
        assert "ZAC" in msg
        # Major-currency listings must NOT leak into the warning
        assert "AAPL" not in msg
        assert "BARC.L" not in msg

    def test_when_mixed_seed_then_run_db_preflight_does_not_raise(
        self,
        engine: Engine,
        db_manager: _FakeDbManager,
        today: datetime.date,
    ) -> None:
        _seed_healthy(engine, today)
        _seed_mixed_currencies(engine)
        # Warning-only output must NOT raise.
        run_db_preflight(db_manager, today=today)

    def test_when_mixed_seed_then_caplog_captures_logger_warning(
        self,
        engine: Engine,
        db_manager: _FakeDbManager,
        today: datetime.date,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging as _logging

        _seed_healthy(engine, today)
        _seed_mixed_currencies(engine)
        with caplog.at_level(_logging.WARNING, logger="research._preflight"):
            run_db_preflight(db_manager, today=today)
        warn_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warn_records, "expected at least one logger.warning emission"
        joined = "\n".join(r.getMessage() for r in warn_records)
        for code in ("GBX", "ILA", "ZAC"):
            assert code in joined


# ---------------------------------------------------------------------------
# Orchestrator — run_db_preflight
# ---------------------------------------------------------------------------


def _seed_healthy(engine: Engine, today: datetime.date) -> None:
    _seed_instruments(engine, n_active=_MIN_INSTRUMENTS)
    _seed_price_history_for_tickers(
        engine,
        [f"a{i}" for i in range(_MIN_PRICE_TICKERS)],
        date=today - datetime.timedelta(days=1),
    )
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

    def test_when_only_fx_warning_then_no_runtime_error_raised(
        self,
        engine: Engine,
        db_manager: _FakeDbManager,
        today: datetime.date,
    ) -> None:
        # Healthy seed with one extra non-major-currency active instrument.
        _seed_healthy(engine, today)
        _seed_instrument_with_currency(engine, "id_brl_warn", "VALE3.SA", "BRL")
        # Must not raise — FX coverage is warning-only.
        run_db_preflight(db_manager, today=today)

    def test_when_fx_warning_only_then_logger_warning_emitted(
        self,
        engine: Engine,
        db_manager: _FakeDbManager,
        today: datetime.date,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _seed_healthy(engine, today)
        _seed_instrument_with_currency(engine, "id_brl_warn2", "VALE3.SA", "BRL")
        import logging as _logging

        with caplog.at_level(_logging.WARNING, logger="research._preflight"):
            run_db_preflight(db_manager, today=today)
        warn_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("BRL" in r.getMessage() for r in warn_records)

    def test_when_warning_present_alongside_failure_then_only_failure_in_runtime(
        self,
        engine: Engine,
        db_manager: _FakeDbManager,
        today: datetime.date,
    ) -> None:
        # Universe failure + non-major currency warning.
        _seed_instruments(engine, n_active=5)
        _seed_price_history(engine, max_date=today - datetime.timedelta(days=1))
        fresh = today - datetime.timedelta(days=2)
        _seed_fred(engine, dict.fromkeys(_REQUIRED, fresh))
        _seed_country(engine, [f"a{i}" for i in range(5)])
        _seed_instrument_with_currency(engine, "id_warn", "VALE3.SA", "BRL")
        with pytest.raises(RuntimeError) as excinfo:
            run_db_preflight(db_manager, today=today)
        msg = str(excinfo.value)
        assert "instruments" in msg.lower()
        # Warning text must NOT pollute the RuntimeError message.
        assert "VALE3.SA" not in msg
        assert "BRL" not in msg


class TestLoadDataIntegration:
    """Verify load_data() raises before assemble_all when DB is unhealthy."""

    def test_when_preflight_fails_then_assemble_all_not_called(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import research.pipeline._load as _load_module
        from research import stock_selection_pipeline as ssp

        sentinel = {"called": False}

        def _fake_assemble_all(*_args: object, **_kwargs: object) -> None:
            sentinel["called"] = True
            raise AssertionError("assemble_all must not run when preflight fails")

        def _fake_preflight(_db: object, today: datetime.date | None = None) -> None:
            raise RuntimeError("Preflight: synthetic failure")

        monkeypatch.setattr(_load_module, "assemble_all", _fake_assemble_all)
        monkeypatch.setattr(_load_module, "_run_db_preflight", _fake_preflight)

        class _StubMgr:
            def initialize(self) -> None:
                pass

        with pytest.raises(RuntimeError, match="synthetic failure"):
            ssp.load_data(_StubMgr())
        assert sentinel["called"] is False
