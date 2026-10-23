"""Unit tests for MacroRegimeService.fetch_country and run_bulk_macro_fetch.

Covers gaps NOT already in test_macro_regime_service_fred_key.py:
  - fetch_country success: counts populated, repo upserts called correctly
  - fetch_country ilsole scraper raises: error surfaced, no propagation
  - fetch_country te scraper raises: error surfaced, no propagation
  - fetch_country te non-success status: errors captured, counts zeroed
  - run_bulk_macro_fetch: progress callback contract, session.commit per-country
"""

from __future__ import annotations

import contextlib
import datetime
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.macro.macro_regime_service import (
    MacroRegimeService,
    run_bulk_macro_fetch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_service(
    ilsole_forecasts: dict | Exception | None = None,
    te_response: dict | Exception | None = None,
) -> tuple[MacroRegimeService, MagicMock]:
    """Return (service, mock_repo) with pre-wired scraper behaviours."""
    mock_repo = MagicMock()
    mock_repo.upsert_economic_indicator.return_value = 1
    mock_repo.upsert_economic_indicator_observation.return_value = 1
    mock_repo.upsert_te_indicators.return_value = 3
    mock_repo.upsert_te_observations.return_value = 3
    mock_repo.upsert_bond_yields.return_value = 2
    mock_repo.upsert_bond_yield_observations.return_value = 2

    mock_ilsole = MagicMock()
    if isinstance(ilsole_forecasts, Exception):
        mock_ilsole.get_forecasts.side_effect = ilsole_forecasts
    else:
        mock_ilsole.get_forecasts.return_value = ilsole_forecasts

    mock_te = MagicMock()
    if isinstance(te_response, Exception):
        mock_te.get_country_indicators.side_effect = te_response
    else:
        mock_te.get_country_indicators.return_value = te_response

    service = MacroRegimeService(
        repo=mock_repo,
        ilsole_scraper=mock_ilsole,
        te_scraper=mock_te,
        fred_scraper=MagicMock(),
    )
    return service, mock_repo


_FORECAST_DATA = {
    "last_inflation": 3.1,
    "inflation_6m": 2.8,
    "gdp_growth_6m": 2.2,
    "timestamp": "2026-01-01T00:00:00",
}

_TE_SUCCESS = {
    "status": "success",
    "indicators": {"inflation_rate": 3.1, "gdp_growth_rate": 2.5},
    "bond_yields": {"10y": 4.2},
}

_TE_FAILED = {
    "status": "error",
    "error": "HTML parse failed",
    "indicators": {},
    "bond_yields": {},
}


# ---------------------------------------------------------------------------
# fetch_country — success
# ---------------------------------------------------------------------------


class TestFetchCountrySuccess:
    def test_when_both_scrapers_succeed_then_counts_all_populated(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=_TE_SUCCESS,
        )

        result = service.fetch_country("USA", include_bonds=True)

        assert result["errors"] == []
        assert result["counts"]["ilsole_forecast"] == 1
        assert result["counts"]["ilsole_observations"] == 1
        assert result["counts"]["te_indicators"] == 3
        assert result["counts"]["te_observations"] == 3
        assert result["counts"]["bond_yields"] == 2
        assert result["counts"]["bond_yield_observations"] == 2

    def test_when_both_scrapers_succeed_then_ilsole_upsert_called_with_country(
        self,
    ) -> None:
        service, mock_repo = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=_TE_SUCCESS,
        )

        service.fetch_country("USA", include_bonds=True)

        mock_repo.upsert_economic_indicator.assert_called_once()
        call_kwargs = mock_repo.upsert_economic_indicator.call_args
        assert call_kwargs.kwargs.get("country") == "USA" or (
            len(call_kwargs.args) > 0 and call_kwargs.args[0] == "USA"
        )

    def test_when_both_scrapers_succeed_then_te_upsert_called_with_country(
        self,
    ) -> None:
        service, mock_repo = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=_TE_SUCCESS,
        )

        service.fetch_country("Germany", include_bonds=True)

        mock_repo.upsert_te_indicators.assert_called_once()
        call_kwargs = mock_repo.upsert_te_indicators.call_args
        assert call_kwargs.kwargs.get("country") == "Germany"

    def test_when_ilsole_returns_none_then_ilsole_counts_zeroed(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=None,
            te_response=_TE_SUCCESS,
        )

        result = service.fetch_country("USA")

        assert result["counts"]["ilsole_forecast"] == 0
        assert result["counts"]["ilsole_observations"] == 0
        assert result["errors"] == []

    def test_when_include_bonds_false_then_bond_counts_zeroed(self) -> None:
        service, mock_repo = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=_TE_SUCCESS,
        )

        result = service.fetch_country("USA", include_bonds=False)

        assert result["counts"]["bond_yields"] == 0
        assert result["counts"]["bond_yield_observations"] == 0
        mock_repo.upsert_bond_yields.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_country — ilsole scraper raises
# ---------------------------------------------------------------------------


class TestFetchCountryIlsoleRaises:
    def test_when_ilsole_raises_then_error_message_contains_ilsole_forecast(
        self,
    ) -> None:
        service, _ = _build_service(
            ilsole_forecasts=RuntimeError("timeout"),
            te_response=_TE_SUCCESS,
        )

        result = service.fetch_country("USA")

        assert any("ilsole_forecast" in e for e in result["errors"])

    def test_when_ilsole_raises_then_no_exception_propagates(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=RuntimeError("boom"),
            te_response=_TE_SUCCESS,
        )

        # Must not raise
        result = service.fetch_country("USA")
        assert isinstance(result, dict)

    def test_when_ilsole_raises_then_te_still_processed(self) -> None:
        service, mock_repo = _build_service(
            ilsole_forecasts=RuntimeError("boom"),
            te_response=_TE_SUCCESS,
        )

        result = service.fetch_country("USA")

        assert result["counts"].get("te_indicators", 0) == 3
        mock_repo.upsert_te_indicators.assert_called_once()


# ---------------------------------------------------------------------------
# fetch_country — ilsole row found but nothing parsed
# ---------------------------------------------------------------------------

_ALL_NONE_FORECAST = {
    "last_inflation": None,
    "inflation_6m": None,
    "inflation_10y_avg": None,
    "gdp_growth_6m": None,
    "earnings_12m": None,
    "eps_expected_12m": None,
    "peg_ratio": None,
    "lt_rate_forecast": None,
    "reference_date": None,
    "timestamp": "2026-01-01T00:00:00",
}


class TestFetchCountryIlsoleUnusableForecast:
    def test_when_every_forecast_value_is_none_then_snapshot_not_overwritten(
        self,
    ) -> None:
        service, mock_repo = _build_service(
            ilsole_forecasts=_ALL_NONE_FORECAST,
            te_response=_TE_SUCCESS,
        )

        service.fetch_country("USA")

        mock_repo.upsert_economic_indicator.assert_not_called()
        mock_repo.upsert_economic_indicator_observation.assert_not_called()

    def test_when_every_forecast_value_is_none_then_error_surfaced(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_ALL_NONE_FORECAST,
            te_response=_TE_SUCCESS,
        )

        result = service.fetch_country("USA")

        assert any(e.startswith("ilsole_forecast") for e in result["errors"])
        assert result["counts"]["ilsole_forecast"] == 0
        assert result["counts"]["ilsole_observations"] == 0

    def test_when_one_forecast_value_parsed_then_row_is_stored(self) -> None:
        partial = {**_ALL_NONE_FORECAST, "lt_rate_forecast": 4.6}
        service, mock_repo = _build_service(
            ilsole_forecasts=partial,
            te_response=_TE_SUCCESS,
        )

        result = service.fetch_country("USA")

        mock_repo.upsert_economic_indicator.assert_called_once()
        assert result["errors"] == []


# ---------------------------------------------------------------------------
# fetch_country — te scraper raises
# ---------------------------------------------------------------------------


class TestFetchCountryTeRaises:
    def test_when_te_raises_then_error_message_contains_trading_economics(
        self,
    ) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=RuntimeError("connection refused"),
        )

        result = service.fetch_country("USA")

        assert any("trading_economics" in e for e in result["errors"])

    def test_when_te_raises_then_no_exception_propagates(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=RuntimeError("boom"),
        )

        result = service.fetch_country("USA")
        assert isinstance(result, dict)

    def test_when_te_raises_then_te_counts_default_to_zero(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=RuntimeError("boom"),
        )

        result = service.fetch_country("USA")

        assert result["counts"].get("te_indicators", 0) == 0
        assert result["counts"].get("te_observations", 0) == 0
        assert result["counts"].get("bond_yields", 0) == 0
        assert result["counts"].get("bond_yield_observations", 0) == 0


# ---------------------------------------------------------------------------
# Write failures on a shared Postgres-style transaction
# ---------------------------------------------------------------------------


class _PgLikeSession:
    """Models the part of a Postgres transaction that matters for error recovery.

    A failed statement aborts the transaction: every later statement raises
    until the failure is rolled back, and a ``commit`` on an aborted
    transaction silently discards everything (Postgres turns it into a
    ROLLBACK). A SAVEPOINT scope confines the damage to its own writes.
    """

    def __init__(self, fail_on: set[str]) -> None:
        self.fail_on = fail_on
        self.pending: list[str] = []
        self.committed: list[str] = []
        self._aborted = False

    def write(self, label: str) -> int:
        if self._aborted:
            raise RuntimeError("InFailedSqlTransaction: transaction is aborted")
        if label in self.fail_on:
            self._aborted = True
            raise RuntimeError("DataError: value too long for type varchar(50)")
        self.pending.append(label)
        return 1

    @contextlib.contextmanager
    def begin_nested(self) -> Iterator[None]:
        mark = len(self.pending)
        try:
            yield
        except BaseException:
            del self.pending[mark:]
            self._aborted = False
            raise

    def commit(self) -> None:
        if not self._aborted:
            self.committed.extend(self.pending)
        self.pending.clear()
        self._aborted = False


class _PgLikeRepo:
    """Repository double whose every upsert is a write on a ``_PgLikeSession``."""

    def __init__(self, session: _PgLikeSession) -> None:
        self.session = session

    def upsert_economic_indicator(self, **_: object) -> int:
        return self.session.write("ilsole_forecast")

    def upsert_economic_indicator_observation(self, **_: object) -> int:
        return self.session.write("ilsole_observations")

    def upsert_te_indicators(self, **_: object) -> int:
        return self.session.write("te_indicators")

    def upsert_te_observations(self, **_: object) -> int:
        return self.session.write("te_observations")

    def upsert_bond_yields(self, **_: object) -> int:
        return self.session.write("bond_yields")

    def upsert_bond_yield_observations(self, **_: object) -> int:
        return self.session.write("bond_yield_observations")

    def get_fred_latest_date(self, series_id: str) -> None:
        return None

    def upsert_fred_observations(self, series_id: str, observations: list) -> int:
        return self.session.write(f"fred:{series_id}")


def _service_on_pg_like_repo(
    fail_on: set[str],
    te_response: dict | None = None,
    fred_scraper: MagicMock | None = None,
) -> tuple[MacroRegimeService, _PgLikeSession]:
    session = _PgLikeSession(fail_on)
    ilsole = MagicMock()
    ilsole.get_forecasts.return_value = _FORECAST_DATA
    te = MagicMock()
    te.get_country_indicators.return_value = te_response or _TE_SUCCESS
    service = MacroRegimeService(
        repo=_PgLikeRepo(session),  # type: ignore[arg-type]
        ilsole_scraper=ilsole,
        te_scraper=te,
        fred_scraper=fred_scraper or MagicMock(),
    )
    return service, session


class TestFetchCountryWriteFailureIsolation:
    def test_when_te_write_fails_then_earlier_ilsole_rows_survive_the_commit(
        self,
    ) -> None:
        service, session = _service_on_pg_like_repo({"te_indicators"})

        result = service.fetch_country("USA")
        session.commit()

        assert session.committed == ["ilsole_forecast", "ilsole_observations"]
        assert any(e.startswith("trading_economics") for e in result["errors"])

    def test_when_te_observation_write_fails_then_te_counts_report_nothing_stored(
        self,
    ) -> None:
        service, session = _service_on_pg_like_repo({"te_observations"})

        result = service.fetch_country("USA")
        session.commit()

        assert "te_indicators" not in session.committed
        assert result["counts"]["te_indicators"] == 0
        assert result["counts"]["te_observations"] == 0

    def test_when_bond_write_fails_then_te_indicators_are_still_committed(
        self,
    ) -> None:
        service, session = _service_on_pg_like_repo({"bond_yields"})

        result = service.fetch_country("USA")
        session.commit()

        assert "te_indicators" in session.committed
        assert "te_observations" in session.committed
        assert "bond_yields" not in session.committed
        assert result["counts"]["te_indicators"] == 1
        assert result["counts"]["bond_yields"] == 0


class TestFetchFredSeriesWriteFailureIsolation:
    @staticmethod
    def _fred_scraper() -> MagicMock:
        scraper = MagicMock()
        scraper.get_series_observations.return_value = {
            "status": "success",
            "observations": [{"date": None, "value": 1.0}],
        }
        return scraper

    def test_when_one_series_write_fails_then_other_series_are_committed(
        self,
    ) -> None:
        service, session = _service_on_pg_like_repo(
            {"fred:VIXCLS"}, fred_scraper=self._fred_scraper()
        )

        result = service.fetch_fred_series(series_ids=["DGS2", "VIXCLS", "DGS10"])
        session.commit()

        assert session.committed == ["fred:DGS2", "fred:DGS10"]
        assert result["counts"] == {"DGS2": 1, "VIXCLS": 0, "DGS10": 1}

    def test_when_one_series_write_fails_then_only_that_series_is_an_error(
        self,
    ) -> None:
        service, _ = _service_on_pg_like_repo(
            {"fred:VIXCLS"}, fred_scraper=self._fred_scraper()
        )

        result = service.fetch_fred_series(series_ids=["DGS2", "VIXCLS", "DGS10"])

        assert [e.split(":")[0] for e in result["errors"]] == ["VIXCLS"]


# ---------------------------------------------------------------------------
# fetch_fred_series — restated observations
# ---------------------------------------------------------------------------


class _FredStore:
    """Repository double with the real upsert contract: ``value`` is overwritten."""

    def __init__(self) -> None:
        self.session = MagicMock()
        self.rows: dict[tuple[str, datetime.date], float | None] = {}

    def get_fred_latest_date(self, series_id: str) -> datetime.date | None:
        dates = [d for (sid, d) in self.rows if sid == series_id]
        return max(dates) if dates else None

    def upsert_fred_observations(self, series_id: str, observations: list) -> int:
        for obs in observations:
            self.rows[(series_id, obs["date"])] = obs["value"]
        return len(observations)


class _FredSource:
    """FRED double: serves the current vintage, honouring ``observation_start``."""

    def __init__(self, truth: dict[datetime.date, float | None]) -> None:
        self.truth = truth

    def get_series_observations(
        self, series_id: str, start_date: str | None = None
    ) -> dict:
        floor = datetime.date.fromisoformat(start_date) if start_date else None
        observations = [
            {"date": d, "value": v}
            for d, v in sorted(self.truth.items())
            if floor is None or d >= floor
        ]
        return {"status": "success", "observations": observations}


class TestFetchFredSeriesRestatements:
    _Q2 = datetime.date(2026, 4, 1)
    _Q3 = datetime.date(2026, 7, 1)

    def _service(self, store: _FredStore, source: _FredSource) -> MacroRegimeService:
        return MacroRegimeService(
            repo=store,  # type: ignore[arg-type]
            ilsole_scraper=MagicMock(),
            te_scraper=MagicMock(),
            fred_scraper=source,  # type: ignore[arg-type]
        )

    def test_when_latest_print_is_revised_then_incremental_run_stores_revision(
        self,
    ) -> None:
        store = _FredStore()
        source = _FredSource({self._Q2: 3.8, self._Q3: 2.9})
        service = self._service(store, source)
        service.fetch_fred_series(series_ids=["A191RL1Q225SBEA"])

        source.truth = {self._Q2: 3.5, self._Q3: 4.3}
        service.fetch_fred_series(series_ids=["A191RL1Q225SBEA"])

        assert store.rows[("A191RL1Q225SBEA", self._Q3)] == 4.3
        assert store.rows[("A191RL1Q225SBEA", self._Q2)] == 3.5

    def test_when_last_observation_was_missing_then_later_run_fills_it_in(
        self,
    ) -> None:
        store = _FredStore()
        source = _FredSource({self._Q2: 3.8, self._Q3: None})
        service = self._service(store, source)
        service.fetch_fred_series(series_ids=["USREC"])

        source.truth = {self._Q2: 3.8, self._Q3: 1.0}
        service.fetch_fred_series(series_ids=["USREC"])

        assert store.rows[("USREC", self._Q3)] == 1.0

    def test_when_incremental_then_request_starts_before_latest_stored_date(
        self,
    ) -> None:
        store = _FredStore()
        store.rows[("DGS10", datetime.date(2026, 7, 1))] = 4.1
        source = MagicMock()
        source.get_series_observations.return_value = {
            "status": "success",
            "observations": [],
        }
        service = self._service(store, source)

        service.fetch_fred_series(series_ids=["DGS10"])

        start = source.get_series_observations.call_args.kwargs["start_date"]
        assert datetime.date.fromisoformat(start) < datetime.date(2026, 7, 1)


# ---------------------------------------------------------------------------
# fetch_country — te non-success status
# ---------------------------------------------------------------------------


class TestFetchCountryTeNonSuccess:
    def test_when_te_status_error_then_errors_contains_trading_economics(
        self,
    ) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=_TE_FAILED,
        )

        result = service.fetch_country("USA")

        assert any("trading_economics" in e for e in result["errors"])

    def test_when_te_status_error_then_te_counts_zeroed(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=_TE_FAILED,
        )

        result = service.fetch_country("USA")

        assert result["counts"]["te_indicators"] == 0
        assert result["counts"]["te_observations"] == 0
        assert result["counts"]["bond_yields"] == 0
        assert result["counts"]["bond_yield_observations"] == 0

    def test_when_te_status_error_then_no_exception_propagates(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=_TE_FAILED,
        )

        result = service.fetch_country("USA")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# fetch_country — te indicators ok but bond-yield page failed
# ---------------------------------------------------------------------------


class TestFetchCountryBondYieldFailure:
    _TE_BONDS_FAILED = {
        "status": "success",
        "indicators": {"inflation_rate": 3.1},
        "bond_yields": {},
        "bond_yields_error": "Parse structure failure: bond table not found",
    }

    def test_when_bond_page_fails_then_error_names_the_bond_source(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=self._TE_BONDS_FAILED,
        )

        result = service.fetch_country("USA", include_bonds=True)

        assert any(
            e.startswith("trading_economics_bonds") and "bond table not found" in e
            for e in result["errors"]
        )

    def test_when_bond_page_fails_then_indicators_are_still_stored(self) -> None:
        service, mock_repo = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=self._TE_BONDS_FAILED,
        )

        result = service.fetch_country("USA", include_bonds=True)

        mock_repo.upsert_te_indicators.assert_called_once()
        mock_repo.upsert_bond_yields.assert_not_called()
        assert result["counts"]["te_indicators"] == 3
        assert result["counts"]["bond_yields"] == 0

    def test_when_bonds_not_requested_then_no_bond_error_reported(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=self._TE_BONDS_FAILED,
        )

        result = service.fetch_country("USA", include_bonds=False)

        assert result["errors"] == []


# ---------------------------------------------------------------------------
# run_bulk_macro_fetch — progress and session.commit contract
# ---------------------------------------------------------------------------


class TestRunBulkMacroFetch:
    def _run_bulk(
        self,
        countries: list[str],
        on_progress: MagicMock | None = None,
    ) -> tuple[MagicMock, MagicMock]:
        """Run run_bulk_macro_fetch with database_manager patched at source."""
        request = SimpleNamespace(countries=countries, include_bonds=True)
        mock_session = MagicMock()
        mock_svc_cls = MagicMock()
        mock_svc_cls.return_value.fetch_country.return_value = {
            "counts": {"te_indicators": 2},
            "errors": [],
        }

        with (
            patch("app.database.database_manager") as mock_db,
            patch("app.services.macro.macro_regime_service.MacroRegimeRepository"),
            patch(
                "app.services.macro.macro_regime_service.MacroRegimeService",
                mock_svc_cls,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            kwargs: dict = {}
            if on_progress is not None:
                kwargs["on_progress"] = on_progress
            run_bulk_macro_fetch(request, **kwargs)

        return mock_session, on_progress or MagicMock()

    def test_when_two_countries_then_on_progress_total_equals_two(self) -> None:
        on_progress = MagicMock()
        self._run_bulk(["USA", "Germany"], on_progress=on_progress)

        first_call_kwargs = on_progress.call_args_list[0].kwargs
        assert first_call_kwargs.get("total") == 2

    def test_when_two_countries_then_on_progress_fired_for_each_country(
        self,
    ) -> None:
        on_progress = MagicMock()
        self._run_bulk(["USA", "Germany"], on_progress=on_progress)

        all_kwargs = [c.kwargs for c in on_progress.call_args_list]
        countries_seen = [kw.get("current_country") for kw in all_kwargs]
        assert "USA" in countries_seen
        assert "Germany" in countries_seen

    def test_when_two_countries_then_session_commit_called_twice(self) -> None:
        mock_session, _ = self._run_bulk(["USA", "Germany"])
        assert mock_session.commit.call_count == 2

    def test_when_no_errors_then_final_progress_status_is_completed(self) -> None:
        on_progress = MagicMock()
        self._run_bulk(["USA", "Germany"], on_progress=on_progress)

        # The final on_progress call carries status
        final_kwargs = on_progress.call_args_list[-1].kwargs
        assert final_kwargs.get("status") == "completed"
