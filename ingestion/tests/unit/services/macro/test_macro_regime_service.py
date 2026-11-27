"""Unit tests for MacroRegimeService.fetch_fred_series.

Covers gaps NOT already in test_macro_regime_service_fred_key.py:
  - fetch_fred_series: one failing series write must not lose the other series
  - fetch_fred_series: restated observations are re-fetched and stored
"""

from __future__ import annotations

import contextlib
import datetime
from collections.abc import Iterator
from unittest.mock import MagicMock

from app.services.macro.macro_regime_service import MacroRegimeService

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

    def rollback(self) -> None:
        self.pending.clear()
        self._aborted = False


class _PgLikeRepo:
    """Repository double whose every upsert is a write on a ``_PgLikeSession``."""

    def __init__(self, session: _PgLikeSession) -> None:
        self.session = session

    def get_fred_latest_date(self, series_id: str) -> None:
        return None

    def upsert_fred_observations(self, series_id: str, observations: list) -> int:
        return self.session.write(f"fred:{series_id}")


def _service_on_pg_like_repo(
    fail_on: set[str],
    fred_scraper: MagicMock | None = None,
) -> tuple[MacroRegimeService, _PgLikeSession]:
    session = _PgLikeSession(fail_on)
    service = MacroRegimeService(
        repo=_PgLikeRepo(session),  # type: ignore[arg-type]
        fred_scraper=fred_scraper or MagicMock(),
    )
    return service, session


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
