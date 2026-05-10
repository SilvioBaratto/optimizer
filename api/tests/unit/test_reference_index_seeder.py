"""Progress-callback ordering contract for seed_reference_indices (#572).

Verifies the pre-loop ``on_progress(total=N)`` sentinel is emitted before
any per-ticker ``on_progress(current=...)`` call. No DB, no network.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from app.services import reference_index_seeder as mod


@contextmanager
def _fake_session_ctx(session: MagicMock):
    yield session


def _build_session_with_nyse(nyse_id: int | None = 1) -> MagicMock:
    session = MagicMock()
    session.execute.return_value.scalar_one_or_none.return_value = nyse_id
    return session


def _build_repo_and_service() -> tuple[MagicMock, MagicMock]:
    repo = MagicMock()
    instrument = MagicMock()
    instrument.id = 42
    repo.get_instrument_by_yfinance_ticker.return_value = instrument

    service = MagicMock()
    service.fetch_and_store.return_value = {"counts": {}, "errors": []}
    return repo, service


def _index_of_first_call_with(spy: MagicMock, key: str) -> int:
    for i, call in enumerate(spy.call_args_list):
        if key in call.kwargs:
            return i
    raise AssertionError(f"no call with kwarg {key!r}")


class TestSeedReferenceIndicesProgressOrdering:
    def test_when_two_tickers_seeded_total_call_precedes_first_current_call(
        self,
    ) -> None:
        spy = MagicMock()
        session = _build_session_with_nyse()
        repo, service = _build_repo_and_service()

        with patch("app.database.database_manager") as db_mgr, patch.object(
            mod, "YFinanceRepository", return_value=repo
        ), patch.object(mod, "YFinanceDataService", return_value=service):
            db_mgr.get_session.return_value = _fake_session_ctx(session)

            mod.seed_reference_indices(
                ["SPY", "QQQ"], yf_client=MagicMock(), on_progress=spy
            )

        total_idx = _index_of_first_call_with(spy, "total")
        current_idx = _index_of_first_call_with(spy, "current")
        assert total_idx < current_idx

    def test_when_two_tickers_seeded_total_kwarg_equals_len_tickers(self) -> None:
        spy = MagicMock()
        session = _build_session_with_nyse()
        repo, service = _build_repo_and_service()

        with patch("app.database.database_manager") as db_mgr, patch.object(
            mod, "YFinanceRepository", return_value=repo
        ), patch.object(mod, "YFinanceDataService", return_value=service):
            db_mgr.get_session.return_value = _fake_session_ctx(session)

            mod.seed_reference_indices(
                ["SPY", "QQQ"], yf_client=MagicMock(), on_progress=spy
            )

        total_idx = _index_of_first_call_with(spy, "total")
        assert spy.call_args_list[total_idx].kwargs["total"] == 2

    def test_when_nyse_missing_no_total_only_call_emitted(self) -> None:
        """Failed NYSE lookup must not emit a misleading total sentinel."""
        spy = MagicMock()
        session = _build_session_with_nyse(nyse_id=None)
        repo, service = _build_repo_and_service()

        with patch("app.database.database_manager") as db_mgr, patch.object(
            mod, "YFinanceRepository", return_value=repo
        ), patch.object(mod, "YFinanceDataService", return_value=service):
            db_mgr.get_session.return_value = _fake_session_ctx(session)

            mod.seed_reference_indices(
                ["SPY", "QQQ"], yf_client=MagicMock(), on_progress=spy
            )

        total_only_calls = [
            c for c in spy.call_args_list
            if set(c.kwargs.keys()) == {"total"}
        ]
        assert total_only_calls == []
