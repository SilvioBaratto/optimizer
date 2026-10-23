"""SPEC B1 — sector/industry market-structure ingestion.

Client parsing patches ``yf.Sector``; the repo is exercised against real SQLite;
the bulk service is driven with a mocked repo + session (its upsert is verified
by call, mirroring the other market-wide services).
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from portopt_db.repositories.market_data.market_structure_repository import (
    MarketStructureRepository,
)
from yfinance.data import YfData
from yfinance.exceptions import YFDataException, YFRateLimitError

from app.services.market_data.market_structure_service import (
    run_market_structure_fetch,
)
from app.services.market_data.yfinance import is_no_data_error
from app.services.market_data.yfinance.market.sectors import (
    SECTOR_KEYS,
    SectorsClient,
)

_AS_OF = dt.date(2026, 8, 27)


def _sector_client() -> SectorsClient:
    return SectorsClient(
        rate_limiter=MagicMock(),
        circuit_breaker=MagicMock(),
        default_max_retries=1,
    )


class TestSectorsClient:
    def test_fetch_sector_parses_overview_industries_and_companies(self) -> None:
        sec = SimpleNamespace(
            key="technology",
            name="Technology",
            symbol="^YH311",
            overview={
                "market_cap": 1.5e13,
                "market_weight": 0.31,
                "companies_count": 800,
                "industries_count": 12,
                "employee_count": 5_000_000,
            },
            industries=pd.DataFrame(
                {"name": ["Semiconductors", "Software"]},
                index=["semiconductors", "software-infrastructure"],
            ),
            top_companies=pd.DataFrame(
                {"name": ["Apple", "Microsoft"], "market weight": [0.07, 0.06]},
                index=["AAPL", "MSFT"],
            ),
        )
        client = _sector_client()
        with patch(
            "app.services.market_data.yfinance.market.sectors.yf.Sector",
            return_value=sec,
        ):
            out = client.fetch_sector("technology", region="US")

        assert out is not None
        assert out["name"] == "Technology"
        assert out["overview"]["companies_count"] == 800
        assert {i["key"] for i in out["industries"]} == {
            "semiconductors",
            "software-infrastructure",
        }
        assert {c["symbol"] for c in out["top_companies"]} == {"AAPL", "MSFT"}
        assert out["top_companies"][0]["weight"] == 0.07

    def test_swallowed_fetch_failure_is_raised_not_returned_as_a_hollow_sector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # yfinance (hide_exceptions=True) swallows the rate-limit error and leaves
        # every Sector field None; that must not come back as a truthy dict.
        monkeypatch.setattr(
            "app.services.infrastructure.retry.time.sleep", lambda _: None
        )
        client = SectorsClient(
            rate_limiter=MagicMock(),
            circuit_breaker=MagicMock(),
            default_max_retries=2,
        )
        with patch.object(
            YfData, "get_raw_json", side_effect=YFRateLimitError()
        ) as fetch:
            with pytest.raises(YFDataException):
                client.fetch_sector("technology", region="US")

        # The empty payload is a final answer, so the failed fetch is not retried:
        # one attempt reads the overview and probes the name, nothing more.
        assert fetch.call_count == 2
        client.circuit_breaker.reset.assert_not_called()


def _empty_sector() -> SimpleNamespace:
    """What ``yf.Sector`` exposes once yfinance has swallowed a failed fetch."""
    return SimpleNamespace(
        key=None,
        name=None,
        symbol=None,
        overview={},
        industries=pd.DataFrame(),
        top_companies=pd.DataFrame(),
    )


class TestSwallowedYahooFailure:
    def test_empty_sector_payload_raises_a_final_no_data_error(self) -> None:
        client = SectorsClient(
            rate_limiter=MagicMock(),
            circuit_breaker=MagicMock(),
            default_max_retries=3,
        )
        with patch(
            "app.services.market_data.yfinance.market.sectors.yf.Sector",
            return_value=_empty_sector(),
        ) as sector_cls:
            with pytest.raises(YFDataException) as excinfo:
                client.fetch_sector("technology", region="GB")

        assert is_no_data_error(excinfo.value)
        assert sector_cls.call_count == 1

    def test_sweep_writes_nothing_and_reports_the_failures(self) -> None:
        repo = MagicMock(name="repo")
        yf = SimpleNamespace(
            sectors=SectorsClient(
                rate_limiter=MagicMock(),
                circuit_breaker=MagicMock(),
                default_max_retries=1,
            )
        )
        with (
            patch(
                "portopt_db.repositories.market_data.market_structure_repository."
                "MarketStructureRepository",
                return_value=repo,
            ),
            patch("app.database.database_manager", _fake_dbm()),
            patch(
                "app.services.market_data.yfinance.market.sectors.yf.Sector",
                return_value=_empty_sector(),
            ),
        ):
            result = run_market_structure_fetch(yf, regions=("GB",))  # type: ignore[arg-type]

        repo.upsert_sector_snapshot.assert_not_called()
        assert result["sectors_written"] == 0
        assert result["error_count"] == len(SECTOR_KEYS)


class TestRepository:
    def test_upserts_are_idempotent(self, db_session) -> None:
        repo = MarketStructureRepository(db_session)
        n_ind = n_co = 0
        for _ in range(2):
            repo.upsert_sector_snapshot(
                "technology",
                "US",
                _AS_OF,
                name="Technology",
                symbol="^YH311",
                market_cap=1.5e13,
                market_weight=0.31,
                companies_count=800,
                industries_count=12,
                employee_count=5_000_000,
            )
            n_ind = repo.upsert_industries(
                "technology",
                "US",
                _AS_OF,
                [{"key": "semiconductors", "name": "Semiconductors"}],
            )
            n_co = repo.upsert_top_companies(
                "technology",
                "US",
                _AS_OF,
                [{"symbol": "AAPL", "name": "Apple", "weight": 0.07, "rating": "buy"}],
            )
        db_session.flush()

        assert n_ind == 1 and n_co == 1
        snap = repo.get_sector_snapshot("technology", "US")
        assert snap is not None and float(snap.market_weight) == 0.31
        assert repo.get_latest_sector_as_of("technology", "US") == _AS_OF


def _fake_dbm() -> MagicMock:
    @contextmanager
    def _cm():
        yield MagicMock(name="session")

    dbm = MagicMock()
    dbm.get_session = _cm
    return dbm


class TestBulkFetch:
    def test_sweeps_sectors_and_regions(self) -> None:
        repo = MagicMock(name="repo")
        repo.upsert_industries.return_value = 2
        repo.upsert_top_companies.return_value = 3
        yf = MagicMock()
        yf.sectors.fetch_sector.return_value = {
            "key": "technology",
            "name": "Technology",
            "symbol": "^YH311",
            "overview": {"market_cap": 1.0, "companies_count": 10},
            "industries": [{"key": "semiconductors", "name": "Semis"}],
            "top_companies": [{"symbol": "AAPL", "name": "Apple", "weight": 0.07}],
        }
        with (
            patch(
                "portopt_db.repositories.market_data.market_structure_repository."
                "MarketStructureRepository",
                return_value=repo,
            ),
            patch("app.database.database_manager", _fake_dbm()),
        ):
            result = run_market_structure_fetch(yf, regions=("US", "GB"))

        expected_calls = len(SECTOR_KEYS) * 2
        assert yf.sectors.fetch_sector.call_count == expected_calls
        assert result["sectors_written"] == expected_calls
        assert result["regions"] == ["US", "GB"]

    def test_bad_sector_is_isolated(self) -> None:
        repo = MagicMock(name="repo")
        repo.upsert_industries.return_value = 0
        repo.upsert_top_companies.return_value = 0
        yf = MagicMock()
        yf.sectors.fetch_sector.side_effect = RuntimeError("boom")
        with (
            patch(
                "portopt_db.repositories.market_data.market_structure_repository."
                "MarketStructureRepository",
                return_value=repo,
            ),
            patch("app.database.database_manager", _fake_dbm()),
        ):
            result = run_market_structure_fetch(yf, regions=("US",))

        assert result["sectors_written"] == 0
        assert result["error_count"] == len(SECTOR_KEYS)

    def test_sector_without_data_is_recorded_as_an_error_and_not_persisted(
        self,
    ) -> None:
        repo = MagicMock(name="repo")
        yf = MagicMock()
        yf.sectors.fetch_sector.return_value = None
        with (
            patch(
                "portopt_db.repositories.market_data.market_structure_repository."
                "MarketStructureRepository",
                return_value=repo,
            ),
            patch("app.database.database_manager", _fake_dbm()),
        ):
            result = run_market_structure_fetch(yf, regions=("US",))

        assert result["sectors_written"] == 0
        assert result["error_count"] == len(SECTOR_KEYS)
        repo.upsert_sector_snapshot.assert_not_called()


class TestFailedFetchAccounting:
    @staticmethod
    def _patches(repo: MagicMock):
        return (
            patch(
                "portopt_db.repositories.market_data.market_structure_repository."
                "MarketStructureRepository",
                return_value=repo,
            ),
            patch("app.database.database_manager", _fake_dbm()),
        )

    def test_none_from_the_client_is_counted_as_an_error(self) -> None:
        repo = MagicMock(name="repo")
        yf = MagicMock()
        yf.sectors.fetch_sector.return_value = None
        progress = MagicMock()
        repo_patch, dbm_patch = self._patches(repo)
        with repo_patch, dbm_patch:
            result = run_market_structure_fetch(
                yf, regions=("GB",), on_progress=progress
            )

        repo.upsert_sector_snapshot.assert_not_called()
        assert result["sectors_written"] == 0
        assert result["error_count"] == len(SECTOR_KEYS)
        errors = progress.call_args_list[-1].kwargs["errors"]
        assert "technology/GB: no data" in errors


class TestRolledBackPairsAreNotCounted:
    @staticmethod
    def _client() -> MagicMock:
        yf = MagicMock()
        yf.sectors.fetch_sector.return_value = {
            "key": "technology",
            "name": "Technology",
            "symbol": "^YH311",
            "overview": {"market_cap": 1.0},
            "industries": [{"key": "semiconductors", "name": "Semis"}],
            "top_companies": [{"symbol": "AAPL", "name": "Apple"}],
        }
        return yf

    def test_later_upsert_failure_leaves_every_counter_at_zero(self) -> None:
        repo = MagicMock(name="repo")
        repo.upsert_industries.return_value = 2
        repo.upsert_top_companies.side_effect = RuntimeError("constraint")
        with (
            patch(
                "portopt_db.repositories.market_data.market_structure_repository."
                "MarketStructureRepository",
                return_value=repo,
            ),
            patch("app.database.database_manager", _fake_dbm()),
        ):
            result = run_market_structure_fetch(self._client(), regions=("US",))

        assert result["error_count"] == len(SECTOR_KEYS)
        assert result["sectors_written"] == 0
        assert result["industries_written"] == 0
        assert result["top_companies_written"] == 0

    def test_commit_failure_leaves_every_counter_at_zero(self) -> None:
        repo = MagicMock(name="repo")
        repo.upsert_industries.return_value = 2
        repo.upsert_top_companies.return_value = 3
        session = MagicMock(name="session")
        session.commit.side_effect = RuntimeError("connection lost")

        @contextmanager
        def _cm():
            yield session

        dbm = MagicMock()
        dbm.get_session = _cm
        with (
            patch(
                "portopt_db.repositories.market_data.market_structure_repository."
                "MarketStructureRepository",
                return_value=repo,
            ),
            patch("app.database.database_manager", dbm),
        ):
            result = run_market_structure_fetch(self._client(), regions=("US",))

        assert result["error_count"] == len(SECTOR_KEYS)
        assert result["sectors_written"] == 0
        assert result["industries_written"] == 0
        assert result["top_companies_written"] == 0


class TestSectorOverviewScope:
    """Yahoo's overview figures are global, so only one region row may carry them."""

    @staticmethod
    def _sweep(regions: tuple[str, ...]) -> MagicMock:
        repo = MagicMock(name="repo")
        repo.upsert_industries.return_value = 1
        repo.upsert_top_companies.return_value = 1
        yf = MagicMock()
        yf.sectors.fetch_sector.return_value = {
            "key": "technology",
            "name": "Technology",
            "symbol": "^YH311",
            "overview": {
                "market_cap": 2.8e13,
                "market_weight": 0.32,
                "companies_count": 866,
                "industries_count": 12,
                "employee_count": 5_000_000,
            },
            "industries": [],
            "top_companies": [],
        }
        with (
            patch(
                "portopt_db.repositories.market_data.market_structure_repository."
                "MarketStructureRepository",
                return_value=repo,
            ),
            patch("app.database.database_manager", _fake_dbm()),
        ):
            run_market_structure_fetch(yf, regions=regions)
        return repo

    def test_overview_is_stored_once_not_under_every_region(self) -> None:
        repo = self._sweep(("US", "GB", "IT"))

        carriers = {
            call.args[1]
            for call in repo.upsert_sector_snapshot.call_args_list
            if call.kwargs["market_cap"] is not None
        }
        assert carriers == {"US"}

    def test_non_us_rows_keep_identity_but_drop_every_overview_column(self) -> None:
        repo = self._sweep(("GB",))

        kwargs = repo.upsert_sector_snapshot.call_args.kwargs
        assert kwargs["name"] == "Technology"
        assert kwargs["symbol"] == "^YH311"
        for column in (
            "market_cap",
            "market_weight",
            "companies_count",
            "industries_count",
            "employee_count",
        ):
            assert kwargs[column] is None

    def test_us_row_keeps_the_overview(self) -> None:
        repo = self._sweep(("US",))

        kwargs = repo.upsert_sector_snapshot.call_args.kwargs
        assert kwargs["market_cap"] == 2.8e13
        assert kwargs["companies_count"] == 866


class TestSchedulerStep:
    def test_composes_run_step(self) -> None:
        M = "app.services.jobs.scheduler"
        with (
            patch(f"{M}._run_step", return_value=True) as run_step,
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            ),
        ):
            from app.services.jobs.scheduler import (
                _market_structure_jobs,
                run_market_structure_step,
            )

            assert run_market_structure_step() is True

        assert run_step.call_args.args[0] == "market_structure"
        assert run_step.call_args.args[1] is _market_structure_jobs
