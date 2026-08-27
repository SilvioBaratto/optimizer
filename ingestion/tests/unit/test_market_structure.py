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
from portopt_db.repositories.market_data.market_structure_repository import (
    MarketStructureRepository,
)

from app.services.market_data.market_structure_service import (
    run_market_structure_fetch,
)
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
