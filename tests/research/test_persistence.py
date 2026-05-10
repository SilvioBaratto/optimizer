"""Tests for ``research/_persistence.py`` — issue #547.

Covers:
- ``_diff_from_default``: dataclass diff vs ``MeanRiskConfig()`` defaults.
- ``_flatten_metrics``: dotted-key flattening of nested metric dicts.
- ``persist_research_run``: full DB write-back with idempotent upsert.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date

import pytest
from app.models._shared.base import Base
from app.models.portfolio import (
    Portfolio,
    PortfolioSnapshot,
    SnapshotOptimizerParam,
    SnapshotSummaryEntry,
    SnapshotWeight,
)
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from optimizer.optimization import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
)
from research._persistence import (
    _diff_from_default,
    _flatten_metrics,
    persist_research_run,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture()
def session_factory(engine):
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    @contextmanager
    def factory() -> Generator[Session, None, None]:
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return factory


@pytest.fixture()
def metrics() -> dict[str, dict[str, float]]:
    return {
        "gross": {"sharpe": 1.5, "vol": 0.2, "max_dd": -0.15},
        "net": {"sharpe": 1.3, "vol": 0.2},
        "after_tax": {"sharpe": 1.0},
    }


@pytest.fixture()
def weights() -> dict[str, float]:
    return {"AAPL": 0.4, "MSFT": 0.35, "GOOG": 0.25}


@pytest.fixture()
def optimizer_cfg() -> MeanRiskConfig:
    return MeanRiskConfig(
        objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
        risk_measure=RiskMeasureType.VARIANCE,
        max_weights=0.10,
        l2_coef=0.05,
    )


@pytest.fixture()
def sector_mapping() -> dict[str, str]:
    return {"AAPL": "Tech", "MSFT": "Tech", "GOOG": "Comms"}


# ---------------------------------------------------------------------------
# _diff_from_default
# ---------------------------------------------------------------------------


class TestDiffFromDefault:
    def test_when_default_config_returns_empty(self) -> None:
        assert _diff_from_default(MeanRiskConfig()) == {}

    def test_when_max_weights_changed_only_that_field_returned(self) -> None:
        cfg = MeanRiskConfig(max_weights=0.10)
        diff = _diff_from_default(cfg)
        assert set(diff.keys()) == {"max_weights"}
        assert diff["max_weights"] == 0.10

    def test_when_multiple_fields_changed_all_returned(self) -> None:
        cfg = MeanRiskConfig(
            objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
            max_weights=0.10,
            l2_coef=0.05,
        )
        diff = _diff_from_default(cfg)
        assert set(diff.keys()) == {"objective", "max_weights", "l2_coef"}


# ---------------------------------------------------------------------------
# _flatten_metrics
# ---------------------------------------------------------------------------


class TestFlattenMetrics:
    def test_when_nested_dict_keys_are_dotted(self) -> None:
        flat = _flatten_metrics(
            {"gross": {"sharpe": 1.5, "vol": 0.2}, "net": {"sharpe": 1.3}}
        )
        assert flat == {"gross.sharpe": 1.5, "gross.vol": 0.2, "net.sharpe": 1.3}

    def test_when_empty_dict_returns_empty(self) -> None:
        assert _flatten_metrics({}) == {}

    def test_when_single_level_passes_through(self) -> None:
        assert _flatten_metrics({"sharpe": 1.5, "vol": 0.2}) == {
            "sharpe": 1.5,
            "vol": 0.2,
        }

    def test_when_three_levels_deep_dotted(self) -> None:
        flat = _flatten_metrics({"a": {"b": {"c": 1}}})
        assert flat == {"a.b.c": 1}


# ---------------------------------------------------------------------------
# persist_research_run
# ---------------------------------------------------------------------------


class TestPersistResearchRun:
    def test_when_called_returns_portfolio_uuid(
        self,
        session_factory,
        metrics,
        weights,
        optimizer_cfg,
        sector_mapping,
    ) -> None:
        portfolio_id = persist_research_run(
            snapshot_date=date(2026, 4, 1),
            weights=weights,
            metrics=metrics,
            optimizer_cfg=optimizer_cfg,
            sector_mapping=sector_mapping,
            turnover=0.1,
            session_factory=session_factory,
        )
        assert isinstance(portfolio_id, uuid.UUID)

    def test_when_called_creates_snapshot_row(
        self,
        engine,
        session_factory,
        metrics,
        weights,
        optimizer_cfg,
        sector_mapping,
    ) -> None:
        persist_research_run(
            snapshot_date=date(2026, 4, 1),
            weights=weights,
            metrics=metrics,
            optimizer_cfg=optimizer_cfg,
            sector_mapping=sector_mapping,
            turnover=0.1,
            session_factory=session_factory,
        )
        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as s:
            count = s.execute(
                select(func.count()).select_from(PortfolioSnapshot)
            ).scalar_one()
            assert count == 1
            snap = s.execute(select(PortfolioSnapshot)).scalar_one()
            assert snap.snapshot_type == "research_run"
            assert snap.snapshot_date == date(2026, 4, 1)
            assert snap.turnover == 0.1
            assert snap.holding_count == 3

    def test_when_called_creates_one_weight_row_per_ticker(
        self,
        engine,
        session_factory,
        metrics,
        weights,
        optimizer_cfg,
        sector_mapping,
    ) -> None:
        persist_research_run(
            snapshot_date=date(2026, 4, 1),
            weights=weights,
            metrics=metrics,
            optimizer_cfg=optimizer_cfg,
            sector_mapping=sector_mapping,
            turnover=0.1,
            session_factory=session_factory,
        )
        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as s:
            count = s.execute(
                select(func.count()).select_from(SnapshotWeight)
            ).scalar_one()
            assert count == 3

    def test_when_called_writes_one_summary_per_leaf_metric(
        self,
        engine,
        session_factory,
        metrics,
        weights,
        optimizer_cfg,
        sector_mapping,
    ) -> None:
        persist_research_run(
            snapshot_date=date(2026, 4, 1),
            weights=weights,
            metrics=metrics,
            optimizer_cfg=optimizer_cfg,
            sector_mapping=sector_mapping,
            turnover=0.1,
            session_factory=session_factory,
        )
        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as s:
            keys = {
                row.key for row in s.execute(select(SnapshotSummaryEntry)).scalars()
            }
            assert keys == {
                "gross.sharpe",
                "gross.vol",
                "gross.max_dd",
                "net.sharpe",
                "net.vol",
                "after_tax.sharpe",
            }

    def test_when_called_writes_optimizer_param_only_for_diff(
        self,
        engine,
        session_factory,
        metrics,
        weights,
        optimizer_cfg,
        sector_mapping,
    ) -> None:
        persist_research_run(
            snapshot_date=date(2026, 4, 1),
            weights=weights,
            metrics=metrics,
            optimizer_cfg=optimizer_cfg,
            sector_mapping=sector_mapping,
            turnover=0.1,
            session_factory=session_factory,
        )
        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as s:
            keys = {
                row.key for row in s.execute(select(SnapshotOptimizerParam)).scalars()
            }
            assert keys == {"objective", "max_weights", "l2_coef"}

    def test_when_called_resolves_portfolio_by_name(
        self,
        engine,
        session_factory,
        metrics,
        weights,
        optimizer_cfg,
        sector_mapping,
    ) -> None:
        persist_research_run(
            snapshot_date=date(2026, 4, 1),
            weights=weights,
            metrics=metrics,
            optimizer_cfg=optimizer_cfg,
            sector_mapping=sector_mapping,
            turnover=0.1,
            session_factory=session_factory,
        )
        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as s:
            portfolios = s.execute(select(Portfolio)).scalars().all()
            assert len(portfolios) == 1
            assert portfolios[0].name == "research_run"

    def test_when_called_twice_one_snapshot_remains(
        self,
        engine,
        session_factory,
        metrics,
        weights,
        optimizer_cfg,
        sector_mapping,
    ) -> None:
        persist_research_run(
            snapshot_date=date(2026, 4, 1),
            weights=weights,
            metrics=metrics,
            optimizer_cfg=optimizer_cfg,
            sector_mapping=sector_mapping,
            turnover=0.1,
            session_factory=session_factory,
        )
        new_metrics = {"gross": {"sharpe": 2.0}}
        persist_research_run(
            snapshot_date=date(2026, 4, 1),
            weights={"AAPL": 1.0},
            metrics=new_metrics,
            optimizer_cfg=optimizer_cfg,
            sector_mapping={"AAPL": "Tech"},
            turnover=0.0,
            session_factory=session_factory,
        )
        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as s:
            count = s.execute(
                select(func.count()).select_from(PortfolioSnapshot)
            ).scalar_one()
            assert count == 1

    def test_when_called_twice_summary_keys_reflect_latest_metrics(
        self,
        engine,
        session_factory,
        metrics,
        weights,
        optimizer_cfg,
        sector_mapping,
    ) -> None:
        persist_research_run(
            snapshot_date=date(2026, 4, 1),
            weights=weights,
            metrics=metrics,
            optimizer_cfg=optimizer_cfg,
            sector_mapping=sector_mapping,
            turnover=0.1,
            session_factory=session_factory,
        )
        persist_research_run(
            snapshot_date=date(2026, 4, 1),
            weights={"AAPL": 1.0},
            metrics={"gross": {"sharpe": 2.0}},
            optimizer_cfg=optimizer_cfg,
            sector_mapping={"AAPL": "Tech"},
            turnover=0.0,
            session_factory=session_factory,
        )
        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as s:
            keys = {
                row.key for row in s.execute(select(SnapshotSummaryEntry)).scalars()
            }
            assert keys == {"gross.sharpe"}
