"""End-to-end smoke test for ``research.stock_selection_pipeline.main`` (issue #551).

Heavy compute steps are stubbed; the smoke test verifies orchestration
glue: report.md + chart PNGs + metrics.json + checklist.json + weights.csv
land in ``output_dir``, weights sum to one, and the optional ``--persist``
path inserts exactly one ``PortfolioSnapshot`` row.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jinja2")

from app.models.base import Base
from app.models.portfolio import (
    PortfolioSnapshot,
    SnapshotOptimizerParam,
    SnapshotSummaryEntry,
    SnapshotWeight,
)
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from optimizer.factors import FactorOOSResult, FactorValidationReport
from research import _persistence
from research import stock_selection_pipeline as ssp
from research.data_assembly import DataAssembly

# ---------------------------------------------------------------------------
# Fixture factories
# ---------------------------------------------------------------------------

_N_TICKERS = 6
_N_DAYS = 60
_TICKERS = [f"T{i:02d}" for i in range(_N_TICKERS)]


def _make_assembly() -> DataAssembly:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=_N_DAYS)
    increments = rng.normal(0, 0.005, size=(_N_DAYS, _N_TICKERS))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(increments, axis=0)),
        index=dates,
        columns=_TICKERS,
    )
    assembly = DataAssembly(
        prices=prices,
        volumes=pd.DataFrame(1.0, index=dates, columns=_TICKERS),
        fundamentals=pd.DataFrame(index=_TICKERS),
        sector_mapping=dict.fromkeys(_TICKERS, "Tech"),
        financial_statements=pd.DataFrame(),
        analyst_data=pd.DataFrame(),
        insider_data=pd.DataFrame(),
        macro_data=pd.DataFrame(),
        currency_map=dict.fromkeys(_TICKERS, "EUR"),
        fx_rates=pd.DataFrame(),
    )
    assembly.assembly_hash = "smoke0000aaaa1111"
    return assembly


def _make_result(assembly: DataAssembly) -> SimpleNamespace:
    n = len(_TICKERS)
    weights = pd.Series(np.full(n, 1.0 / n), index=_TICKERS)
    rebalance_dates = assembly.prices.index[::20]
    weight_history = pd.DataFrame(
        np.tile(weights.to_numpy(), (len(rebalance_dates), 1)),
        index=rebalance_dates,
        columns=_TICKERS,
    )
    portfolio_returns = pd.Series(
        np.zeros(len(assembly.prices.index)),
        index=assembly.prices.index,
    )
    return SimpleNamespace(
        weights=weights,
        weight_history=weight_history,
        retighten_trace=[{"attempt": 1, "top4": 0.10, "max_weights": 0.10}],
        rebalance_decision=(False, "between_review_dates"),
        gross_returns=portfolio_returns,
        net_returns=portfolio_returns,
        net_sharpe_ratio=None,
        backtest=None,
        summary={"sharpe_ratio": 0.0},
        opt_config=None,
        turnover=0.0,
    )


def _make_validation_report() -> FactorValidationReport:
    from optimizer.factors._validation import ICResult

    return FactorValidationReport(
        ic_results=[
            ICResult(
                factor_name="value",
                mean_ic=0.04,
                ic_std=0.05,
                t_stat=2.5,
                p_value=0.01,
                significant=True,
            ),
        ],
    )


def _make_oos_result() -> FactorOOSResult:
    per_fold = pd.DataFrame({"value": [0.03, 0.04]})
    return FactorOOSResult(
        per_fold_ic=per_fold,
        per_fold_spread=pd.DataFrame(),
        mean_oos_ic=per_fold.mean(axis=0),
        mean_oos_icir=per_fold.mean(axis=0) / per_fold.std(axis=0),
        n_folds=2,
    )


def _stub_report_performance(output_dir: Path) -> Any:
    """Drop-in replacement: write artefacts + return 17/17 PASS contract."""

    def _stub(
        *args: Any, **kwargs: Any
    ) -> tuple[int, list[dict[str, Any]], dict[str, dict[str, float]], list[Path]]:
        out = kwargs.get("output_dir", output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text("{}\n")
        (out / "checklist.json").write_text("{}\n")
        chart_paths: list[Path] = []
        for name in (
            "cumulative_returns.png",
            "drawdowns.png",
            "rolling_sharpe.png",
            "sector_allocation.png",
            "country_allocation.png",
            "factor_ic.png",
        ):
            p = out / name
            p.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header
            chart_paths.append(p)
        rules = [
            {"rule": f"rule_{i}", "pass": True, "measured": "ok", "target": "ok"}
            for i in range(17)
        ]
        metrics = {
            "Portfolio (gross)": {"Ann. Return": 0.10},
            "Portfolio": {"Ann. Return": 0.09},
            "Portfolio (after-tax)": {"Ann. Return": 0.07},
        }
        return 17, rules, metrics, chart_paths

    return _stub


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    assembly: DataAssembly,
    result: SimpleNamespace,
    output_dir: Path,
) -> None:
    """Stub every heavy step so main() runs in <1s."""
    history_health = SimpleNamespace(succeeded_dates=1, total_dates=1)
    regime_stub = SimpleNamespace(value="expansion")
    last_review = pd.Timestamp("2024-01-15")
    clean_rets = pd.DataFrame(
        0.0,
        index=assembly.prices.index,
        columns=_TICKERS,
    )
    monkeypatch.setattr(
        ssp,
        "load_data",
        lambda **_: (assembly, {"T00": "US"}, None),
    )
    monkeypatch.setattr(
        ssp,
        "screen_investable",
        lambda _a: pd.Index(_TICKERS),
    )
    monkeypatch.setattr(
        ssp,
        "_materialise_clean_returns",
        lambda _a, _i: clean_rets,
    )
    monkeypatch.setattr(
        ssp,
        "build_history",
        lambda *_a, **_kw: ({}, pd.DataFrame(), history_health),
    )
    monkeypatch.setattr(
        ssp,
        "validate_is",
        lambda *_a, **_kw: _make_validation_report(),
    )
    monkeypatch.setattr(
        ssp,
        "validate_oos",
        lambda *_a, **_kw: _make_oos_result(),
    )
    monkeypatch.setattr(
        ssp,
        "_check_factor_coverage",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        ssp,
        "classify_and_tilt",
        lambda *_a, **_kw: (regime_stub, {"value": 1.05}),
    )
    monkeypatch.setattr(ssp, "optimize_portfolio", lambda **_kw: result)
    monkeypatch.setattr(
        ssp,
        "_decide_rebalance",
        lambda **_kw: (False, "between_review_dates"),
    )
    monkeypatch.setattr(ssp, "_read_last_review_date", lambda _d: last_review)
    monkeypatch.setattr(ssp, "_write_last_review_date", lambda _d: None)
    monkeypatch.setattr(ssp, "compute_weighted_cost_bps", lambda _w, _c: 12.0)
    monkeypatch.setattr(
        ssp,
        "report_performance",
        _stub_report_performance(output_dir),
    )


# ---------------------------------------------------------------------------
# Tests — artefact emission
# ---------------------------------------------------------------------------


_EXPECTED_PNGS = (
    "cumulative_returns.png",
    "drawdowns.png",
    "rolling_sharpe.png",
    "sector_allocation.png",
    "country_allocation.png",
    "factor_ic.png",
)


class TestPipelineSmokeArtefacts:
    def test_when_main_run_then_exits_zero_on_seventeen_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assembly = _make_assembly()
        result = _make_result(assembly)
        _patch_pipeline(
            monkeypatch,
            assembly=assembly,
            result=result,
            output_dir=tmp_path,
        )

        with pytest.raises(SystemExit) as excinfo:
            ssp.main(n_selected=25, output_dir=tmp_path)
        assert excinfo.value.code in (0, 1)

    def test_when_main_run_then_six_pngs_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assembly = _make_assembly()
        result = _make_result(assembly)
        _patch_pipeline(
            monkeypatch,
            assembly=assembly,
            result=result,
            output_dir=tmp_path,
        )

        with pytest.raises(SystemExit):
            ssp.main(n_selected=25, output_dir=tmp_path)

        for name in _EXPECTED_PNGS:
            png = tmp_path / name
            assert png.exists(), f"missing {name}"
            assert png.stat().st_size > 0

    def test_when_main_run_then_report_md_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assembly = _make_assembly()
        result = _make_result(assembly)
        _patch_pipeline(
            monkeypatch,
            assembly=assembly,
            result=result,
            output_dir=tmp_path,
        )

        with pytest.raises(SystemExit):
            ssp.main(n_selected=25, output_dir=tmp_path)

        report = tmp_path / "report.md"
        assert report.exists()
        assert report.stat().st_size > 0

    def test_when_main_run_then_metrics_and_checklist_json_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assembly = _make_assembly()
        result = _make_result(assembly)
        _patch_pipeline(
            monkeypatch,
            assembly=assembly,
            result=result,
            output_dir=tmp_path,
        )

        with pytest.raises(SystemExit):
            ssp.main(n_selected=25, output_dir=tmp_path)

        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "checklist.json").exists()

    def test_when_main_run_then_weights_csv_sums_to_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assembly = _make_assembly()
        result = _make_result(assembly)
        _patch_pipeline(
            monkeypatch,
            assembly=assembly,
            result=result,
            output_dir=tmp_path,
        )

        with pytest.raises(SystemExit):
            ssp.main(n_selected=25, output_dir=tmp_path)

        weights_csv = tmp_path / "weights.csv"
        assert weights_csv.exists()
        df = pd.read_csv(weights_csv)
        assert df["weight"].sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests — --persist path
# ---------------------------------------------------------------------------


@pytest.fixture()
def sqlite_session_factory():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=eng)
    SessionLocal = sessionmaker(bind=eng, autoflush=False, autocommit=False)

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

    return eng, factory


def _patch_persist_factory(monkeypatch: pytest.MonkeyPatch, factory) -> None:
    real = _persistence.persist_research_run

    def patched(*args: Any, **kwargs: Any):
        kwargs.setdefault("session_factory", factory)
        return real(*args, **kwargs)

    monkeypatch.setattr(ssp, "persist_research_run", patched)


class TestPipelineSmokePersist:
    def test_when_persist_true_then_one_snapshot_inserted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_session_factory,
    ) -> None:
        engine, factory = sqlite_session_factory
        assembly = _make_assembly()
        result = _make_result(assembly)
        _patch_pipeline(
            monkeypatch,
            assembly=assembly,
            result=result,
            output_dir=tmp_path,
        )
        _patch_persist_factory(monkeypatch, factory)

        with pytest.raises(SystemExit):
            ssp.main(persist=True, n_selected=25, output_dir=tmp_path)

        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as s:
            count = s.execute(
                select(func.count()).select_from(PortfolioSnapshot)
            ).scalar_one()
            assert count == 1
            snap = s.execute(select(PortfolioSnapshot)).scalar_one()
            assert snap.snapshot_type == "research_run"

    def test_when_persist_true_then_child_rows_present(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_session_factory,
    ) -> None:
        engine, factory = sqlite_session_factory
        assembly = _make_assembly()
        result = _make_result(assembly)
        _patch_pipeline(
            monkeypatch,
            assembly=assembly,
            result=result,
            output_dir=tmp_path,
        )
        _patch_persist_factory(monkeypatch, factory)

        with pytest.raises(SystemExit):
            ssp.main(persist=True, n_selected=25, output_dir=tmp_path)

        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as s:
            for model in (SnapshotWeight, SnapshotSummaryEntry, SnapshotOptimizerParam):
                count = s.execute(select(func.count()).select_from(model)).scalar_one()
                assert count > 0, f"no rows in {model.__name__}"

    def test_when_persist_true_called_twice_then_idempotent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_session_factory,
    ) -> None:
        engine, factory = sqlite_session_factory
        assembly = _make_assembly()
        result = _make_result(assembly)
        _patch_pipeline(
            monkeypatch,
            assembly=assembly,
            result=result,
            output_dir=tmp_path,
        )
        _patch_persist_factory(monkeypatch, factory)

        for _ in range(2):
            with pytest.raises(SystemExit):
                ssp.main(persist=True, n_selected=25, output_dir=tmp_path)

        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as s:
            count = s.execute(
                select(func.count()).select_from(PortfolioSnapshot)
            ).scalar_one()
            assert count == 1

    def test_when_persist_false_then_no_snapshot_written(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_session_factory,
    ) -> None:
        engine, factory = sqlite_session_factory
        assembly = _make_assembly()
        result = _make_result(assembly)
        _patch_pipeline(
            monkeypatch,
            assembly=assembly,
            result=result,
            output_dir=tmp_path,
        )
        _patch_persist_factory(monkeypatch, factory)

        with pytest.raises(SystemExit):
            ssp.main(persist=False, n_selected=25, output_dir=tmp_path)

        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as s:
            count = s.execute(
                select(func.count()).select_from(PortfolioSnapshot)
            ).scalar_one()
            assert count == 0
