"""Tests for ``research/_report.py`` — issue #549 (Jinja report renderer)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import FactorValidationReport
from optimizer.factors._validation import ICResult
from research._report import compute_binding_constraints, render_report

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def regime() -> str:
    return "expansion"


@pytest.fixture()
def tilts() -> dict[str, float]:
    return {"value": 1.10, "momentum": 0.95, "quality": 1.00}


@pytest.fixture()
def validation_report() -> FactorValidationReport:
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
            ICResult(
                factor_name="momentum",
                mean_ic=0.06,
                ic_std=0.04,
                t_stat=3.1,
                p_value=0.001,
                significant=True,
            ),
        ]
    )


@pytest.fixture()
def oos_per_fold_ic() -> pd.DataFrame:
    return pd.DataFrame({"value": [0.03, 0.025, 0.04], "momentum": [0.05, 0.07, 0.04]})


@pytest.fixture()
def optimizer_diff() -> dict[str, object]:
    return {"max_weights": 0.10, "l2_coef": 0.05}


@pytest.fixture()
def binding_constraints() -> list[str]:
    return ["region:Europe ≤ 0.60 (binding, w=0.5998)"]


@pytest.fixture()
def retighten_trace() -> list[dict[str, float | int]]:
    return [
        {"attempt": 1, "top4": 0.32, "max_weights": 0.10},
        {"attempt": 2, "top4": 0.29, "max_weights": 0.095},
    ]


@pytest.fixture()
def rebalance_decision() -> tuple[bool, str]:
    return (True, "review_date_drift_5pct")


@pytest.fixture()
def checklist_rows() -> list[dict[str, object]]:
    return [
        {"rule": f"rule_{i}", "pass": True, "measured": 0.5, "target": 0.4}
        for i in range(17)
    ]


@pytest.fixture()
def metrics() -> dict[str, dict[str, float]]:
    return {
        "Portfolio (gross)": {
            "Ann. Return": 0.12,
            "Ann. Vol": 0.18,
            "Sharpe (rf)": 0.65,
            "Sortino": 0.85,
            "Info Ratio": 0.50,
            "Downside Vol": 0.12,
            "Max Drawdown": -0.15,
        },
        "Portfolio": {
            "Ann. Return": 0.10,
            "Ann. Vol": 0.18,
            "Sharpe (rf)": 0.55,
            "Sortino": 0.72,
            "Info Ratio": 0.40,
            "Downside Vol": 0.12,
            "Max Drawdown": -0.16,
        },
        "Portfolio (after-tax)": {
            "Ann. Return": 0.08,
            "Ann. Vol": 0.18,
            "Sharpe (rf)": 0.45,
            "Sortino": 0.60,
            "Info Ratio": 0.30,
            "Downside Vol": 0.12,
            "Max Drawdown": -0.17,
        },
    }


@pytest.fixture()
def chart_paths(tmp_path: Path) -> list[Path]:
    return [
        tmp_path / "cumulative_returns.png",
        tmp_path / "drawdowns.png",
        tmp_path / "rolling_sharpe.png",
        tmp_path / "sector_allocation.png",
        tmp_path / "country_allocation.png",
        tmp_path / "factor_ic.png",
    ]


@pytest.fixture()
def assembly_hash() -> str:
    return "deadbeef12345678"


# ---------------------------------------------------------------------------
# compute_binding_constraints
# ---------------------------------------------------------------------------


class TestComputeBindingConstraints:
    def test_when_row_tight_label_returned(self) -> None:
        A = np.array([[1.0, 1.0, 0.0]])
        b = np.array([0.6])
        w = np.array([0.30, 0.30, 0.40])  # 0.60 - 0.60 = 0.0 → tight
        result = compute_binding_constraints(A, b, w, labels=["region:Europe ≤ 0.60"])
        assert len(result) == 1
        assert "region:Europe" in result[0]

    def test_when_row_slack_no_label_returned(self) -> None:
        A = np.array([[1.0, 1.0, 0.0]])
        b = np.array([0.6])
        w = np.array([0.10, 0.10, 0.80])  # 0.20 - 0.60 = -0.40 → slack
        assert (
            compute_binding_constraints(A, b, w, labels=["region:Europe ≤ 0.60"]) == []
        )

    def test_when_some_rows_tight_only_those_returned(self) -> None:
        A = np.array(
            [
                [1.0, 1.0, 0.0],  # Europe
                [0.0, 0.0, 1.0],  # Asia
            ]
        )
        b = np.array([0.6, 0.4])
        w = np.array([0.30, 0.30, 0.10])  # Europe tight, Asia slack
        result = compute_binding_constraints(
            A, b, w, labels=["region:Europe ≤ 0.60", "region:Asia ≤ 0.40"]
        )
        assert len(result) == 1
        assert "Europe" in result[0]

    def test_when_within_tolerance_treated_as_tight(self) -> None:
        A = np.array([[1.0, 1.0]])
        b = np.array([0.6])
        w = np.array([0.30, 0.29995])  # diff -5e-5 inside tol 1e-4
        result = compute_binding_constraints(
            A,
            b,
            w,
            labels=["region:Europe ≤ 0.60"],
            tol=1e-4,
        )
        assert len(result) == 1

    def test_when_no_labels_supplied_default_row_labels_used(self) -> None:
        A = np.array([[1.0, 1.0]])
        b = np.array([0.6])
        w = np.array([0.30, 0.30])
        result = compute_binding_constraints(A, b, w)
        assert len(result) == 1
        assert "row_0" in result[0]


# ---------------------------------------------------------------------------
# render_report — full render
# ---------------------------------------------------------------------------


class TestRenderReportFull:
    def _render(self, tmp_path: Path, **kwargs):
        return render_report(output_dir=tmp_path, **kwargs)

    def _full_kwargs(
        self,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        retighten_trace,
        rebalance_decision,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ):
        return {
            "regime": regime,
            "tilts": tilts,
            "validation_report": validation_report,
            "oos_per_fold_ic": oos_per_fold_ic,
            "optimizer_diff": optimizer_diff,
            "binding_constraints": binding_constraints,
            "retighten_trace": retighten_trace,
            "rebalance_decision": rebalance_decision,
            "checklist_rows": checklist_rows,
            "metrics": metrics,
            "chart_paths": chart_paths,
            "assembly_hash": assembly_hash,
        }

    def test_when_called_returns_path_to_report_md(
        self,
        tmp_path: Path,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        retighten_trace,
        rebalance_decision,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ) -> None:
        path = self._render(
            tmp_path,
            **self._full_kwargs(
                regime,
                tilts,
                validation_report,
                oos_per_fold_ic,
                optimizer_diff,
                binding_constraints,
                retighten_trace,
                rebalance_decision,
                checklist_rows,
                metrics,
                chart_paths,
                assembly_hash,
            ),
        )
        assert path == tmp_path / "report.md"
        assert path.exists()

    def test_when_called_assembly_hash_present_in_header(
        self,
        tmp_path: Path,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        retighten_trace,
        rebalance_decision,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ) -> None:
        path = self._render(
            tmp_path,
            **self._full_kwargs(
                regime,
                tilts,
                validation_report,
                oos_per_fold_ic,
                optimizer_diff,
                binding_constraints,
                retighten_trace,
                rebalance_decision,
                checklist_rows,
                metrics,
                chart_paths,
                assembly_hash,
            ),
        )
        text = path.read_text()
        assert assembly_hash in text

    def test_when_called_regime_and_tilts_rendered(
        self,
        tmp_path: Path,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        retighten_trace,
        rebalance_decision,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ) -> None:
        path = self._render(
            tmp_path,
            **self._full_kwargs(
                regime,
                tilts,
                validation_report,
                oos_per_fold_ic,
                optimizer_diff,
                binding_constraints,
                retighten_trace,
                rebalance_decision,
                checklist_rows,
                metrics,
                chart_paths,
                assembly_hash,
            ),
        )
        text = path.read_text()
        assert "expansion" in text
        assert "value" in text
        assert "momentum" in text

    def test_when_called_factor_ic_table_renders_factors(
        self,
        tmp_path: Path,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        retighten_trace,
        rebalance_decision,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ) -> None:
        path = self._render(
            tmp_path,
            **self._full_kwargs(
                regime,
                tilts,
                validation_report,
                oos_per_fold_ic,
                optimizer_diff,
                binding_constraints,
                retighten_trace,
                rebalance_decision,
                checklist_rows,
                metrics,
                chart_paths,
                assembly_hash,
            ),
        )
        text = path.read_text()
        assert "value" in text
        assert "momentum" in text

    def test_when_called_optimizer_diff_keys_rendered(
        self,
        tmp_path: Path,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        retighten_trace,
        rebalance_decision,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ) -> None:
        path = self._render(
            tmp_path,
            **self._full_kwargs(
                regime,
                tilts,
                validation_report,
                oos_per_fold_ic,
                optimizer_diff,
                binding_constraints,
                retighten_trace,
                rebalance_decision,
                checklist_rows,
                metrics,
                chart_paths,
                assembly_hash,
            ),
        )
        text = path.read_text()
        assert "max_weights" in text
        assert "l2_coef" in text

    def test_when_called_binding_constraints_rendered(
        self,
        tmp_path: Path,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        retighten_trace,
        rebalance_decision,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ) -> None:
        path = self._render(
            tmp_path,
            **self._full_kwargs(
                regime,
                tilts,
                validation_report,
                oos_per_fold_ic,
                optimizer_diff,
                binding_constraints,
                retighten_trace,
                rebalance_decision,
                checklist_rows,
                metrics,
                chart_paths,
                assembly_hash,
            ),
        )
        text = path.read_text()
        assert "region:Europe" in text

    def test_when_called_retighten_trace_attempts_rendered(
        self,
        tmp_path: Path,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        retighten_trace,
        rebalance_decision,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ) -> None:
        path = self._render(
            tmp_path,
            **self._full_kwargs(
                regime,
                tilts,
                validation_report,
                oos_per_fold_ic,
                optimizer_diff,
                binding_constraints,
                retighten_trace,
                rebalance_decision,
                checklist_rows,
                metrics,
                chart_paths,
                assembly_hash,
            ),
        )
        text = path.read_text()
        assert "attempt" in text.lower()
        assert "0.32" in text

    def test_when_called_rebalance_reason_rendered(
        self,
        tmp_path: Path,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        retighten_trace,
        rebalance_decision,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ) -> None:
        path = self._render(
            tmp_path,
            **self._full_kwargs(
                regime,
                tilts,
                validation_report,
                oos_per_fold_ic,
                optimizer_diff,
                binding_constraints,
                retighten_trace,
                rebalance_decision,
                checklist_rows,
                metrics,
                chart_paths,
                assembly_hash,
            ),
        )
        text = path.read_text()
        assert "review_date_drift_5pct" in text

    def test_when_called_checklist_rows_rendered(
        self,
        tmp_path: Path,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        retighten_trace,
        rebalance_decision,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ) -> None:
        path = self._render(
            tmp_path,
            **self._full_kwargs(
                regime,
                tilts,
                validation_report,
                oos_per_fold_ic,
                optimizer_diff,
                binding_constraints,
                retighten_trace,
                rebalance_decision,
                checklist_rows,
                metrics,
                chart_paths,
                assembly_hash,
            ),
        )
        text = path.read_text()
        for i in range(17):
            assert f"rule_{i}" in text

    def test_when_called_metrics_three_groups_rendered(
        self,
        tmp_path: Path,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        retighten_trace,
        rebalance_decision,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ) -> None:
        path = self._render(
            tmp_path,
            **self._full_kwargs(
                regime,
                tilts,
                validation_report,
                oos_per_fold_ic,
                optimizer_diff,
                binding_constraints,
                retighten_trace,
                rebalance_decision,
                checklist_rows,
                metrics,
                chart_paths,
                assembly_hash,
            ),
        )
        text = path.read_text()
        assert "Portfolio (gross)" in text
        assert "Portfolio (after-tax)" in text

    def test_when_called_chart_links_rendered_relative(
        self,
        tmp_path: Path,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        retighten_trace,
        rebalance_decision,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ) -> None:
        path = self._render(
            tmp_path,
            **self._full_kwargs(
                regime,
                tilts,
                validation_report,
                oos_per_fold_ic,
                optimizer_diff,
                binding_constraints,
                retighten_trace,
                rebalance_decision,
                checklist_rows,
                metrics,
                chart_paths,
                assembly_hash,
            ),
        )
        text = path.read_text()
        assert "./cumulative_returns.png" in text
        assert "./factor_ic.png" in text


# ---------------------------------------------------------------------------
# render_report — edge cases
# ---------------------------------------------------------------------------


class TestRenderReportEdgeCases:
    def test_when_retighten_trace_empty_section_still_renders(
        self,
        tmp_path: Path,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        rebalance_decision,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ) -> None:
        path = render_report(
            output_dir=tmp_path,
            regime=regime,
            tilts=tilts,
            validation_report=validation_report,
            oos_per_fold_ic=oos_per_fold_ic,
            optimizer_diff=optimizer_diff,
            binding_constraints=binding_constraints,
            retighten_trace=[],
            rebalance_decision=rebalance_decision,
            checklist_rows=checklist_rows,
            metrics=metrics,
            chart_paths=chart_paths,
            assembly_hash=assembly_hash,
        )
        assert path.exists()
        text = path.read_text()
        assert "etighten" in text  # section heading still present

    def test_when_rebalance_false_between_review_dates_rendered(
        self,
        tmp_path: Path,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        retighten_trace,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ) -> None:
        path = render_report(
            output_dir=tmp_path,
            regime=regime,
            tilts=tilts,
            validation_report=validation_report,
            oos_per_fold_ic=oos_per_fold_ic,
            optimizer_diff=optimizer_diff,
            binding_constraints=binding_constraints,
            retighten_trace=retighten_trace,
            rebalance_decision=(False, "between_review_dates"),
            checklist_rows=checklist_rows,
            metrics=metrics,
            chart_paths=chart_paths,
            assembly_hash=assembly_hash,
        )
        text = path.read_text()
        assert "between_review_dates" in text

    def test_when_called_twice_overwrites_previous(
        self,
        tmp_path: Path,
        regime,
        tilts,
        validation_report,
        oos_per_fold_ic,
        optimizer_diff,
        binding_constraints,
        retighten_trace,
        rebalance_decision,
        checklist_rows,
        metrics,
        chart_paths,
        assembly_hash,
    ) -> None:
        kwargs: dict[str, Any] = {
            "output_dir": tmp_path,
            "regime": regime,
            "tilts": tilts,
            "validation_report": validation_report,
            "oos_per_fold_ic": oos_per_fold_ic,
            "optimizer_diff": optimizer_diff,
            "binding_constraints": binding_constraints,
            "retighten_trace": retighten_trace,
            "rebalance_decision": rebalance_decision,
            "checklist_rows": checklist_rows,
            "metrics": metrics,
            "chart_paths": chart_paths,
            "assembly_hash": assembly_hash,
        }
        render_report(**kwargs)
        kwargs2 = {**kwargs, "assembly_hash": "cafebabe87654321"}
        path = render_report(**kwargs2)
        text = path.read_text()
        assert "cafebabe87654321" in text
        assert "deadbeef12345678" not in text
