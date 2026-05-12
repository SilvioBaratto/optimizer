"""End-to-end pipeline parity test vs pre-migration baseline (issue #688).

Terminal gate (always runs, no DB needed):
  TestTerminalGateOutputOnFailure — asserts failure table emitted when a rule fails.

Parity (@pytest.mark.slow, skipped when OPTIMIZER_SKIP_E2E=1 or baseline missing):
  TestPipelineParity — runs research.cli.main with fixed historical window,
  diffs artifacts against baseline.

  Binary (PNG/SVG): SHA-256 comparison.
  Numeric JSON (metrics.json): element-wise absolute tolerance 1e-12.
  CSV (weights.csv): ticker order + numpy allclose(atol=1e-12).
  Text (report.md): tokenized diff ignoring lines containing ISO dates.
  Exit code stored in baseline/.exit_code.

To capture the baseline:
  MPLBACKEND=Agg PYTHONHASHSEED=0 \\
    python tests/research/baselines/cycle0/capture_baseline.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from rich.console import Console as RichConsole

import research.pipeline._checklist as _checklist_mod
from research.pipeline._checklist import (
    _CHECKLIST_TOTAL,
    _apply_terminal_gate,
    _rule,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASELINE_DIR = Path(__file__).parent / "baselines" / "cycle0"
_SKIP_E2E: bool = os.environ.get("OPTIMIZER_SKIP_E2E") == "1"

_PARITY_START = datetime.date(2022, 1, 1)
_PARITY_END = datetime.date(2023, 12, 31)
_PARITY_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare_json_values(
    actual: Any,
    baseline: Any,
    *,
    atol: float = 1e-12,
    path: str = "root",
) -> None:
    if isinstance(baseline, dict):
        for key in baseline:
            assert key in actual, f"JSON key '{path}.{key}' missing from actual output"
            _compare_json_values(
                actual[key], baseline[key], atol=atol, path=f"{path}.{key}"
            )
    elif isinstance(baseline, (int, float)) and not isinstance(baseline, bool):
        if baseline is None or actual is None:
            assert actual == baseline, f"{path}: {actual!r} != {baseline!r}"
        else:
            assert abs(float(actual) - float(baseline)) <= atol, (
                f"{path}: |{actual} - {baseline}| = "
                f"{abs(float(actual) - float(baseline))} > {atol}"
            )
    else:
        assert actual == baseline, f"{path}: {actual!r} != {baseline!r}"


def _strip_dates(text: str) -> str:
    """Remove ISO-date tokens so report.md diffs ignore run-date lines."""
    return re.sub(r"\d{4}-\d{2}-\d{2}", "DATE", text)


def _all_pass_rules() -> list[dict[str, Any]]:
    return [
        _rule(f"Rule {i}", ok=True, measured="ok", target="ok")
        for i in range(1, _CHECKLIST_TOTAL + 1)
    ]


def _one_fail_rules() -> list[dict[str, Any]]:
    rules = _all_pass_rules()
    rules[0] = _rule("No single region > 60%", ok=False, measured="75%", target="≤ 60%")
    return rules


def _stub_weights() -> pd.Series:
    tickers = [f"T{i:02d}" for i in range(25)]
    return pd.Series([1.0 / 25] * 25, index=tickers)


def _baseline_has_artifacts() -> bool:
    """Return True if baseline has been captured (.exit_code sentinel present)."""
    return (_BASELINE_DIR / ".exit_code").exists()


def _baseline_exit_code() -> int:
    ec_file = _BASELINE_DIR / ".exit_code"
    if ec_file.exists():
        return int(ec_file.read_text().strip())
    return 1  # conservative default — most runs fail checklist


def _run_pipeline_once(output_dir: Path) -> int:
    """Run the pipeline with parity parameters; return exit code."""
    import matplotlib

    matplotlib.use("Agg")

    from research.cli._main import main

    try:
        main(
            seed=_PARITY_SEED,
            start_date=_PARITY_START,
            end_date=_PARITY_END,
            output_dir=output_dir,
        )
        return 0
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0


# ---------------------------------------------------------------------------
# Terminal gate output tests  (always run — no DB needed)
# ---------------------------------------------------------------------------


class TestTerminalGateOutputOnFailure:
    """Verify _apply_terminal_gate emits a failure table and raises SystemExit(1)."""

    def test_when_one_rule_fails_then_exit_1_and_no_weights_csv(
        self, tmp_path: Path
    ) -> None:
        weights = _stub_weights()
        rules = _one_fail_rules()
        with pytest.raises(SystemExit) as exc:
            _apply_terminal_gate(rules=rules, weights=weights, output_dir=tmp_path)
        assert exc.value.code == 1
        assert not (tmp_path / "weights.csv").exists()

    def test_when_one_rule_fails_then_failure_table_emitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        buf = StringIO()
        patched_console = RichConsole(file=buf, force_terminal=False, highlight=False)
        monkeypatch.setattr(_checklist_mod, "console", patched_console)

        weights = _stub_weights()
        rules = _one_fail_rules()
        with pytest.raises(SystemExit):
            _apply_terminal_gate(rules=rules, weights=weights, output_dir=tmp_path)

        output = buf.getvalue()
        assert "No single region" in output
        assert "75%" in output

    def test_when_all_17_pass_then_exit_0_and_weights_csv_written(
        self, tmp_path: Path
    ) -> None:
        weights = _stub_weights()
        rules = _all_pass_rules()
        with pytest.raises(SystemExit) as exc:
            _apply_terminal_gate(rules=rules, weights=weights, output_dir=tmp_path)
        assert exc.value.code == 0
        assert (tmp_path / "weights.csv").exists()

    def test_when_all_17_pass_then_no_failure_table_in_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        buf = StringIO()
        patched_console = RichConsole(file=buf, force_terminal=False, highlight=False)
        monkeypatch.setattr(_checklist_mod, "console", patched_console)

        weights = _stub_weights()
        rules = _all_pass_rules()
        with pytest.raises(SystemExit):
            _apply_terminal_gate(rules=rules, weights=weights, output_dir=tmp_path)

        output = buf.getvalue()
        assert "Failing" not in output


# ---------------------------------------------------------------------------
# Pipeline parity test  (slow — skipped in CI via OPTIMIZER_SKIP_E2E=1)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPipelineParity:
    """Bit-exact (within tolerance) parity between current HEAD and baseline."""

    @pytest.fixture(scope="class")
    def pipeline_run(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> tuple[int, Path]:
        """Run the pipeline exactly once per class; cache (exit_code, output_dir).

        Guard checks (OPTIMIZER_SKIP_E2E and baseline presence) are performed
        here so they fire before the expensive pipeline execution, regardless of
        fixture evaluation order.
        """
        if _SKIP_E2E:
            pytest.skip("OPTIMIZER_SKIP_E2E=1 — skipping E2E parity test")
        if not _baseline_has_artifacts():
            pytest.skip(
                "No baseline artifacts found in tests/research/baselines/cycle0/. "
                "Run: MPLBACKEND=Agg PYTHONHASHSEED=0 "
                "python tests/research/baselines/cycle0/capture_baseline.py"
            )
        out = tmp_path_factory.mktemp("parity")
        ec = _run_pipeline_once(out)
        return ec, out

    def test_exit_code_matches_baseline(self, pipeline_run: tuple[int, Path]) -> None:
        exit_code, _ = pipeline_run
        expected = _baseline_exit_code()
        assert exit_code == expected, (
            f"Pipeline exit code {exit_code} != baseline {expected}"
        )

    def test_metrics_json_matches_baseline(
        self, pipeline_run: tuple[int, Path]
    ) -> None:
        baseline = _BASELINE_DIR / "metrics.json"
        if not baseline.exists():
            pytest.skip("metrics.json not in baseline")
        _, output_dir = pipeline_run
        actual = output_dir / "metrics.json"
        assert actual.exists(), "metrics.json not generated"
        _compare_json_values(
            json.loads(actual.read_text()),
            json.loads(baseline.read_text()),
        )

    def test_png_charts_sha256_match_baseline(
        self, pipeline_run: tuple[int, Path]
    ) -> None:
        baseline_pngs = list(_BASELINE_DIR.glob("*.png"))
        if not baseline_pngs:
            pytest.skip("no PNG charts in baseline")
        _, output_dir = pipeline_run
        mismatches: list[str] = []
        for bp in baseline_pngs:
            ap = output_dir / bp.name
            if not ap.exists():
                mismatches.append(f"MISSING {bp.name}")
            elif _sha256(ap) != _sha256(bp):
                mismatches.append(f"CHANGED {bp.name}")
        assert not mismatches, "Chart SHA-256 mismatches:\n" + "\n".join(mismatches)

    def test_weights_csv_matches_baseline(self, pipeline_run: tuple[int, Path]) -> None:
        baseline = _BASELINE_DIR / "weights.csv"
        if not baseline.exists():
            pytest.skip(
                "weights.csv not in baseline — checklist may not have passed 17/17"
            )
        _, output_dir = pipeline_run
        actual = output_dir / "weights.csv"
        assert actual.exists(), (
            "weights.csv not generated (checklist not passing 17/17)"
        )
        a_df = pd.read_csv(actual).sort_values("ticker").reset_index(drop=True)
        b_df = pd.read_csv(baseline).sort_values("ticker").reset_index(drop=True)
        assert list(a_df["ticker"]) == list(b_df["ticker"]), "ticker order mismatch"
        np.testing.assert_allclose(
            a_df["weight"].to_numpy(),
            b_df["weight"].to_numpy(),
            atol=1e-12,
            err_msg="Portfolio weights drifted beyond 1e-12 tolerance",
        )

    def test_report_md_matches_baseline(self, pipeline_run: tuple[int, Path]) -> None:
        baseline = _BASELINE_DIR / "report.md"
        if not baseline.exists():
            pytest.skip("report.md not in baseline")
        _, output_dir = pipeline_run
        actual = output_dir / "report.md"
        assert actual.exists(), "report.md not generated"
        a_text = _strip_dates(actual.read_text())
        b_text = _strip_dates(baseline.read_text())
        assert a_text == b_text, (
            "report.md content drifted (after stripping ISO-date tokens). "
            "Run diff to investigate."
        )
