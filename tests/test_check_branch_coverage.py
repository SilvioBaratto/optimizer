"""Tests for the branch-aware coverage gate (scripts/check_branch_coverage.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_branch_coverage.py"


def _write_coverage_xml(path: Path, line_rate: float, branch_rate: float) -> Path:
    xml = path / "coverage.xml"
    xml.write_text(
        f'<?xml version="1.0" ?>\n'
        f'<coverage line-rate="{line_rate}" branch-rate="{branch_rate}"></coverage>\n'
    )
    return xml


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_when_both_metrics_above_floor_then_exit_zero(tmp_path: Path) -> None:
    xml = _write_coverage_xml(tmp_path, line_rate=0.957, branch_rate=0.88)
    result = _run(str(xml), "0.80")
    assert result.returncode == 0
    assert "Coverage gate passed." in result.stdout


def test_when_branch_rate_below_floor_then_exit_one(tmp_path: Path) -> None:
    xml = _write_coverage_xml(tmp_path, line_rate=0.95, branch_rate=0.70)
    result = _run(str(xml), "0.80")
    assert result.returncode == 1
    assert "branch-rate" in result.stdout


def test_when_line_rate_below_floor_then_exit_one(tmp_path: Path) -> None:
    xml = _write_coverage_xml(tmp_path, line_rate=0.50, branch_rate=0.95)
    result = _run(str(xml), "0.80")
    assert result.returncode == 1
    assert "line-rate" in result.stdout


def test_when_no_args_then_usage_exit_two() -> None:
    result = _run()
    assert result.returncode == 2
    assert "usage" in result.stdout


def test_when_threshold_omitted_then_defaults_to_080(tmp_path: Path) -> None:
    # 0.79 branch-rate must fail against the default 0.80 floor.
    xml = _write_coverage_xml(tmp_path, line_rate=0.95, branch_rate=0.79)
    result = _run(str(xml))
    assert result.returncode == 1


@pytest.mark.parametrize("rate", ["0.80", "0.800"])
def test_when_exactly_at_floor_then_exit_zero(tmp_path: Path, rate: str) -> None:
    xml = _write_coverage_xml(tmp_path, line_rate=float(rate), branch_rate=float(rate))
    result = _run(str(xml), "0.80")
    assert result.returncode == 0
