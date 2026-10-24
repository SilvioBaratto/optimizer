"""Where the TUI writes exports: one private directory, never the working directory.

Audit CSVs, Ask logs and daily briefs hold client capital, mandates and adviser free
text. Relative paths dropped them into whatever directory the TUI started in: the
container's writable layer (gone on recreate) or the repo root of a host run.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest
import yaml

from fund.tui import exports

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_exports_default_to_a_private_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FUND_EXPORT_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path / "..")

    path = exports.export_path("ask-fund-1.md")

    assert path == tmp_path / "fund" / "exports" / "ask-fund-1.md"
    assert path.is_absolute()
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_the_export_directory_can_be_chosen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "chosen" / "out"
    monkeypatch.setenv("FUND_EXPORT_DIR", str(target))

    assert exports.export_path("a.csv") == target / "a.csv"
    assert target.is_dir()


def test_git_ignores_exports_that_land_in_a_checkout() -> None:
    lines = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    for pattern in ("audit-export-*.csv", "ask-*.md", "daily-brief-*.md", "/exports/"):
        assert pattern in lines, pattern


def test_the_fund_container_mounts_a_host_visible_export_directory() -> None:
    fund = yaml.safe_load((_REPO_ROOT / "docker-compose.yml").read_text())["services"][
        "fund"
    ]

    assert fund["environment"]["FUND_EXPORT_DIR"] == "/app/exports"
    assert any(str(volume).endswith(":/app/exports") for volume in fund["volumes"])
