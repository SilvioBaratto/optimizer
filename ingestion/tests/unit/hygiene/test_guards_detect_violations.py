"""Checkpoint: every hygiene guard actually bites (issues #7, #15).

Each guard is an absence assertion over a tree that is already clean, so it
passes trivially on day one — indistinguishable from a broken pattern, wrong
anchoring, or too-broad exclusion that would let it pass forever while
catching nothing. This module drives each guard's scanner against a
synthetic tree seeded with known violations (in-process, six guards' worth,
issue #15) and, for the two guards whose contract spans working directories,
against the real repository from two different process working directories.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.hygiene.test_migrations_reversible import _find_downgrade_violation
from tests.unit.hygiene.test_no_http_surface import (
    _INGESTION_ROOT,
    find_http_violations,
)
from tests.unit.hygiene.test_no_legacy_api_references import (
    _REPO_ROOT,
    find_legacy_api_violations,
)
from tests.unit.hygiene.test_no_optimizer_import import (
    find_optimizer_import_violations,
)
from tests.unit.hygiene.test_shared_init_exports import find_reexport_violation
from tests.unit.hygiene.test_synchronous_only import find_async_violations

# (injected token, substring expected in the guard's finding for it) — the
# route decorator's finding names the offense category rather than quoting
# the decorator text back, so its expected substring differs from the token.
_HTTP_SURFACE_TOKENS = (
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("TestClient", "TestClient"),
    ("from app.main", "from app.main"),
    ('@app.get("/health")', "route decorator"),
)

_LEGACY_PATH_TOKENS = (
    "api/v1/",
    "api/app/",
    "api/scripts/",
    "api/tests/",
    "api/requirements.txt",
)

_SAFE_LEGACY_LOOKALIKES = (
    "/api/v0/equity",
    "discord.com/api/webhooks",
    "docs/api/api_key.html",
)

_GUARD_MODULE_RELATIVE_PATHS = (
    "ingestion/tests/unit/hygiene/test_no_http_surface.py",
    "ingestion/tests/unit/hygiene/test_no_legacy_api_references.py",
)


def _write_one_line_per_file(root: Path, lines: tuple[str, ...]) -> None:
    for index, line in enumerate(lines):
        (root / f"violation_{index}.py").write_text(f"{line}\n", encoding="utf-8")


@pytest.mark.criterion("scope-1")
def test_when_synthetic_tree_carries_each_http_token_then_one_finding_per_token_is_reported(
    tmp_path,
):
    _write_one_line_per_file(
        tmp_path, tuple(token for token, _ in _HTTP_SURFACE_TOKENS)
    )

    violations = find_http_violations(tmp_path)

    assert len(violations) == len(_HTTP_SURFACE_TOKENS)
    for _token, expected_substring in _HTTP_SURFACE_TOKENS:
        assert any(expected_substring in finding for finding in violations)


@pytest.mark.criterion("scope-2")
def test_when_synthetic_tree_carries_each_legacy_path_shape_then_each_is_reported(
    tmp_path,
):
    _write_one_line_per_file(tmp_path, _LEGACY_PATH_TOKENS)

    violations = find_legacy_api_violations(tmp_path)

    assert len(violations) == len(_LEGACY_PATH_TOKENS)


@pytest.mark.criterion("scope-2")
def test_when_synthetic_tree_carries_safe_api_lookalikes_then_none_are_reported(
    tmp_path,
):
    (tmp_path / "safe.py").write_text(
        "\n".join(_SAFE_LEGACY_LOOKALIKES) + "\n", encoding="utf-8"
    )

    violations = find_legacy_api_violations(tmp_path)

    assert violations == []


@pytest.mark.criterion("scope-3")
def test_when_real_repository_is_scanned_then_both_guards_report_it_clean():
    assert find_http_violations(_INGESTION_ROOT) == []
    assert find_legacy_api_violations(_REPO_ROOT) == []


_SYNC_ONLY_TOKENS = ("AsyncSession", "asyncpg", "asyncio.", "async def ")


@pytest.mark.criterion("scope-7")
def test_when_synthetic_tree_carries_each_async_marker_then_each_is_reported(tmp_path):
    _write_one_line_per_file(tmp_path, _SYNC_ONLY_TOKENS)

    violations = find_async_violations(tmp_path)

    assert len(violations) == len(_SYNC_ONLY_TOKENS)


@pytest.mark.criterion("scope-7")
def test_when_synthetic_tree_carries_an_optimizer_import_then_it_is_reported(tmp_path):
    (tmp_path / "offender.py").write_text("import optimizer\n", encoding="utf-8")

    violations = find_optimizer_import_violations(tmp_path)

    assert violations == ["offender.py"]


@pytest.mark.criterion("scope-7")
def test_when_bootstrap_benchmarks_is_reexported_then_the_shared_init_guard_fails():
    reexported_source = "from ._benchmark_bootstrap import bootstrap_benchmarks\n"

    assert find_reexport_violation(reexported_source) is True


@pytest.mark.criterion("scope-7")
def test_when_a_non_bare_upgrade_has_a_bare_downgrade_then_the_migrations_guard_fails():
    text = (
        "def upgrade() -> None:\n    op.create_table('t')\n\n"
        "def downgrade() -> None:\n    pass\n"
    )

    assert _find_downgrade_violation(text) == "downgrade() body is bare"


def _run_guard_module(
    cwd: Path, guard_relative_path: str
) -> subprocess.CompletedProcess:
    target = str(_REPO_ROOT / guard_relative_path)
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", "-q", target],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.criterion("scope-4")
@pytest.mark.parametrize("guard_relative_path", _GUARD_MODULE_RELATIVE_PATHS)
def test_when_guard_module_runs_from_repo_root_or_ingestion_dir_then_outcome_is_identical(
    guard_relative_path,
):
    ingestion_root = _REPO_ROOT / "ingestion"

    from_repo_root = _run_guard_module(_REPO_ROOT, guard_relative_path)
    from_ingestion = _run_guard_module(ingestion_root, guard_relative_path)

    assert from_repo_root.returncode == from_ingestion.returncode == 0
