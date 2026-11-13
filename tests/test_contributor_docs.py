"""Guard: the contributor documents describe the workflow the repository really uses.

``CONTRIBUTING.md`` once told contributors to build a pip virtualenv that installs only
the library, to branch from ``main`` and to commit a release straight onto ``main``.
None of that matched the uv workspace, ``development`` as the integration branch or
branch protection, and the migration process for ``portopt-db`` was three lines that
covered only ``upgrade``.
"""

from __future__ import annotations

import configparser
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONTRIBUTING = (_REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
_DB_README = (_REPO_ROOT / "packages" / "portopt-db" / "README.md").read_text(
    encoding="utf-8"
)
_ALEMBIC_INI = _REPO_ROOT / "packages" / "portopt-db" / "alembic.ini"


def test_contributing_sets_up_through_uv_not_a_pip_virtualenv() -> None:
    """A pip venv would install only the library; the workspace is a uv one."""
    assert "uv sync" in _CONTRIBUTING
    assert "python -m venv" not in _CONTRIBUTING
    assert 'pip install -e ".[dev]"' not in _CONTRIBUTING


def test_contributing_runs_each_member_suite_from_its_own_directory() -> None:
    """Run from the root, pytest collects only the library suite."""
    for member in ("cd ingestion", "cd fund", "cd packages/portopt-db"):
        assert member in _CONTRIBUTING, member


def test_contributing_branches_from_development_with_a_standard_prefix() -> None:
    """Work starts on a prefixed branch cut from the integration branch."""
    assert "from `development`" in _CONTRIBUTING
    assert "feature/" in _CONTRIBUTING
    assert "from `main`" not in _CONTRIBUTING


def test_contributing_releases_through_a_release_branch() -> None:
    """Direct commits to ``main`` are blocked, so a release goes through a branch."""
    assert "release/" in _CONTRIBUTING
    assert "git push origin main" not in _CONTRIBUTING


def test_contributing_points_at_the_decision_that_sets_the_policy() -> None:
    """The branching and review rules live in ADR-0007; the guide links it."""
    assert "docs/adr/0007-branching-review-and-secrets-policy.md" in _CONTRIBUTING


def test_db_readme_documents_the_migration_process() -> None:
    """Rollback, validation, one-way revisions and the Postgres suite are documented."""
    for required in (
        "alembic check",
        "alembic downgrade",
        "IRREVERSIBLE_REASON",
        "PORTOPT_TEST_DATABASE_URL",
        "erDiagram",
    ):
        assert required in _DB_README, required


def test_db_readme_names_every_consumer() -> None:
    """``fund`` uses the package too, not only ``ingestion`` and 'future siblings'."""
    assert "fund" in _DB_README
    assert "future siblings" not in _DB_README


def test_alembic_ini_dates_new_revision_files() -> None:
    """A dated prefix makes file names sort in execution order for new revisions."""
    parser = configparser.RawConfigParser()
    parser.read(_ALEMBIC_INI, encoding="utf-8")

    template = parser.get("alembic", "file_template", fallback="")

    assert "%%(year)d" in template
    assert "%%(rev)s" in template
