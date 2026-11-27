"""Guard: no tracked file names an integration the project removed.

The Trading 212 broker integration was taken out of the code, the configuration and
the database, the ISIN lookup that scraped a third-party news site through yfinance
was taken out of the ingestion, and so were the two macro web scrapers (a financial
newspaper's forecast page and an economics data site). A name that survives in a
comment, a doc or a column definition says the integration still exists, and the
first new reference to it is the start of growing it back. The scan covers every
tracked file, its path and its content, so an untracked ``.venv``, ``design/`` or
``__pycache__`` never takes part and the result does not depend on the working
directory.

A later removal appends its names to ``_REMOVED_INTEGRATION_TOKENS``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Regular-expression fragments, matched case-insensitively and joined with "|".
# "trading[ _-]?212" covers the prose name, the TRADING_212_* settings and the
# trading212 package name; "business[ _-]?insider" covers the prose name of the site
# the ISIN lookup scraped and its domain-style spelling; "il[ _-]?sole" covers the
# newspaper's prose name and the IlSole24Ore, ilsole_* and ILSOLE_* spellings, and
# "trading[ _-]?economics" the economics data site's prose name and its
# TRADINGECONOMICS_* and trading_economics_* spellings.
_REMOVED_INTEGRATION_TOKENS = (
    r"trading[ _-]?212",
    r"t212",
    r"tradeable_only",
    r"business[ _-]?insider",
    r"il[ _-]?sole",
    r"trading[ _-]?economics",
)

_REMOVED_INTEGRATION_PATTERN = re.compile(
    "|".join(f"(?:{token})" for token in _REMOVED_INTEGRATION_TOKENS), re.IGNORECASE
)

# Longest stretch of an offending line shown in a failure message.
_EXCERPT_CHARS = 100

# The six macro tables the two scrapers filled stay in the schema until a later
# change drops them, so the models, repository and tests built on them still carry the
# scrapers' names; that change deletes these entries together with the files.
_MACRO_TABLES_DROPPED_LATER = "belongs to the macro tables a later change drops"
_DB_SRC = "packages/portopt-db/src/portopt_db/"
_DB_TESTS = "packages/portopt-db/tests/"

# Tracked files that may name a removed integration, each with the reason.
_ALLOWED_FILES = {
    "CHANGELOG.md": "records the removal of the integration, release by release",
    "tests/test_removed_integrations.py": "names the tokens it forbids",
    "packages/portopt-db/tests/test_is_t212_tradeable_migration.py": (
        "tests the historical revision that added the instruments flag column"
    ),
    "packages/portopt-db/tests/test_t212_ticker_migration.py": (
        "tests the historical revision that added the instruments ticker column"
    ),
    "packages/portopt-db/tests/test_drop_trading_212_columns_migration.py": (
        "tests the revision that drops the three columns, so it must name them"
    ),
    "fund/tests/unit/test_schemas_mandate.py": (
        "regression test that a mandate saved with the removed broker-scope key "
        "still loads"
    ),
    "ingestion/tests/unit/setup/test_wizard.py": (
        "regression test that setup ignores the removed broker's stored secrets"
    ),
    "ingestion/tests/unit/test_universe_builder_imports.py": (
        "regression test that the removed broker package does not come back"
    ),
    "ingestion/tests/unit/test_external_calls.py": (
        "regression test that the removed scrapers' breakers and settings do not "
        "come back"
    ),
    f"{_DB_SRC}models/__init__.py": (
        f"re-exports a macro model: {_MACRO_TABLES_DROPPED_LATER}"
    ),
    f"{_DB_SRC}models/macro/__init__.py": (
        f"re-exports the macro models: {_MACRO_TABLES_DROPPED_LATER}"
    ),
    f"{_DB_SRC}models/macro/macro_regime.py": (
        f"defines the macro models: {_MACRO_TABLES_DROPPED_LATER}"
    ),
    f"{_DB_SRC}repositories/macro/macro_regime_repository.py": (
        f"reads and writes the macro tables: {_MACRO_TABLES_DROPPED_LATER}"
    ),
    f"{_DB_SRC}repositories/database_admin.py": (
        f"lists the macro tables it truncates: {_MACRO_TABLES_DROPPED_LATER}"
    ),
    f"{_DB_TESTS}test_macro_regime_repository_branches.py": (
        f"tests the macro repository: {_MACRO_TABLES_DROPPED_LATER}"
    ),
    f"{_DB_TESTS}test_upsert_updated_at.py": (
        f"tests the macro observation upsert: {_MACRO_TABLES_DROPPED_LATER}"
    ),
    f"{_DB_TESTS}test_purge_non_comparable_te_rows_migration.py": (
        f"tests a revision on the macro tables: {_MACRO_TABLES_DROPPED_LATER}"
    ),
    f"{_DB_TESTS}test_migrations_offline_sql.py": (
        f"renders a revision on the macro tables: {_MACRO_TABLES_DROPPED_LATER}"
    ),
    f"{_DB_TESTS}test_redundant_unique_indexes_migration.py": (
        f"checks a revision on the macro tables: {_MACRO_TABLES_DROPPED_LATER}"
    ),
    f"{_DB_TESTS}test_downgrades_on_chain_built_database.py": (
        f"downgrades a revision on the macro tables: {_MACRO_TABLES_DROPPED_LATER}"
    ),
}

# Tracked directories whose files may name a removed integration, each with the reason.
_ALLOWED_PREFIXES = {
    "packages/portopt-db/alembic/versions/": (
        "applied revisions are history and are never edited"
    ),
}


def _tracked_files() -> list[str]:
    git = shutil.which("git")
    assert git, "git is required to list the tracked files"
    listing = subprocess.run(  # noqa: S603 - fixed argv, no user input
        [git, "ls-files", "-z"],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=True,
    )
    return [name for name in listing.stdout.decode("utf-8").split("\0") if name]


def _is_allowed(name: str) -> bool:
    return name in _ALLOWED_FILES or name.startswith(tuple(_ALLOWED_PREFIXES))


def _findings(name: str) -> list[str]:
    """Return one line per place in a tracked file that names a removed integration."""
    found: list[str] = []
    if _REMOVED_INTEGRATION_PATTERN.search(name):
        found.append(f"{name}: the file name")
    try:
        text = (_REPO_ROOT / name).read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return found
    for number, line in enumerate(text.splitlines(), start=1):
        if _REMOVED_INTEGRATION_PATTERN.search(line):
            found.append(f"{name}:{number}: {line.strip()[:_EXCERPT_CHARS]}")
    return found


@pytest.mark.criterion("REQ-OPS-005")
def test_no_removed_integration_names_in_tracked_files() -> None:
    """Validates every file ``git ls-files`` lists: none names a removed integration."""
    offenders = [
        finding
        for name in _tracked_files()
        if not _is_allowed(name)
        for finding in _findings(name)
    ]
    assert offenders == []


@pytest.mark.criterion("REQ-OPS-005")
def test_the_scan_flags_a_revision_that_names_a_removed_integration() -> None:
    """Validates ``_findings``: it reports the names a history revision holds."""
    revision = (
        "packages/portopt-db/alembic/versions/"
        "f4b6d8a0c2e3_add_is_t212_tradeable_to_instruments.py"
    )

    findings = _findings(revision)

    assert any("the file name" in finding for finding in findings)
    assert any("Trading 212" in finding for finding in findings)


@pytest.mark.criterion("REQ-OPS-005")
@pytest.mark.parametrize(
    "spelling",
    [
        "Business Insider",
        "business insider",
        "BusinessInsider",
        "businessinsider.com",
        "business_insider",
        "business-insider",
    ],
)
def test_pattern_flags_each_spelling_of_the_scraped_news_site(spelling: str) -> None:
    """Validates ``_REMOVED_INTEGRATION_PATTERN``: it matches every spelling."""
    assert _REMOVED_INTEGRATION_PATTERN.search(f"ISINs come from {spelling}")


@pytest.mark.criterion("REQ-OPS-005")
@pytest.mark.parametrize(
    "spelling",
    [
        "Il Sole 24 Ore",
        "IlSole24Ore",
        "ILSOLE_REQUEST_ATTEMPTS",
        "ilsole_forecast",
        "il-sole",
        "Trading Economics",
        "TradingEconomics",
        "tradingeconomics.com",
        "TRADINGECONOMICS_REQUEST_ATTEMPTS",
        "trading_economics_indicators",
        "trading-economics",
    ],
)
def test_pattern_flags_each_spelling_of_the_removed_macro_scrapers(
    spelling: str,
) -> None:
    """Validates ``_REMOVED_INTEGRATION_PATTERN``: it matches every spelling of the
    two removed macro scrapers."""
    assert _REMOVED_INTEGRATION_PATTERN.search(f"macro data comes from {spelling}")


@pytest.mark.criterion("REQ-OPS-005")
def test_no_allowlist_entry_is_stale() -> None:
    """Validates ``_ALLOWED_FILES`` and ``_ALLOWED_PREFIXES``: no entry is stale.

    An entry that is no longer tracked, or that no longer names a removed
    integration, exempts nothing and should be deleted.
    """
    tracked = set(_tracked_files())

    assert sorted(set(_ALLOWED_FILES) - tracked) == []
    assert sorted(name for name in _ALLOWED_FILES if not _findings(name)) == []
    for prefix in _ALLOWED_PREFIXES:
        under_prefix = [name for name in tracked if name.startswith(prefix)]
        assert any(_findings(name) for name in under_prefix), prefix


@pytest.mark.criterion("REQ-OPS-005")
def test_guard_does_not_allow_production_paths() -> None:
    """Validates ``_is_allowed``: production code is never exempt from the scan."""
    production_paths = (
        "packages/portopt-db/src/portopt_db/ports.py",
        "packages/portopt-db/alembic/env.py",
        "fund/src/fund/schemas/mandate.py",
        "ingestion/app/config.py",
    )

    assert [name for name in production_paths if _is_allowed(name)] == []
    assert _is_allowed("CHANGELOG.md")
