"""``portopt_db.migration_state``: tell an empty database from one with pending revisions.

The pure classification runs against a fake revision tree and a fake
``MigrationContext``, so the state and the pending count are pinned without a
database. The I/O shell runs against SQLite files that carry a hand-written
``alembic_version`` row and the real migration tree; the live PostgreSQL path is
covered by ``tests/integration/test_migration_state_postgres.py``.
"""

from __future__ import annotations

import runpy
import sqlite3
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from alembic.script import ScriptDirectory
from alembic.script.revision import ResolutionError

from portopt_db import migration_state
from portopt_db.migration_state import (
    EXIT_ERROR,
    EXIT_PENDING,
    EXIT_PROCEED,
    MigrationState,
    MigrationStateError,
    MigrationStatus,
    PendingRevision,
    auto_migrate_enabled,
    classify,
    inspect_migration_state,
    main,
    pending_refusal_message,
)

_ALEMBIC_DIR = Path(__file__).resolve().parents[1] / "alembic"
_CHAIN = ("aaa1", "bbb2", "ccc3", "ddd4")


@dataclass(frozen=True)
class _FakeRevision:
    revision: str
    doc: str


class _FakeTree:
    """A linear revision chain, oldest first, answering like ``ScriptDirectory``."""

    def __init__(self, *chain: str, heads: tuple[str, ...] | None = None) -> None:
        self._chain = [
            _FakeRevision(rev, f"step   {rev}\n  continued") for rev in chain
        ]
        self._heads = list(heads if heads is not None else chain[-1:])

    def get_heads(self) -> list[str]:
        """Return the configured heads (the chain's tip unless overridden)."""
        return list(self._heads)

    def iterate_revisions(
        self, upper: str, lower: tuple[str, ...]
    ) -> Iterator[_FakeRevision]:
        """Yield the revisions above ``lower`` up to ``upper``, newest first.

        Raises:
            ResolutionError: A ``lower`` revision is not in the chain, as Alembic
                raises for a revision its tree does not hold.
        """
        ids = [step.revision for step in self._chain]
        for rev in lower:
            if rev not in ids:
                raise ResolutionError(f"No such revision {rev!r}", rev)
        start = max((ids.index(rev) + 1 for rev in lower), default=0)
        stop = ids.index(upper) + 1
        yield from reversed(self._chain[start:stop])


@dataclass(frozen=True)
class _FakeContext:
    heads: tuple[str, ...]

    def get_current_heads(self) -> tuple[str, ...]:
        """Return the revisions the fake ``alembic_version`` holds."""
        return self.heads


def _classify(
    *current: str, tables: tuple[str, ...] = ("public.instruments",)
) -> MigrationStatus:
    return classify(
        user_tables=tables, context=_FakeContext(current), script=_FakeTree(*_CHAIN)
    )


@pytest.mark.criterion("REQ-DB-006")
def test_a_database_without_user_tables_is_empty() -> None:
    """Validates portopt_db.migration_state.classify on a brand-new database."""
    status = _classify(tables=())

    assert status.state is MigrationState.EMPTY
    assert status.pending == ()
    assert status.head == "ddd4"


@pytest.mark.criterion("REQ-DB-006")
def test_a_database_with_a_stamp_but_no_user_tables_is_empty() -> None:
    """Validates portopt_db.migration_state.classify: nothing to lose means EMPTY."""
    assert _classify("bbb2", tables=()).state is MigrationState.EMPTY


@pytest.mark.criterion("REQ-DB-006")
def test_a_database_at_the_tree_head_is_at_head() -> None:
    """Validates portopt_db.migration_state.classify when nothing is left to apply."""
    status = _classify("ddd4")

    assert status.state is MigrationState.AT_HEAD
    assert status.count == 0
    assert status.current == ("ddd4",)


@pytest.mark.criterion("REQ-DB-006")
def test_a_database_behind_the_head_lists_the_pending_revisions_oldest_first() -> None:
    """Validates portopt_db.migration_state.classify: the count and the run order."""
    status = _classify("bbb2")

    assert status.state is MigrationState.PENDING
    assert status.count == 2
    assert [step.revision for step in status.pending] == ["ccc3", "ddd4"]
    assert status.pending[0] == PendingRevision("ccc3", "step ccc3 continued")


@pytest.mark.criterion("REQ-DB-006")
def test_user_tables_without_a_version_are_pending_from_base() -> None:
    """Validates portopt_db.migration_state.classify on tables Alembic never stamped:
    data is present, so the start must stop instead of replaying the whole tree."""
    status = _classify()

    assert status.state is MigrationState.PENDING
    assert status.count == len(_CHAIN)
    assert status.current_label == "base"


@pytest.mark.criterion("REQ-DB-006")
def test_a_revision_the_tree_does_not_know_is_an_error() -> None:
    """Validates portopt_db.migration_state.classify on a database a newer version
    migrated: the older code cannot tell how far behind it is."""
    with pytest.raises(MigrationStateError, match="eee5"):
        _classify("eee5")


@pytest.mark.criterion("REQ-DB-006")
@pytest.mark.parametrize("heads", [(), ("ccc3", "ddd4")])
def test_a_tree_without_exactly_one_head_is_an_error(heads: tuple[str, ...]) -> None:
    """Validates portopt_db.migration_state.classify against an ambiguous target."""
    tree = _FakeTree(*_CHAIN, heads=heads)

    with pytest.raises(MigrationStateError, match="head"):
        classify(user_tables=(), context=_FakeContext(()), script=tree)


@pytest.mark.criterion("REQ-DB-006")
def test_the_refusal_names_both_revisions_the_count_and_the_two_ways_out() -> None:
    """Validates portopt_db.migration_state.pending_refusal_message."""
    message = pending_refusal_message(_classify("ccc3"))

    assert "ccc3" in message
    assert "ddd4" in message
    assert "1 revision behind" in message
    assert "portopt migrate" in message
    assert "PORTOPT_AUTO_MIGRATE=true" in message


@pytest.mark.criterion("REQ-DB-006")
def test_the_refusal_counts_several_revisions_in_the_plural() -> None:
    """Validates portopt_db.migration_state.pending_refusal_message for two or more."""
    assert "2 revisions behind" in pending_refusal_message(_classify("bbb2"))


@pytest.mark.criterion("REQ-DB-006")
@pytest.mark.parametrize(
    ("value", "enabled"),
    [
        ("true", True),
        ("TRUE", True),
        (" 1 ", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("", False),
        ("maybe", False),
    ],
)
def test_auto_migrate_is_on_only_for_an_explicit_yes(value: str, enabled: bool) -> None:
    """Validates portopt_db.migration_state.auto_migrate_enabled."""
    assert auto_migrate_enabled({"PORTOPT_AUTO_MIGRATE": value}) is enabled


@pytest.mark.criterion("REQ-DB-006")
def test_auto_migrate_is_off_when_unset() -> None:
    """Validates portopt_db.migration_state.auto_migrate_enabled without the variable."""
    assert auto_migrate_enabled({}) is False


def _real_head() -> str:
    head = ScriptDirectory(str(_ALEMBIC_DIR)).get_current_head()
    assert head is not None
    return head


def _one_behind_head() -> str:
    revision = ScriptDirectory(str(_ALEMBIC_DIR)).get_revision(_real_head())
    assert revision is not None
    assert isinstance(revision.down_revision, str)
    return revision.down_revision


def _sqlite_database(
    path: Path, *, version: str | None, user_table: bool = True
) -> str:
    with sqlite3.connect(path) as connection:
        if version is not None:
            connection.execute(
                "CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"
            )
            connection.execute("INSERT INTO alembic_version VALUES (?)", (version,))
        if user_table:
            connection.execute("CREATE TABLE instruments (id INTEGER PRIMARY KEY)")
    return f"sqlite:///{path}"


@pytest.mark.criterion("REQ-DB-006")
def test_inspect_reports_an_empty_sqlite_file_as_empty(tmp_path: Path) -> None:
    """Validates portopt_db.migration_state.inspect_migration_state on a new file."""
    url = _sqlite_database(tmp_path / "db.sqlite", version=None, user_table=False)

    status = inspect_migration_state(url, _ALEMBIC_DIR)

    assert status.state is MigrationState.EMPTY
    assert status.tables == ()


@pytest.mark.criterion("REQ-DB-006")
def test_inspect_reports_the_real_head_as_at_head(tmp_path: Path) -> None:
    """Validates portopt_db.migration_state.inspect_migration_state against the
    shipped tree; the version table is not counted as a user table."""
    url = _sqlite_database(tmp_path / "db.sqlite", version=_real_head())

    status = inspect_migration_state(url, _ALEMBIC_DIR)

    assert status.state is MigrationState.AT_HEAD
    assert status.tables == ("main.instruments",)


@pytest.mark.criterion("REQ-DB-006")
def test_inspect_counts_the_revisions_after_the_stamped_one(tmp_path: Path) -> None:
    """Validates portopt_db.migration_state.inspect_migration_state with the shipped
    tree: a database one revision behind has exactly the head left to apply."""
    url = _sqlite_database(tmp_path / "db.sqlite", version=_one_behind_head())

    status = inspect_migration_state(url, _ALEMBIC_DIR)

    assert status.state is MigrationState.PENDING
    assert [step.revision for step in status.pending] == [_real_head()]


@pytest.mark.criterion("REQ-DB-006")
def test_inspect_wraps_an_unreachable_database(tmp_path: Path) -> None:
    """Validates portopt_db.migration_state.inspect_migration_state on a failed
    connect: the caller gets one error type, never a driver traceback."""
    url = f"sqlite:///{tmp_path / 'missing-dir' / 'db.sqlite'}"

    with pytest.raises(MigrationStateError, match="could not read"):
        inspect_migration_state(url, _ALEMBIC_DIR)


@pytest.mark.criterion("REQ-DB-006")
def test_inspect_wraps_a_missing_migration_tree(tmp_path: Path) -> None:
    """Validates portopt_db.migration_state.inspect_migration_state without a tree."""
    url = _sqlite_database(tmp_path / "db.sqlite", version=None)

    with pytest.raises(MigrationStateError, match="migration tree"):
        inspect_migration_state(url, tmp_path / "no-alembic")


def _run_main(
    *argv: str, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    code = main(["--alembic-dir", str(_ALEMBIC_DIR), *argv], environ=env)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.mark.criterion("REQ-DB-006")
def test_main_refuses_a_pending_database_with_exit_three(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validates portopt_db.migration_state.main: the entrypoints branch on code 3."""
    url = _sqlite_database(tmp_path / "db.sqlite", version=_one_behind_head())

    code, _, err = _run_main(env={"DATABASE_URL": url}, capsys=capsys)

    assert code == EXIT_PENDING == 3
    assert "portopt migrate" in err


@pytest.mark.criterion("REQ-DB-006")
@pytest.mark.parametrize("at_head", [False, True])
def test_main_lets_an_empty_or_current_database_proceed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], at_head: bool
) -> None:
    """Validates portopt_db.migration_state.main for the two states that migrate on
    start exactly as before."""
    stamp = _real_head() if at_head else None
    url = _sqlite_database(tmp_path / "db.sqlite", version=stamp, user_table=at_head)

    code, out, _ = _run_main(env={"DATABASE_URL": url}, capsys=capsys)

    assert code == EXIT_PROCEED == 0
    assert out.strip()


@pytest.mark.criterion("REQ-DB-006")
def test_main_with_auto_migrate_skips_the_check(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validates portopt_db.migration_state.main with PORTOPT_AUTO_MIGRATE=true: it
    proceeds without even needing a database URL, as the old entrypoint did."""
    code, out, _ = _run_main(env={"PORTOPT_AUTO_MIGRATE": "true"}, capsys=capsys)

    assert code == EXIT_PROCEED
    assert "PORTOPT_AUTO_MIGRATE" in out


@pytest.mark.criterion("REQ-DB-006")
def test_main_without_a_database_url_is_an_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validates portopt_db.migration_state.main when DATABASE_URL is unset."""
    code, _, err = _run_main(env={}, capsys=capsys)

    assert code == EXIT_ERROR == 1
    assert "DATABASE_URL" in err


@pytest.mark.criterion("REQ-DB-006")
def test_main_reports_an_unreadable_database_as_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validates portopt_db.migration_state.main: a failed check never lets the
    container start."""
    url = f"sqlite:///{tmp_path / 'missing-dir' / 'db.sqlite'}"

    code, _, err = _run_main(env={"DATABASE_URL": url}, capsys=capsys)

    assert code == EXIT_ERROR
    assert "could not read" in err


@pytest.mark.criterion("REQ-DB-006")
def test_main_never_prints_the_database_password(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validates portopt_db.migration_state.main on a refused Postgres connect."""
    url = "postgresql://postgres:s3cret-pw@127.0.0.1:1/optimizer_db"

    code, out, err = _run_main(env={"DATABASE_URL": url}, capsys=capsys)

    assert code == EXIT_ERROR
    assert "s3cret-pw" not in out + err


@pytest.mark.criterion("REQ-DB-006")
def test_the_module_runs_as_a_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates the ``python -m portopt_db.migration_state`` entry the shells call."""
    monkeypatch.setattr(sys, "argv", ["migration_state"])
    monkeypatch.setenv("PORTOPT_AUTO_MIGRATE", "true")
    monkeypatch.delitem(sys.modules, migration_state.__name__)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module(migration_state.__name__, run_name="__main__")

    assert excinfo.value.code == EXIT_PROCEED
