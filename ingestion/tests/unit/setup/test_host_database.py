"""``app.setup.host_database``: the compose database as the host-side CLI sees it.

``inspect_migration_state`` is replaced by a recording fake in every test, so no
database is read; the unit suite's autouse guard fails any test that forgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from portopt_db.migration_state import (
    MigrationState,
    MigrationStateError,
    MigrationStatus,
    PendingRevision,
)

from app.setup import host_database

_HEAD = "8ed6504aee6a"


def _status(state: MigrationState) -> MigrationStatus:
    pending = (
        (PendingRevision(_HEAD, "drop the scraper tables"),)
        if state is MigrationState.PENDING
        else ()
    )
    current = ("2266e24c509f",) if pending else (_HEAD,)
    return MigrationStatus(state, _HEAD, current, pending, ("public.instruments",))


@dataclass
class _StateReads:
    """Stands in for ``inspect_migration_state``: records each call, then answers."""

    answer: MigrationStatus | Exception = field(
        default_factory=lambda: _status(MigrationState.AT_HEAD)
    )
    calls: list[tuple[str, Path]] = field(default_factory=list)

    def __call__(self, url: str, alembic_dir: Path) -> MigrationStatus:
        self.calls.append((url, alembic_dir))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> _StateReads:
    """Route every state read of ``host_database`` to a fresh recorder."""
    recorder = _StateReads()
    monkeypatch.setattr(host_database, "inspect_migration_state", recorder)
    return recorder


@pytest.mark.criterion("REQ-ING-030")
def test_the_db_package_sits_under_packages_portopt_db(tmp_path: Path) -> None:
    """Validates app.setup.host_database.db_package_dir."""
    assert host_database.db_package_dir(tmp_path) == tmp_path / "packages/portopt-db"


@pytest.mark.criterion("REQ-ING-030")
def test_read_state_reads_the_compose_database_against_the_checkouts_tree(
    reads: _StateReads, tmp_path: Path
) -> None:
    """Validates app.setup.host_database.read_state: the published port and the tree
    the images are built from, whatever DATABASE_URL the shell exports."""
    host_database.read_state(tmp_path)

    assert reads.calls == [(host_database.HOST_DB_URL, tmp_path / "alembic")]
    assert "localhost:54320/optimizer_db" in host_database.HOST_DB_URL


@pytest.mark.criterion("REQ-ING-030")
def test_pending_revisions_over_data_are_refused(
    reads: _StateReads, tmp_path: Path
) -> None:
    """Validates app.setup.host_database.pending_refusal on a PENDING database."""
    reads.answer = _status(MigrationState.PENDING)

    refusal = host_database.pending_refusal(tmp_path, environ={})

    assert refusal is not None
    assert "portopt migrate" in refusal
    assert _HEAD in refusal


@pytest.mark.criterion("REQ-ING-030")
@pytest.mark.parametrize("state", [MigrationState.EMPTY, MigrationState.AT_HEAD])
def test_an_empty_or_current_database_is_not_refused(
    reads: _StateReads, tmp_path: Path, state: MigrationState
) -> None:
    """Validates app.setup.host_database.pending_refusal for the states that migrate
    on start as before."""
    reads.answer = _status(state)

    assert host_database.pending_refusal(tmp_path, environ={}) is None


@pytest.mark.criterion("REQ-ING-030")
def test_auto_migrate_skips_the_read(reads: _StateReads, tmp_path: Path) -> None:
    """Validates app.setup.host_database.pending_refusal with PORTOPT_AUTO_MIGRATE:
    the opt-out keeps the old behaviour, so the database is not even asked."""
    reads.answer = _status(MigrationState.PENDING)

    refusal = host_database.pending_refusal(
        tmp_path, environ={"PORTOPT_AUTO_MIGRATE": "true"}
    )

    assert refusal is None
    assert reads.calls == []


@pytest.mark.criterion("REQ-ING-030")
def test_an_unreadable_state_is_left_to_the_caller(
    reads: _StateReads, tmp_path: Path
) -> None:
    """Validates app.setup.host_database.pending_refusal: a failed read raises, so
    start, setup and migrate each decide what a failure means for them."""
    reads.answer = MigrationStateError("connection refused")

    with pytest.raises(MigrationStateError, match="connection refused"):
        host_database.pending_refusal(tmp_path, environ={})
