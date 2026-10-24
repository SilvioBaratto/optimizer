"""Behaviour of ``alembic/env.py``, driven through the real Alembic command API.

``env.py`` is a script that executes on import, so it is exercised the way
``alembic`` itself runs it: a ``Config`` pointing at the package's script
directory, ``DATABASE_URL`` set in the environment, and a command that makes
Alembic load the environment. SQLite files (or offline SQL rendering) stand in
for Postgres, so no live database is needed.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text

from alembic import command

_PKG_ROOT = Path(__file__).resolve().parents[1]


def _config(output: io.StringIO | None = None) -> Config:
    # No ini file: env.py would otherwise call fileConfig(), which disables
    # every existing logger and leaks into unrelated tests.
    cfg = Config(output_buffer=output)
    cfg.set_main_option(
        "script_location", str(_PKG_ROOT / "alembic").replace("%", "%%")
    )
    return cfg


def _head() -> str:
    head = ScriptDirectory.from_config(_config()).get_current_head()
    assert head is not None
    return head


def test_offline_run_accepts_percent_encoded_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A '%' in DATABASE_URL (an encoded password) must not abort env.py."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p%40ss@localhost:1/db")
    out = io.StringIO()

    command.stamp(_config(out), "head", sql=True)

    assert _head() in out.getvalue()


def test_online_run_hands_percent_encoded_url_to_engine_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The engine must receive the URL exactly as exported, '%' included."""
    db_file = tmp_path / "p%40ss.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")

    command.stamp(_config(), "head")

    # SQLite takes the database path verbatim, so the file only appears under
    # this exact name when env.py neither decoded nor doubled the '%'.
    assert db_file.exists()
    with create_engine(f"sqlite:///{db_file}").connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version"))
        assert version.scalar_one() == _head()
