"""``alembic upgrade head --sql`` must render the whole revision chain.

``alembic/env.py`` documents offline mode as a supported way to render the
migration SQL for review or a DBA hand-off. Offline mode runs every revision
against a ``MockConnection`` that cannot be introspected, so any revision that
calls ``sa.inspect(op.get_bind())`` unguarded aborts the render part-way down
the chain. These tests drive the real chain through the Alembic command API
with no database behind it.
"""

from __future__ import annotations

import io
import re
from functools import cache
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from alembic import command

_PKG_ROOT = Path(__file__).resolve().parents[1]


def _config(buffer: io.StringIO) -> Config:
    # No ini file: env.py would otherwise call fileConfig() and reconfigure the
    # logging of the whole pytest session.
    cfg = Config(output_buffer=buffer)
    cfg.set_main_option("script_location", str(_PKG_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", "postgresql://offline.invalid/render")
    return cfg


@cache
def _render_offline() -> str:
    """Render ``upgrade head`` offline and return the SQL script.

    Raises whatever the chain raises, so a revision that cannot render shows up
    as a failure of every test that needs the script.
    """
    mp = pytest.MonkeyPatch()
    mp.delenv("DATABASE_URL", raising=False)
    try:
        buffer = io.StringIO()
        command.upgrade(_config(buffer), "head", sql=True)
        return buffer.getvalue()
    finally:
        mp.undo()


def test_when_upgrade_head_is_rendered_offline_then_the_chain_reaches_head() -> None:
    """The last emitted statement stamps the current head revision."""
    head = ScriptDirectory.from_config(_config(io.StringIO())).get_current_head()

    assert f"version_num='{head}'" in _render_offline()


def test_when_upgrade_head_is_rendered_offline_then_the_macro_snapshot_seed_is_kept() -> (
    None
):
    """The observation seed is part of the hand-off SQL, not silently skipped.

    ``trading_economics_indicators`` exists at that point of the chain, so the
    seed must render; ``bond_yields`` never exists there, so it must not.
    """
    sql = _render_offline()

    assert re.search(
        r"INSERT INTO trading_economics_observations.*?"
        r"FROM trading_economics_indicators",
        sql,
        re.DOTALL,
    )
    assert "FROM bond_yields" not in sql


def test_when_upgrade_head_is_rendered_offline_then_st_rate_forecast_is_dropped() -> (
    None
):
    """The column drop is rendered even though no inspector could confirm it."""
    assert re.search(
        r"ALTER TABLE economic_indicators DROP COLUMN (IF EXISTS )?st_rate_forecast",
        _render_offline(),
    )
