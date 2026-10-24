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


def _revision_block(sql: str, down: str, up: str) -> str:
    """Return the statements rendered for the ``down -> up`` revision step."""
    start = sql.index(f"-- Running upgrade {down} -> {up}")
    end = sql.find("-- Running upgrade", start + 1)
    return sql[start : end if end != -1 else len(sql)]


def test_when_normalize_models_is_rendered_then_financial_value_is_not_narrowed() -> (
    None
):
    """``financial_statements.value`` must never drop below the width it ends with.

    The next revision widens the column to ``NUMERIC(38, 6)``; narrowing it to
    ``NUMERIC(20, 6)`` first overflows on any populated table holding a value
    of 1e14 or more, so the chain stalls on the very first upgrade.
    """
    block = _revision_block(_render_offline(), "ecf0a9a2bfdc", "b0b30daa80c5")

    widened = re.search(
        r"ALTER TABLE financial_statements ALTER COLUMN value TYPE (NUMERIC\(\d+, \d+\))",
        block,
    )
    assert widened is not None
    assert widened.group(1) == "NUMERIC(38, 6)"


def test_when_normalize_models_is_rendered_then_null_start_date_rows_are_deduped_first() -> (
    None
):
    """Duplicate NULL-date rows are removed before they share one sentinel date.

    ``uq_insider_tx_row`` treats NULLs as distinct, so a populated table may
    hold several NULL-date rows of one insider and type; the sentinel backfill
    would put them all on the same key.
    """
    block = _revision_block(_render_offline(), "ecf0a9a2bfdc", "b0b30daa80c5")

    delete_at = block.index("DELETE FROM insider_transactions")
    backfill_at = block.index("UPDATE insider_transactions SET start_date")
    assert delete_at < backfill_at
