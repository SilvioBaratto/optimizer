"""``fund.database`` wiring: server-side lock timeout reaches the DbConfig."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from fund.config import FundConfig, load_config


def test_lock_timeout_defaults_on_and_is_env_sourced():
    assert FundConfig().db_lock_timeout_ms == 60_000
    assert load_config(env={}).db_lock_timeout_ms == 60_000
    assert (
        load_config(env={"DATABASE_LOCK_TIMEOUT_MS": "5000"}).db_lock_timeout_ms
        == 5_000
    )
    # 0 is the documented "off" switch and must survive the env round-trip.
    assert load_config(env={"DATABASE_LOCK_TIMEOUT_MS": "0"}).db_lock_timeout_ms == 0


def test_build_config_passes_lock_timeout_to_dbconfig():
    import fund.database as db_mod

    with patch.object(
        db_mod, "settings", replace(FundConfig(), db_lock_timeout_ms=7_000)
    ):
        cfg = db_mod._build_config()

    assert cfg.lock_timeout_ms == 7_000
    assert cfg.connect_args()["options"] == "-c lock_timeout=7000"


def test_default_fund_sessions_are_protected_from_unbounded_lock_waits():
    import fund.database as db_mod

    with patch.object(db_mod, "settings", FundConfig()):
        options = db_mod._build_config().connect_args()["options"]

    assert options == "-c lock_timeout=60000"
