"""The database circuit breaker's thresholds come from fund's environment.

Validates ``fund.config.load_config`` reading ``DATABASE_BREAKER_FAILURE_THRESHOLD``
and ``DATABASE_BREAKER_RESET_TIMEOUT_S`` into ``FundConfig``, and
``fund.database._build_config`` passing them into ``DbConfig``. The variables and
defaults are the ingestion daemon's, so both processes share one setting.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fund.config import load_config

_THRESHOLD_VAR = "DATABASE_BREAKER_FAILURE_THRESHOLD"
_TIMEOUT_VAR = "DATABASE_BREAKER_RESET_TIMEOUT_S"


def _db_config_for(env: dict[str, str]):
    from fund.database import _build_config

    with patch("fund.database.settings", load_config(env=env)):
        return _build_config()


def test_defaults_are_five_failures_and_thirty_seconds() -> None:
    """Unset, fund keeps the breaker's defaults all the way into ``DbConfig``."""
    config = load_config(env={})

    assert config.db_breaker_failure_threshold == 5
    assert config.db_breaker_reset_timeout_s == 30.0
    db_config = _db_config_for({})
    assert db_config.breaker_failure_threshold == 5
    assert db_config.breaker_reset_timeout_s == 30.0


def test_environment_overrides_reach_the_database_config() -> None:
    """Both variables travel from the environment into ``DbConfig``."""
    env = {_THRESHOLD_VAR: "8", _TIMEOUT_VAR: "12.5"}

    db_config = _db_config_for(env)

    assert db_config.breaker_failure_threshold == 8
    assert db_config.breaker_reset_timeout_s == 12.5


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        (_THRESHOLD_VAR, "0"),
        (_THRESHOLD_VAR, "-2"),
        (_THRESHOLD_VAR, "abc"),
        (_THRESHOLD_VAR, "2.5"),
        (_TIMEOUT_VAR, "0"),
        (_TIMEOUT_VAR, "-1"),
        (_TIMEOUT_VAR, "abc"),
        (_TIMEOUT_VAR, "inf"),
        (_TIMEOUT_VAR, "nan"),
    ],
)
def test_a_value_that_cannot_drive_the_breaker_is_refused(
    variable: str, value: str
) -> None:
    """A zero, negative, non-numeric or non-finite value fails at load, named."""
    with pytest.raises(ValueError, match=variable):
        load_config(env={variable: value})
