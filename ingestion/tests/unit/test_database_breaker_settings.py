"""The database circuit breaker's thresholds come from the daemon's environment.

Validates the ``database_breaker_*`` fields of ``app.config.Settings``, read from
``DATABASE_BREAKER_FAILURE_THRESHOLD`` and ``DATABASE_BREAKER_RESET_TIMEOUT_S``,
and their path into ``DbConfig`` through ``app.database._build_config``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config import Settings

_THRESHOLD_VAR = "DATABASE_BREAKER_FAILURE_THRESHOLD"
_TIMEOUT_VAR = "DATABASE_BREAKER_RESET_TIMEOUT_S"


@pytest.fixture(autouse=True)
def _unset_breaker_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from the defaults, whatever the shell exports."""
    monkeypatch.delenv(_THRESHOLD_VAR, raising=False)
    monkeypatch.delenv(_TIMEOUT_VAR, raising=False)


def _build_config_from(settings: Settings):
    from app.database import _build_config

    with patch("app.database.settings", settings):
        return _build_config()


def test_defaults_are_five_failures_and_thirty_seconds() -> None:
    """Unset, the daemon keeps the breaker's defaults."""
    settings = Settings(_env_file=None)

    assert settings.database_breaker_failure_threshold == 5
    assert settings.database_breaker_reset_timeout_s == 30.0
    config = _build_config_from(settings)
    assert config.breaker_failure_threshold == 5
    assert config.breaker_reset_timeout_s == 30.0


def test_environment_overrides_reach_the_database_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both variables travel from the environment into ``DbConfig``."""
    monkeypatch.setenv(_THRESHOLD_VAR, "8")
    monkeypatch.setenv(_TIMEOUT_VAR, "12.5")

    config = _build_config_from(Settings(_env_file=None))

    assert config.breaker_failure_threshold == 8
    assert config.breaker_reset_timeout_s == 12.5


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
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    """A zero, negative, non-numeric or non-finite value fails at load."""
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


# The spellings fund is tested against too: (text, threshold, timeout); None = refused.
_SHARED_EDGE_CASES = [
    ("5.0", None, 5.0),
    ("", None, None),
    (" 7 ", 7, 7.0),
    ("1_0", None, None),
    ("+3", None, None),
    ("true", None, None),
    ("inf", None, None),
    ("nan", None, None),
]


@pytest.mark.parametrize(
    ("variable", "text", "expected"),
    [(_THRESHOLD_VAR, text, threshold) for text, threshold, _ in _SHARED_EDGE_CASES]
    + [(_TIMEOUT_VAR, text, timeout) for text, _, timeout in _SHARED_EDGE_CASES],
)
def test_the_daemon_reads_the_same_spellings_as_fund(
    monkeypatch: pytest.MonkeyPatch, variable: str, text: str, expected: float | None
) -> None:
    """Plain digits (a fraction only for the timeout) or a refusal naming the variable."""
    monkeypatch.setenv(variable, text)
    field = {
        _THRESHOLD_VAR: "database_breaker_failure_threshold",
        _TIMEOUT_VAR: "database_breaker_reset_timeout_s",
    }[variable]

    if expected is None:
        with pytest.raises(ValidationError, match=f"{variable} must be"):
            Settings(_env_file=None)
    else:
        assert getattr(Settings(_env_file=None), field) == expected
