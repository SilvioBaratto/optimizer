"""Which handler ``apply_action`` runs for each action, with exactly which arguments.

Validates the dispatch of ``fund/src/fund/actions.py`` (``apply_action``): the twelve
action strings it accepts, the private ``_apply_*`` function each one reaches and the
exact arguments it receives (defaults included), that the handler's result is
returned as it is, that a repeated idempotency key is replayed before the action is
even looked at, and the exact error for any other string. The handlers are replaced
by recorders; ``test_actions.py`` drives each action against a database. No
requirement id is registered for the adviser actions yet.
"""

from __future__ import annotations

import datetime as dt
import re
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from fund import actions
from fund.config import settings

SESSION = SimpleNamespace(name="session")
RUN = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
PID = uuid.UUID("11111111-2222-3333-4444-555555555555")
KEY = "gate-key"
CONFIG = SimpleNamespace(name="config")
# A distinct value for every keyword, so a value forwarded under the wrong name shows.
EVERY_KEYWORD: dict[str, Any] = {
    "run_id": RUN,
    "portfolio_id": PID,
    "checkpointer": SimpleNamespace(name="checkpointer"),
    "store": SimpleNamespace(name="store"),
    "model": SimpleNamespace(name="model"),
    "fallback": SimpleNamespace(name="fallback"),
    "mandate": SimpleNamespace(name="mandate"),
    "questionnaire": SimpleNamespace(name="questionnaire"),
    "asof": dt.date(2026, 11, 13),
    "base_currency": "EUR",
    "target": "max_position",
    "prev_value": 0.1,
    "new_value": {"AAA": 1.0},
    "actor": "adviser",
    "source": "tui",
    "override_flag": True,
    "suitability_flag": True,
    "idempotency_key": KEY,
    "rationale": "because",
    "pm_directive": "trim AAA",
    "scope": "portfolio",
    "config": CONFIG,
}
# The keywords each handler takes, in the order apply_action passes them.
_RESUME = (
    "session",
    "run_id",
    "portfolio_id",
    "checkpointer",
    "store",
    "model",
    "fallback",
    "pm_directive",
    "actor",
    "source",
    "override_flag",
    "suitability_flag",
    "idempotency_key",
    "rationale",
    "config",
)
HANDLER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "_apply_resume": _RESUME,
    "_apply_edit": (
        "session",
        "run_id",
        "portfolio_id",
        "checkpointer",
        "store",
        "model",
        "fallback",
        "new_value",
        "actor",
        "source",
        "idempotency_key",
        "rationale",
        "config",
    ),
    "_apply_freeze": (
        "session",
        "scope",
        "run_id",
        "portfolio_id",
        "actor",
        "source",
        "idempotency_key",
        "rationale",
    ),
    "_apply_rearm": (
        "session",
        "scope",
        "portfolio_id",
        "actor",
        "source",
        "idempotency_key",
        "rationale",
    ),
    "_apply_mandate_edit": (
        "session",
        "portfolio_id",
        "mandate",
        "actor",
        "source",
        "prev_value",
        "idempotency_key",
        "rationale",
    ),
    "_apply_guardrail_override": (
        "session",
        "portfolio_id",
        "target",
        "prev_value",
        "new_value",
        "actor",
        "source",
        "override_flag",
        "suitability_flag",
        "idempotency_key",
        "rationale",
    ),
    "_apply_new_portfolio": (
        "session",
        "portfolio_id",
        "mandate",
        "actor",
        "source",
        "idempotency_key",
        "rationale",
    ),
    "_apply_profile": (
        "session",
        "portfolio_id",
        "questionnaire",
        "model",
        "fallback",
        "checkpointer",
        "store",
        "base_currency",
        "asof",
        "actor",
        "source",
        "idempotency_key",
        "rationale",
        "config",
    ),
    "_apply_run": (
        "session",
        "portfolio_id",
        "mandate",
        "model",
        "fallback",
        "checkpointer",
        "store",
        "asof",
        "actor",
        "source",
        "idempotency_key",
        "rationale",
        "config",
    ),
}
# Every action string accepted today: the handler it reaches and, for the four
# resume-style actions, the positional action and the resume verb.
DISPATCH: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]] = [
    ("approve", "_apply_resume", ("approve",), {"decision": "approve"}),
    ("reject", "_apply_resume", ("reject",), {"decision": "reject"}),
    ("respond", "_apply_resume", ("respond",), {"decision": "respond"}),
    ("killswitch", "_apply_resume", ("killswitch",), {"decision": "reject"}),
    ("edit", "_apply_edit", (), {}),
    ("freeze", "_apply_freeze", (), {}),
    ("rearm", "_apply_rearm", (), {}),
    ("mandate_edit", "_apply_mandate_edit", (), {}),
    ("guardrail_override", "_apply_guardrail_override", (), {}),
    ("new_portfolio", "_apply_new_portfolio", (), {}),
    ("profile", "_apply_profile", (), {}),
    ("run", "_apply_run", (), {}),
]
_DEFAULTS: dict[str, Any] = {
    "run_id": None,
    "portfolio_id": None,
    "checkpointer": None,
    "store": None,
    "model": None,
    "fallback": None,
    "mandate": None,
    "questionnaire": None,
    "asof": None,
    "base_currency": None,
    "target": None,
    "prev_value": None,
    "new_value": None,
    "actor": "human",
    "source": None,
    "override_flag": False,
    "suitability_flag": False,
    "idempotency_key": None,
    "rationale": None,
    "pm_directive": None,
    "scope": None,
    "config": settings,
}


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Replace every handler and the replay lookup with recorders."""
    recorded: list[Any] = []

    def recorder(name: str) -> Any:
        """A stand-in for one handler that records its call."""

        def fake(*args: Any, **kwargs: Any) -> Any:
            """Record the call and return a result unique to it."""
            result = SimpleNamespace(handler=name)
            recorded.append((name, args, kwargs, result))
            return result

        return fake

    for name in HANDLER_KEYWORDS:
        monkeypatch.setattr(actions, name, recorder(name))

    def no_replay(session: Any, key: str) -> None:
        """Record the lookup and find nothing to replay."""
        recorded.append(("_replayed_action", (session, key), {}, None))

    monkeypatch.setattr(actions, "_replayed_action", no_replay)
    return recorded


def _expected_kwargs(
    handler: str, values: dict[str, Any], extra: dict[str, Any]
) -> dict[str, Any]:
    forwarded = {name: values[name] for name in HANDLER_KEYWORDS[handler]}
    return {**extra, **forwarded}


@pytest.mark.parametrize(("action", "handler", "args", "extra"), DISPATCH)
def test_each_action_reaches_its_handler_with_exactly_its_arguments(
    calls, action, handler, args, extra
) -> None:
    """Every keyword the handler takes is forwarded under its own name, no other."""
    result = actions.apply_action(action, session=SESSION, **EVERY_KEYWORD)

    values = {"session": SESSION, **EVERY_KEYWORD}
    (replay, call) = calls
    assert replay == ("_replayed_action", (SESSION, KEY), {}, None)
    assert call[:3] == (handler, args, _expected_kwargs(handler, values, extra))
    assert result is call[3]


@pytest.mark.parametrize(("action", "handler", "args", "extra"), DISPATCH)
def test_each_action_forwards_the_defaults_when_given_only_a_session(
    calls, action, handler, args, extra
) -> None:
    """The defaults reach the handler as they are; no replay is looked up."""
    actions.apply_action(action, session=SESSION)

    values = {"session": SESSION, **_DEFAULTS}
    ((name, got_args, got_kwargs, _),) = calls
    assert (name, got_args) == (handler, args)
    assert got_kwargs == _expected_kwargs(handler, values, extra)
    assert got_kwargs.get("config", settings) is settings


@pytest.mark.parametrize(
    "action", ["frobnicate", "Approve", " approve", "approve ", "", "edit_weights"]
)
def test_any_other_action_is_refused_with_its_name(calls, action) -> None:
    """The error names the action as written, and no handler runs."""
    with pytest.raises(
        ValueError, match=f"^{re.escape(f'unknown action {action!r}')}$"
    ):
        actions.apply_action(action, session=SESSION)

    assert calls == []


def test_the_unknown_action_error_carries_no_chained_cause(calls) -> None:
    """The refusal is raised on its own, not out of another error."""
    with pytest.raises(ValueError, match="unknown action") as raised:
        actions.apply_action("frobnicate", session=SESSION)

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_a_non_string_action_is_refused_by_its_repr(calls) -> None:
    """Whatever the caller passed is named in the error."""
    with pytest.raises(ValueError, match=r"^unknown action None$"):
        actions.apply_action(None, session=SESSION)  # type: ignore[arg-type]


def test_an_unknown_action_with_a_key_is_looked_up_for_replay_first(calls) -> None:
    """The replay lookup comes before the action is checked."""
    with pytest.raises(ValueError, match="unknown action 'frobnicate'"):
        actions.apply_action("frobnicate", session=SESSION, idempotency_key=KEY)

    assert calls == [("_replayed_action", (SESSION, KEY), {}, None)]


@pytest.mark.parametrize("action", ["approve", "frobnicate"])
def test_a_repeated_key_is_replayed_without_running_any_handler(
    monkeypatch, calls, action
) -> None:
    """The first result of a key is returned again, whatever the action now says."""
    first = SimpleNamespace(name="first result")
    monkeypatch.setattr(actions, "_replayed_action", lambda session, key: first)

    result = actions.apply_action(
        action, session=SESSION, run_id=RUN, idempotency_key=KEY
    )

    assert result is first
    assert calls == []


def test_the_model_free_actions_are_the_two_stops() -> None:
    """Only reject and killswitch settle without a model, store or checkpointer."""
    assert frozenset({"reject", "killswitch"}) == actions.MODEL_FREE_ACTIONS
