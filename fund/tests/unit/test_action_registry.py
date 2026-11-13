"""The ``ActionKind`` registry ``apply_action`` dispatches through.

Validates ``fund/src/fund/actions.py`` (``ActionKind``, ``ACTION_HANDLERS``,
``ActionRequest`` and its ``ActionTarget`` / ``ActionPayload`` / ``ActionDeps``
groups): every kind has a handler, the kinds are exactly the strings
``apply_action`` accepts, an unknown kind keeps its error text, the request carries
every keyword of ``apply_action`` in exactly one group, and ``apply_action`` hands
the handler of the kind a request built from its keywords.
``test_apply_action_dispatch_contract.py`` pins what each handler then receives. No
requirement id is registered for the adviser actions yet.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import uuid
from types import SimpleNamespace

import pytest

from fund import actions
from fund.actions import (
    ACTION_HANDLERS,
    ActionDeps,
    ActionKind,
    ActionPayload,
    ActionRequest,
    ActionTarget,
    apply_action,
)

SESSION = SimpleNamespace(name="session")
ACCEPTED = {
    "approve",
    "reject",
    "respond",
    "killswitch",
    "edit",
    "freeze",
    "rearm",
    "mandate_edit",
    "guardrail_override",
    "new_portfolio",
    "profile",
    "run",
}
GROUPS = (ActionTarget, ActionPayload, ActionDeps)


def test_apply_action_rejects_unknown_kind() -> None:
    """An action no kind names keeps today's error text, and nothing is handled."""
    unknown = "unknown action 'frobnicate'"

    with pytest.raises(ValueError, match=f"^{re.escape(unknown)}$"):
        apply_action("frobnicate", session=SESSION, run_id=uuid.uuid4())


def test_every_action_kind_has_a_handler() -> None:
    """The registry has exactly one callable handler per kind."""
    assert set(ACTION_HANDLERS) == set(ActionKind)
    assert all(callable(handler) for handler in ACTION_HANDLERS.values())


def test_the_kinds_are_the_strings_apply_action_accepts() -> None:
    """One kind per accepted action string, each equal to its string."""
    assert {kind.value for kind in ActionKind} == ACCEPTED
    assert all(kind == kind.value for kind in ActionKind)


def test_the_request_carries_every_keyword_of_apply_action_once() -> None:
    """The three groups split the keywords between them with nothing left over."""
    keywords = set(inspect.signature(apply_action).parameters) - {"action"}
    fields = [field.name for group in GROUPS for field in dataclasses.fields(group)]

    assert sorted(fields) == sorted(keywords)
    assert [field.name for field in dataclasses.fields(ActionRequest)] == [
        "target",
        "payload",
        "deps",
    ]


@pytest.mark.parametrize("group", [*GROUPS, ActionRequest])
def test_the_request_and_its_groups_are_frozen(group) -> None:
    """A handler cannot change what the caller asked for."""
    assert group.__dataclass_params__.frozen


@pytest.mark.parametrize("kind", list(ActionKind))
def test_apply_action_hands_the_kinds_handler_a_request_of_its_keywords(
    monkeypatch, kind
) -> None:
    """Each keyword lands in its group under its own name."""
    received: list[ActionRequest] = []
    answer = SimpleNamespace(name="result")

    def handler(request: ActionRequest) -> SimpleNamespace:
        """Record the request and answer."""
        received.append(request)
        return answer

    monkeypatch.setitem(ACTION_HANDLERS, kind, handler)
    run_id, portfolio_id = uuid.uuid4(), uuid.uuid4()

    result = apply_action(
        kind.value,
        session=SESSION,
        run_id=run_id,
        portfolio_id=portfolio_id,
        scope="fund",
        rationale="because",
        model=actions,
    )

    assert result is answer
    (request,) = received
    assert request.target == ActionTarget(
        run_id=run_id, portfolio_id=portfolio_id, scope="fund"
    )
    assert request.payload.rationale == "because"
    assert request.payload.actor == "human"
    assert request.deps.session is SESSION
    assert request.deps.model is actions
