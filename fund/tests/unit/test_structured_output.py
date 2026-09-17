"""Task 6 — ``fund.schemas.structured.structured_call`` (validate → retry → fallback).

Every structured LLM step in the fund routes through ``structured_call``. It binds
``with_structured_output(schema, method="function_calling")`` (D4 pin), invokes the
model, validates the result with pydantic, retries the primary on failure, then
hands off once to an optional ``fallback`` model, and only then raises
``StructuredOutputError`` — never a bare ``pydantic.ValidationError``.

The helper is model-agnostic (duck-typed over a ``Protocol``); these tests drive
all four paths with an in-file **mock** chat model that records its call counts,
so no live LLM is touched and "no network" is structurally guaranteed (SPEC §5).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pydantic
import pytest
from pydantic import BaseModel, ConfigDict

from fund.schemas import structured
from fund.schemas.structured import StructuredOutputError, structured_call

_MESSAGES = [{"role": "user", "content": "pick a number"}]

# A malformed payload: missing the required ``value`` field → ValidationError.
_MALFORMED: dict[str, object] = {"note": "no value field"}


class _Answer(BaseModel):
    """Trivial local schema exercised by the helper (frozen, like the real ones)."""

    model_config = ConfigDict(frozen=True)

    value: int


class _MockRunnable:
    """What ``with_structured_output`` returns: a runnable with ``.invoke``."""

    def __init__(self, model: _MockChatModel) -> None:
        self.model = model

    def invoke(self, messages: object) -> object:
        return self.model.yield_next(messages)


class _MockChatModel:
    """A scripted, fully local stand-in for a chat model — records call counts.

    ``outcomes`` is consumed one per ``.invoke``: a ``BaseModel`` instance or a
    ``dict`` is returned, a ``BaseException`` is raised. No network, ever.
    """

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.with_structured_output_calls = 0
        self.invoke_calls = 0
        self.methods: list[str] = []
        self.messages_seen: list[object] = []

    def with_structured_output(
        self, schema: type[BaseModel], *, method: str = "function_calling"
    ) -> _MockRunnable:
        self.with_structured_output_calls += 1
        self.methods.append(method)
        self.last_schema = schema
        return _MockRunnable(self)

    def yield_next(self, messages: object) -> object:
        self.invoke_calls += 1
        self.messages_seen.append(messages)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


# -- success -----------------------------------------------------------------


def test_returns_validated_instance_on_success():
    model = _MockChatModel([_Answer(value=7)])
    result = structured_call(model, _Answer, _MESSAGES)
    assert result == _Answer(value=7)
    assert model.invoke_calls == 1
    assert model.with_structured_output_calls == 1


def test_binds_function_calling_method_by_default():
    # D4 pin: json_schema fell through in Phase-0 probing.
    model = _MockChatModel([_Answer(value=7)])
    structured_call(model, _Answer, _MESSAGES)
    assert model.methods == ["function_calling"]


def test_coerces_a_valid_dict_payload_to_the_schema():
    model = _MockChatModel([{"value": 3}])
    result = structured_call(model, _Answer, _MESSAGES)
    assert result == _Answer(value=3)


def test_forwards_the_messages_to_the_model():
    model = _MockChatModel([_Answer(value=7)])
    structured_call(model, _Answer, _MESSAGES)
    assert model.messages_seen == [_MESSAGES]


# -- retry on the primary ----------------------------------------------------


def test_retries_the_primary_once_then_returns_the_valid_instance():
    model = _MockChatModel([dict(_MALFORMED), _Answer(value=7)])
    result = structured_call(model, _Answer, _MESSAGES)
    assert result == _Answer(value=7)
    assert model.invoke_calls == 2


def test_retries_when_the_model_raises():
    model = _MockChatModel([RuntimeError("model exploded"), _Answer(value=7)])
    result = structured_call(model, _Answer, _MESSAGES)
    assert result == _Answer(value=7)
    assert model.invoke_calls == 2


def test_retries_argument_controls_the_number_of_primary_attempts():
    model = _MockChatModel([dict(_MALFORMED), dict(_MALFORMED), _Answer(value=5)])
    result = structured_call(model, _Answer, _MESSAGES, retries=2)
    assert result == _Answer(value=5)
    assert model.invoke_calls == 3


def test_zero_retries_makes_a_single_primary_attempt():
    model = _MockChatModel([dict(_MALFORMED)])
    with pytest.raises(StructuredOutputError):
        structured_call(model, _Answer, _MESSAGES, retries=0)
    assert model.invoke_calls == 1


# -- fallback ----------------------------------------------------------------


def test_falls_back_when_the_primary_is_always_malformed():
    primary = _MockChatModel([dict(_MALFORMED), dict(_MALFORMED)])
    fallback = _MockChatModel([_Answer(value=9)])
    result = structured_call(primary, _Answer, _MESSAGES, fallback=fallback)
    assert result == _Answer(value=9)
    assert primary.invoke_calls == 2  # retries=1 → two primary attempts
    assert fallback.invoke_calls == 1  # fallback tried exactly once


# -- exhaustion --------------------------------------------------------------


def test_raises_structured_output_error_when_all_attempts_are_exhausted():
    primary = _MockChatModel([dict(_MALFORMED), dict(_MALFORMED)])
    fallback = _MockChatModel([dict(_MALFORMED)])
    with pytest.raises(StructuredOutputError):
        structured_call(primary, _Answer, _MESSAGES, fallback=fallback)
    assert primary.invoke_calls == 2
    assert fallback.invoke_calls == 1


def test_raises_without_a_fallback_when_the_primary_is_exhausted():
    model = _MockChatModel([dict(_MALFORMED), dict(_MALFORMED)])
    with pytest.raises(StructuredOutputError):
        structured_call(model, _Answer, _MESSAGES)
    assert model.invoke_calls == 2


def test_exhaustion_raises_structured_error_not_bare_validation_error():
    model = _MockChatModel([dict(_MALFORMED), dict(_MALFORMED)])
    with pytest.raises(StructuredOutputError) as excinfo:
        structured_call(model, _Answer, _MESSAGES)
    # Not a leaked pydantic error, but it chains the last failure as the cause.
    assert not isinstance(excinfo.value, pydantic.ValidationError)
    assert isinstance(excinfo.value.__cause__, pydantic.ValidationError)


# -- no network --------------------------------------------------------------


def test_makes_no_calls_beyond_the_scripted_invocations():
    # The mock is fully local: the only outbound calls it can make are the
    # scripted `.invoke`s it counts. Asserting the exact counts proves the helper
    # performs no hidden (network) round-trips beyond the retry/fallback budget.
    primary = _MockChatModel([dict(_MALFORMED), dict(_MALFORMED)])
    fallback = _MockChatModel([_Answer(value=1)])
    structured_call(primary, _Answer, _MESSAGES, fallback=fallback)
    assert primary.invoke_calls + fallback.invoke_calls == 3
    assert primary.with_structured_output_calls == 2
    assert fallback.with_structured_output_calls == 1


# -- import hygiene (Protocol only, no agent/ingestion imports) ---------------


def test_module_imports_no_agent_or_ingestion_packages():
    tree = ast.parse(Path(structured.__file__).read_text(encoding="utf-8"))
    top_level: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert {"deepagents", "langchain_ollama", "app"}.isdisjoint(top_level)
