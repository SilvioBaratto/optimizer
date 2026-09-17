"""Model-agnostic structured-output helper (Task 6): validate → retry → fallback.

Every structured LLM step in the fund routes through ``structured_call``: it binds
``with_structured_output(schema, method="function_calling")`` (D4 pin —
``json_schema`` fell through in Phase-0 probing), invokes the model, and validates
the result with pydantic. A weak model that returns malformed output (or raises) is
retried on the primary ``retries`` times, then handed once to an optional
``fallback`` model. Only when every attempt fails does it raise
``StructuredOutputError`` (chaining the last failure) — never a bare
``pydantic.ValidationError`` leaking to the caller.

The helper is deliberately **model-agnostic**: it is duck-typed over a
``SupportsStructuredOutput`` ``Protocol`` and imports nothing from ``deepagents`` /
``langchain_ollama`` / ``app`` (the concrete DeepSeek-on-Ollama primary/fallback
instances are wired at the call site in Fase 7). A mock model drives every test
path with zero network.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

__all__ = [
    "StructuredOutputError",
    "SupportsStructuredOutput",
    "structured_call",
]

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_DEFAULT_METHOD = "function_calling"


class StructuredOutputError(RuntimeError):
    """Raised when every attempt (primary retries + fallback) fails to validate.

    Chains the last underlying failure as ``__cause__`` so the caller can inspect
    what went wrong without having to catch a bare ``pydantic.ValidationError``.
    """


class _SupportsInvoke(Protocol):
    """The runnable ``with_structured_output`` returns — anything with ``invoke``."""

    def invoke(self, messages: Any, /) -> Any: ...


class SupportsStructuredOutput(Protocol):
    """A chat model exposing langchain's ``with_structured_output`` binding.

    Structural type only: any object with this method satisfies it, so the
    concrete ``ChatOllama`` primary/fallback are never imported here.
    """

    def with_structured_output(
        self, schema: type[BaseModel], *, method: str = _DEFAULT_METHOD
    ) -> _SupportsInvoke: ...


def _invoke_once(
    model: SupportsStructuredOutput,
    schema: type[SchemaT],
    messages: Any,
    method: str,
) -> SchemaT:
    """One attempt: bind, invoke, coerce to a validated ``schema`` instance.

    Re-raises whatever the model raises, or ``pydantic.ValidationError`` when the
    returned payload does not validate against ``schema``.
    """
    runnable = model.with_structured_output(schema, method=method)
    result = runnable.invoke(messages)
    if isinstance(result, schema):
        return result
    # A weak model may hand back a dict / partial payload instead of the instance.
    return schema.model_validate(result)


def structured_call(
    model: SupportsStructuredOutput,
    schema: type[SchemaT],
    messages: Any,
    *,
    method: str = _DEFAULT_METHOD,
    retries: int = 1,
    fallback: SupportsStructuredOutput | None = None,
) -> SchemaT:
    """Return a validated ``schema`` instance; retry the primary, then the fallback.

    Attempts the primary ``model`` ``retries + 1`` times, then (when given) the
    ``fallback`` model exactly once. Any failure — a model error or a payload that
    fails pydantic validation — is swallowed and the next attempt tried; only after
    every attempt is exhausted does it raise ``StructuredOutputError`` (chaining the
    last failure). ``method`` defaults to ``"function_calling"`` (D4 pin).
    """
    candidates: list[SupportsStructuredOutput] = [model] * (retries + 1)
    if fallback is not None:
        candidates.append(fallback)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return _invoke_once(candidate, schema, messages, method)
        except Exception as exc:  # duck-typed model may raise anything; validation too
            last_error = exc
    raise StructuredOutputError(
        f"structured output for {schema.__name__} failed after "
        f"{len(candidates)} attempt(s)"
    ) from last_error
