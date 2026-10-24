"""Model-agnostic structured-output helper: validate → retry → fallback.

Every structured LLM step in the fund routes through ``structured_call``: it binds
``with_structured_output(schema, method="function_calling")`` (``json_schema``
fell through in initial probing), invokes the model, and validates
the result with pydantic. A weak model that returns malformed output (or raises) is
retried on the primary ``retries`` times, then handed once to an optional
``fallback`` model. A retry after a validation failure carries the validation errors
back to the model (an identical request at temperature 0 would fail identically); a
failure that no retry can fix (an HTTP 401/403/404: bad key, forbidden, unknown
model) skips the remaining retries of that model; a transient one (429, 5xx, timeout,
connection) backs off first. Only when every attempt fails does it raise
``StructuredOutputError`` (naming and chaining the last failure) — never a bare
``pydantic.ValidationError`` leaking to the caller.

The helper is deliberately **model-agnostic**: it is duck-typed over a
``SupportsStructuredOutput`` ``Protocol`` and imports nothing from ``deepagents`` /
``langchain_ollama`` / ``app`` (the concrete DeepSeek-on-Ollama primary/fallback
instances are wired at the call site). A mock model drives every test
path with zero network.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

__all__ = [
    "StructuredOutputError",
    "SupportsStructuredOutput",
    "structured_call",
]

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_DEFAULT_METHOD = "function_calling"

_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 8.0
_ERROR_DETAIL_MAX_CHARS = 300

# 4xx statuses a later attempt can clear: request timeout, too early, rate limited.
_RETRYABLE_CLIENT_STATUSES = frozenset({408, 425, 429})

# Statuses that mean the credentials or the model name are wrong, never the sample.
# Other 4xx stay retryable: providers answer a weak model's malformed tool call with
# one (Groq returns 400 ``tool_use_failed``), and the next attempt can clear it.
_UNRECOVERABLE_CLIENT_STATUSES = frozenset({401, 403, 404})


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


def _http_status(exc: BaseException) -> int | None:
    """The HTTP status an SDK exception carries, whichever attribute it uses."""
    response = getattr(exc, "response", None)
    for holder, attr in (
        (exc, "status_code"),
        (response, "status_code"),
        (exc, "code"),
    ):
        value = getattr(holder, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    return None


def _is_non_retryable(exc: BaseException) -> bool:
    """True for a failure the same request can never recover from.

    A 401/403/404 (bad key, forbidden, unknown model) or a plain
    ``PermissionError``: repeating the call only repeats the failure, so the
    remaining retries of that model are skipped.
    """
    if isinstance(exc, PermissionError):
        return True
    return _http_status(exc) in _UNRECOVERABLE_CLIENT_STATUSES


def _is_transient(exc: BaseException) -> bool:
    """True for a failure that usually clears when the call is repeated later.

    Matched on status or builtin types, then on the class name because the
    provider SDKs' own timeout / connection errors cannot be imported here.
    """
    status = _http_status(exc)
    if status is not None:
        return status in _RETRYABLE_CLIENT_STATUSES or status >= 500
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__
    return any(token in name for token in ("Timeout", "Connect", "RateLimit"))


def _validation_summary(exc: ValidationError) -> str:
    """One ``field: message`` line per error — no echoed input, no doc URL."""
    return "\n".join(
        f"- {'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors(include_url=False, include_context=False)
    )


def _with_validation_feedback(messages: Any, exc: ValidationError) -> Any:
    """``messages`` plus a user turn that names what the last payload got wrong.

    Returns ``messages`` unchanged for an input shape that is neither a string nor a
    list, since no turn can be appended to it safely. Never mutates the caller's list.
    """
    feedback = {
        "role": "user",
        "content": (
            "Your previous structured output was rejected by validation:\n"
            f"{_validation_summary(exc)}\n"
            "Return a corrected object that fixes exactly these problems."
        ),
    }
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}, feedback]
    if isinstance(messages, list):
        return [*messages, feedback]
    return messages


def _describe(exc: BaseException | None) -> str:
    """A bounded, human-readable rendering of a failure for the raised message."""
    if exc is None:
        return "no attempt was made"
    if isinstance(exc, ValidationError):
        detail = _validation_summary(exc).replace("\n", "; ").lstrip("- ")
    else:
        detail = f"{type(exc).__name__}: {exc}"
    if len(detail) > _ERROR_DETAIL_MAX_CHARS:
        detail = detail[:_ERROR_DETAIL_MAX_CHARS] + "..."
    return detail


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

    Attempts the primary ``model`` up to ``retries + 1`` times, then (when given) the
    ``fallback`` model exactly once. Any failure — a model error or a payload that
    fails pydantic validation — is swallowed and the next attempt tried; only after
    every attempt is exhausted does it raise ``StructuredOutputError``, naming the
    last failure in its message and chaining it as ``__cause__``.

    A retry directly after a validation failure appends the errors to ``messages`` so
    the model can correct itself; any other retry, and the fallback, start from
    ``messages`` unchanged. A non-retryable failure (HTTP 401/403/404, or a
    ``PermissionError``) abandons the remaining retries of that model, which still
    leaves the fallback its turn; a transient one (429, 5xx, timeout, connection)
    sleeps with exponential backoff before the next attempt on the same model.
    ``method`` defaults to ``"function_calling"``.

    Args:
        model: The primary chat model.
        schema: The pydantic model the output must validate against.
        messages: The prompt: a string or a list of chat messages. A list is copied,
            never mutated.
        method: The ``with_structured_output`` binding method.
        retries: Extra primary attempts after the first.
        fallback: Optional model tried once after the primary is exhausted.

    Returns:
        A validated ``schema`` instance.

    Raises:
        StructuredOutputError: Every attempt failed.
    """
    groups: list[tuple[SupportsStructuredOutput, int]] = [(model, retries + 1)]
    if fallback is not None:
        groups.append((fallback, 1))

    attempts = 0
    last_error: Exception | None = None
    for candidate, budget in groups:
        # Feedback describes the payload the previous attempt produced, so it is
        # meaningful only to the very next attempt on that same model.
        prompt = messages
        for attempt in range(budget):
            attempts += 1
            try:
                return _invoke_once(candidate, schema, prompt, method)
            # A duck-typed model may raise anything, and so can validation.
            except Exception as exc:
                last_error = exc
                if _is_non_retryable(exc):
                    break
                if isinstance(exc, ValidationError):
                    prompt = _with_validation_feedback(messages, exc)
                else:
                    prompt = messages
                    if attempt + 1 < budget and _is_transient(exc):
                        time.sleep(
                            min(
                                _BACKOFF_MAX_SECONDS,
                                _BACKOFF_BASE_SECONDS * 2**attempt,
                            )
                        )
    raise StructuredOutputError(
        f"structured output for {schema.__name__} failed after "
        f"{attempts} attempt(s): {_describe(last_error)}"
    ) from last_error
