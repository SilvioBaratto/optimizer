"""Per-scope tally of how Yahoo answered each yfinance request.

yfinance requests end in ``None`` both when Yahoo has nothing to say and when
the request never got an answer, so a caller that only looks at return values
cannot tell a quiet day from an outage. The resilience layer records each
request's outcome here instead. The tally lives in a ``ContextVar`` so
concurrent workers keep separate scopes and the timeout watchdog, which runs a
call in another thread, can be given the caller's scope.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class RequestOutcomes:
    """How Yahoo responded to the requests of one tracking scope.

    Attributes:
        answered: Requests Yahoo responded to, with data, an empty result or
            content that could not be parsed.
        failures: One ``"ErrorType: message"`` line per request that got no
            answer after its retries (unreachable, rate limited, timed out).
    """

    answered: int = 0
    failures: list[str] = field(default_factory=list)


_current: ContextVar[RequestOutcomes | None] = ContextVar(
    "yfinance_request_outcomes", default=None
)


@contextmanager
def track_request_outcomes() -> Iterator[RequestOutcomes]:
    """Record the outcome of every yfinance request made inside the block.

    Yields:
        The live tally, complete once the block exits. Requests made on other
        threads count only when those threads run in a copy of this context.
    """
    outcomes = RequestOutcomes()
    token = _current.set(outcomes)
    try:
        yield outcomes
    finally:
        _current.reset(token)


def record_answer() -> None:
    """Count a request Yahoo answered; a no-op outside a tracking scope."""
    outcomes = _current.get()
    if outcomes is not None:
        outcomes.answered += 1


def record_failure(reason: str) -> None:
    """Record a request that got no answer; a no-op outside a tracking scope.

    Args:
        reason: Short description of why no answer arrived.
    """
    outcomes = _current.get()
    if outcomes is not None:
        outcomes.failures.append(reason)
