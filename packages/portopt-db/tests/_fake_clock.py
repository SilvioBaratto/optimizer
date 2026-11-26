"""A hand-moved clock for ``DbCircuitBreaker(clock=...)`` in the breaker tests.

A test advances it past a reset timeout instead of waiting for one.
"""

from __future__ import annotations


class FakeClock:
    """Monotonic fake clock the test moves by hand.

    Calling it returns the current reading, as ``time.monotonic`` does.
    """

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds``."""
        self.now += seconds
