"""Settable wall clock for asserting when a service stamps a timestamp.

No DB, no I/O — imported directly by tests (not used as a fixture).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

_START = datetime(2026, 10, 1, 4, 0, tzinfo=timezone.utc)


class FakeClock:
    """A frozen clock that only moves when the code under test does work.

    Patch ``datetime_cls`` over a service module's ``datetime`` and wrap the
    service's collaborators in ``advancing`` so that time passes mid-run; a
    timestamp taken before the work then differs from one taken after it.
    """

    def __init__(self, start: datetime = _START) -> None:
        self.start = start
        self.current = start

    @property
    def datetime_cls(self) -> type[datetime]:
        """A ``datetime`` subclass whose ``now()`` reads this clock."""
        clock = self

        class _Fake(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
                return clock.current

        return _Fake

    def advancing(
        self, result: Any, step: timedelta = timedelta(hours=1)
    ) -> Callable[..., Any]:
        """Wrap *result* in a mock side effect that moves the clock when called.

        Args:
            result: Value to return, or a callable receiving the call arguments.
            step: How far each call advances the clock.

        Returns:
            A callable suitable for ``Mock.side_effect``.
        """

        def _call(*args: Any, **kwargs: Any) -> Any:
            self.current += step
            return result(*args, **kwargs) if callable(result) else result

        return _call

    def assert_finished_at_marks_run_end(self, on_progress: Any) -> None:
        """Assert the completed ``finished_at`` is the clock reading after the work."""
        assert self.current > self.start, "the run never advanced the clock"
        for call in on_progress.call_args_list:
            if call.kwargs.get("status") == "completed":
                finished = datetime.fromisoformat(call.kwargs["finished_at"])
                assert finished == self.current, (
                    f"finished_at {finished.isoformat()} is not the end of the run "
                    f"({self.current.isoformat()})"
                )
                return
        raise AssertionError("service never reported status='completed'")
