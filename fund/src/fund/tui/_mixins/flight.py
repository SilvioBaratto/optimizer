"""Single-flight guards for the shell's thread workers.

One gate-mutating action, and one profile / run / stream round, may run at a
time; a late launch is refused with a status line instead of racing the first.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Concatenate, ParamSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from fund.tui.app import ShellApp


_P = ParamSpec("_P")


def _single_flight(
    label: str,
    *,
    refused: Callable[Concatenate[ShellApp, _P], str] | None = None,
) -> Callable[
    [Callable[Concatenate[ShellApp, _P], None]],
    Callable[Concatenate[ShellApp, _P], None],
]:
    """Run a gate-mutating thread worker only while no other one is in flight.

    Textual's ``exclusive=True`` merely flags the *older* ``Worker`` cancelled; its
    thread keeps running. Without this guard a repeat or conflicting keypress during
    the seconds a graph resume takes starts a second thread on the same paused run, so
    approve + reject both execute and whichever commits last decides the run's final
    state. A non-blocking lock refuses the late action with a status line rather than
    queueing a decision made against a stale screen, and it is released on every exit
    path, failures included. The portfolio / fund freeze is deliberately not guarded:
    see :meth:`ShellApp._freeze_worker`.

    Args:
        label: Present-tense phrase shown on the status line while the action runs.
        refused: Called with the worker's own arguments when the action is turned away;
            returns what was not done (for example the adviser's typed directive), which
            is appended to the status line so the refusal does not swallow it.

    Returns:
        A decorator for a ``ShellApp`` worker body, applied beneath ``@work``.
    """

    def decorate(
        body: Callable[Concatenate[ShellApp, _P], None],
    ) -> Callable[Concatenate[ShellApp, _P], None]:
        @functools.wraps(body)
        def guarded(self: ShellApp, *args: _P.args, **kwargs: _P.kwargs) -> None:
            if not self._action_lock.acquire(blocking=False):
                message = "another action is still in progress — wait for it to finish"
                if refused is not None:
                    message += f" ({refused(self, *args, **kwargs)})"
                self.call_from_thread(self._set_status, message)
                return
            try:
                self.call_from_thread(self._set_status, f"{label}…")
                body(self, *args, **kwargs)
            finally:
                self._action_lock.release()

        return guarded

    return decorate


def _round_flight(
    body: Callable[Concatenate[ShellApp, _P], None],
) -> Callable[Concatenate[ShellApp, _P], None]:
    """Run a profile / run / stream thread worker only while no other one is alive.

    Those workers used to rely on ``exclusive=True``, which cancels the older Textual
    handle (so it reads as finished) while its thread keeps running: a second round or
    a re-profile could then start next to the first and replace the ConstraintSet it is
    using. The lock is held by the thread itself, so :meth:`ShellApp._round_in_flight`
    stays true for as long as the work really runs.

    Args:
        body: A ``ShellApp`` worker body, applied beneath ``@work``.

    Returns:
        The guarded body; a late launch sets a status line and does nothing.
    """

    @functools.wraps(body)
    def guarded(self: ShellApp, *args: _P.args, **kwargs: _P.kwargs) -> None:
        if not self._round_lock.acquire(blocking=False):
            self.call_from_thread(self._set_status, _ROUND_BUSY)
            return
        try:
            body(self, *args, **kwargs)
        finally:
            self._round_lock.release()

    return guarded


_ROUND_BUSY = "a run or profile is already in progress — wait for it to finish"
