"""Grey out the shell's write actions while the database circuit breaker is open.

Reads need no gate: they go through the same breaker, which refuses them at once,
and each view keeps its last paint. A write would be refused the same way, but
only after the adviser had read a gate and answered a confirm, so the bindings
that start one are disabled instead and the status bar says why. A confirm that
was already open when the breaker opened still ends in a refused write, which the
action worker reports like any other failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from portopt_db.resilience import BreakerState

from fund.tui._mixins.host import ShellHost
from fund.tui.safe_text import describe_error

if TYPE_CHECKING:
    from portopt_db.resilience import CircuitBreaker

# Bindings whose action writes through the database, directly or through a run.
DB_WRITE_ACTIONS = frozenset(
    {
        "decide_approve",
        "decide_reject",
        "decide_killswitch",
        "rearm",
        "save_mandate",
        "override_guardrail",
        "new_portfolio",
        "profile_portfolio",
        "run_portfolio",
        "barge_in",
        "edit_decision",
        "steer_decision",
    }
)

# An open breaker whose timeout has passed refuses nothing until a call becomes
# the probe, yet the database is not known to be back; the next poll read probes.
_PROBE_PENDING = "database unavailable: the next read tests the connection"
_DISABLED_SUFFIX = " - write actions are disabled until it answers again"
_RECOVERED = "database reachable again - write actions are enabled"


class DbGateMixin(ShellHost):
    """Disables write bindings while the database breaker is not closed.

    Mixed into :class:`~fund.tui.app.ShellApp`. Off until
    :meth:`watch_db_breaker` names a breaker. Runs on the event loop only: the
    poll tick calls :meth:`_sync_db_gate`, and Textual calls :meth:`check_action`
    when it renders or runs a binding.
    """

    _db_breaker: CircuitBreaker | None = None
    # Why the database is unavailable as of the last sync; ``None`` keeps every
    # action enabled.
    _db_gate_text: str | None = None

    def watch_db_breaker(self, breaker: CircuitBreaker | None) -> None:
        """Disable the write actions whenever ``breaker`` is not closed.

        Safe before the shell is mounted: the next poll tick applies it.

        Args:
            breaker: The breaker of the manager the shell's ``session_factory``
                opens sessions through, or ``None`` to stop gating.
        """
        self._db_breaker = breaker

    def _db_refusal_text(self) -> str | None:
        """Say why the database is unavailable, or return None when it is not.

        Returns:
            One safe line for the status bar while the breaker is open or
            half-open, else ``None``.
        """
        breaker = self._db_breaker
        if breaker is None or breaker.state is BreakerState.CLOSED:
            return None
        refusal = breaker.refusal()
        return _PROBE_PENDING if refusal is None else describe_error(refusal)

    def _db_write_refusal(self) -> str | None:
        """The status line explaining a refused write, or None while writes run.

        Reads the gate as of the last poll, the same state ``check_action`` reads,
        so a write started any other way (an Ask decision verb) is refused exactly
        when the keys are greyed out.
        """
        text = self._db_gate_text
        return None if text is None else text + _DISABLED_SUFFIX

    def _sync_db_gate(self) -> None:
        """Re-read the breaker; when the gate flips, repaint the keys and say why."""
        text = self._db_refusal_text()
        flipped = (text is None) != (self._db_gate_text is None)
        self._db_gate_text = text
        if not flipped:
            return
        self.refresh_bindings()
        self._set_status(self._db_write_refusal() or _RECOVERED)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Report a write action as disabled while the database is unavailable.

        Args:
            action: Name of the action a binding or ``run_action`` would run.
            parameters: The action's arguments.

        Returns:
            ``None`` (shown greyed out, not run) for a write action while the gate
            is closed, else whatever Textual's default check decides.
        """
        if action in DB_WRITE_ACTIONS and self._db_gate_text is not None:
            return None
        return super().check_action(action, parameters)
