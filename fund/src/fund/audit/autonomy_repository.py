"""``AutonomyRepository`` — fund-side behavior over ``fund_autonomy_state``.

The ``fund_autonomy_state`` *model* lives in ``portopt_db`` (shared schema,
Alembic-owned); its *behavior* lives here, mirroring the ``portfolio_mandates`` /
``human_audit_log`` model/repo split. Sits on ``portopt_db.repository``'s
``RepositoryBase`` and opens no session of its own — the caller injects a sync
session. No ``commit``: the caller owns the transaction boundary.

This module is deliberately agent-stack-free (``portopt_db`` + ``sqlalchemy`` +
stdlib only). Callers that must stay free of ``deepagents``/``langgraph`` — the
scheduler and worker entrypoints — import it from this submodule directly rather
than via ``fund.audit`` (whose package ``__init__`` eagerly loads the langgraph
checkpointer bootstrap).
"""

from __future__ import annotations

import datetime as dt
import uuid
from enum import Enum
from typing import Any, cast

from portopt_db.models import FundAutonomyState
from portopt_db.repository import RepositoryBase
from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult


class AutonomyScope(str, Enum):
    """The two autonomy scopes; fund scope wins over portfolio scope."""

    FUND = "fund"
    PORTFOLIO = "portfolio"


class AutonomyState(str, Enum):
    """A scope's posture: autonomous (``ARMED``) or halted (``FROZEN``)."""

    ARMED = "armed"
    FROZEN = "frozen"


class AutonomyRepository(RepositoryBase):
    """Read/flip the per-scope autonomy posture on an injected sync session."""

    def fund_state(self) -> FundAutonomyState:
        """Return the single fund-scope row, creating it armed+gated if absent."""
        return self._get_or_create(AutonomyScope.FUND, None)

    def portfolio_state(self, portfolio_id: uuid.UUID) -> FundAutonomyState:
        """Return a portfolio's row, creating it armed+gated if absent.

        Args:
            portfolio_id: The portfolio whose autonomy row to fetch or create.

        Returns:
            The portfolio's (possibly freshly created) autonomy-state row.
        """
        return self._get_or_create(AutonomyScope.PORTFOLIO, portfolio_id)

    def fund_posture(self) -> str:
        """The fund scope's state string **without** creating the row (defaults armed).

        A read-only accessor for display paths (the header/Posture glyph) that must not
        mutate the DB — unlike :meth:`fund_state`, it never inserts a row, so it is safe
        to call from a short read-only session interleaved with other work.
        """
        row = self._get(AutonomyScope.FUND, None)
        return row.state if row is not None else AutonomyState.ARMED.value

    def portfolio_posture(self, portfolio_id: uuid.UUID) -> str:
        """A portfolio's state string **without** creating the row (defaults armed).

        The read-only counterpart to :meth:`portfolio_state`; see
        :meth:`fund_posture`.

        Args:
            portfolio_id: The portfolio whose posture to read.

        Returns:
            ``"frozen"`` when the portfolio is explicitly frozen, else ``"armed"``.
        """
        row = self._get(AutonomyScope.PORTFOLIO, portfolio_id)
        return row.state if row is not None else AutonomyState.ARMED.value

    def set_state(
        self,
        scope: AutonomyScope,
        portfolio_id: uuid.UUID | None,
        state: AutonomyState,
        reason: str | None = None,
    ) -> FundAutonomyState:
        """Set a scope's posture, stamping or clearing the frozen provenance.

        Freezing records ``reason`` and ``frozen_at``; re-arming clears both, so
        the frozen provenance never lingers on an armed row.

        Args:
            scope: Fund or portfolio scope.
            portfolio_id: The portfolio for ``PORTFOLIO`` scope; ``None`` for the
                single fund row.
            state: The posture to set (``ARMED`` or ``FROZEN``).
            reason: Free-text reason recorded when freezing; ignored on re-arm.

        Returns:
            The updated, flushed row.
        """
        row = self._get_or_create(scope, portfolio_id)
        row.state = state.value
        if state is AutonomyState.FROZEN:
            row.frozen_at = dt.datetime.now(dt.timezone.utc)
            row.frozen_reason = reason
        else:
            row.frozen_at = None
            row.frozen_reason = None
        self.session.flush()
        return row

    def consume_first_round(self, portfolio_id: uuid.UUID) -> bool:
        """Atomically consume a portfolio's one-time first-round gate.

        The portfolio's row is created armed+gated if absent, then a single
        ``WHERE first_round_gated`` guarded UPDATE flips the flag. The guard makes
        the flip idempotent and race-safe: exactly one caller sees ``True``.

        Args:
            portfolio_id: The portfolio whose first round to consume.

        Returns:
            ``True`` the first time (the round was gated and is now consumed),
            ``False`` on every later call.
        """
        row = self.portfolio_state(portfolio_id)
        result = self.session.execute(
            update(FundAutonomyState)
            .where(
                FundAutonomyState.scope == AutonomyScope.PORTFOLIO.value,
                FundAutonomyState.portfolio_id == portfolio_id,
                FundAutonomyState.first_round_gated.is_(True),
            )
            .values(first_round_gated=False)
            # The Core UPDATE bypasses the identity map; expire the stale ORM row
            # below so a later read reflects the flipped flag.
            .execution_options(synchronize_session=False)
        )
        self.session.expire(row)
        # SQLAlchemy's ``Result`` stub omits ``rowcount`` — it lives on the
        # ``CursorResult`` a DML ``execute`` actually returns — so the cast restores it.
        return bool(cast("CursorResult[Any]", result).rowcount == 1)

    def _get(
        self, scope: AutonomyScope, portfolio_id: uuid.UUID | None
    ) -> FundAutonomyState | None:
        stmt = select(FundAutonomyState).where(FundAutonomyState.scope == scope.value)
        # The fund row carries a genuine NULL portfolio_id — match with IS NULL,
        # never ``== None`` (which SQL evaluates to UNKNOWN, matching nothing).
        if portfolio_id is None:
            stmt = stmt.where(FundAutonomyState.portfolio_id.is_(None))
        else:
            stmt = stmt.where(FundAutonomyState.portfolio_id == portfolio_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def _get_or_create(
        self, scope: AutonomyScope, portfolio_id: uuid.UUID | None
    ) -> FundAutonomyState:
        row = self._get(scope, portfolio_id)
        if row is None:
            row = FundAutonomyState(scope=scope.value, portfolio_id=portfolio_id)
            self.session.add(row)
            self.session.flush()
        return row


__all__ = ["AutonomyRepository", "AutonomyScope", "AutonomyState"]
