"""``AutonomyRepository`` — fund-side behavior over ``fund_autonomy_state``.

The ``fund_autonomy_state`` *model* lives in ``portopt_db`` (shared schema,
Alembic-owned); its *behavior* lives here, mirroring the ``portfolio_mandates`` /
``human_audit_log`` model/repo split. Sits on ``portopt_db.repository``'s
``RepositoryBase`` and opens no session of its own — the caller injects a sync
session. No ``commit``: the caller owns the transaction boundary.

This module is deliberately agent-stack-free (``portopt_db`` + ``sqlalchemy`` +
stdlib only), and so is importing it: ``fund.audit``'s package ``__init__`` loads the
langgraph checkpointer bootstrap only on demand. Callers that must stay free of
``deepagents``/``langgraph`` — the scheduler and worker entrypoints — rely on that.
"""

from __future__ import annotations

import datetime as dt
import uuid
from enum import Enum
from typing import Any, cast

from portopt_db.models import FundAutonomyState
from portopt_db.repository import RepositoryBase
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult

# ``fund_autonomy_state.frozen_reason`` is VARCHAR(500). Postgres rejects a longer
# value outright (SQLite stores it silently), so an oversized reason would abort the
# very freeze that carries it.
_FROZEN_REASON_MAX = 500


def _fit_frozen_reason(reason: str | None) -> str | None:
    """Clip ``reason`` to the column width, marking the cut with an ellipsis."""
    if reason is None or len(reason) <= _FROZEN_REASON_MAX:
        return reason
    return reason[: _FROZEN_REASON_MAX - 1] + "\u2026"


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
        to call from a short read-only session interleaved with other work. It always
        reads the database, never the session's cached copy of the row, so a freeze
        committed by another session is seen by a session that loaded the row earlier.
        """
        row = self._get(AutonomyScope.FUND, None, fresh=True)
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
        row = self._get(AutonomyScope.PORTFOLIO, portfolio_id, fresh=True)
        return row.state if row is not None else AutonomyState.ARMED.value

    def frozen_portfolio_ids(self) -> list[uuid.UUID]:
        """Ids of the portfolios whose own autonomy row is FROZEN, sorted ascending.

        Read-only like :meth:`portfolio_posture`: it never creates a row. The fund
        scope is not consulted — a frozen fund halts every portfolio regardless, and
        callers that care read :meth:`fund_posture` separately.

        Returns:
            The explicitly frozen portfolio ids; ``[]`` when none is frozen.
        """
        stmt = (
            select(FundAutonomyState.portfolio_id)
            .where(
                FundAutonomyState.scope == AutonomyScope.PORTFOLIO.value,
                FundAutonomyState.state == AutonomyState.FROZEN.value,
            )
            .order_by(FundAutonomyState.portfolio_id)
        )
        ids = self.session.execute(stmt).scalars().all()
        return [pid for pid in ids if pid is not None]  # portfolio rows carry an id

    def frozen_scope(self, portfolio_id: uuid.UUID | None) -> AutonomyScope | None:
        """The scope currently halting ``portfolio_id``, or ``None`` when it is armed.

        Fund scope wins over portfolio scope. Read-only like :meth:`fund_posture`: it
        never creates a row, so it is safe at the end of an in-flight round.

        Args:
            portfolio_id: The portfolio to check, or ``None`` to check the fund only.

        Returns:
            ``AutonomyScope.FUND`` / ``AutonomyScope.PORTFOLIO`` for the frozen scope,
            else ``None``.
        """
        frozen = AutonomyState.FROZEN.value
        if self.fund_posture() == frozen:
            return AutonomyScope.FUND
        if portfolio_id is not None and self.portfolio_posture(portfolio_id) == frozen:
            return AutonomyScope.PORTFOLIO
        return None

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
                Clipped to the column width, so keep the full text in the audit log.

        Returns:
            The updated, flushed row.
        """
        row = self._get_or_create(scope, portfolio_id)
        row.state = state.value
        if state is AutonomyState.FROZEN:
            row.frozen_at = dt.datetime.now(dt.timezone.utc)
            row.frozen_reason = _fit_frozen_reason(reason)
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

    def restore_first_round(self, portfolio_id: uuid.UUID) -> None:
        """Re-arm a first-round gate that a round consumed without reaching the adviser.

        The gate is one-shot so a portfolio's first rebalance gets every PM gate for
        maximum oversight. A round that dies before the adviser sees anything has not
        provided that oversight, so its caller hands the gate back. Only call this
        for a round whose own :meth:`consume_first_round` returned ``True``.

        Args:
            portfolio_id: The portfolio whose first-round gate to restore.
        """
        self.portfolio_state(portfolio_id).first_round_gated = True
        self.session.flush()

    def _get(
        self,
        scope: AutonomyScope,
        portfolio_id: uuid.UUID | None,
        *,
        fresh: bool = False,
    ) -> FundAutonomyState | None:
        stmt = select(FundAutonomyState).where(FundAutonomyState.scope == scope.value)
        if fresh:
            # Without this the ORM hands back the already-loaded row with its old
            # attributes, hiding a state flipped by another session since.
            stmt = stmt.execution_options(populate_existing=True)
        # The fund row carries a genuine NULL portfolio_id — match with IS NULL,
        # never ``== None`` (which SQL evaluates to UNKNOWN, matching nothing).
        if portfolio_id is None:
            stmt = stmt.where(FundAutonomyState.portfolio_id.is_(None))
        else:
            stmt = stmt.where(FundAutonomyState.portfolio_id == portfolio_id)
        # A round holds one session open for minutes and loaded this row at its start;
        # without populate_existing the identity map would hide a freeze committed by
        # another session in the meantime.
        return self.session.execute(
            stmt.execution_options(populate_existing=True)
        ).scalar_one_or_none()

    def _get_or_create(
        self, scope: AutonomyScope, portfolio_id: uuid.UUID | None
    ) -> FundAutonomyState:
        row = self._get(scope, portfolio_id)
        if row is not None:
            return row
        # A row another transaction inserted but has not committed is invisible to the
        # read above, and a plain INSERT would wait on the unique index until that
        # transaction ends and then raise. Skipping the conflict lets the loser adopt
        # the winner's row once it commits (a freeze racing a portfolio's first round).
        insert = (
            pg_insert
            if self.session.get_bind().dialect.name == "postgresql"
            else sqlite_insert
        )
        self.session.execute(
            insert(FundAutonomyState)
            .values(scope=scope.value, portfolio_id=portfolio_id)
            .on_conflict_do_nothing()
        )
        created = self._get(scope, portfolio_id)
        if created is None:  # pragma: no cover - the insert above guarantees a row
            raise RuntimeError(f"autonomy row for scope {scope.value!r} not created")
        return created


__all__ = ["AutonomyRepository", "AutonomyScope", "AutonomyState"]
