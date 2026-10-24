"""``MandateRepository`` — fund-side behavior over the ``portfolio_mandates`` model.

The ``portfolio_mandates`` *model* lives in ``portopt_db`` (shared schema,
Alembic-owned); its *behavior* lives here, mirroring the ``agent_runs`` /
``paper_orders`` model/repo split. Sits on ``portopt_db.repository``'s
``RepositoryBase`` and opens no session of its own — the caller injects a sync
session (D1). No ``commit``: the caller owns the transaction boundary.

One row per portfolio (UNIQUE ``portfolio_id``): :meth:`upsert` is idempotent on
that key — it inserts the mandate once, then updates the same row on subsequent
calls. The pydantic ``fund.schemas.PortfolioMandate`` is the input; the DB model
(imported qualified as ``PortfolioMandateModel`` to dodge the name clash, SPEC
R5) stores the full mandate in the ``mandate`` JSON column — the **source of
truth** — with ``base_currency`` / ``capital`` / ``drift_l1_threshold`` /
``benchmark`` mirrored into indexed scalar columns for the state panel.

``nav_history`` / ``positions`` / ``paper_orders`` store no currency, so once a
portfolio has any of them its ``base_currency`` is locked: :meth:`upsert` raises
:class:`BaseCurrencyLockedError` rather than mix two currencies in one series. Every
writer (CLI, TUI, actions) goes through :meth:`upsert`, so the lock covers them all.
"""

from __future__ import annotations

import uuid

from portopt_db.models import NavHistory, PaperOrder, Position
from portopt_db.models import PortfolioMandate as PortfolioMandateModel
from portopt_db.repository import RepositoryBase
from sqlalchemy import exists, select

from fund.schemas.mandate import PortfolioMandate


class BaseCurrencyLockedError(ValueError):
    """A base-currency change was refused because the portfolio has ledger history."""


class MandateRepository(RepositoryBase):
    """Upsert/read one mandate per portfolio on an injected sync session."""

    def upsert(self, mandate: PortfolioMandate) -> PortfolioMandateModel:
        """Insert (or update in place) the portfolio's mandate; return the row.

        Idempotent on the UNIQUE ``portfolio_id``: the first call inserts, later
        calls update the same row's scalar mirror + JSON source of truth.
        ``status`` is left untouched on update (only stamped ``"active"`` on
        insert, mirroring the model's server default).

        Args:
            mandate: Mandate to upsert. ``portfolio_id`` is a string in the
                Pydantic schema; it is coerced to ``uuid.UUID`` before the
                DB write.

        Returns:
            The flushed, not-yet-committed model row.

        Raises:
            BaseCurrencyLockedError: ``mandate.base_currency`` differs from the stored
                one and the portfolio already has NAV history, positions or paper
                orders (none of which record a currency).
        """
        portfolio_id = uuid.UUID(mandate.portfolio_id)
        payload = mandate.model_dump(mode="json")
        row = self.get(portfolio_id)
        if (
            row is not None
            and row.base_currency != mandate.base_currency
            and self.has_ledger_history(portfolio_id)
        ):
            raise BaseCurrencyLockedError(
                f"base currency cannot change from {row.base_currency} to "
                f"{mandate.base_currency}: portfolio {portfolio_id} already has NAV "
                f"history, positions or paper orders, which carry no currency"
            )
        if row is None:
            row = PortfolioMandateModel(
                portfolio_id=portfolio_id,
                base_currency=mandate.base_currency,
                capital=float(mandate.capital),
                drift_l1_threshold=mandate.drift_l1_threshold,
                benchmark=mandate.benchmark,
                mandate=payload,
                status="active",
            )
            self.session.add(row)
        else:
            row.base_currency = mandate.base_currency
            row.capital = float(mandate.capital)
            row.drift_l1_threshold = mandate.drift_l1_threshold
            row.benchmark = mandate.benchmark
            row.mandate = payload
        # `flush` populates the client-side UUID default; no `refresh` — a PK
        # re-read mis-maps this model's shape (pg UUID + JSON + Float) on SQLite,
        # and every attribute is already set on the instance (mirrors OrderRepo).
        self.session.flush()
        return row

    def get(self, portfolio_id: uuid.UUID) -> PortfolioMandateModel | None:
        """Return the portfolio's mandate row, or ``None``."""
        stmt = select(PortfolioMandateModel).where(
            PortfolioMandateModel.portfolio_id == portfolio_id
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def has_ledger_history(self, portfolio_id: uuid.UUID) -> bool:
        """Whether the portfolio already has NAV marks, positions or paper orders.

        These tables record amounts without a currency, so any row in them pins the
        portfolio's ``base_currency``.
        """
        for model in (NavHistory, Position, PaperOrder):
            stmt = select(exists().where(model.portfolio_id == portfolio_id))
            if self.session.execute(stmt).scalar():
                return True
        return False

    def list_portfolio_ids(self) -> list[uuid.UUID]:
        """Distinct portfolio ids that carry a mandate (one row per portfolio).

        Order is unspecified; ``observe.list_portfolios`` dedups across sources
        and sorts.
        """
        stmt = select(PortfolioMandateModel.portfolio_id).distinct()
        return list(self.session.execute(stmt).scalars().all())

    def list_active(self) -> list[PortfolioMandateModel]:
        """Return every active mandate row, ordered by ``portfolio_id``.

        The enumeration the Phase-9 sweeps walk. Trigger filtering is *not* in
        SQL — the ``portfolio_mandates`` table has no scalar trigger column — so
        callers rehydrate ``PortfolioMandate.from_stored(row.mandate)`` and
        read ``.triggers.cron`` / ``.triggers.drift``. Deterministic order keeps
        the sequential sweep repeatable.
        """
        stmt = (
            select(PortfolioMandateModel)
            .where(PortfolioMandateModel.status == "active")
            .order_by(PortfolioMandateModel.portfolio_id)
        )
        return list(self.session.execute(stmt).scalars().all())


__all__ = ["BaseCurrencyLockedError", "MandateRepository"]
