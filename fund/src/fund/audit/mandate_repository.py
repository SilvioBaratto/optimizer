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
"""

from __future__ import annotations

import uuid

from portopt_db.models import PortfolioMandate as PortfolioMandateModel
from portopt_db.repository import RepositoryBase
from sqlalchemy import select

from fund.schemas.mandate import PortfolioMandate


class MandateRepository(RepositoryBase):
    """Upsert/read one mandate per portfolio on an injected sync session."""

    def upsert(self, mandate: PortfolioMandate) -> PortfolioMandateModel:
        """Insert (or update in place) the portfolio's mandate; return the row.

        Idempotent on the UNIQUE ``portfolio_id``: the first call inserts, later
        calls update the same row's scalar mirror + JSON source of truth. The
        pydantic ``portfolio_id`` is a string; it maps to the model's ``Uuid``
        column. ``status`` is left untouched on update (only stamped ``"active"``
        on insert, mirroring the model's server default).
        """
        portfolio_id = uuid.UUID(mandate.portfolio_id)
        payload = mandate.model_dump(mode="json")
        row = self.get(portfolio_id)
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


__all__ = ["MandateRepository"]
