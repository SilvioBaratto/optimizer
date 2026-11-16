"""Repository for an instrument's analyst estimates, earnings and ratings."""

from collections.abc import Sequence
from datetime import date, datetime, timezone
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import delete, select

from portopt_db.models.market_data.yfinance_data import (
    AnalystAction,
    AnalystPriceTarget,
    AnalystRecommendation,
    EarningsDate,
    EarningsEstimate,
    EarningsHistory,
    EpsRevisions,
    EpsTrend,
    GrowthEstimate,
    RevenueEstimate,
)
from portopt_db.repositories.market_data._mapping import (
    _last_wins,
    _refreshable_columns,
    _safe_date,
    _safe_float,
    _safe_int,
    _safe_str,
    truncate,
)
from portopt_db.repository import RepositoryBase


def _estimate_update_columns(rows: list[dict[str, Any]]) -> list[str] | None:
    """Conflict-update columns for a batch of forward estimates.

    A row that carries no currency must not turn a label stored by an earlier
    fetch into NULL, so ``currency`` is left out of its update.

    Args:
        rows: Rows of one batch; callers pass rows that all do, or all do not,
            carry a currency (see :func:`_split_by_currency`).

    Returns:
        ``None`` (update every column) when the rows carry a currency, otherwise
        every column except ``currency``.
    """
    if not rows or any(r["currency"] is not None for r in rows):
        return None
    return _refreshable_columns(rows[0], frozenset({"currency"}))


def _split_by_currency(
    rows: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Separate the rows that carry a currency from those that do not.

    One ``ON CONFLICT`` update applies a single column list to every row of its
    statement, so a batch mixing both kinds would overwrite the stored label of
    a currency-less row with NULL.

    Returns:
        The non-empty groups, rows with a currency first.
    """
    labelled = [r for r in rows if r["currency"] is not None]
    unlabelled = [r for r in rows if r["currency"] is None]
    return [group for group in (labelled, unlabelled) if group]


class AnalystRepository(RepositoryBase):
    """Analyst estimates, earnings, rating changes, recommendations and targets."""

    def _upsert_estimates(
        self, model: type, rows: list[dict[str, Any]], *, constraint_name: str
    ) -> int:
        """Upsert forward-estimate rows without erasing stored currency labels.

        Rows with and without a currency go in separate statements because a
        conflict update applies one column list to all rows of its statement.

        Returns:
            Number of rows processed.
        """
        return sum(
            self._upsert(
                model,
                group,
                constraint_name=constraint_name,
                update_columns=_estimate_update_columns(group),
            )
            for group in _split_by_currency(rows)
        )

    def upsert_earnings_estimate(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert forward-period earnings estimates (index = period labels)."""
        rows = [
            {
                "instrument_id": instrument_id,
                "period": truncate(EarningsEstimate, "period", period),
                "num_analysts": _safe_int(row.get("numberOfAnalysts")),
                "avg": _safe_float(row.get("avg")),
                "low": _safe_float(row.get("low")),
                "high": _safe_float(row.get("high")),
                "year_ago_eps": _safe_float(row.get("yearAgoEps")),
                "growth": _safe_float(row.get("growth")),
                "currency": truncate(EarningsEstimate, "currency", row.get("currency")),
            }
            for period, row in df.iterrows()
        ]
        rows = _last_wins(rows, "period")
        return self._upsert_estimates(
            EarningsEstimate,
            rows,
            constraint_name="uq_earnings_estimate_instrument_period",
        )

    def upsert_revenue_estimate(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert forward-period revenue estimates (index = period labels)."""
        rows = [
            {
                "instrument_id": instrument_id,
                "period": truncate(RevenueEstimate, "period", period),
                "num_analysts": _safe_int(row.get("numberOfAnalysts")),
                "avg": _safe_float(row.get("avg")),
                "low": _safe_float(row.get("low")),
                "high": _safe_float(row.get("high")),
                "year_ago_revenue": _safe_float(row.get("yearAgoRevenue")),
                "growth": _safe_float(row.get("growth")),
                "currency": truncate(RevenueEstimate, "currency", row.get("currency")),
            }
            for period, row in df.iterrows()
        ]
        rows = _last_wins(rows, "period")
        return self._upsert_estimates(
            RevenueEstimate,
            rows,
            constraint_name="uq_revenue_estimate_instrument_period",
        )

    def upsert_growth_estimates(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert per-period growth estimates (stock + index trend).

        yfinance 1.6.0 emits only ``stockTrend`` / ``indexTrend`` (industry and
        sector trend were dropped upstream). Column names vary across versions, so
        read defensively.
        """

        def _col(row: Any, *names: str) -> Any:
            for name in names:
                if name in row:
                    return row.get(name)
            return None

        rows = [
            {
                "instrument_id": instrument_id,
                "period": truncate(GrowthEstimate, "period", period),
                "stock_trend": _safe_float(_col(row, "stockTrend", "stock")),
                "index_trend": _safe_float(_col(row, "indexTrend", "index")),
            }
            for period, row in df.iterrows()
        ]
        return self._upsert(
            GrowthEstimate,
            _last_wins(rows, "period"),
            constraint_name="uq_growth_estimate_instrument_period",
        )

    def upsert_eps_trend(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert the EPS estimate trend (index = period labels "0q"/"+1q"/...).

        Columns are snapshot ages (``current``/``7daysAgo``/``30daysAgo``/...) plus
        a per-period ``currency`` column (yfinance 1.2.2+).
        """
        rows = [
            {
                "instrument_id": instrument_id,
                "period": truncate(EpsTrend, "period", period),
                "current_estimate": _safe_float(row.get("current")),
                "days_ago_7": _safe_float(row.get("7daysAgo")),
                "days_ago_30": _safe_float(row.get("30daysAgo")),
                "days_ago_60": _safe_float(row.get("60daysAgo")),
                "days_ago_90": _safe_float(row.get("90daysAgo")),
                "currency": truncate(EpsTrend, "currency", row.get("currency")),
            }
            for period, row in df.iterrows()
        ]
        rows = _last_wins(rows, "period")
        return self._upsert_estimates(
            EpsTrend, rows, constraint_name="uq_eps_trend_instrument_period"
        )

    def upsert_eps_revisions(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert EPS revision counts (index = period labels "0q"/"+1q"/...).

        Columns are up/down revision counts over the trailing 7/30-day windows.
        Yahoo emits ``downLast7Days`` (capital D); the lowercase spelling is
        accepted too.
        """
        rows = [
            {
                "instrument_id": instrument_id,
                "period": truncate(EpsRevisions, "period", period),
                "up_last_7days": _safe_int(row.get("upLast7days")),
                "up_last_30days": _safe_int(row.get("upLast30days")),
                # Yahoo capitalises the D only on this column.
                "down_last_7days": _safe_int(
                    row.get("downLast7Days", row.get("downLast7days"))
                ),
                "down_last_30days": _safe_int(row.get("downLast30days")),
                "currency": truncate(EpsRevisions, "currency", row.get("currency")),
            }
            for period, row in df.iterrows()
        ]
        rows = _last_wins(rows, "period")
        return self._upsert_estimates(
            EpsRevisions, rows, constraint_name="uq_eps_revisions_instrument_period"
        )

    def upsert_earnings_history(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert historical EPS surprise (index = past-quarter dates).

        Deduped in-batch on ``period_date`` (last-wins): two source rows on the
        same calendar day would otherwise collide on ``uq_earnings_history_...``
        within one ON CONFLICT and abort the whole write.

        ``surprisePercent`` arrives as a fraction (0.04 = 4%) and is stored in
        percent points, the unit ``earnings_dates`` and the earnings calendar
        already use.
        """
        by_date: dict[date, dict[str, Any]] = {}
        for idx, row in df.iterrows():
            period_date = _safe_date(idx)
            if period_date is None:
                continue
            surprise = _safe_float(row.get("surprisePercent"))
            by_date[period_date] = {
                "instrument_id": instrument_id,
                "period_date": period_date,
                "eps_estimate": _safe_float(row.get("epsEstimate")),
                "eps_actual": _safe_float(row.get("epsActual")),
                "eps_difference": _safe_float(row.get("epsDifference")),
                "surprise_percent": None if surprise is None else surprise * 100.0,
            }
        rows = list(by_date.values())
        return self._upsert(
            EarningsHistory,
            rows,
            constraint_name="uq_earnings_history_instrument_period",
        )

    def upsert_earnings_dates(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert past/upcoming earnings dates (index = tz-aware datetimes).

        Column labels vary across yfinance versions; read defensively. A stored
        upcoming date with no reported EPS that the frame no longer lists is
        deleted: Yahoo moved the report, and keeping both dates would make the
        stale one read as the next earnings date.
        """

        def _col(row: Any, *names: str) -> Any:
            for name in names:
                if name in row:
                    return row.get(name)
            return None

        by_date: dict[date, dict[str, Any]] = {}
        for idx, row in df.iterrows():
            earnings_date = _safe_date(idx)
            if earnings_date is None:
                continue
            by_date[earnings_date] = {
                "instrument_id": instrument_id,
                "earnings_date": earnings_date,
                "eps_estimate": _safe_float(_col(row, "EPS Estimate", "epsEstimate")),
                "eps_actual": _safe_float(_col(row, "Reported EPS", "epsActual")),
                "surprise_percent": _safe_float(
                    _col(row, "Surprise(%)", "surprisePercent")
                ),
            }
        rows = list(by_date.values())
        written = self._upsert(
            EarningsDate,
            rows,
            constraint_name="uq_earnings_date_instrument_date",
        )
        if rows:
            # An empty frame says nothing about the upcoming date, so only a
            # populated one may retire a stored projection, and only after the
            # new dates are in.
            with self.session.begin_nested():
                self.session.execute(
                    delete(EarningsDate).where(
                        EarningsDate.instrument_id == instrument_id,
                        EarningsDate.eps_actual.is_(None),
                        EarningsDate.earnings_date >= datetime.now(timezone.utc).date(),
                        EarningsDate.earnings_date.not_in(list(by_date)),
                    )
                )
        return written

    def upsert_analyst_actions(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert analyst upgrade/downgrade actions (index = grade dates)."""

        def _col(row: Any, *names: str) -> Any:
            for name in names:
                if name in row:
                    return row.get(name)
            return None

        rows: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for idx, row in df.iterrows():
            action_date = _safe_date(idx)
            if action_date is None:
                continue
            firm = _safe_str(_col(row, "Firm", "firm"), 200) or ""
            to_grade = _safe_str(_col(row, "ToGrade", "toGrade"), 100) or ""
            key = (action_date, firm, to_grade)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "action_date": action_date,
                    "firm": firm,
                    "from_grade": _safe_str(_col(row, "FromGrade", "fromGrade"), 100),
                    "to_grade": to_grade,
                    "action": _safe_str(_col(row, "Action", "action"), 50),
                }
            )
        return self._upsert(AnalystAction, rows, constraint_name="uq_analyst_action")

    def upsert_recommendations(self, instrument_id: UUID, rec_df: pd.DataFrame) -> int:
        """Upsert analyst recommendations from recommendations_summary DataFrame."""
        rows = []
        for _, row_data in rec_df.iterrows():
            period = truncate(AnalystRecommendation, "period", row_data.get("period"))
            if not period:
                continue
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "period": period,
                    "strong_buy": _safe_int(row_data.get("strongBuy")),
                    "buy": _safe_int(row_data.get("buy")),
                    "hold": _safe_int(row_data.get("hold")),
                    "sell": _safe_int(row_data.get("sell")),
                    "strong_sell": _safe_int(row_data.get("strongSell")),
                }
            )

        return self._upsert(
            AnalystRecommendation,
            _last_wins(rows, "period"),
            constraint_name="uq_analyst_rec_instrument_period",
        )

    def get_recommendations(
        self, instrument_id: UUID
    ) -> Sequence[AnalystRecommendation]:
        """Return analyst recommendation summary rows for an instrument."""
        stmt = (
            select(AnalystRecommendation)
            .where(AnalystRecommendation.instrument_id == instrument_id)
            .order_by(AnalystRecommendation.period)
        )
        return self.session.execute(stmt).scalars().all()

    def upsert_price_targets(self, instrument_id: UUID, targets: dict[str, Any]) -> int:
        """Upsert analyst price targets from dict."""
        row = {
            "instrument_id": instrument_id,
            "current": _safe_float(targets.get("current")),
            "low": _safe_float(targets.get("low")),
            "high": _safe_float(targets.get("high")),
            "mean": _safe_float(targets.get("mean")),
            "median": _safe_float(targets.get("median")),
        }

        return self._upsert(
            AnalystPriceTarget,
            [row],
            constraint_name="uq_analyst_pt_instrument",
        )

    def get_price_targets(self, instrument_id: UUID) -> AnalystPriceTarget | None:
        """Return the analyst price target row for an instrument, or None."""
        stmt = select(AnalystPriceTarget).where(
            AnalystPriceTarget.instrument_id == instrument_id
        )
        return self.session.execute(stmt).scalar_one_or_none()
