"""Drop the non-ingestion tables.

The repo was stripped to an ingestion daemon: the API layer, the portfolio /
optimization / backtest / factor / risk / rebalancing services, and the API-key
auth middleware are gone. Nothing reads or writes these 17 tables any more.

Surviving schema is the ingestion set: ``background_jobs``,
``background_job_errors``, ``exchanges``, ``instruments``, the yfinance tables
(``ticker_profiles``, ``price_history``, ``financial_statements``,
``dividends``, ``stock_splits``, ``analyst_recommendations``,
``analyst_price_targets``, ``institutional_holders``, ``mutual_fund_holders``,
``insider_transactions``, ``ticker_news``), and the macro tables
(``economic_indicators``, ``economic_indicator_observations``,
``fred_observations``, ``trading_economics_indicators``,
``trading_economics_observations``, ``bond_yields``,
``bond_yield_observations``, ``macro_news``, ``macro_news_summaries``,
``macro_news_themes``, ``macro_calibrations``).

Drop order is child-before-parent so no FK blocks the drop:
``activity_event_metadata`` → ``activity_events`` → ``portfolios``, and the
four ``snapshot_*`` tables → ``portfolio_snapshots`` → ``portfolios``.
``optimization_runs`` / ``backtest_runs`` also reference ``background_jobs``,
which survives.

This migration is destructive and one-way — see ``downgrade``.

Revision ID: d1e2f3a4b5c6
Revises: c7d8e9f0a1b2
Create Date: 2026-07-11
"""

from alembic import op

revision: str = "d1e2f3a4b5c6"
down_revision: str | None = "c7d8e9f0a1b2"
branch_labels = None
depends_on = None


# Child-before-parent. Order matters: reversing it trips FK constraints.
_TABLES_IN_DROP_ORDER: tuple[str, ...] = (
    # portfolio snapshot detail tables → portfolio_snapshots
    "snapshot_optimizer_params",
    "snapshot_summary_entries",
    "snapshot_sector_mappings",
    "snapshot_weights",
    # activity detail → activity_events
    "activity_event_metadata",
    "activity_events",
    # portfolio children
    "portfolio_snapshots",
    "broker_positions",
    "broker_account_snapshots",
    "risk_limits",
    "rebalancing_policies",
    "optimization_runs",
    "backtest_runs",
    # portfolio root
    "portfolios",
    # standalone
    "factor_scores",
    "factor_validation_reports",
    "api_keys",
)


def upgrade() -> None:
    for table in _TABLES_IN_DROP_ORDER:
        op.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')


def downgrade() -> None:
    """Not reversible.

    Recreating empty tables would be a lie: the rows are gone, and the models,
    repositories, and services that gave these tables meaning were deleted from
    the codebase in the same change. Restore from a dump taken before the
    upgrade instead.
    """
    raise NotImplementedError(
        "d1e2f3a4b5c6 is destructive and one-way — restore from a pre-upgrade "
        "dump rather than downgrading."
    )
