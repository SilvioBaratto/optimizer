"""Service for syncing broker state from Trading 212 into the portfolio tables."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models.universe import Instrument
from app.repositories.portfolio_repository import PortfolioRepository
from app.services.trading212.client import Trading212Client

logger = logging.getLogger(__name__)


@dataclass
class BrokerSyncResult:
    positions_synced: int = 0
    positions_removed: int = 0
    account_synced: bool = False
    orders_fetched: int = 0
    dividends_fetched: int = 0
    errors: list[str] | None = None


def _map_t212_ticker_to_yfinance(
    ticker: str, session: Session,
) -> str | None:
    """Look up the yfinance_ticker for a T212 ticker via the instruments table."""
    from sqlalchemy import select

    stmt = select(Instrument.yfinance_ticker).where(Instrument.ticker == ticker)
    result = session.execute(stmt).scalar_one_or_none()
    return result


def sync_portfolio(
    client: Trading212Client,
    portfolio_id: Any,
    session: Session,
    *,
    on_progress: Any = None,
) -> BrokerSyncResult:
    """Sync positions and account state from Trading 212 into DB.

    Args:
        client: Authenticated T212 client.
        portfolio_id: UUID of the portfolio to sync into.
        session: Active SQLAlchemy session.
        on_progress: Optional callback for progress updates.
    """
    repo = PortfolioRepository(session)
    result = BrokerSyncResult()
    errors: list[str] = []
    synced_at = datetime.now(timezone.utc)

    # Step 1: Sync positions
    if on_progress:
        on_progress(current=0, total=4, extra={"step": "positions"})

    try:
        raw_positions = client.get_portfolio_positions()
        position_rows: list[dict[str, Any]] = []
        current_tickers: set[str] = set()

        for pos in raw_positions:
            ticker = pos.get("ticker", "")
            current_tickers.add(ticker)
            yf_ticker = _map_t212_ticker_to_yfinance(ticker, session)

            position_rows.append({
                "ticker": ticker,
                "yfinance_ticker": yf_ticker,
                "name": pos.get("name") or pos.get("shortName"),
                "quantity": pos.get("quantity", 0),
                "average_price": pos.get("averagePrice", 0),
                "current_price": pos.get("currentPrice"),
                "ppl": pos.get("ppl"),
                "fx_ppl": pos.get("fxPpl"),
                "initial_fill_date": pos.get("initialFillDate"),
            })

        result.positions_synced = repo.upsert_positions(
            portfolio_id, position_rows, synced_at,
        )
        result.positions_removed = repo.delete_stale_positions(
            portfolio_id, current_tickers,
        )
        logger.info(
            "Synced %d positions, removed %d stale",
            result.positions_synced, result.positions_removed,
        )
    except Exception as e:
        errors.append(f"Position sync failed: {e}")
        logger.error("Position sync error: %s", e)

    # Step 2: Sync account cash
    if on_progress:
        on_progress(current=1, total=4, extra={"step": "account"})

    try:
        cash_data = client.get_account_cash()
        account_info = client.get_account_info()

        repo.upsert_account_snapshot(
            portfolio_id,
            {
                "total": cash_data.get("total", 0),
                "free": cash_data.get("free", 0),
                "invested": cash_data.get("invested", 0),
                "blocked": cash_data.get("blocked"),
                "result": cash_data.get("result"),
                "currency": account_info.get("currencyCode", "EUR"),
            },
            synced_at,
        )
        result.account_synced = True
        logger.info("Account snapshot saved")
    except Exception as e:
        errors.append(f"Account sync failed: {e}")
        logger.error("Account sync error: %s", e)

    # Step 3: Fetch order history (for activity events)
    if on_progress:
        on_progress(current=2, total=4, extra={"step": "orders"})

    try:
        orders = client.get_all_order_history()
        result.orders_fetched = len(orders)

        # Create activity events for recent filled orders
        for order in orders[:20]:  # Only latest 20
            if order.get("status") == "FILLED":
                side = "Bought" if order.get("type") == "BUY" else "Sold"
                ticker = order.get("ticker", "?")
                qty = order.get("filledQuantity", order.get("quantity", 0))
                price = order.get("fillPrice", order.get("limitPrice", 0))
                repo.add_event(
                    event_type="trade",
                    title=f"{side} {qty} {ticker}",
                    portfolio_id=portfolio_id,
                    description=f"Fill price: {price}",
                    metadata={"order_id": order.get("id"), "ticker": ticker},
                )
    except Exception as e:
        errors.append(f"Order history fetch failed: {e}")
        logger.error("Order history error: %s", e)

    # Step 4: Fetch dividend history
    if on_progress:
        on_progress(current=3, total=4, extra={"step": "dividends"})

    try:
        dividends = client.get_all_dividend_history()
        result.dividends_fetched = len(dividends)
    except Exception as e:
        errors.append(f"Dividend history fetch failed: {e}")
        logger.error("Dividend history error: %s", e)

    if on_progress:
        on_progress(current=4, total=4, extra={"step": "done"})

    result.errors = errors if errors else None
    session.commit()
    return result
