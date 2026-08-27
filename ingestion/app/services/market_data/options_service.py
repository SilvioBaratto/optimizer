"""Options-chain ingestion (SPEC A10 / OQ1).

Full option chains are high-volume, so they are fetched by their own
low-frequency scheduler step (``run_options_step``) rather than inside the
daily per-ticker ``fetch_and_store`` loop, and gated by their own staleness
window: an instrument whose most recent snapshot is younger than
``staleness_hours`` is skipped.

``yf.Ticker.option_chain(expiry)`` returns an ``Options`` namedtuple whose
``calls`` / ``puts`` are DataFrames; every contract row is flattened to one
``options_chain`` row stamped with the snapshot date (``as_of``).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from app.repositories.market_data.yfinance_repository import YFinanceRepository
from app.services._shared import ProgressCallback, _noop
from app.services.market_data.yfinance import YFinanceClient

logger = logging.getLogger(__name__)

_DEFAULT_STALENESS_HOURS = 168  # weekly


def _f(v: Any) -> float | None:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> int | None:
    f = _f(v)
    return int(f) if f is not None else None


def _flatten_chain(chain: Any, as_of: date, expiry: date) -> list[dict[str, Any]]:
    """Flatten an Options(calls, puts, ...) namedtuple to option_chain rows."""
    rows: list[dict[str, Any]] = []
    for option_type, frame in (
        ("call", getattr(chain, "calls", None)),
        ("put", getattr(chain, "puts", None)),
    ):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        for _, r in frame.iterrows():
            symbol = r.get("contractSymbol")
            strike = _f(r.get("strike"))
            if not symbol or strike is None:
                continue
            itm = r.get("inTheMoney")
            rows.append(
                {
                    "as_of": as_of,
                    "expiry": expiry,
                    "option_type": option_type,
                    "strike": strike,
                    "contract_symbol": str(symbol)[:50],
                    "last_price": _f(r.get("lastPrice")),
                    "bid": _f(r.get("bid")),
                    "ask": _f(r.get("ask")),
                    "volume": _i(r.get("volume")),
                    "open_interest": _i(r.get("openInterest")),
                    "implied_volatility": _f(r.get("impliedVolatility")),
                    "in_the_money": (
                        None if itm is None or pd.isna(itm) else bool(itm)
                    ),
                }
            )
    return rows


def _is_fresh(as_of: date | None, staleness_hours: int, now: datetime) -> bool:
    if as_of is None:
        return False
    age_hours = (now.date() - as_of).days * 24
    return age_hours < staleness_hours


def run_bulk_options_fetch(
    yf_client: YFinanceClient,
    *,
    staleness_hours: int = _DEFAULT_STALENESS_HOURS,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Fetch + persist full option chains for every instrument.

    Own staleness gate (skip instruments with a fresh snapshot); logs total row
    volume written. Best-effort per instrument — a ticker with no options simply
    contributes nothing.
    """
    from app.database import database_manager

    now = datetime.now(timezone.utc)
    as_of = now.date()

    with database_manager.get_session() as session:
        repo = YFinanceRepository(session)
        instruments = repo.get_instruments_with_yfinance_ticker()
        total = len(instruments)
        on_progress(total=total)

        errors: list[str] = []
        total_rows = 0
        processed = 0
        skipped = 0

        for idx, instrument in enumerate(instruments, 1):
            ticker = instrument.yfinance_ticker
            on_progress(current=idx, current_ticker=ticker)
            if not ticker:
                continue

            if _is_fresh(repo.get_options_as_of(instrument.id), staleness_hours, now):
                skipped += 1
                continue

            try:
                expiries = yf_client.metadata.fetch_options_expirations(ticker) or ()
                inst_rows = 0
                for expiry in expiries:
                    chain = yf_client.metadata.fetch_option_chain(ticker, date=expiry)
                    if chain is None:
                        continue
                    expiry_date = pd.Timestamp(expiry).date()
                    rows = _flatten_chain(chain, as_of, expiry_date)
                    if rows:
                        inst_rows += repo.upsert_option_chain(instrument.id, rows)
                total_rows += inst_rows
                processed += 1
                session.commit()
            except Exception as e:  # one bad ticker must not abort the sweep
                logger.warning("Failed options for %s: %s", ticker, e)
                errors.append(f"{ticker}: {e}")
                session.rollback()

        result = {
            "instruments_total": total,
            "instruments_processed": processed,
            "instruments_skipped_fresh": skipped,
            "contract_rows": total_rows,
            "error_count": len(errors),
        }
        logger.info(
            "Bulk options fetch: %d instruments, %d fresh-skipped, %d contract rows",
            processed,
            skipped,
            total_rows,
        )
        on_progress(
            status="completed",
            finished_at=now.isoformat(),
            errors=errors,
            result=result,
        )

    return result
