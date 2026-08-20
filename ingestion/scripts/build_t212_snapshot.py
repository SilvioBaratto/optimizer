"""One-off: pull live T212 positions, FX-normalize, write a portfolio_snapshots row.

Run: docker compose exec scheduler python scripts/build_t212_snapshot.py
"""

from __future__ import annotations

from datetime import date

from app.repositories.portfolio.portfolio_repository import PortfolioRepository
from app.services.portfolio.t212_position_normalizer.normalizer_service import (
    normalize_live_positions,
)

from app.database import database_manager
from app.services.universe.trading212.client import Trading212Client

PORTFOLIO_NAME = "t212"


def main() -> None:
    client = Trading212Client.from_settings()
    if client is None:
        raise SystemExit("T212 credentials missing (TRADING_212_API_KEY/SECRET).")

    with database_manager.get_session() as session:
        repo = PortfolioRepository(session)
        portfolio = repo.get_by_name(PORTFOLIO_NAME)
        if portfolio is None:
            raise SystemExit(f"Portfolio '{PORTFOLIO_NAME}' not found.")

        result = normalize_live_positions(client, session)

        weights = {
            p.ticker: p.eur_weight
            for p in result.positions
            if p.ticker and p.eur_weight > 0.0
        }
        if not weights:
            raise SystemExit("No usable weights (all positions unmapped/zero).")

        snap = repo.create_snapshot(
            portfolio_id=portfolio.id,
            snapshot_date=date.today(),
            snapshot_type="manual",
            weights=weights,
            summary={
                "source": "trading212_live",
                "base_currency": result.base_currency,
                "reconciliation_ok": result.reconciliation_ok,
                "reconciliation_delta_pct": result.reconciliation_delta_pct,
                "unmapped_count": result.unmapped_count,
                "fx_missing_count": result.fx_missing_count,
                "stale_price_count": result.stale_price_count,
                "dropped_raw_tickers": result.dropped_raw_tickers,
                "n_positions": len(weights),
            },
        )
        session.commit()

        print(f"Snapshot {snap.id} created for '{PORTFOLIO_NAME}'")
        print(
            f"  base={result.base_currency} positions={len(weights)} "
            f"recon_ok={result.reconciliation_ok} "
            f"delta_pct={result.reconciliation_delta_pct}"
        )
        print(
            f"  unmapped={result.unmapped_count} fx_missing={result.fx_missing_count} "
            f"stale={result.stale_price_count} dropped={result.dropped_raw_tickers}"
        )
        for t, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {t:<12} {w:.4f}")


if __name__ == "__main__":
    main()
