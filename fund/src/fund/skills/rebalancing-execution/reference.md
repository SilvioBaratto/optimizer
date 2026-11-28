# Rebalancing & execution — reference

On-demand support for `rebalancing-execution`. Points to the theory; never copies it.

## When to trade: drift band (no-trade band) + turnover
- Portfolio instability → revise only when worthwhile: `07:70`, `07:82`, `07:201`.
- Turnover from signal decay; trade PARTIALLY toward the aim portfolio: `11:14`,
  `11:133`, `11:135`, `11:155`.
- Implementation status: every candidate is a plain long-only solve — no
  `transaction_costs`, no `previous_weights`, and `l1_coef` is constant under
  long-only budget 1 (the L1 norm equals the budget) — so no turnover term shapes the
  weights. The band only describes how far each candidate moves the book; the
  placed ticket is always the chosen candidate's full target book.

## How to trade: cost-aware execution
- Implementation shortfall & market impact: `22:14`.
- Almgren-Chriss trading frontier (schedule vs impact): `22:42`, `22:59`.
- The aim / weights link from conditioned forecasts: `30:14`, `30:34`.

## Fill model (D30, paper)
- Fill at the NEXT close (a future bar → no look-ahead) + estimated slippage +
  commissions. Simple slippage now; VWAP / Almgren-Chriss later (needs intraday).
- The fill bar must be priced strictly AFTER the decision date (`asof`). A scheduled
  run uses `asof` = today, so an approval given before the next bar is ingested finds
  no fill bar: `place_orders` returns `{ok: false}` and writes no ticket. The run is
  then finalised `incomplete` (reason `no_order`) and the holdings snapshot is left
  untouched — re-run once a later bar exists. Never report such a run as traded.
- `place_orders` is idempotent: re-running a run does not double-book the ticket.
