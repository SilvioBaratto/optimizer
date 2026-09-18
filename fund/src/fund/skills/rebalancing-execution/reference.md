# Rebalancing & execution — reference

On-demand support for `rebalancing-execution`. Points to the theory; never copies it.

## When to trade: no-trade band + turnover
- Portfolio instability → revise only when worthwhile: `07:70`, `07:82`, `07:197`.
- Turnover from signal decay; trade PARTIALLY toward the aim portfolio: `11:14`,
  `11:133`, `11:135`, `11:155`.

## How to trade: cost-aware execution
- Implementation shortfall & market impact: `22:14`.
- Almgren-Chriss trading frontier (schedule vs impact): `22:42`, `22:59`.
- The aim / weights link from conditioned forecasts: `30:14`, `30:34`.

## Fill model (D30, paper)
- Fill at the NEXT close (a future bar → no look-ahead) + estimated slippage +
  commissions. Simple slippage now; VWAP / Almgren-Chriss later (needs intraday).
- `place_orders` is idempotent: re-running a run does not double-book the ticket.
