# Research CLI

End-to-end stock selection and portfolio research pipeline. Loads data from the DB, runs factor-based selection with macro regime overlay, optimizes a portfolio, and renders charts + `report.md`.

## Prerequisites

- Docker DB running: `docker compose up -d` (PostgreSQL on port 54320)
- Library installed: `pip install -e ".[dev]"` from repo root
- Working directory: repo root (not `research/`)

## Run

```bash
python -m research.cli [OPTIONS]
```

All flags are optional — defaults produce a production-quality run.

## Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--rebalance-freq` | int | `63` | Rebalancing frequency in trading days (63 = quarterly) |
| `--n-selected` | int | `50` | Target number of stocks to select (must be in [25, 50]) |
| `--cost-bps` | float | `10.0` | Transaction cost assumption in basis points |
| `--tax-rate` | float | `0.26` | Capital-gains tax rate (Italian default: 26 %) |
| `--base-currency` | str | `EUR` | FX base currency for return conversion (`EUR`, `GBP`, `USD`) |
| `--robust` | flag | off | Use robust MeanRisk with mu uncertainty set |
| `--persist` | flag | off | Write final snapshot to DB on 17/17 checklist PASS |
| `--start-date` | YYYY-MM-DD | full history | Slice prices on or after this date |
| `--end-date` | YYYY-MM-DD | full history | Slice prices on or before this date |
| `--seed` | int | `42` | RNG seed for bootstrap uncertainty sets |

## Examples

```bash
# Default run — full history, EUR base, no DB write
python -m research.cli

# Custom date window
python -m research.cli --start-date 2022-01-01 --end-date 2024-12-31

# Robust optimizer, persist to DB on PASS
python -m research.cli --robust --persist

# Smaller universe, higher cost assumption, GBP base
python -m research.cli --n-selected 30 --cost-bps 15 --base-currency GBP
```

## Pipeline steps

1. **Load** — pulls prices, fundamentals, FX rates, macro data, risk-free rate from DB
2. **Screen** — investability filter (market cap, ADDV, trading frequency, listing age, …)
3. **Factor history** — builds rolling factor scores (value, momentum, quality, …) at each rebalance date
4. **IS validation** — in-sample IC / t-stat / VIF / Benjamini-Hochberg correction
5. **OOS validation** — rolling block OOS ICIR; aborts if fewer than 4 factors pass both gates
6. **Regime + tilts** — rule-based macro regime classification (GDP / yield-spread); applies multiplicative factor-group tilts; writes regime to DB
7. **Optimize** — IC-weighted composite score → stock selection → MeanRisk (or RobustMeanRisk) optimization with region caps
8. **Rebalance decision** — hybrid calendar/threshold gate vs last review date
9. **Report** — performance metrics (Sharpe, Sortino, IR, drawdown), diversification tables, 17-rule checklist, backtest charts, `report.md`

## Outputs

All artefacts land in `research/output/`:

| File | Description |
|------|-------------|
| `report.md` | Full research report (regime, factors, metrics, checklist) |
| `weights.csv` | Final portfolio weights (only written on 17/17 checklist PASS) |
| `metrics.json` | Gross / net / after-tax performance metrics |
| `checklist.json` | 17-rule checklist result with pass/fail per rule |
| `cumulative_returns.png` | Portfolio vs SPY cumulative return chart |
| `drawdowns.png` | Drawdown chart |
| `rolling_sharpe.png` | Rolling 12-month Sharpe ratio |
| `sector_allocation.png` | Sector weight breakdown |

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | 17/17 checklist PASS — `weights.csv` written |
| `1` | Checklist FAIL or factor coverage error — inspect `report.md` |

## Notes

- Run from repo root, not from `research/`. Imports resolve relative to `optimizer/` package.
- `--persist` only writes to DB when all 17 checklist rules pass. Safe to pass on any run.
- `--robust` adds a mu-uncertainty ellipsoid; slower but more conservative weights.
- Date slicing is applied to prices only — factor history and validation still use the full available window up to `--end-date`.
- The rebalance decision (`should_rebalance`) compares current weights against `research/output/last_review_date.txt`. Delete this file to force a cold-start rebalance.
