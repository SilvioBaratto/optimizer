# Data Inventory — Live Volumes

Snapshot of the optimizer PostgreSQL database (`postgresql://postgres:postgres@localhost:54320/optimizer_db`).
57 base tables (55 documented below + `alembic_version`, `apscheduler_jobs` infra). Row counts
are live-at-snapshot; treat as orders of magnitude, not exact current values.

## 1. Table Volumes (rows, descending)

| Table | Rows |
|-------|-----:|
| financial_statements | 11,253,341 |
| price_history | 9,786,726 |
| shares_outstanding | 789,921 |
| analyst_actions | 419,930 |
| sec_filings | 389,385 |
| dividends | 372,607 |
| insider_transactions | 256,439 |
| earnings_dates | 106,647 |
| ticker_news | 61,898 |
| institutional_holders | 58,417 |
| mutual_fund_holders | 52,398 |
| insider_roster | 44,559 |
| growth_estimates | 40,203 |
| revenue_estimate | 30,408 |
| earnings_estimate | 23,656 |
| earnings_history | 17,956 |
| analyst_recommendations | 17,100 |
| instruments | 8,898 |
| ticker_profiles | 8,895 |
| ticker_profile_extras | 8,895 |
| insider_purchases | 8,024 |
| analyst_price_targets | 7,949 |
| stock_splits | 7,794 |
| major_holders | 7,734 |
| etf_bond_ratings | 6,549 |
| earnings_calendar | 3,996 |
| etf_sector_weights | 2,200 |
| economic_event_calendar | 1,496 |
| etf_asset_classes | 839 |
| etf_equity_holdings | 839 |
| etf_fund_operations | 839 |
| etf_metadata | 839 |
| etf_bond_holdings | 774 |
| etf_holdings | 770 |
| split_calendar | 477 |
| trading_economics_indicators | 138 |
| trading_economics_observations | 138 |
| macro_news_themes | 66 |
| macro_news | 36 |
| bond_yields | 16 |
| bond_yield_observations | 16 |
| exchanges | 14 |
| background_job_errors | 12 |
| ipo_calendar | 11 |
| background_jobs | 6 |
| economic_indicator_observations | 4 |
| economic_indicators | 4 |
| macro_news_summaries | 4 |
| fred_observations | 0 |
| macro_calibrations | 0 |
| market_summaries | 0 |
| options_chain | 0 |
| sector_industries | 0 |
| sector_snapshots | 0 |
| sector_top_companies | 0 |

## 2. Empty Tables and Their Populating Scheduler Step

All seven empty tables are populated by low-frequency (monthly / weekly / market-wide)
steps, not the daily per-ticker loop — so empty is a schedule state, not a failure.

| Table | Populating step | Job / trigger |
|-------|-----------------|---------------|
| fred_observations | `run_fred_step` (CLI `fred`) | `fred_monthly` (0 8 1 * *); **also requires `FRED_API_KEY`** — absent key = no-op |
| macro_calibrations | `run_calibrate_step` (CLI `calibrate`) | tail of `daily_pipeline`; writes only when macro indicator rows exist and BAML LLM is invoked |
| market_summaries | `run_market_summary_step` | `weekly_market_wide` (Sat); iterates the 8 `MARKET_IDENTIFIERS` |
| options_chain | `run_options_step` | `weekly_market_wide` (Sat, after weekly refetch); own ~weekly staleness gate |
| sector_industries | `run_market_structure_step` | `weekly_market_wide` (Sat 04:00, `market_structure_fetch`) |
| sector_snapshots | `run_market_structure_step` | `weekly_market_wide` (Sat 04:00, `market_structure_fetch`) |
| sector_top_companies | `run_market_structure_step` | `weekly_market_wide` (Sat 04:00, `market_structure_fetch`) |

## 3. Notable Specifics

- **`financial_statements` — the EAV giant (~11.3M rows).** Long/EAV layout: yfinance
  statement DataFrames (columns = period dates, index = line-item names) melted to one row
  per cell. Overloaded `statement_type` also carries `valuation_measures`, `eps_trend`,
  `eps_revisions`, `earnings` — filter on `statement_type` + `period_type`, not type alone.
  `line_item` labels are raw yfinance strings, not a controlled vocabulary. `value` is
  `Numeric(38,6)` (Decimal on read).

- **`price_history` — ~9.8M daily OHLCV bars.** One row per (instrument, date), plus sparse
  corporate-action columns (dividends / stock_splits / capital_gains). `price_unit` records
  listing currency as-is (e.g. `GBX` = pence) — never normalized; FX/scale is the reader's
  job. yfinance repair path (sklearn DBSCAN) matters upstream: without it ~22% of tickers
  return empty history and never land here.

- **The 8-table ETF family** (written together in `_fetch_etf_metadata`, yfinance step):
  `etf_metadata` (1:1 headline, 839), `etf_asset_classes` (839), `etf_equity_holdings`
  (839), `etf_fund_operations` (839), `etf_bond_holdings` (774), `etf_holdings` (770),
  `etf_bond_ratings` (6,549), `etf_sector_weights` (2,200). All composition tables are
  point-in-time keyed on `(instrument_id, as_of, ...)`; only ETF instruments get rows.

- **The 4 calendar tables** (market-wide, no instrument FK; `weekly_market_wide` sweep via
  `run_calendars_step`): `earnings_calendar` (3,996), `economic_event_calendar` (1,496),
  `split_calendar` (477), `ipo_calendar` (11). Distinct from the per-instrument
  `earnings_dates` / `stock_splits` tables.

- **Macro tables pending the fred / macro / calibrate steps.** `fred_observations` and
  `macro_calibrations` are empty (see §2). The scraped macro set is thinly populated:
  `trading_economics_indicators`/`_observations` (138 each), `bond_yields`/`_observations`
  (16 each), `economic_indicators`/`_observations` and `macro_news_summaries` (4 each),
  `macro_news` (36). These fill via `run_macro_step` (daily_pipeline / weekly_refetch),
  `run_summarize_step`, and — for calibrations — `run_calibrate_step` invoking the BAML LLM.
