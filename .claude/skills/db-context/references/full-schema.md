# Full Column-by-Column Schema Reference

Every table in the optimizer database — 26 modelled tables across four domains, plus
`alembic_version` and `apscheduler_jobs` which have no model. All inherit `id` (UUID PK),
`created_at` (DateTime TZ), `updated_at` (DateTime TZ) from `BaseModel`.

Everything here is ingestion. The portfolio, execution, factor, risk, rebalancing and auth
tables were dropped by migration `d1e2f3a4b5c6` (one-way — its `downgrade()` raises), along with
the code that owned them.

Verified against the live database on 2026-07-11 (head `d1e2f3a4b5c6`).

---

## Core Domain

### exchanges
**Model**: `Exchange` in `ingestion/app/models/universe.py`
**Unique**: name (column-level `unique=True`)

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| name | String(255) | NO | Exchange display name |
| t212_id | Integer | YES | Trading 212 exchange identifier |

**Relationships**: `instruments` (1→many, CASCADE, delete-orphan)

---

### instruments
**Model**: `Instrument` in `ingestion/app/models/universe.py`
**Unique**: `uq_instrument_ticker_exchange` ON (ticker, exchange_id)
**Indexes**: ix_instrument_exchange_id, ix_instrument_isin, ix_instrument_yfinance_ticker, ticker (column-level), delisted_at (column-level)

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| ticker | String(100) | NO | Trading 212 ticker symbol |
| short_name | String(100) | NO | Short display name |
| name | String(500) | YES | Full instrument name |
| isin | String(20) | YES | International Securities ID |
| instrument_type | String(50) | YES | e.g., "EQUITY" |
| currency_code | String(10) | YES | Trading currency |
| yfinance_ticker | String(100) | YES | Mapped yfinance symbol |
| delisted_at | Date | YES | Survivorship bias: when dropped from T212 |
| delisting_return | Float | YES | Terminal return at delisting |
| exchange_id | UUID FK→exchanges.id | NO | CASCADE on delete |

**Relationships**: exchange (many→1), plus 11 yfinance child relationships (all passive_deletes)

---

## yfinance Market Data

### ticker_profiles
**Model**: `TickerProfile` in `ingestion/app/models/yfinance_data.py`
**Unique**: `uq_ticker_profile_instrument` ON (instrument_id) — **1:1 relationship**
**Indexes**: ix_ticker_profiles_sector, ix_ticker_profiles_industry, ix_ticker_profiles_country

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| instrument_id | UUID FK→instruments.id | NO | CASCADE |
| symbol | String(50) | YES | yfinance symbol |
| short_name | String(500) | YES | |
| long_name | String(500) | YES | |
| isin | String(20) | YES | |
| exchange | String(50) | YES | Exchange code from yfinance |
| quote_type | String(50) | YES | "EQUITY", "ETF", etc. |
| currency | String(10) | YES | |
| sector | String(200) | YES | GICS sector (yfinance names) |
| industry | String(200) | YES | Sub-industry |
| country | String(100) | YES | Country of incorporation |
| website | String(500) | YES | |
| long_business_summary | Text | YES | |
| market_cap | BigInteger | YES | |
| enterprise_value | BigInteger | YES | |
| shares_outstanding | BigInteger | YES | |
| float_shares | BigInteger | YES | |
| implied_shares_outstanding | BigInteger | YES | |
| current_price | Float | YES | |
| previous_close | Float | YES | |
| open_price | Float | YES | |
| day_low | Float | YES | |
| day_high | Float | YES | |
| fifty_two_week_low | Float | YES | |
| fifty_two_week_high | Float | YES | |
| fifty_day_average | Float | YES | |
| two_hundred_day_average | Float | YES | |
| average_volume | BigInteger | YES | |
| average_volume_10days | BigInteger | YES | |
| regular_market_volume | BigInteger | YES | |
| bid | Float | YES | |
| ask | Float | YES | |
| bid_size | Integer | YES | |
| ask_size | Integer | YES | |
| beta | Float | YES | |
| trailing_pe | Float | YES | |
| forward_pe | Float | YES | |
| trailing_eps | Float | YES | |
| forward_eps | Float | YES | |
| price_to_sales_trailing_12months | Float | YES | |
| price_to_book | Float | YES | |
| enterprise_to_revenue | Float | YES | |
| enterprise_to_ebitda | Float | YES | |
| peg_ratio | Float | YES | |
| book_value | Float | YES | |
| profit_margins | Float | YES | |
| operating_margins | Float | YES | |
| gross_margins | Float | YES | |
| ebitda_margins | Float | YES | |
| return_on_assets | Float | YES | |
| return_on_equity | Float | YES | |
| total_revenue | BigInteger | YES | |
| revenue_per_share | Float | YES | |
| revenue_growth | Float | YES | |
| earnings_growth | Float | YES | |
| earnings_quarterly_growth | Float | YES | |
| ebitda | BigInteger | YES | |
| gross_profits | BigInteger | YES | |
| free_cashflow | BigInteger | YES | |
| operating_cashflow | BigInteger | YES | |
| total_cash | BigInteger | YES | |
| total_cash_per_share | Float | YES | |
| total_debt | BigInteger | YES | |
| debt_to_equity | Float | YES | |
| current_ratio | Float | YES | |
| quick_ratio | Float | YES | |
| dividend_rate | Float | YES | |
| dividend_yield | Float | YES | |
| ex_dividend_date | Date | YES | |
| payout_ratio | Float | YES | |
| five_year_avg_dividend_yield | Float | YES | |
| trailing_annual_dividend_rate | Float | YES | |
| trailing_annual_dividend_yield | Float | YES | |
| last_dividend_value | Float | YES | |
| target_high_price | Float | YES | |
| target_low_price | Float | YES | |
| target_mean_price | Float | YES | |
| target_median_price | Float | YES | |
| number_of_analyst_opinions | Integer | YES | |
| recommendation_key | String(50) | YES | e.g., "buy", "hold" |
| recommendation_mean | Float | YES | 1.0=strong buy to 5.0=sell |
| full_time_employees | Integer | YES | |

---

### price_history
**Model**: `PriceHistory` in `ingestion/app/models/yfinance_data.py`
**Unique**: `uq_price_history_instrument_date` ON (instrument_id, date)
**Indexes**: ix_price_history_instrument_id, ix_price_history_date

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| instrument_id | UUID FK→instruments.id | NO | CASCADE |
| date | Date | NO | Trading date |
| open | Numeric(20,6) | YES | |
| high | Numeric(20,6) | YES | |
| low | Numeric(20,6) | YES | |
| close | Numeric(20,6) | YES | |
| volume | BigInteger | YES | |
| dividends | Numeric(20,6) | YES | Ex-dividend amount on this date |
| stock_splits | Numeric(20,6) | YES | Split ratio on this date |

---

### financial_statements
**Model**: `FinancialStatement` in `ingestion/app/models/yfinance_data.py`
**Unique**: `uq_financial_statement_row` ON (instrument_id, statement_type, period_type, period_date, line_item)
**Indexes**: ix_financial_statements_instrument_id, ix_financial_statements_type_period, ix_financial_statements_period_date

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| instrument_id | UUID FK→instruments.id | NO | CASCADE |
| statement_type | String(50) | NO | "income_statement", "balance_sheet", "cashflow", "earnings" |
| period_type | String(20) | NO | "annual", "quarterly" |
| period_date | Date | NO | End of fiscal period |
| line_item | String(200) | NO | e.g., "TotalRevenue", "NetIncome" |
| value | Numeric(38,6) | YES | Monetary amount |
| currency_code | String(10) | YES | e.g., "USD", "EUR" |

---

### dividends
**Model**: `Dividend` in `ingestion/app/models/yfinance_data.py`
**Unique**: `uq_dividend_instrument_date` ON (instrument_id, date)
**Indexes**: ix_dividends_instrument_id

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| instrument_id | UUID FK→instruments.id | NO | CASCADE |
| date | Date | NO | Ex-dividend date |
| amount | Numeric(20,6) | NO | Per-share dividend |

---

### stock_splits
**Model**: `StockSplit` in `ingestion/app/models/yfinance_data.py`
**Unique**: `uq_stock_split_instrument_date` ON (instrument_id, date)
**Indexes**: ix_stock_splits_instrument_id

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| instrument_id | UUID FK→instruments.id | NO | CASCADE |
| date | Date | NO | Split date |
| ratio | Numeric(20,6) | NO | e.g., 4.0 for 4:1 split |

---

### analyst_recommendations
**Model**: `AnalystRecommendation` in `ingestion/app/models/yfinance_data.py`
**Unique**: `uq_analyst_rec_instrument_period` ON (instrument_id, period)
**Indexes**: ix_analyst_recommendations_instrument_id

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| instrument_id | UUID FK→instruments.id | NO | CASCADE |
| period | String(50) | NO | e.g., "0m", "-1m", "-2m", "-3m" |
| strong_buy | Integer | YES | |
| buy | Integer | YES | |
| hold | Integer | YES | |
| sell | Integer | YES | |
| strong_sell | Integer | YES | |

---

### analyst_price_targets
**Model**: `AnalystPriceTarget` in `ingestion/app/models/yfinance_data.py`
**Unique**: `uq_analyst_pt_instrument` ON (instrument_id) — **1:1**
**Indexes**: ix_analyst_price_targets_instrument_id

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| instrument_id | UUID FK→instruments.id | NO | CASCADE |
| current | Numeric(20,6) | YES | |
| low | Numeric(20,6) | YES | |
| high | Numeric(20,6) | YES | |
| mean | Numeric(20,6) | YES | |
| median | Numeric(20,6) | YES | |

---

### institutional_holders
**Model**: `InstitutionalHolder` in `ingestion/app/models/yfinance_data.py`
**Unique**: `uq_inst_holder_instrument_name` ON (instrument_id, holder_name)
**Indexes**: ix_institutional_holders_instrument_id

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| instrument_id | UUID FK→instruments.id | NO | CASCADE |
| holder_name | String(500) | NO | |
| date_reported | Date | YES | |
| shares | BigInteger | YES | |
| value | BigInteger | YES | Total position value |
| pct_held | Float | YES | Percentage of outstanding |

---

### mutual_fund_holders
**Model**: `MutualFundHolder` in `ingestion/app/models/yfinance_data.py`
**Unique**: `uq_mutual_fund_holder_instrument_name` ON (instrument_id, holder_name)
**Indexes**: ix_mutual_fund_holders_instrument_id

Same column structure as institutional_holders.

---

### insider_transactions
**Model**: `InsiderTransaction` in `ingestion/app/models/yfinance_data.py`
**Unique**: `uq_insider_tx_row` ON (instrument_id, insider_name, start_date, transaction_type)
**Indexes**: ix_insider_transactions_instrument_id, ix_insider_transactions_start_date

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| instrument_id | UUID FK→instruments.id | NO | CASCADE |
| insider_name | String(500) | NO | |
| position | String(500) | YES | e.g., "CEO", "CFO" |
| transaction_type | String(200) | NO | e.g., "Insider Purchase" |
| shares | BigInteger | YES | |
| value | BigInteger | YES | |
| start_date | Date | NO | Sentinel: 1970-01-01 if missing |
| ownership | String(50) | YES | "D" (direct) or "I" (indirect) |

---

### ticker_news
**Model**: `TickerNews` in `ingestion/app/models/yfinance_data.py`
**Unique**: `uq_ticker_news_instrument_uuid` ON (instrument_id, news_uuid)
**Indexes**: ix_ticker_news_instrument_id, ix_ticker_news_publish_time

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| instrument_id | UUID FK→instruments.id | NO | CASCADE |
| news_uuid | String(200) | NO | yfinance article UUID |
| title | Text | YES | |
| publisher | String(500) | YES | |
| link | Text | YES | |
| publish_time | DateTime(tz) | YES | Historical: may lack TZ info |
| news_type | String(100) | YES | |
| ticker_name | String(500) | YES | Denormalized ticker display name |
| full_content | Text | YES | Scraped article body |

---

## Macro Regime Domain

### economic_indicators
**Model**: `EconomicIndicator` in `ingestion/app/models/macro_regime.py`
**Unique**: `uq_economic_indicator_country` ON (country)
**Indexes**: ix_economic_indicators_country

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| country | String(100) | NO | USA, France, Germany, UK |
| last_inflation | Float | YES | IlSole24Ore forecast |
| inflation_6m | Float | YES | |
| inflation_10y_avg | Float | YES | |
| gdp_growth_6m | Float | YES | |
| earnings_12m | Float | YES | |
| eps_expected_12m | Float | YES | |
| peg_ratio | Float | YES | |
| lt_rate_forecast | Float | YES | Long-term rate forecast |
| reference_date | Date | YES | |

---

### economic_indicator_observations
**Model**: `EconomicIndicatorObservation` in `ingestion/app/models/macro_regime.py`
**Unique**: `uq_econ_obs_country_date` ON (country, date)
**Indexes**: ix_econ_observations_country, ix_econ_observations_date

Same columns as economic_indicators plus `date` (Date, NOT NULL). One row per (country, date) preserving daily snapshots.

---

### trading_economics_indicators
**Model**: `TradingEconomicsIndicator` in `ingestion/app/models/macro_regime.py`
**Unique**: `uq_te_indicator_country_key` ON (country, indicator_key)
**Indexes**: ix_trading_economics_indicators_country

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| country | String(100) | NO | |
| indicator_key | String(100) | NO | e.g., "gdp_growth_rate", "inflation_rate" |
| value | Float | YES | Latest value |
| previous | Float | YES | Previous period value |
| unit | String(50) | YES | e.g., "percent", "USD Billion" |
| reference | String(100) | YES | Reference period |
| raw_name | String(200) | YES | Original name from source |

---

### trading_economics_observations
**Model**: `TradingEconomicsObservation` in `ingestion/app/models/macro_regime.py`
**Unique**: `uq_te_obs_country_key_date` ON (country, indicator_key, date)
**Indexes**: ix_te_observations_country, ix_te_observations_date, ix_te_obs_country_key_date (composite)

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| country | String(100) | NO | |
| indicator_key | String(100) | NO | |
| date | Date | NO | |
| value | Float | YES | |

---

### bond_yields
**Model**: `BondYield` in `ingestion/app/models/macro_regime.py`
**Unique**: `uq_bond_yield_country_maturity` ON (country, maturity)
**Indexes**: ix_bond_yields_country

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| country | String(100) | NO | |
| maturity | String(10) | NO | "2Y", "5Y", "10Y", "30Y" |
| yield_value | Float | YES | Current yield |
| day_change | Float | YES | |
| month_change | Float | YES | |
| year_change | Float | YES | |
| reference_date | Date | YES | |

---

### bond_yield_observations
**Model**: `BondYieldObservation` in `ingestion/app/models/macro_regime.py`
**Unique**: `uq_bond_obs_country_mat_date` ON (country, maturity, date)
**Indexes**: ix_bond_observations_country, ix_bond_observations_date, ix_bond_obs_country_maturity_date (composite)

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| country | String(100) | NO | |
| maturity | String(10) | NO | |
| date | Date | NO | |
| yield_value | Float | YES | |

---

### fred_observations
**Model**: `FredObservation` in `ingestion/app/models/macro_regime.py`
**Unique**: `uq_fred_observation_series_date` ON (series_id, date)
**Indexes**: ix_fred_observations_series_id, ix_fred_observations_date

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| series_id | String(50) | NO | FRED series code |
| date | Date | NO | Observation date |
| value | Float | YES | |

---

### macro_news
**Model**: `MacroNews` in `ingestion/app/models/macro_regime.py`
**Unique**: `uq_macro_news_id` ON (news_id)
**Indexes**: ix_macro_news_publish_time

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| news_id | String(200) | NO | Deduplication key |
| title | Text | YES | |
| publisher | String(500) | YES | |
| link | Text | YES | |
| publish_time | DateTime(tz) | YES | |
| source_ticker | String(50) | YES | Ticker used to find this article |
| source_query | String(200) | YES | Search query used |
| snippet | Text | YES | Short preview |
| full_content | Text | YES | Scraped full article |

**Relationships**: `theme_entries` (1→many MacroNewsTheme, CASCADE, delete-orphan, lazy="selectin")
**Property**: `themes` → comma-joined sorted theme strings

---

### macro_news_themes
**Model**: `MacroNewsTheme` in `ingestion/app/models/macro_regime.py`
**Unique**: `uq_macro_news_theme` ON (news_id, theme)
**Indexes**: ix_macro_news_themes_news_id, ix_macro_news_themes_theme

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| news_id | UUID FK→macro_news.id | NO | CASCADE |
| theme | String(50) | NO | e.g., "inflation", "trade", "monetary_policy" |

---

### macro_news_summaries
**Model**: `MacroNewsSummary` in `ingestion/app/models/macro_regime.py`
**Unique**: `uq_macro_news_summary_country_date` ON (country, summary_date)
**Indexes**: ix_macro_news_summaries_country, ix_macro_news_summaries_summary_date

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| country | String(100) | NO | |
| summary_date | Date | NO | |
| summary | Text | YES | AI-generated summary |
| sentiment | String(50) | YES | e.g., "positive", "negative", "neutral" |
| sentiment_score | Float | YES | Numeric sentiment score |
| article_count | Integer | YES | Number of articles summarized |
| news_summary | Text | YES | Alternative/extended summary |

---

### macro_calibrations
**Model**: `MacroCalibration` in `ingestion/app/models/macro_regime.py`
**Unique**: `uq_macro_calibration_country` ON (country)
**Indexes**: ix_macro_calibrations_country

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| country | String(100) | NO | |
| phase | String(50) | NO | Regime phase from BAML ClassifyMacroRegime |
| delta | Float | NO | Risk adjustment parameter |
| tau | Float | NO | Uncertainty parameter |
| confidence | Float | NO | LLM confidence score |
| rationale | Text | YES | LLM reasoning |
| macro_summary | Text | YES | Underlying data summary |

---

## Operations Domain

### background_jobs
**Model**: `BackgroundJob` in `ingestion/app/models/background_job.py`
**Indexes**: ix_background_jobs_type_status (composite), job_type (column-level)

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| job_type | String(100) | NO | | e.g., "macro_fetch", "yfinance_fetch" |
| status | String(20) | NO | "pending" | "pending", "running", "completed", "failed" |
| current | Integer | NO | 0 | Progress numerator |
| total | Integer | NO | 0 | Progress denominator |
| extra | JSON/JSONB | YES | | Custom progress metadata |
| result | JSON/JSONB | YES | | Operation results (counts, etc.) |
| error | Text | YES | | Primary error message |
| started_at | DateTime(tz) | NO | now() | |
| finished_at | DateTime(tz) | YES | | |

**Note**: `extra` and `result` use `JSON().with_variant(JSONB, "postgresql")` for SQLite test compatibility.
**Property**: `errors` → list of error messages from error_entries

---

### background_job_errors
**Model**: `BackgroundJobError` in `ingestion/app/models/background_job.py`
**Unique**: `uq_bg_job_error_index` ON (job_id, error_index)
**Indexes**: ix_background_job_errors_job_id

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| job_id | UUID FK→background_jobs.id | NO | CASCADE |
| error_index | Integer | NO | Ordered position (0-based) |
| message | Text | NO | Error message text |
