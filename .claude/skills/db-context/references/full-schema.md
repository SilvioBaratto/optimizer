# Full schema reference — optimizer PostgreSQL

Column-by-column reference for all 57 base tables (55 modelled + 2 infra), grouped by
domain. Live row counts are from the last snapshot. Nullable `no` = `NOT NULL`.

Connection: `postgresql://postgres:postgres@localhost:54320/optimizer_db`
Models: `packages/portopt-db/src/portopt_db/models/{jobs,macro,market_data,universe}/`
Alembic head: `b6c7d8e9f0a1` (62 migrations, single tree at `packages/portopt-db/alembic/`)

---

## Core

### instruments — 8,898 rows
- PK: `id`
- Unique: `uq_instrument_ticker_exchange` (ticker, exchange_id)
- FK: `exchange_id` → exchanges.id

| Column | Type | Nullable |
|--------|------|----------|
| ticker | character varying | no |
| short_name | character varying | no |
| name | character varying | yes |
| isin | character varying | yes |
| instrument_type | character varying | yes |
| currency_code | character varying | yes |
| yfinance_ticker | character varying | yes |
| exchange_id | uuid | no |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |
| delisted_at | date | yes |
| delisting_return | double precision | yes |
| asset_class | character varying | no |
| fi_subclass | character varying | yes |
| duration_bucket | character varying | yes |
| t212_ticker | character varying | yes |

---

## Universe

### exchanges — 14 rows
- PK: `id`
- Unique: `exchanges_name_key` (name)
- FK: none

| Column | Type | Nullable |
|--------|------|----------|
| name | character varying | no |
| t212_id | integer | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

---

## Prices & Actions

### price_history — 9,786,726 rows
- PK: `id`
- Unique: `uq_price_history_instrument_date` (instrument_id, date)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| date | date | no |
| open | numeric | yes |
| high | numeric | yes |
| low | numeric | yes |
| close | numeric | yes |
| volume | bigint | yes |
| dividends | numeric | yes |
| stock_splits | numeric | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |
| price_unit | character varying | yes |
| capital_gains | numeric | yes |

### dividends — 372,607 rows
- PK: `id`
- Unique: `uq_dividend_instrument_date` (instrument_id, date)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| date | date | no |
| amount | numeric | no |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### stock_splits — 7,794 rows
- PK: `id`
- Unique: `uq_stock_split_instrument_date` (instrument_id, date)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| date | date | no |
| ratio | numeric | no |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### options_chain — 0 rows
- PK: `id`
- Unique: `uq_option_contract` (instrument_id, as_of, contract_symbol)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| as_of | date | no |
| expiry | date | no |
| option_type | character varying | no |
| strike | numeric | no |
| contract_symbol | character varying | no |
| last_price | numeric | yes |
| bid | numeric | yes |
| ask | numeric | yes |
| volume | bigint | yes |
| open_interest | bigint | yes |
| implied_volatility | numeric | yes |
| in_the_money | boolean | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

---

## Fundamentals

### financial_statements — 11,253,341 rows
- PK: `id`
- Unique: `uq_financial_statement_row` (instrument_id, statement_type, period_type, period_date, line_item)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| statement_type | character varying | no |
| period_type | character varying | no |
| period_date | date | no |
| line_item | character varying | no |
| value | numeric | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |
| currency_code | character varying | yes |

### sec_filings — 389,385 rows
- PK: `id`
- Unique: `uq_sec_filing` (instrument_id, filing_date, form_type, title)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| filing_date | date | no |
| form_type | character varying | no |
| title | character varying | no |
| url | text | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### ticker_profiles — 8,895 rows
- PK: `id`
- Unique: `uq_ticker_profile_instrument` (instrument_id)
- FK: `instrument_id` → instruments.id
- 1:1 with instruments. 85 columns — the wide yfinance `.info` snapshot (identity, valuation, margins, cash-flow, dividends, analyst targets).

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| symbol | character varying | yes |
| short_name | character varying | yes |
| long_name | character varying | yes |
| isin | character varying | yes |
| exchange | character varying | yes |
| quote_type | character varying | yes |
| currency | character varying | yes |
| sector | character varying | yes |
| industry | character varying | yes |
| country | character varying | yes |
| website | character varying | yes |
| long_business_summary | text | yes |
| market_cap | bigint | yes |
| enterprise_value | bigint | yes |
| shares_outstanding | bigint | yes |
| float_shares | bigint | yes |
| implied_shares_outstanding | bigint | yes |
| current_price | double precision | yes |
| previous_close | double precision | yes |
| open_price | double precision | yes |
| day_low | double precision | yes |
| day_high | double precision | yes |
| fifty_two_week_low | double precision | yes |
| fifty_two_week_high | double precision | yes |
| fifty_day_average | double precision | yes |
| two_hundred_day_average | double precision | yes |
| average_volume | bigint | yes |
| average_volume_10days | bigint | yes |
| regular_market_volume | bigint | yes |
| bid | double precision | yes |
| ask | double precision | yes |
| bid_size | integer | yes |
| ask_size | integer | yes |
| beta | double precision | yes |
| trailing_pe | double precision | yes |
| forward_pe | double precision | yes |
| trailing_eps | double precision | yes |
| forward_eps | double precision | yes |
| price_to_sales_trailing_12months | double precision | yes |
| price_to_book | double precision | yes |
| enterprise_to_revenue | double precision | yes |
| enterprise_to_ebitda | double precision | yes |
| peg_ratio | double precision | yes |
| book_value | double precision | yes |
| profit_margins | double precision | yes |
| operating_margins | double precision | yes |
| gross_margins | double precision | yes |
| ebitda_margins | double precision | yes |
| return_on_assets | double precision | yes |
| return_on_equity | double precision | yes |
| total_revenue | bigint | yes |
| revenue_per_share | double precision | yes |
| revenue_growth | double precision | yes |
| earnings_growth | double precision | yes |
| earnings_quarterly_growth | double precision | yes |
| ebitda | bigint | yes |
| gross_profits | bigint | yes |
| free_cashflow | bigint | yes |
| operating_cashflow | bigint | yes |
| total_cash | bigint | yes |
| total_cash_per_share | double precision | yes |
| total_debt | bigint | yes |
| debt_to_equity | double precision | yes |
| current_ratio | double precision | yes |
| quick_ratio | double precision | yes |
| dividend_rate | double precision | yes |
| dividend_yield | double precision | yes |
| ex_dividend_date | date | yes |
| payout_ratio | double precision | yes |
| five_year_avg_dividend_yield | double precision | yes |
| trailing_annual_dividend_rate | double precision | yes |
| trailing_annual_dividend_yield | double precision | yes |
| last_dividend_value | double precision | yes |
| target_high_price | double precision | yes |
| target_low_price | double precision | yes |
| target_mean_price | double precision | yes |
| target_median_price | double precision | yes |
| number_of_analyst_opinions | integer | yes |
| recommendation_key | character varying | yes |
| recommendation_mean | double precision | yes |
| full_time_employees | integer | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### ticker_profile_extras — 8,895 rows
- PK: `id`
- Unique: `uq_ticker_profile_extras_instrument` (instrument_id)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| shares_short | bigint | yes |
| shares_short_prior_month | bigint | yes |
| short_ratio | double precision | yes |
| short_percent_of_float | double precision | yes |
| shares_percent_shares_out | double precision | yes |
| held_percent_insiders | double precision | yes |
| held_percent_institutions | double precision | yes |
| fifty_two_week_change | double precision | yes |
| sandp_52_week_change | double precision | yes |
| sector_key | character varying | yes |
| industry_key | character varying | yes |
| audit_risk | integer | yes |
| board_risk | integer | yes |
| compensation_risk | integer | yes |
| shareholder_rights_risk | integer | yes |
| overall_risk | integer | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### etf_fund_operations — 839 rows
- PK: `id`
- Unique: `uq_etf_fund_operations_instrument_asof` (instrument_id, as_of)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| as_of | date | no |
| annual_report_expense_ratio | numeric | yes |
| annual_holdings_turnover | numeric | yes |
| total_net_assets | numeric | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

---

## Estimates

### earnings_estimate — 23,656 rows
- PK: `id`
- Unique: `uq_earnings_estimate_instrument_period` (instrument_id, period)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| period | character varying | no |
| num_analysts | integer | yes |
| avg | numeric | yes |
| low | numeric | yes |
| high | numeric | yes |
| year_ago_eps | numeric | yes |
| growth | numeric | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### revenue_estimate — 30,408 rows
- PK: `id`
- Unique: `uq_revenue_estimate_instrument_period` (instrument_id, period)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| period | character varying | no |
| num_analysts | integer | yes |
| avg | numeric | yes |
| low | numeric | yes |
| high | numeric | yes |
| year_ago_revenue | numeric | yes |
| growth | numeric | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### earnings_history — 17,956 rows
- PK: `id`
- Unique: `uq_earnings_history_instrument_period` (instrument_id, period_date)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| period_date | date | no |
| eps_estimate | numeric | yes |
| eps_actual | numeric | yes |
| eps_difference | numeric | yes |
| surprise_percent | numeric | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### growth_estimates — 40,203 rows
- PK: `id`
- Unique: `uq_growth_estimate_instrument_period` (instrument_id, period)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| period | character varying | no |
| stock_trend | numeric | yes |
| index_trend | numeric | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

---

## Analyst

### analyst_actions — 419,930 rows
- PK: `id`
- Unique: `uq_analyst_action` (instrument_id, action_date, firm, to_grade)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| action_date | date | no |
| firm | character varying | no |
| from_grade | character varying | yes |
| to_grade | character varying | no |
| action | character varying | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### analyst_price_targets — 7,949 rows
- PK: `id`
- Unique: `uq_analyst_pt_instrument` (instrument_id)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| current | numeric | yes |
| low | numeric | yes |
| high | numeric | yes |
| mean | numeric | yes |
| median | numeric | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### analyst_recommendations — 17,100 rows
- PK: `id`
- Unique: `uq_analyst_rec_instrument_period` (instrument_id, period)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| period | character varying | no |
| strong_buy | integer | yes |
| buy | integer | yes |
| hold | integer | yes |
| sell | integer | yes |
| strong_sell | integer | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

---

## Ownership

### insider_transactions — 256,439 rows
- PK: `id`
- Unique: `uq_insider_tx_row` (instrument_id, insider_name, transaction_type, start_date)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| insider_name | character varying | no |
| position | character varying | yes |
| transaction_type | character varying | no |
| shares | bigint | yes |
| value | bigint | yes |
| start_date | date | no |
| ownership | character varying | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### institutional_holders — 58,417 rows
- PK: `id`
- Unique: `uq_inst_holder_instrument_name` (instrument_id, holder_name)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| holder_name | character varying | no |
| date_reported | date | yes |
| shares | bigint | yes |
| value | bigint | yes |
| pct_held | double precision | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### mutual_fund_holders — 52,398 rows
- PK: `id`
- Unique: `uq_mutual_fund_holder_instrument_name` (instrument_id, holder_name)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| holder_name | character varying | no |
| date_reported | date | yes |
| shares | bigint | yes |
| value | bigint | yes |
| pct_held | double precision | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### insider_roster — 44,559 rows
- PK: `id`
- Unique: `uq_insider_roster_instrument_name` (instrument_id, insider_name)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| insider_name | character varying | no |
| position | character varying | yes |
| most_recent_transaction | character varying | yes |
| latest_transaction_date | date | yes |
| shares_owned_directly | bigint | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### insider_purchases — 8,024 rows
- PK: `id`
- Unique: `uq_insider_purchases_instrument` (instrument_id)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| purchase_shares | bigint | yes |
| sale_shares | bigint | yes |
| net_shares | bigint | yes |
| total_insider_shares | bigint | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### major_holders — 7,734 rows
- PK: `id`
- Unique: `uq_major_holders_instrument` (instrument_id)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| insiders_percent_held | double precision | yes |
| institutions_percent_held | double precision | yes |
| institutions_float_percent_held | double precision | yes |
| institutions_count | bigint | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### shares_outstanding — 789,921 rows
- PK: `id`
- Unique: `uq_shares_outstanding_instrument_date` (instrument_id, date)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| date | date | no |
| shares | bigint | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

---

## ETF

### etf_metadata — 839 rows
- PK: `id`
- Unique: `uq_etf_metadata_instrument` (instrument_id)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| instrument_id | uuid | no |
| aum | numeric | yes |
| nav | numeric | yes |
| fund_family | character varying | yes |
| legal_type | character varying | yes |
| expense_ratio | numeric | yes |
| base_currency | character varying | yes |
| as_of | date | yes |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |
| category | character varying | yes |
| description | text | yes |

### etf_asset_classes — 839 rows
- PK: `id`
- Unique: `uq_etf_asset_classes_instrument_asof` (instrument_id, as_of)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| instrument_id | uuid | no |
| as_of | date | no |
| stock_pct | numeric | yes |
| bond_pct | numeric | yes |
| cash_pct | numeric | yes |
| other_pct | numeric | yes |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### etf_equity_holdings — 839 rows
- PK: `id`
- Unique: `uq_etf_equity_holdings_instrument_asof` (instrument_id, as_of)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| as_of | date | no |
| price_to_earnings | numeric | yes |
| price_to_book | numeric | yes |
| price_to_sales | numeric | yes |
| price_to_cashflow | numeric | yes |
| median_market_cap | numeric | yes |
| three_year_earnings_growth | numeric | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### etf_bond_holdings — 774 rows
- PK: `id`
- Unique: `uq_etf_bond_holdings_instrument_asof` (instrument_id, as_of)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| as_of | date | no |
| duration | numeric | yes |
| maturity | numeric | yes |
| credit_quality | numeric | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### etf_bond_ratings — 6,549 rows
- PK: `id`
- Unique: `uq_etf_bond_ratings_instrument_asof_rating` (instrument_id, as_of, rating)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| as_of | date | no |
| rating | character varying | no |
| weight | numeric | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### etf_holdings — 770 rows
- PK: `id`
- Unique: `uq_etf_holdings_instrument_asof_symbol` (instrument_id, as_of, holding_symbol)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| instrument_id | uuid | no |
| as_of | date | no |
| holding_symbol | character varying | no |
| holding_name | character varying | yes |
| weight | numeric | yes |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### etf_sector_weights — 2,200 rows
- PK: `id`
- Unique: `uq_etf_sector_weights_instrument_asof_sector` (instrument_id, as_of, sector)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| instrument_id | uuid | no |
| as_of | date | no |
| sector | character varying | no |
| weight | numeric | yes |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

---

## Calendars

### earnings_dates — 106,647 rows
- PK: `id`
- Unique: `uq_earnings_date_instrument_date` (instrument_id, earnings_date)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| earnings_date | date | no |
| eps_estimate | numeric | yes |
| eps_actual | numeric | yes |
| surprise_percent | numeric | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### earnings_calendar — 3,996 rows
- PK: `id`
- Unique: `uq_earnings_calendar` (ticker, event_date)
- FK: none (market-wide)

| Column | Type | Nullable |
|--------|------|----------|
| ticker | character varying | no |
| event_date | date | no |
| company_name | character varying | yes |
| eps_estimate | numeric | yes |
| eps_actual | numeric | yes |
| eps_surprise_pct | numeric | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### economic_event_calendar — 1,496 rows
- PK: `id`
- Unique: `uq_economic_event_calendar` (event, country, event_date)
- FK: none (market-wide)

| Column | Type | Nullable |
|--------|------|----------|
| event | character varying | no |
| country | character varying | no |
| event_date | date | no |
| actual | character varying | yes |
| forecast | character varying | yes |
| prior | character varying | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### split_calendar — 477 rows
- PK: `id`
- Unique: `uq_split_calendar` (ticker, split_date)
- FK: none (market-wide)

| Column | Type | Nullable |
|--------|------|----------|
| ticker | character varying | no |
| split_date | date | no |
| company_name | character varying | yes |
| ratio | character varying | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### ipo_calendar — 11 rows
- PK: `id`
- Unique: `uq_ipo_calendar` (ticker, ipo_date)
- FK: none (market-wide)

| Column | Type | Nullable |
|--------|------|----------|
| ticker | character varying | no |
| ipo_date | date | no |
| company_name | character varying | yes |
| exchange | character varying | yes |
| currency | character varying | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

---

## Sector

### sector_snapshots — 0 rows
- PK: `id`
- Unique: `uq_sector_snapshot` (sector_key, region, as_of)
- FK: none (market-wide)

| Column | Type | Nullable |
|--------|------|----------|
| sector_key | character varying | no |
| region | character varying | no |
| as_of | date | no |
| name | character varying | yes |
| symbol | character varying | yes |
| market_cap | numeric | yes |
| market_weight | numeric | yes |
| companies_count | integer | yes |
| industries_count | integer | yes |
| employee_count | bigint | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### sector_industries — 0 rows
- PK: `id`
- Unique: `uq_sector_industry` (sector_key, region, as_of, industry_key)
- FK: none (market-wide)

| Column | Type | Nullable |
|--------|------|----------|
| sector_key | character varying | no |
| region | character varying | no |
| as_of | date | no |
| industry_key | character varying | no |
| industry_name | character varying | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### sector_top_companies — 0 rows
- PK: `id`
- Unique: `uq_sector_top_company` (sector_key, region, as_of, symbol)
- FK: none (market-wide)

| Column | Type | Nullable |
|--------|------|----------|
| sector_key | character varying | no |
| region | character varying | no |
| as_of | date | no |
| symbol | character varying | no |
| name | character varying | yes |
| weight | numeric | yes |
| rating | character varying | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

---

## Market

### market_summaries — 0 rows
- PK: `id`
- Unique: `uq_market_summary` (market, symbol, as_of)
- FK: none (market-wide)

| Column | Type | Nullable |
|--------|------|----------|
| market | character varying | no |
| symbol | character varying | no |
| as_of | date | no |
| short_name | character varying | yes |
| price | numeric | yes |
| change | numeric | yes |
| change_percent | numeric | yes |
| previous_close | numeric | yes |
| market_state | character varying | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

---

## News

### ticker_news — 61,898 rows
- PK: `id`
- Unique: `uq_ticker_news_instrument_uuid` (instrument_id, news_uuid)
- FK: `instrument_id` → instruments.id

| Column | Type | Nullable |
|--------|------|----------|
| instrument_id | uuid | no |
| news_uuid | character varying | no |
| title | text | yes |
| publisher | character varying | yes |
| link | text | yes |
| publish_time | timestamp with time zone | yes |
| news_type | character varying | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |
| full_content | text | yes |
| ticker_name | character varying | yes |

### macro_news — 36 rows
- PK: `id`
- Unique: `uq_macro_news_id` (news_id)
- FK: none

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| news_id | character varying | no |
| title | text | yes |
| publisher | character varying | yes |
| link | text | yes |
| publish_time | timestamp with time zone | yes |
| source_ticker | character varying | yes |
| source_query | character varying | yes |
| snippet | text | yes |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |
| full_content | text | yes |

---

## Macro

### trading_economics_indicators — 138 rows
- PK: `id`
- Unique: `uq_te_indicator_country_key` (country, indicator_key)
- FK: none

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| country | character varying | no |
| indicator_key | character varying | no |
| value | double precision | yes |
| previous | double precision | yes |
| unit | character varying | yes |
| reference | character varying | yes |
| raw_name | character varying | yes |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### trading_economics_observations — 138 rows
- PK: `id`
- Unique: `uq_te_obs_country_key_date` (country, indicator_key, date)
- FK: none

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| country | character varying | no |
| indicator_key | character varying | no |
| date | date | no |
| value | double precision | yes |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### bond_yields — 16 rows
- PK: `id`
- Unique: `uq_bond_yield_country_maturity` (country, maturity)
- FK: none

| Column | Type | Nullable |
|--------|------|----------|
| country | character varying | no |
| maturity | character varying | no |
| yield_value | double precision | yes |
| day_change | double precision | yes |
| month_change | double precision | yes |
| year_change | double precision | yes |
| reference_date | date | yes |
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### bond_yield_observations — 16 rows
- PK: `id`
- Unique: `uq_bond_obs_country_mat_date` (country, maturity, date)
- FK: none

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| country | character varying | no |
| maturity | character varying | no |
| date | date | no |
| yield_value | double precision | yes |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### economic_indicators — 4 rows
- PK: `id`
- Unique: `uq_economic_indicator_country` (country)
- FK: none

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| country | character varying | no |
| last_inflation | double precision | yes |
| inflation_6m | double precision | yes |
| inflation_10y_avg | double precision | yes |
| gdp_growth_6m | double precision | yes |
| earnings_12m | double precision | yes |
| eps_expected_12m | double precision | yes |
| peg_ratio | double precision | yes |
| lt_rate_forecast | double precision | yes |
| reference_date | date | yes |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### economic_indicator_observations — 4 rows
- PK: `id`
- Unique: `uq_econ_obs_country_date` (country, date)
- FK: none

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| country | character varying | no |
| date | date | no |
| last_inflation | double precision | yes |
| inflation_6m | double precision | yes |
| inflation_10y_avg | double precision | yes |
| gdp_growth_6m | double precision | yes |
| earnings_12m | double precision | yes |
| eps_expected_12m | double precision | yes |
| peg_ratio | double precision | yes |
| lt_rate_forecast | double precision | yes |
| reference_date | date | yes |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### fred_observations — 0 rows
- PK: `id`
- Unique: `uq_fred_observation_series_date` (series_id, date)
- FK: none

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| series_id | character varying | no |
| date | date | no |
| value | double precision | yes |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### macro_calibrations — 0 rows
- PK: `id`
- Unique: `uq_macro_calibration_country` (country)
- FK: none

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| country | character varying | no |
| phase | character varying | no |
| delta | double precision | no |
| tau | double precision | no |
| confidence | double precision | no |
| rationale | text | yes |
| macro_summary | text | yes |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |
| regime_classification | character varying | yes |

### macro_news_summaries — 4 rows
- PK: `id`
- Unique: `uq_macro_news_summary_country_date` (country, summary_date)
- FK: none

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| country | character varying | no |
| summary_date | date | no |
| summary | text | yes |
| sentiment | character varying | yes |
| sentiment_score | double precision | yes |
| article_count | integer | yes |
| news_summary | text | yes |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |

### macro_news_themes — 66 rows
- PK: `id`
- Unique: `uq_macro_news_theme` (news_id, theme)
- FK: `news_id` → macro_news.id

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |
| news_id | uuid | no |
| theme | character varying | no |

---

## Operations

### background_jobs — 6 rows
- PK: `id`
- Unique: none
- FK: none

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| job_type | character varying | no |
| status | character varying | no |
| current | integer | no |
| total | integer | no |
| extra | jsonb | yes |
| result | jsonb | yes |
| error | text | yes |
| started_at | timestamp with time zone | no |
| finished_at | timestamp with time zone | yes |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |
| worker_pid | integer | yes |
| worker_host | text | yes |
| last_heartbeat_at | timestamp with time zone | yes |
| attempt | integer | no |

### background_job_errors — 12 rows
- PK: `id`
- Unique: `uq_bg_job_error_index` (job_id, error_index)
- FK: `job_id` → background_jobs.id

| Column | Type | Nullable |
|--------|------|----------|
| id | uuid | no |
| created_at | timestamp with time zone | no |
| updated_at | timestamp with time zone | no |
| job_id | uuid | no |
| error_index | integer | no |
| message | text | no |

---

## Infrastructure (not modelled)

These two tables exist in the live DB but are not SQLAlchemy models in `portopt_db` —
they are managed by external tooling.

### alembic_version
Single-row table owned by Alembic; holds the current migration revision. Current head:
`b6c7d8e9f0a1`. Column: `version_num` (character varying, PK).

### apscheduler_jobs
Owned by APScheduler's `SQLAlchemyJobStore` (in-process scheduler in `app/worker.py`);
persists serialized scheduled-job state so misfired runs execute at next startup within the
grace window. Typical columns: `id` (varchar, PK), `next_run_time` (double precision,
indexed), `job_state` (bytea). Not part of the domain schema.
