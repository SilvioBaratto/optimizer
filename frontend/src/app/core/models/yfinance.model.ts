// ---------------------------------------------------------------------------
// Shapes returned by /api/v1/yfinance-data/instruments/{id}/*
// Field names mirror the backend Pydantic schemas in
// api/app/schemas/yfinance_data.py — keep snake_case, do not rename.
// ---------------------------------------------------------------------------

export interface TickerProfile {
  id: string;
  instrument_id: string;
  symbol: string | null;
  short_name: string | null;
  long_name: string | null;
  isin: string | null;
  exchange: string | null;
  quote_type: string | null;
  currency: string | null;
  sector: string | null;
  industry: string | null;
  country: string | null;
  website: string | null;
  long_business_summary: string | null;
  market_cap: number | null;
  enterprise_value: number | null;
  shares_outstanding: number | null;
  float_shares: number | null;
  current_price: number | null;
  previous_close: number | null;
  fifty_two_week_low: number | null;
  fifty_two_week_high: number | null;
  fifty_day_average: number | null;
  two_hundred_day_average: number | null;
  average_volume: number | null;
  beta: number | null;
  trailing_pe: number | null;
  forward_pe: number | null;
  trailing_eps: number | null;
  forward_eps: number | null;
  price_to_sales_trailing_12months: number | null;
  price_to_book: number | null;
  enterprise_to_revenue: number | null;
  enterprise_to_ebitda: number | null;
  peg_ratio: number | null;
  book_value: number | null;
  profit_margins: number | null;
  operating_margins: number | null;
  gross_margins: number | null;
  return_on_assets: number | null;
  return_on_equity: number | null;
  total_revenue: number | null;
  revenue_growth: number | null;
  earnings_growth: number | null;
  ebitda: number | null;
  free_cashflow: number | null;
  operating_cashflow: number | null;
  total_debt: number | null;
  debt_to_equity: number | null;
  current_ratio: number | null;
  dividend_rate: number | null;
  dividend_yield: number | null;
  payout_ratio: number | null;
  recommendation_key: string | null;
  recommendation_mean: number | null;
  full_time_employees: number | null;
  created_at: string;
  updated_at: string;
}

export interface PriceHistory {
  id: string;
  instrument_id: string;
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
  dividends: number | null;
  stock_splits: number | null;
  created_at: string;
  updated_at: string;
}

export interface FinancialStatement {
  id: string;
  instrument_id: string;
  statement_type: string;
  period_type: string;
  period_date: string;
  line_item: string;
  value: number | null;
  created_at: string;
  updated_at: string;
}

export interface Dividend {
  id: string;
  instrument_id: string;
  date: string;
  amount: number;
  created_at: string;
  updated_at: string;
}

export interface StockSplit {
  id: string;
  instrument_id: string;
  date: string;
  ratio: number;
  created_at: string;
  updated_at: string;
}

export interface AnalystRecommendation {
  id: string;
  instrument_id: string;
  period: string;
  strong_buy: number | null;
  buy: number | null;
  hold: number | null;
  sell: number | null;
  strong_sell: number | null;
  created_at: string;
  updated_at: string;
}

export interface AnalystPriceTarget {
  id: string;
  instrument_id: string;
  current: number | null;
  low: number | null;
  high: number | null;
  mean: number | null;
  median: number | null;
  created_at: string;
  updated_at: string;
}

export interface InstitutionalHolder {
  id: string;
  instrument_id: string;
  holder_name: string;
  date_reported: string | null;
  shares: number | null;
  value: number | null;
  pct_held: number | null;
  created_at: string;
  updated_at: string;
}

export interface MutualFundHolder {
  id: string;
  instrument_id: string;
  holder_name: string;
  date_reported: string | null;
  shares: number | null;
  value: number | null;
  pct_held: number | null;
  created_at: string;
  updated_at: string;
}

export interface InsiderTransaction {
  id: string;
  instrument_id: string;
  insider_name: string;
  position: string | null;
  transaction_type: string;
  shares: number | null;
  value: number | null;
  start_date: string;
  ownership: string | null;
  created_at: string;
  updated_at: string;
}

export interface TickerNews {
  id: string;
  instrument_id: string;
  news_uuid: string;
  title: string | null;
  publisher: string | null;
  link: string | null;
  publish_time: string | null;
  news_type: string | null;
  ticker_name: string | null;
  full_content: string | null;
  created_at: string;
  updated_at: string;
}

export interface FetchProgress {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current: number;
  total: number;
  detail: string;
}
