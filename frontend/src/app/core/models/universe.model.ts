export interface UniverseStats {
  total_exchanges: number;
  total_instruments: number;
  last_updated: string;
}

export interface Exchange {
  id: string;
  name: string;
  mic: string;
  country: string;
  instrument_count: number;
}

export interface Instrument {
  id: string;
  ticker: string;
  short_name: string;
  name: string | null;
  isin: string | null;
  instrument_type: string | null;
  currency_code: string | null;
  yfinance_ticker: string | null;
  exchange_name: string | null;
}

export interface InstrumentList {
  items: Instrument[];
  total: number;
  page: number;
  page_size: number;
}

export interface BuildProgress {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current: number;
  total: number;
  detail: string;
}

export interface BuildResult {
  exchanges_added: number;
  instruments_added: number;
  duration_seconds: number;
}

// ── Universe screen (POST /universe/screen) ────────────────────────────────
// Mirrors api/app/schemas/universe_screen.py.

export type ScreenPreset =
  | 'developed_markets'
  | 'broad_universe'
  | 'small_cap'
  | 'large_cap';

export interface HysteresisThreshold {
  entry: number;
  exit: number;
}

export interface UniverseScreenRequest {
  preset?: ScreenPreset;
  market_cap?: HysteresisThreshold;
  addv_12m?: HysteresisThreshold;
  addv_3m?: HysteresisThreshold;
  trading_frequency?: HysteresisThreshold;
  price_us?: HysteresisThreshold;
  price_europe?: HysteresisThreshold;
  min_trading_history?: number;
  min_ipo_seasoning?: number;
  min_annual_reports?: number;
  min_quarterly_reports?: number;
  exchange_region?: string;
  mcap_percentile_entry?: number;
  mcap_percentile_exit?: number;
  current_members?: string[];
}

export interface UniverseScreenResponse {
  passingTickers: string[];
  totalScreened: number;
  diagnostics: Record<string, Record<string, boolean>>;
}
