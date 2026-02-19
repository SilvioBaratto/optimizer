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
  name: string;
  exchange: string;
  type: string;
  currency: string;
  isin: string;
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
