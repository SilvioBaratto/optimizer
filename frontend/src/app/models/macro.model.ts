export interface EconomicIndicator {
  id: string;
  name: string;
  country: string;
  category: string;
  value: number;
  previous: number;
  unit: string;
  frequency: string;
  last_updated: string;
}

export interface BondYield {
  maturity: string;
  yield_pct: number;
  change_bps: number;
  country: string;
  date: string;
}

export interface TradingEconomicsIndicator {
  id: string;
  country: string;
  category: string;
  title: string;
  latest_value: number;
  previous_value: number;
  unit: string;
  frequency: string;
  last_updated: string;
}

export interface CountryMacroSummary {
  country: string;
  gdp_growth: number;
  inflation: number;
  unemployment: number;
  interest_rate: number;
  debt_to_gdp: number;
}

export interface MacroFetchProgress {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current: number;
  total: number;
  detail: string;
}
