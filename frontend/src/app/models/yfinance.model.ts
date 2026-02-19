export interface TickerProfile {
  ticker: string;
  name: string;
  sector: string;
  industry: string;
  market_cap: number;
  pe_ratio: number | null;
  dividend_yield: number | null;
  beta: number | null;
  fifty_two_week_high: number;
  fifty_two_week_low: number;
  avg_volume: number;
  description: string;
}

export interface PriceHistory {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface AnalystRecommendation {
  period: string;
  strong_buy: number;
  buy: number;
  hold: number;
  sell: number;
  strong_sell: number;
}

export interface InsiderTransaction {
  date: string;
  insider: string;
  position: string;
  transaction_type: string;
  shares: number;
  value: number;
}

export interface TickerNews {
  title: string;
  publisher: string;
  link: string;
  published: string;
  summary: string;
}

export interface FetchProgress {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current: number;
  total: number;
  detail: string;
}
