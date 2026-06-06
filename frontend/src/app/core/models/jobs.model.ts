export type JobStatus = 'pending' | 'running' | 'completed' | 'failed';

/**
 * Backend `BackgroundJobService` job types. Must match the exact strings the
 * API returns in `domain` (set by `_job_to_summary()` in
 * `api/app/api/v1/jobs.py`). Frontend uses underscore_case — do NOT short-name.
 */
export type JobDomain =
  // Data pipeline
  | 'yfinance_fetch'
  | 'macro_fetch'
  | 'fred_fetch'
  | 'macro_news_fetch'
  | 'news_summarize'
  | 'macro_calibrate'
  // Portfolio ops
  | 'portfolio_sync'
  | 'universe_build'
  | 'reference_index_seed'
  // Execution
  | 'optimize'
  | 'backtest'
  | 'validate'
  | 'tune'
  | 'factors_compute'
  // Reports
  | 'report_generate';

export type FreshnessLevel = 'fresh' | 'stale' | 'critical' | 'unknown';

export interface JobSummary {
  id: string;
  domain: string;
  status: JobStatus;
  current: number;
  total: number;
  error: string | null;
  errors_count: number;
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number | null;
}

export interface JobListResponse {
  jobs: JobSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface DomainMeta {
  domain: JobDomain | string;
  label: string;
  icon: string;
  staleThresholdHours: number;
  criticalThresholdHours: number;
}

export interface DomainStatus {
  meta: DomainMeta;
  lastSuccess: JobSummary | null;
  running: JobSummary | null;
  recentFailures: JobSummary[];
  freshness: FreshnessLevel;
  lastSuccessAgeHours: number | null;
}

// Threshold presets by scheduling cadence.
const DAILY = { staleThresholdHours: 26, criticalThresholdHours: 72 };
const MONTHLY = { staleThresholdHours: 744, criticalThresholdHours: 1440 };
const ON_DEMAND = { staleThresholdHours: 168, criticalThresholdHours: 720 };

export const DOMAIN_META: DomainMeta[] = [
  // Data pipeline (daily, except fred_fetch which is monthly)
  { domain: 'yfinance_fetch',       label: 'Market Data',       icon: 'trending-up', ...DAILY },
  { domain: 'macro_fetch',          label: 'Macro Scraper',     icon: 'globe',       ...DAILY },
  { domain: 'fred_fetch',           label: 'FRED Data',         icon: 'bar-chart-3', ...MONTHLY },
  { domain: 'macro_news_fetch',     label: 'News Fetch',        icon: 'newspaper',   ...DAILY },
  { domain: 'news_summarize',       label: 'News Summarize',    icon: 'file-text',   ...DAILY },
  { domain: 'macro_calibrate',      label: 'Calibration',       icon: 'settings-2',  ...DAILY },
  // Portfolio ops (on-demand)
  { domain: 'portfolio_sync',       label: 'Portfolio Sync',    icon: 'refresh-cw',  ...ON_DEMAND },
  { domain: 'universe_build',       label: 'Universe Build',    icon: 'layers',      ...ON_DEMAND },
  { domain: 'reference_index_seed', label: 'Reference Seed',    icon: 'database',    ...ON_DEMAND },
  // Execution (on-demand)
  { domain: 'optimize',             label: 'Optimizer',         icon: 'target',         ...ON_DEMAND },
  { domain: 'backtest',             label: 'Backtest',          icon: 'play',           ...ON_DEMAND },
  { domain: 'validate',             label: 'Validation',        icon: 'check-circle',   ...ON_DEMAND },
  { domain: 'tune',                 label: 'Tuning',            icon: 'sliders',        ...ON_DEMAND },
  { domain: 'factors_compute',      label: 'Factor Compute',    icon: 'function-square', ...ON_DEMAND },
  // Reports (on-demand)
  { domain: 'report_generate',      label: 'Report Generate',   icon: 'printer',     ...ON_DEMAND },
];
