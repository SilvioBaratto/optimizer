export type JobStatus = 'pending' | 'running' | 'completed' | 'failed';
export type JobDomain = 'yfinance' | 'macro' | 'news' | 'calibrate';
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
  domain: JobDomain;
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

export const DOMAIN_META: DomainMeta[] = [
  { domain: 'yfinance',  label: 'Market Data',  icon: 'trending-up', staleThresholdHours: 26,  criticalThresholdHours: 72 },
  { domain: 'macro',     label: 'Macro Scraper', icon: 'globe',       staleThresholdHours: 26,  criticalThresholdHours: 72 },
  { domain: 'news',      label: 'News Pipeline', icon: 'newspaper',   staleThresholdHours: 26,  criticalThresholdHours: 72 },
  { domain: 'calibrate', label: 'Calibration',   icon: 'settings-2',  staleThresholdHours: 48,  criticalThresholdHours: 120 },
];
