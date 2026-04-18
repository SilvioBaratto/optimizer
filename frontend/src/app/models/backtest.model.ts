export interface BacktestConfig {
  startDate: string;
  endDate: string;
  initialCapital: number;
  rebalanceFrequency: 'daily' | 'weekly' | 'monthly' | 'quarterly';
  transactionCostBps: number;
  benchmark: string;
}

export interface BacktestEquityPoint {
  date: string;
  portfolio: number;
  benchmark: number;
}

export interface BacktestResult {
  equity: BacktestEquityPoint[];
  metrics: BacktestMetrics;
  /**
   * Benchmark KPIs aligned with `metrics`. Optional because the backend
   * doesn't compute them today (issue #434); when absent, the UI falls back
   * to a dash for every benchmark cell rather than synthetic placeholders.
   */
  benchmarkMetrics?: BacktestMetrics;
  drawdowns: Drawdown[];
  monthlyReturns: MonthlyReturnCell[];
  rollingMetrics: RollingMetric[];
  returnDistribution: ReturnDistributionBin[];
  factorLoadings: FactorLoading[];
}

export interface BacktestMetrics {
  totalReturn: number;
  annualizedReturn: number;
  annualizedVol: number;
  sharpe: number;
  sortino: number;
  maxDrawdown: number;
  calmar: number;
  cvar95: number;
  trackingError: number;
  informationRatio: number;
  winRate: number;
  profitFactor: number;
}

export interface Drawdown {
  start: string;
  trough: string;
  end: string | null;
  depth: number;
  duration: number;
  recovery: number | null;
}

export interface MonthlyReturnCell {
  year: number;
  month: number;
  value: number;
}

export interface RollingMetric {
  date: string;
  sharpe: number;
  volatility: number;
  beta: number;
}

export interface ReturnDistributionBin {
  binStart: number;
  binEnd: number;
  count: number;
  frequency: number;
}

export interface FactorLoading {
  factor: string;
  loading: number;
  tStat: number;
  pValue: number;
}

// ── Backend API DTOs (mirror api/app/schemas/{backtest,validation}.py) ──────

export interface BacktestApiRequest {
  tickers: string[];
  start_date: string;
  end_date: string;
  pipeline_config?: Record<string, unknown>;
}

export interface BacktestAsyncResponse {
  jobId: string;
  runId: string;
  status: string;
  message: string;
}

export interface BacktestProgressResponse {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current: number;
  total: number;
  errors: string[];
  result: Record<string, unknown> | null;
  error: string | null;
}

export type CvType = 'walk_forward' | 'cpcv' | 'multiple_randomized';

export interface ValidateApiRequest {
  tickers: string[];
  start_date: string;
  end_date: string;
  cv_type?: CvType;
  cv_config?: Record<string, unknown>;
  optimizer_type?: string;
  optimizer_config?: Record<string, unknown>;
}

export interface ValidateAsyncResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface ValidateFoldResult {
  weights: Record<string, number>;
  measures: Record<string, number>;
}

export interface ValidateProgressResponse {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current: number;
  total: number;
  current_fold: number;
  total_folds: number;
  errors: string[];
  result: {
    folds?: ValidateFoldResult[];
    aggregate?: Record<string, number>;
  } | null;
  error: string | null;
}

export interface EquityCurvePoint {
  date: string;
  value: number;
}
