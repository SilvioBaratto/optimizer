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

// Intentional: POST /api/v1/backtest returns CamelCaseModel (jobId, runId) unlike
// the snake_case responses used by /optimize and /factors/compute.
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
    fold_results?: ValidateFoldResult[];
    aggregate?: Record<string, number>;
    aggregate_score?: number;
  } | null;
  error: string | null;
}

export interface EquityCurvePoint {
  date: string;
  value: number;
}

/**
 * Response payload of `GET /api/v1/backtest/runs/{run_id}` (issue #464/#465).
 *
 * Mirrors the backend `BacktestRunResponse` (CamelCaseModel): every field
 * is camelCase in the JSON. Numeric dict values may be `null` for NaN/Inf
 * sanitised at the service layer; consumers must handle that.
 */
export interface BacktestRunResponse {
  id: string;
  portfolioId: string | null;
  jobId: string | null;
  status: 'pending' | 'running' | 'completed' | 'failed';
  config: Record<string, unknown>;
  equityCurve: Record<string, number | null>;
  drawdowns: Record<string, number | null>;
  monthlyReturns: Record<string, number | null>;
  yearlyReturns: Record<string, number | null>;
  rollingMetrics: Record<string, Record<string, number | null>>;
  turnoverHistory: Record<string, number | null>;
  cvFoldMetrics: Array<Record<string, number | null>> | null;
  summaryStats: Record<string, number | null>;
  errorMessage: string | null;
  durationSeconds: number | null;
  createdAt: string;
  updatedAt: string;
}
