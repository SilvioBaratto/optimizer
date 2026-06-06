export type RiskMeasureType =
  | 'variance'
  | 'semi_variance'
  | 'standard_deviation'
  | 'semi_deviation'
  | 'mean_absolute_deviation'
  | 'first_lower_partial_moment'
  | 'cvar'
  | 'evar'
  | 'worst_realization'
  | 'cdar'
  | 'max_drawdown'
  | 'average_drawdown'
  | 'edar'
  | 'ulcer_index'
  | 'gini_mean_difference';

export type ExtraRiskMeasureType = 'value_at_risk' | 'drawdown_at_risk';

export type MuEstimatorType =
  | 'empirical'
  | 'shrunk'
  | 'ew'
  | 'equilibrium';

export type CovEstimatorType =
  | 'empirical'
  | 'ledoit_wolf'
  | 'oas'
  | 'shrunk'
  | 'ew'
  | 'gerber'
  | 'graphical_lasso_cv'
  | 'denoise'
  | 'detone'
  | 'implied';

// Mirrors `OptimizeRequest.optimizer_type` Literal in api/app/models/optimization.py.
export type OptimizerType = 'mean_risk' | 'regime_blended';

// Mirrors ObjectiveFunctionType in optimizer/optimization/_config.py.
export type ObjectiveFunctionType =
  | 'minimize_risk'
  | 'maximize_return'
  | 'maximize_utility'
  | 'maximize_ratio';

export type PipelineNodeStatus = 'pending' | 'running' | 'completed' | 'error';

export interface PipelineNode {
  id: string;
  label: string;
  status: PipelineNodeStatus;
  duration_ms?: number;
  detail?: string;
}

export interface MomentEstimationOutput {
  muEstimator: MuEstimatorType;
  covEstimator: CovEstimatorType;
  expectedReturns: { ticker: string; value: number }[];
  topEigenvalues: number[];
}

export type ViewType = 'absolute' | 'relative';

export interface ViewFormation {
  id: string;
  type: ViewType;
  assets: string[];
  value: number;
  confidence: number;
  source: string;
}

export interface EfficientFrontierPoint {
  risk: number;
  return: number;
  sharpe: number;
  label?: string;
}

export interface OptimizationConfig {
  strategy: string;
  riskMeasure: RiskMeasureType;
  muEstimator: MuEstimatorType;
  covEstimator: CovEstimatorType;
  riskAversion: number;
  cvarBeta: number;
  robustKappa: number;
  preSelection: boolean;
}

export interface StrategyComparison {
  strategy: string;
  annualizedReturn: number;
  annualizedVol: number;
  sharpe: number;
  maxDrawdown: number;
  cvar95: number;
}

// ── Backend request / response DTOs ──────────────────────────────────────────

export interface OptimizeRequest {
  tickers: string[];
  start_date: string;
  end_date: string;
  optimizer_type: OptimizerType | string;
  config?: Record<string, unknown>;
  constraints?: Record<string, unknown> | null;
}

export interface OptimizationRunResponse {
  id: string;
  portfolioId: string | null;
  jobId: string | null;
  status: 'pending' | 'running' | 'completed' | 'failed';
  optimizerType: string;
  universeTickers: string[];
  config: Record<string, unknown>;
  weights: Record<string, number>;
  metrics: Record<string, unknown>;
  riskContributions: Record<string, number>;
  efficientFrontier: EfficientFrontierPoint[] | null;
  errorMessage: string | null;
  solverLog: string | null;
  durationSeconds: number | null;
  createdAt: string;
  updatedAt: string;
}

export interface OptimizationRunListResponse {
  items: OptimizationRunResponse[];
  total: number;
}

export interface OptimizeAsyncResponse {
  job_id: string;
  run_id: string;
}

export interface TuneRequest {
  tickers: string[];
  start_date: string;
  end_date: string;
  optimizer_type: OptimizerType | string;
  search_space: Record<string, unknown>;
  search_type?: 'grid' | 'randomized';
  scorer?: string;
  cv_config?: Record<string, unknown>;
}

export interface TuneJobCreateResponse {
  job_id: string;
  status: string;
  message: string;
}

// ── LLM moments request / response shapes ────────────────────────────────────

export interface CalibrateDeltaRequest {
  macro_text: string;
}

export interface CalibrateDeltaResponse {
  delta: number;
  rationale: string;
}

export interface AdaptFactorWeightsRequest {
  macro_indicators: string;
  factor_groups: string[];
}

export interface AdaptFactorWeightsResponse {
  phase: string;
  weights: Record<string, number>;
  rationale: string;
}

export interface SelectCovRegimeRequest {
  news_headlines: string[];
  avg_sentiment_score: number;
  realized_vol_30d: number;
}

export interface SelectCovRegimeResponse {
  estimator_type: CovEstimatorType;
  rationale: string;
}

// ── Views request / response shapes ──────────────────────────────────────────

export interface GenerateViewsRequest {
  tickers: string[];
}

// Wire shape mirrors api/app/schemas/views/views.py (CamelCaseModel: `P`→`p`,
// `Q`→`q`). `idzorekAlphas` is a ticker→alpha dict (NOT an array); the backend
// also returns the per-view `viewConfidences` list separately.
export interface AssetViewResponse {
  asset: string;
  direction: number;
  magnitudeBps: number;
  confidence: number;
  reasoning: string;
}

export interface GenerateViewsResponse {
  nViews: number;
  nAssets: number;
  viewStrings: string[];
  p: number[][];
  q: number[];
  viewConfidences: number[];
  idzorekAlphas: Record<string, number>;
  views: AssetViewResponse[];
  rationale: string;
  tickersWithData: string[];
  tickersMissingData: string[];
}

export interface EntropyPoolingRequest {
  tickers: string[];
  start_date: string;
  end_date: string;
  mean_views?: string[];
  variance_views?: string[];
  correlation_views?: string[];
  skew_views?: string[];
  cvar_views?: string[];
}

export interface EntropyPoolingResponse {
  tickers: string[];
  mu: number[];
  covariance: number[][];
}

export interface OpinionPoolRequest {
  tickers: string[];
  personas: string[];
  ic_histories?: Record<string, number[]>;
}

// Per-expert summary inside OpinionPoolResponse.experts (wire $def).
export interface ExpertViewSummary {
  persona: string;
  name: string;
  nViews: number;
  viewStrings: string[];
  idzorekAlphas: Record<string, number>;
  icWeight: number;
}

// Wire shape mirrors OpinionPoolResponse in api/app/schemas/views/views.py.
// The endpoint returns IC-weighted expert summaries, NOT a P/Q pick matrix.
export interface OpinionPoolResponse {
  nExperts: number;
  tickers: string[];
  tickersWithData: string[];
  tickersMissingData: string[];
  experts: ExpertViewSummary[];
  icWeights: number[];
  poolingType: string;
  totalViews: number;
}
