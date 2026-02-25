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
  | 'equilibrium'
  | 'hmm_blended';

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
  | 'implied'
  | 'hmm_blended';

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
