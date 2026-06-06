// DTOs for /api/v1/pipeline-builder/*. Mirrors the Pydantic schemas in
// api/app/schemas/pipeline_builder/pipeline_builder.py.
//
// `StepRunResponse` keeps `jobId` even though the wire format is `job_id`
// (AsyncJobCreateResponse is plain BaseModel, no alias generator). The
// service maps the wire shape to this camelCase contract at the HTTP
// boundary so downstream consumers never see snake_case.

export type BaseCurrency = 'EUR' | 'GBP' | 'USD';

export interface RunLevelConfig {
  rebalance_freq: number;
  n_selected: number;
  cost_bps: number;
  tax_rate: number;
  base_currency: BaseCurrency;
  robust: boolean;
  persist: boolean;
  start_date: string | null;
  end_date: string | null;
  seed: number;
}

export enum StepStatus {
  Pending = 'pending',
  Locked = 'locked',
  Ready = 'ready',
  Running = 'running',
  Completed = 'completed',
  Error = 'error',
}

export interface PipelineStep {
  readonly id: PipelineStepId;
  readonly label: string;
  readonly index: number;
  readonly name: string;
  readonly description: string;
}

export type PipelineStepId =
  | 'load'
  | 'screen'
  | 'clean_returns'
  | 'build_history'
  | 'validate_is'
  | 'validate_oos'
  | 'coverage_gate'
  | 'regime'
  | 'optimize'
  | 'rebalance_decision'
  | 'cost'
  | 'report'
  | 'persist';

export type PollStepStatus =
  | 'not_started'
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed';

export interface CreateSessionResponse {
  sessionId: string;
}

export interface StepRunResponse {
  jobId: string;
  status: string;
  message: string;
}

export interface StepPollResponse {
  status: PollStepStatus;
  progress: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string | null;
  gateReason: string | null;
}

export const PIPELINE_STEPS: readonly PipelineStep[] = [
  {
    id: 'load',
    label: '1',
    index: 0,
    name: 'Load',
    description: 'Fetch price & macro history from the database',
  },
  {
    id: 'screen',
    label: '1b',
    index: 1,
    name: 'Screen',
    description: 'Apply the investability universe screen',
  },
  {
    id: 'clean_returns',
    label: '2',
    index: 2,
    name: 'Clean returns',
    description: 'Validate, winsorize and impute the return series',
  },
  {
    id: 'build_history',
    label: '2.5',
    index: 3,
    name: 'Build history',
    description: 'Assemble factor scores over the rebalance grid',
  },
  {
    id: 'validate_is',
    label: '3',
    index: 4,
    name: 'Validate in-sample',
    description: 'Factor IC, t-stats and VIF on the full sample',
  },
  {
    id: 'validate_oos',
    label: '4',
    index: 5,
    name: 'Validate out-of-sample',
    description: 'Rolling out-of-sample information coefficient by fold',
  },
  {
    id: 'coverage_gate',
    label: '5',
    index: 6,
    name: 'Coverage gate',
    description: 'Require enough factors passing IS ∩ OOS to continue',
  },
  {
    id: 'regime',
    label: '5b',
    index: 7,
    name: 'Regime',
    description: 'Classify the macro regime and apply group tilts',
  },
  {
    id: 'optimize',
    label: '6',
    index: 8,
    name: 'Optimize',
    description: 'Solve portfolio weights for the selected universe',
  },
  {
    id: 'rebalance_decision',
    label: '7',
    index: 9,
    name: 'Rebalance decision',
    description: 'Decide whether to rebalance at the current date',
  },
  {
    id: 'cost',
    label: '7b',
    index: 10,
    name: 'Cost',
    description: 'Estimate per-country transaction costs',
  },
  {
    id: 'report',
    label: '7c',
    index: 11,
    name: 'Report',
    description: 'Run the acceptance checklist and write artifacts',
  },
  {
    id: 'persist',
    label: 'Final',
    index: 12,
    name: 'Persist',
    description: 'Save final weights and report if checks pass',
  },
] as const;

// Step-result payload shapes mirroring the dict returns in
// api/app/services/pipeline_builder/steps.py (no wire-side Pydantic
// schema enforces them). Panels narrow StepPollResponse.result at the
// consumption point — `result() as LoadStepResult` etc.

export interface LoadStepResult {
  n_tickers: number;
  n_trading_days: number;
  assembly_hash: string;
  base_currency: BaseCurrency;
  price_start: string;
  price_end: string;
}

export interface ScreenStepResult {
  n_investable: number;
  preset: string;
  band_warning: boolean;
  band_low: number;
  band_high: number;
}

export interface CleanReturnsStepResult {
  n_days: number;
  n_tickers: number;
  return_start: string;
  return_end: string;
  preprocessing_steps: string[];
}

export interface BuildHistoryStepResult {
  succeeded_dates: number;
  total_dates: number;
  failed_dates: number;
  n_factors: number;
  rebalance_freq: number;
  market_proxy_loaded: boolean;
}

export interface IcResultItem {
  factor_name: string;
  mean_ic: number | null;
  ic_std: number | null;
  t_stat: number | null;
  p_value: number | null;
  significant: boolean;
}

export interface ValidateIsStepResult {
  ic_results: IcResultItem[];
  n_significant: number;
  significant_factors: string[];
  high_vif_factors: string[];
  config: Record<string, number>;
}

export interface OosResultItem {
  factor_name: string;
  oos_mean_ic: number;
  oos_icir: number | null;
}

export interface ValidateOosStepResult {
  n_folds: number;
  oos_results: OosResultItem[];
  config: Record<string, number>;
}

export interface CoverageGateStepResult {
  passing_factors: string[];
  is_only_factors: string[];
  oos_only_factors: string[];
  n_passing: number;
  min_factors: number;
}

export interface RegimeStepResult {
  regime: string;
  tilts: Record<string, number>;
  persisted: boolean;
  note: string;
}

export interface WeightItem {
  ticker: string;
  weight: number;
}

export interface OptimizeStepResult {
  weights: WeightItem[];
  n_selected: number;
  is_sharpe: number | null;
  net_sharpe: number | null;
  hockey_stick_warning: boolean;
  sector_breakdown: Record<string, number>;
  country_breakdown: Record<string, number>;
}

export interface RebalanceDecisionStepResult {
  decision: boolean;
  reason: string;
  n_weights: number;
  weight_count_warning: boolean;
  missing_sectors: string[];
  current_date: string;
  last_review_date: string;
}

export interface CostCountryItem {
  country: string;
  stamp: number;
  spread: number;
  fx: number;
  total: number;
}

export interface CostStepResult {
  cost_bps_actual: number;
  cost_bps_assumed: number;
  exceeds_assumed: boolean;
  per_country: CostCountryItem[];
}

export interface ChecklistRule {
  rule: string;
  pass: boolean;
  measured: string | number | null;
  target: string;
}

export interface ReportArtifactPaths {
  report_md: string;
  weights_csv: string;
  metrics_json: string;
  checklist_json: string;
  // Written only on a checklist near-miss; the path is always returned but
  // the file exists only when the run failed the 17/17 gate.
  weights_diagnostic: string;
}

export interface ReportStepResult {
  pass_count: number;
  checklist_total: number;
  checklist_passed: boolean;
  checklist_rules: ChecklistRule[];
  metrics: Record<string, Record<string, number>>;
  artifact_paths: ReportArtifactPaths;
  chart_paths: string[];
  output_dir: string;
}

export interface PersistStepResult {
  persisted: boolean;
  reason: string;
  pass_count: number;
  checklist_passed: boolean;
}
