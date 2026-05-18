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
  { id: 'load', label: '1', index: 0 },
  { id: 'screen', label: '1b', index: 1 },
  { id: 'clean_returns', label: '2', index: 2 },
  { id: 'build_history', label: '2.5', index: 3 },
  { id: 'validate_is', label: '3', index: 4 },
  { id: 'validate_oos', label: '4', index: 5 },
  { id: 'coverage_gate', label: '5', index: 6 },
  { id: 'regime', label: '5b', index: 7 },
  { id: 'optimize', label: '6', index: 8 },
  { id: 'rebalance_decision', label: '7', index: 9 },
  { id: 'cost', label: '7b', index: 10 },
  { id: 'report', label: '7c', index: 11 },
  { id: 'persist', label: 'Final', index: 12 },
] as const;
