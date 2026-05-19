// Per-step parameter registry collected once in the upfront config form and
// replayed by the host on every dispatch (initial / Continue / Retry / re-run).
// Defaults mirror the Pydantic *StepRequest defaults in
// api/app/schemas/pipeline_builder/pipeline_builder.py.

import type {
  PipelineStepId,
  RunLevelConfig,
} from '../../models/pipeline-builder.model';
import type { ScreenPreset } from './step-screen-panel';

export interface StepParamsConfig {
  load: { include_delisted: boolean; macro_country: string };
  screen: { preset: ScreenPreset };
  build_history: { market_proxy_ticker: string };
  coverage_gate: { min_factors: number };
  regime: { enable_tilts: boolean; persist_regime: boolean };
}

export interface PipelineConfigSubmit {
  config: RunLevelConfig;
  steps: StepParamsConfig;
}

export const STEP_PARAM_DEFAULTS: StepParamsConfig = {
  load: { include_delisted: true, macro_country: 'USA' },
  screen: { preset: 'developed_markets' },
  build_history: { market_proxy_ticker: 'URTH' },
  coverage_gate: { min_factors: 2 },
  regime: { enable_tilts: true, persist_regime: false },
};

const PARAM_STEP_IDS = [
  'load',
  'screen',
  'build_history',
  'coverage_gate',
  'regime',
] as const;
type ParamStepId = (typeof PARAM_STEP_IDS)[number];

function isParamStep(id: PipelineStepId): id is ParamStepId {
  return (PARAM_STEP_IDS as readonly string[]).includes(id);
}

/** Body for a step's POST: the upfront params for the 5 param-bearing steps,
 *  `{}` for the other 8. */
export function buildStepParams(
  cfg: StepParamsConfig | null,
  id: PipelineStepId,
): Record<string, unknown> {
  if (!cfg || !isParamStep(id)) return {};
  return { ...cfg[id] };
}
