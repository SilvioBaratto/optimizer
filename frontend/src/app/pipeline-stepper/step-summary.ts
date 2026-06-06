// Pure per-step formatter producing the one-line summary shown in a collapsed
// section header. Narrows StepPollResponse.result to the typed *StepResult.
// Returns '' for a missing/absent result (section shows no summary yet).

import type {
  BuildHistoryStepResult,
  CleanReturnsStepResult,
  CostStepResult,
  CoverageGateStepResult,
  LoadStepResult,
  OptimizeStepResult,
  PersistStepResult,
  PipelineStepId,
  RebalanceDecisionStepResult,
  RegimeStepResult,
  ReportStepResult,
  ScreenStepResult,
  ValidateIsStepResult,
  ValidateOosStepResult,
} from '../models/pipeline-builder.model';

type RawResult = Record<string, unknown>;

const FORMATTERS: Record<PipelineStepId, (r: RawResult) => string> = {
  load: (r) => {
    const x = r as unknown as LoadStepResult;
    return `${x.n_tickers} tickers · ${x.n_trading_days} days · ${x.base_currency}`;
  },
  screen: (r) => {
    const x = r as unknown as ScreenStepResult;
    return `${x.n_investable} investable · preset ${x.preset}${x.band_warning ? ' ⚠ band' : ''}`;
  },
  clean_returns: (r) => {
    const x = r as unknown as CleanReturnsStepResult;
    return `${x.n_tickers} tickers · ${x.n_days} days`;
  },
  build_history: (r) => {
    const x = r as unknown as BuildHistoryStepResult;
    return `${x.succeeded_dates}/${x.total_dates} dates · ${x.n_factors} factors`;
  },
  validate_is: (r) => {
    const x = r as unknown as ValidateIsStepResult;
    return `${x.n_significant} significant factors`;
  },
  validate_oos: (r) => {
    const x = r as unknown as ValidateOosStepResult;
    return `${x.n_folds} folds · ${x.oos_results?.length ?? 0} factors`;
  },
  coverage_gate: (r) => {
    const x = r as unknown as CoverageGateStepResult;
    return `${x.n_passing} passing (min ${x.min_factors})`;
  },
  regime: (r) => {
    const x = r as unknown as RegimeStepResult;
    return `${x.regime}${x.persisted ? ' · persisted' : ''}`;
  },
  optimize: (r) => {
    const x = r as unknown as OptimizeStepResult;
    const sharpe = x.net_sharpe != null ? x.net_sharpe.toFixed(2) : '—';
    return `${x.n_selected} holdings · net Sharpe ${sharpe}${x.hockey_stick_warning ? ' ⚠' : ''}`;
  },
  rebalance_decision: (r) => {
    const x = r as unknown as RebalanceDecisionStepResult;
    return `${x.decision ? 'rebalance' : 'hold'} · ${x.n_weights} weights`;
  },
  cost: (r) => {
    const x = r as unknown as CostStepResult;
    return `${x.cost_bps_actual} bps actual${x.exceeds_assumed ? ' ⚠ exceeds' : ''}`;
  },
  report: (r) => {
    const x = r as unknown as ReportStepResult;
    return `${x.pass_count}/${x.checklist_total} checks · ${x.checklist_passed ? 'PASS' : 'FAIL'}`;
  },
  persist: (r) => {
    const x = r as unknown as PersistStepResult;
    return x.persisted ? 'persisted' : `not persisted — ${x.reason}`;
  },
};

export function stepSummary(
  stepId: PipelineStepId,
  result: Record<string, unknown> | null,
): string {
  if (!result) return '';
  try {
    return FORMATTERS[stepId](result);
  } catch {
    return '';
  }
}
