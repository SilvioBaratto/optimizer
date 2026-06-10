import { sectionSummary, stageSummary } from './chip-summary';
import type {
  LoadStepResult,
  OptimizeStepResult,
  PipelineStepId,
  ScreenStepResult,
} from '../core/models/pipeline-builder.model';

function loadResult(
  overrides: Partial<LoadStepResult> = {},
): Record<string, unknown> {
  return {
    n_tickers: 42,
    n_trading_days: 1260,
    assembly_hash: 'h',
    base_currency: 'EUR',
    price_start: '2020-01-01',
    price_end: '2025-01-01',
    ...overrides,
  } as unknown as Record<string, unknown>;
}

function screenResult(
  overrides: Partial<ScreenStepResult> = {},
): Record<string, unknown> {
  return {
    n_investable: 30,
    preset: 'developed_markets',
    band_warning: false,
    band_low: 25,
    band_high: 50,
    ...overrides,
  } as unknown as Record<string, unknown>;
}

function optimizeResult(
  overrides: Partial<OptimizeStepResult> = {},
): Record<string, unknown> {
  return {
    weights: [],
    n_selected: 25,
    is_sharpe: 1.2,
    net_sharpe: 1.1,
    hockey_stick_warning: false,
    sector_breakdown: {},
    country_breakdown: {},
    ...overrides,
  } as unknown as Record<string, unknown>;
}

describe('sectionSummary', () => {
  it('when results map is empty, returns ""', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>();
    expect(sectionSummary('universe', results)).toBe('');
  });

  it('when all mapped steps have null results, returns ""', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['load', null],
      ['screen', null],
    ]);
    expect(sectionSummary('universe', results)).toBe('');
  });

  it('when one step has a result, returns that step\'s summary verbatim', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['load', loadResult()],
    ]);
    expect(sectionSummary('universe', results)).toBe(
      '42 tickers · 1260 days · EUR',
    );
  });

  it('when multiple steps have results, joins them with " · " in step order', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['load', loadResult()],
      ['screen', screenResult()],
    ]);
    expect(sectionSummary('universe', results)).toBe(
      '42 tickers · 1260 days · EUR · 30 investable · preset developed_markets',
    );
  });

  it('when only the second step has a result, returns only that step\'s summary', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['screen', screenResult()],
    ]);
    expect(sectionSummary('universe', results)).toBe(
      '30 investable · preset developed_markets',
    );
  });
});

describe('stageSummary', () => {
  it('when results map is empty for the stage, returns ""', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>();
    expect(stageSummary('optimize', results)).toBe('');
  });

  it('when an unrelated step has a result, returns "" for that stage', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['load', loadResult()],
    ]);
    expect(stageSummary('optimize', results)).toBe('');
  });

  it('when the optimize stage step has a result, returns its summary', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['optimize', optimizeResult()],
    ]);
    expect(stageSummary('optimize', results)).toBe(
      '25 holdings · net Sharpe 1.10',
    );
  });

  it('when the universe stage has both load and screen results, joins them with " · "', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['load', loadResult()],
      ['screen', screenResult()],
    ]);
    expect(stageSummary('universe', results)).toBe(
      '42 tickers · 1260 days · EUR · 30 investable · preset developed_markets',
    );
  });

  it('when a step in the mapping is missing from the results map, it is silently skipped', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['load', loadResult()],
    ]);
    expect(stageSummary('universe', results)).toBe(
      '42 tickers · 1260 days · EUR',
    );
  });
});

describe('sectionSummary — non-universe sections', () => {
  // objective: ['validate_is', 'validate_oos', 'coverage_gate']
  it('when objective steps have results, joins them with " · " in section order', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['validate_is', { n_significant: 5 }],
      ['validate_oos', { n_folds: 4, oos_results: [{ factor_name: 'MOM' }, { factor_name: 'VAL' }] }],
      ['coverage_gate', { n_passing: 6, min_factors: 3 }],
    ]);
    expect(sectionSummary('objective', results)).toBe(
      '5 significant factors · 4 folds · 2 factors · 6 passing (min 3)',
    );
  });

  it('when only one objective step has a result, returns only that step\'s summary', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['validate_is', { n_significant: 3 }],
    ]);
    expect(sectionSummary('objective', results)).toBe('3 significant factors');
  });

  // constraints: ['regime', 'rebalance_decision']
  it('when constraints steps have results, joins them with " · " in section order', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['regime', { regime: 'contraction', persisted: false }],
      ['rebalance_decision', { decision: true, n_weights: 25 }],
    ]);
    expect(sectionSummary('constraints', results)).toBe(
      'contraction · rebalance · 25 weights',
    );
  });

  it('when only the regime step has a result in constraints, returns only regime summary', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['regime', { regime: 'expansion', persisted: true }],
    ]);
    expect(sectionSummary('constraints', results)).toBe('expansion · persisted');
  });

  // moments: ['clean_returns', 'build_history']
  it('when moments steps have results, joins them with " · " in section order', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['clean_returns', { n_tickers: 50, n_days: 500 }],
      ['build_history', { succeeded_dates: 48, total_dates: 52, n_factors: 7 }],
    ]);
    expect(sectionSummary('moments', results)).toBe(
      '50 tickers · 500 days · 48/52 dates · 7 factors',
    );
  });

  it('when only build_history has a result in moments, returns only that summary', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['build_history', { succeeded_dates: 10, total_dates: 12, n_factors: 4 }],
    ]);
    expect(sectionSummary('moments', results)).toBe('10/12 dates · 4 factors');
  });

  // horizon: ['optimize', 'cost', 'report', 'persist']
  it('when all horizon steps have results, joins them with " · " in section order', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['optimize', optimizeResult()],
      ['cost', { cost_bps_actual: 8, exceeds_assumed: false }],
      ['report', { pass_count: 17, checklist_total: 17, checklist_passed: true }],
      ['persist', { persisted: true, reason: '' }],
    ]);
    expect(sectionSummary('horizon', results)).toBe(
      '25 holdings · net Sharpe 1.10 · 8 bps actual · 17/17 checks · PASS · persisted',
    );
  });

  it('when only optimize has a result in horizon, returns only optimize summary', () => {
    const results = new Map<PipelineStepId, Record<string, unknown> | null>([
      ['optimize', optimizeResult({ n_selected: 10, net_sharpe: 0.75 })],
    ]);
    expect(sectionSummary('horizon', results)).toBe('10 holdings · net Sharpe 0.75');
  });
});
