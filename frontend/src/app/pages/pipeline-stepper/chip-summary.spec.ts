import { sectionSummary, stageSummary } from './chip-summary';
import type {
  LoadStepResult,
  OptimizeStepResult,
  PipelineStepId,
  ScreenStepResult,
} from '../../models/pipeline-builder.model';

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
