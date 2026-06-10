import { stepSummary } from './step-summary';
import type { PipelineStepId } from '../core/models/pipeline-builder.model';

describe('stepSummary', () => {
  it('when the result is null, an empty summary is returned', () => {
    expect(stepSummary('load', null)).toBe('');
  });

  it('when a load result is given, the ticker/day/currency line is built', () => {
    expect(stepSummary('load', { n_tickers: 8, n_trading_days: 252, base_currency: 'EUR' }))
      .toBe('8 tickers · 252 days · EUR');
  });

  it('when an optimize result has a net sharpe, it is formatted to two decimals', () => {
    expect(stepSummary('optimize', { n_selected: 20, net_sharpe: 1.234, hockey_stick_warning: false }))
      .toBe('20 holdings · net Sharpe 1.23');
  });

  it('when an optimize result has no net sharpe, a dash is shown', () => {
    expect(stepSummary('optimize', { n_selected: 20, net_sharpe: null }))
      .toBe('20 holdings · net Sharpe —');
  });

  it('when a persist result is not persisted, the reason is appended', () => {
    expect(stepSummary('persist', { persisted: false, reason: 'dry run' }))
      .toBe('not persisted — dry run');
  });

  it('when the step id has no formatter, the call throws and degrades to empty', () => {
    expect(stepSummary('bogus_step' as PipelineStepId, { any: 1 })).toBe('');
  });

  // ── screen ──────────────────────────────────────────────────────────────────
  describe('screen formatter', () => {
    const cases = [
      {
        desc: 'when band_warning is false, no warning suffix is appended',
        result: { n_investable: 30, preset: 'developed_markets', band_warning: false },
        expected: '30 investable · preset developed_markets',
      },
      {
        desc: 'when band_warning is true, the warning-band suffix is appended',
        result: { n_investable: 30, preset: 'developed_markets', band_warning: true },
        expected: '30 investable · preset developed_markets ⚠ band',
      },
    ];
    cases.forEach(({ desc, result, expected }) => {
      it(desc, () => {
        expect(stepSummary('screen', result)).toBe(expected);
      });
    });
  });

  // ── clean_returns ────────────────────────────────────────────────────────────
  it('when a clean_returns result is given, the ticker/day line is built', () => {
    expect(stepSummary('clean_returns', { n_tickers: 50, n_days: 500 }))
      .toBe('50 tickers · 500 days');
  });

  // ── build_history ────────────────────────────────────────────────────────────
  it('when a build_history result is given, the dates/factors line is built', () => {
    expect(stepSummary('build_history', { succeeded_dates: 48, total_dates: 52, n_factors: 7 }))
      .toBe('48/52 dates · 7 factors');
  });

  // ── validate_is ──────────────────────────────────────────────────────────────
  it('when a validate_is result is given, the significant-factors count is shown', () => {
    expect(stepSummary('validate_is', { n_significant: 5 }))
      .toBe('5 significant factors');
  });

  // ── validate_oos ─────────────────────────────────────────────────────────────
  describe('validate_oos formatter', () => {
    const cases = [
      {
        desc: 'when oos_results is null, factor count falls back to 0',
        result: { n_folds: 4, oos_results: null },
        expected: '4 folds · 0 factors',
      },
      {
        desc: 'when oos_results is a non-empty array, its length is used',
        result: { n_folds: 4, oos_results: [{ factor_name: 'MOM' }, { factor_name: 'VAL' }] },
        expected: '4 folds · 2 factors',
      },
    ];
    cases.forEach(({ desc, result, expected }) => {
      it(desc, () => {
        expect(stepSummary('validate_oos', result)).toBe(expected);
      });
    });
  });

  // ── coverage_gate ────────────────────────────────────────────────────────────
  it('when a coverage_gate result is given, the passing/min line is built', () => {
    expect(stepSummary('coverage_gate', { n_passing: 6, min_factors: 3 }))
      .toBe('6 passing (min 3)');
  });

  // ── regime ───────────────────────────────────────────────────────────────────
  describe('regime formatter', () => {
    const cases = [
      {
        desc: 'when persisted is false, only the regime name is shown',
        result: { regime: 'growth', persisted: false },
        expected: 'growth',
      },
      {
        desc: 'when persisted is true, the persisted suffix is appended',
        result: { regime: 'growth', persisted: true },
        expected: 'growth · persisted',
      },
    ];
    cases.forEach(({ desc, result, expected }) => {
      it(desc, () => {
        expect(stepSummary('regime', result)).toBe(expected);
      });
    });
  });

  // ── optimize (hockey_stick_warning arm) ──────────────────────────────────────
  it('when hockey_stick_warning is true, the warning glyph is appended', () => {
    expect(stepSummary('optimize', { n_selected: 15, net_sharpe: 0.85, hockey_stick_warning: true }))
      .toBe('15 holdings · net Sharpe 0.85 ⚠');
  });

  // ── rebalance_decision ───────────────────────────────────────────────────────
  describe('rebalance_decision formatter', () => {
    const cases = [
      {
        desc: 'when decision is false, hold is shown',
        result: { decision: false, n_weights: 20 },
        expected: 'hold · 20 weights',
      },
      {
        desc: 'when decision is true, rebalance is shown',
        result: { decision: true, n_weights: 20 },
        expected: 'rebalance · 20 weights',
      },
    ];
    cases.forEach(({ desc, result, expected }) => {
      it(desc, () => {
        expect(stepSummary('rebalance_decision', result)).toBe(expected);
      });
    });
  });

  // ── cost ─────────────────────────────────────────────────────────────────────
  describe('cost formatter', () => {
    const cases = [
      {
        desc: 'when exceeds_assumed is false, no exceeds suffix is appended',
        result: { cost_bps_actual: 12, exceeds_assumed: false },
        expected: '12 bps actual',
      },
      {
        desc: 'when exceeds_assumed is true, the exceeds suffix is appended',
        result: { cost_bps_actual: 12, exceeds_assumed: true },
        expected: '12 bps actual ⚠ exceeds',
      },
    ];
    cases.forEach(({ desc, result, expected }) => {
      it(desc, () => {
        expect(stepSummary('cost', result)).toBe(expected);
      });
    });
  });

  // ── report ───────────────────────────────────────────────────────────────────
  describe('report formatter', () => {
    const cases = [
      {
        desc: 'when checklist_passed is true, PASS is shown',
        result: { pass_count: 17, checklist_total: 17, checklist_passed: true },
        expected: '17/17 checks · PASS',
      },
      {
        desc: 'when checklist_passed is false, FAIL is shown',
        result: { pass_count: 14, checklist_total: 17, checklist_passed: false },
        expected: '14/17 checks · FAIL',
      },
    ];
    cases.forEach(({ desc, result, expected }) => {
      it(desc, () => {
        expect(stepSummary('report', result)).toBe(expected);
      });
    });
  });

  // ── persist (persisted: true arm) ────────────────────────────────────────────
  it('when a persist result is persisted, "persisted" is returned', () => {
    expect(stepSummary('persist', { persisted: true, reason: '' }))
      .toBe('persisted');
  });
});
