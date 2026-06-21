/**
 * Source-blind contract tests — criterion:
 *   "The 15-risk-measure / multi-objective raw controls are removed;
 *    each method maps to a fixed curated config."
 *
 * Tests are derived from the acceptance criteria alone. They will remain
 * RED until the implementation satisfies the spec.
 */

import { OPTIMIZE_METHODS } from './optimize-methods';

const EXPECTED_IDS = ['min_variance', 'max_sharpe', 'risk_parity_erc', 'hrp'] as const;

describe('OptimizeMethodFixedConfigs', () => {

  // ── cardinality ─────────────────────────────────────────────────────────────

  it('when all methods are listed, exactly four are returned', () => {
    expect(OPTIMIZE_METHODS.length).toBe(4);
  });

  // ── membership (one test per named method) ───────────────────────────────────

  for (const id of EXPECTED_IDS) {
    it(`when methods are listed, "${id}" is among them`, () => {
      const ids = OPTIMIZE_METHODS.map(m => m.id);
      expect(ids).toContain(id);
    });
  }

  // ── fixed config — each method must carry exactly one optimizer_type string ──

  it('when each method config is resolved, it carries a fixed optimizer_type string', () => {
    for (const method of OPTIMIZE_METHODS) {
      expect(typeof method.config.optimizer_type).toBe('string');
    }
  });

  it('when min-variance config is resolved, optimizer_type is mean_risk', () => {
    const m = OPTIMIZE_METHODS.find(x => x.id === 'min_variance')!;
    expect(m.config.optimizer_type).toBe('mean_risk');
  });

  it('when max-sharpe config is resolved, optimizer_type is mean_risk', () => {
    const m = OPTIMIZE_METHODS.find(x => x.id === 'max_sharpe')!;
    expect(m.config.optimizer_type).toBe('mean_risk');
  });

  // ── "raw controls removed" — no method exposes a selectable risk-measure list

  it('when each method config is inspected, no method exposes a selectable risk-measure array', () => {
    for (const method of OPTIMIZE_METHODS) {
      const cfg = method.config as Record<string, unknown>;
      // A fixed config stores a single string, never an array of choices
      expect(Array.isArray(cfg['risk_measures'])).toBeFalse();
    }
  });

  // ── "multi-objective raw controls removed" — no method exposes an objectives list

  it('when each method config is inspected, no method exposes a selectable objectives array', () => {
    for (const method of OPTIMIZE_METHODS) {
      const cfg = method.config as Record<string, unknown>;
      expect(Array.isArray(cfg['objectives'])).toBeFalse();
    }
  });

  // ── uniqueness — all four ids are distinct

  it('when method ids are collected, all four are unique', () => {
    const ids = OPTIMIZE_METHODS.map(m => m.id);
    const unique = new Set(ids);
    expect(unique.size).toBe(4);
  });

  // ── property-covering examples for the invariant
  //    "for every method in the four, config.optimizer_type is always a non-empty string"
  //    (approximates a property test across the full domain of methods)

  it('when min-variance config is resolved, optimizer_type is a non-empty string', () => {
    const m = OPTIMIZE_METHODS.find(x => x.id === 'min_variance')!;
    expect(m.config.optimizer_type.length).toBeGreaterThan(0);
  });

  it('when risk-parity-erc config is resolved, optimizer_type is a non-empty string', () => {
    const m = OPTIMIZE_METHODS.find(x => x.id === 'risk_parity_erc')!;
    expect(m.config.optimizer_type.length).toBeGreaterThan(0);
  });

  it('when hrp config is resolved, optimizer_type is a non-empty string', () => {
    const m = OPTIMIZE_METHODS.find(x => x.id === 'hrp')!;
    expect(m.config.optimizer_type.length).toBeGreaterThan(0);
  });
});
