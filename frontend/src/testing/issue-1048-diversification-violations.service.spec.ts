/**
 * Issue #1048 – DiversificationViolationsService unit tests.
 *
 * These tests target the client-side violation-computation logic:
 *   - geographic: no region > 60-70 %
 *   - sector concentration: no sector > 15 %, target HHI < 0.12
 *   - top-4 holdings combined < 30 %
 *   - healthcare 8-12 %, technology 10-12 %, no absent major sector
 *
 * Invariants (property-based style – multiple boundary examples per invariant):
 *   INV-1  Any portfolio where one region ≥ 70 % of weight → region-concentration flag
 *   INV-2  Any portfolio where one sector > 15 % of weight → sector-concentration flag
 *   INV-3  Any portfolio whose equal-weight HHI ≥ 0.12 → hhi flag
 *   INV-4  escalated = true iff regime is 'stress' or 'bear'
 */

import { DiversificationViolationsService } from '../app/portfolio-builder/diversification-preview/diversification-violations.service';
import type { ViolationFlag, SimpleProfile } from '../app/portfolio-builder/diversification-preview/diversification-violations.service';

// ── Pure helpers ──────────────────────────────────────────────────────────────

function buildHomogeneous(n: number, sector = 'Technology', region = 'US'): SimpleProfile[] {
  return Array.from({ length: n }, () => ({ sector, region }));
}

function buildMixed(
  total: number,
  dominant: { sector: string; region: string },
  dominantCount: number,
): SimpleProfile[] {
  const rest = total - dominantCount;
  return [
    ...Array.from({ length: dominantCount }, () => ({ ...dominant })),
    ...Array.from({ length: rest }, () => ({ sector: 'Financials', region: 'Europe' })),
  ];
}

// ── Suite ─────────────────────────────────────────────────────────────────────

describe('DiversificationViolationsService (issue-1048)', () => {
  let svc: DiversificationViolationsService;

  beforeEach(() => {
    svc = new DiversificationViolationsService();
  });

  // ======================================================================
  // Region-concentration (> 60 % threshold)
  // ======================================================================

  describe('region-concentration violation', () => {

    it('when one region is 100 % (single region), returns a region-concentration flag', () => {
      const profiles = buildHomogeneous(3, 'Technology', 'US');
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'region-concentration')).toBeTrue();
    });

    it('when one region is exactly at the 70 % boundary, returns a region-concentration flag', () => {
      // 7 US out of 10 = 70 %
      const profiles = buildMixed(10, { sector: 'Technology', region: 'US' }, 7);
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'region-concentration')).toBeTrue();
    });

    it('when one region is 50 % (below 60 % threshold), does NOT return a region-concentration flag', () => {
      // 5 US, 5 Europe → max region = 50 % < 60 %
      const profiles = buildMixed(10, { sector: 'Technology', region: 'US' }, 5);
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'region-concentration')).toBeFalse();
    });

    // INV-1: multiple boundary examples
    [
      { dominantCount: 8, total: 10, label: '80 %' },
      { dominantCount: 9, total: 10, label: '90 %' },
      { dominantCount: 10, total: 10, label: '100 %' },
    ].forEach(({ dominantCount, total, label }) => {
      it(`INV-1: when dominant region is ${label}, region-concentration flag is present`, () => {
        const profiles = buildMixed(total, { sector: 'Technology', region: 'US' }, dominantCount);
        const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
        expect(flags.some(f => f.kind === 'region-concentration'))
          .withContext(`region ${label} must trigger flag`)
          .toBeTrue();
      });
    });
  });

  // ======================================================================
  // Sector-concentration (> 15 %)
  // ======================================================================

  describe('sector-concentration violation', () => {

    it('when one sector is 75 % of the portfolio, returns a sector-concentration flag', () => {
      const profiles = buildMixed(4, { sector: 'Technology', region: 'US' }, 3);
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'sector-concentration')).toBeTrue();
    });

    it('when no single sector exceeds 15 %, does NOT return a sector-concentration flag', () => {
      // 8 equally-weighted distinct sectors = 12.5 % each < 15 %
      const profiles: SimpleProfile[] = [
        { sector: 'Technology',  region: 'US' },
        { sector: 'Healthcare',  region: 'US' },
        { sector: 'Financials',  region: 'US' },
        { sector: 'Energy',      region: 'US' },
        { sector: 'Consumer',    region: 'US' },
        { sector: 'Industrials', region: 'US' },
        { sector: 'Materials',   region: 'US' },
        { sector: 'Utilities',   region: 'US' },
      ];
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'sector-concentration')).toBeFalse();
    });

    // INV-2: multiple boundary examples
    [
      { dominantCount: 2, total: 4,  label: '50 %' },
      { dominantCount: 3, total: 10, label: '30 %' },
      { dominantCount: 2, total: 10, label: '20 %' },
    ].forEach(({ dominantCount, total, label }) => {
      it(`INV-2: when dominant sector is ${label} (> 15 %), sector-concentration flag is present`, () => {
        const profiles = buildMixed(total, { sector: 'Technology', region: 'US' }, dominantCount);
        const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
        expect(flags.some(f => f.kind === 'sector-concentration'))
          .withContext(`sector ${label} must trigger flag`)
          .toBeTrue();
      });
    });
  });

  // ======================================================================
  // HHI (≥ 0.12)
  // HHI (equal-weight) = 1/N
  // HHI ≥ 0.12 when N ≤ 8
  // ======================================================================

  describe('HHI violation', () => {

    it('when portfolio has 2 equal-weight assets (HHI = 0.5), returns an HHI flag', () => {
      const profiles = buildHomogeneous(2);
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'hhi')).toBeTrue();
    });

    it('when portfolio has 8 equal-weight assets (HHI = 0.125 ≥ 0.12), returns an HHI flag', () => {
      const profiles = buildHomogeneous(8);
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'hhi')).toBeTrue();
    });

    it('when portfolio has 9 equal-weight assets (HHI ≈ 0.111 < 0.12), does NOT return an HHI flag', () => {
      const profiles = buildHomogeneous(9);
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'hhi')).toBeFalse();
    });

    // INV-3: boundary sweep
    [1, 2, 3, 4, 5, 6, 7, 8].forEach(n => {
      it(`INV-3: when N=${n} equal-weight assets (HHI = ${(1/n).toFixed(3)} ≥ 0.12), HHI flag present`, () => {
        const profiles = buildHomogeneous(n);
        const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
        expect(flags.some(f => f.kind === 'hhi'))
          .withContext(`N=${n} must trigger HHI flag`)
          .toBeTrue();
      });
    });
  });

  // ======================================================================
  // Top-4 concentration (≥ 30 %)
  // ======================================================================

  describe('top-4 concentration violation', () => {

    it('when portfolio has 2 assets (top-4 = 100 % ≥ 30 %), returns a top4-concentration flag', () => {
      const profiles = buildHomogeneous(2);
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'top4-concentration')).toBeTrue();
    });

    it('when portfolio has 20 equal-weight assets (top-4 = 20 % < 30 %), does NOT return a top4-concentration flag', () => {
      const profiles = buildHomogeneous(20);
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'top4-concentration')).toBeFalse();
    });
  });

  // ======================================================================
  // Healthcare weight (8-12 %)
  // ======================================================================

  describe('healthcare violation', () => {

    it('when healthcare weight is 0 % (below 8 %), returns a healthcare flag', () => {
      const profiles: SimpleProfile[] = [
        { sector: 'Technology', region: 'US' },
        { sector: 'Financials', region: 'US' },
      ];
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'healthcare')).toBeTrue();
    });

    it('when healthcare weight is 75 % (above 12 %), returns a healthcare flag', () => {
      const profiles: SimpleProfile[] = [
        { sector: 'Healthcare', region: 'US' },
        { sector: 'Healthcare', region: 'US' },
        { sector: 'Healthcare', region: 'US' },
        { sector: 'Technology', region: 'US' },
      ];
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'healthcare')).toBeTrue();
    });
  });

  // ======================================================================
  // Technology weight (10-12 %)
  // ======================================================================

  describe('technology violation', () => {

    it('when technology weight is 0 % (below 10 %), returns a technology flag', () => {
      const profiles: SimpleProfile[] = [
        { sector: 'Financials', region: 'US' },
        { sector: 'Healthcare', region: 'US' },
      ];
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'technology')).toBeTrue();
    });

    it('when technology weight is 100 % (above 12 %), returns a technology flag', () => {
      const profiles = buildHomogeneous(3, 'Technology', 'US');
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'technology')).toBeTrue();
    });
  });

  // ======================================================================
  // Absent major sector
  // ======================================================================

  describe('absent-sector violation', () => {

    it('when only one sector is present, returns an absent-sector flag', () => {
      const profiles = buildHomogeneous(5, 'Technology', 'US');
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'absent-sector')).toBeTrue();
    });

    it('when all major sectors are represented, does NOT return an absent-sector flag', () => {
      const profiles: SimpleProfile[] = [
        { sector: 'Technology',  region: 'US' },
        { sector: 'Healthcare',  region: 'US' },
        { sector: 'Financials',  region: 'US' },
        { sector: 'Energy',      region: 'US' },
        { sector: 'Consumer',    region: 'US' },
        { sector: 'Industrials', region: 'US' },
        { sector: 'Materials',   region: 'US' },
        { sector: 'Utilities',   region: 'US' },
      ];
      const flags: ViolationFlag[] = svc.computeViolations(profiles, 'neutral');
      expect(flags.some(f => f.kind === 'absent-sector')).toBeFalse();
    });
  });

  // ======================================================================
  // Regime escalation (INV-4)
  // escalated = true iff regime ∈ {'stress', 'bear', 'CONTRACTION'}
  // ======================================================================

  describe('regime escalation', () => {

    const VIOLATION_PROFILES = buildHomogeneous(3, 'Technology', 'US');

    it('when regime is "stress", all flags are escalated', () => {
      const flags: ViolationFlag[] = svc.computeViolations(VIOLATION_PROFILES, 'stress');
      expect(flags.length).toBeGreaterThan(0);
      expect(flags.every(f => f.escalated))
        .withContext('all flags must be escalated in stress regime')
        .toBeTrue();
    });

    it('when regime is "bear", all flags are escalated', () => {
      const flags: ViolationFlag[] = svc.computeViolations(VIOLATION_PROFILES, 'bear');
      expect(flags.length).toBeGreaterThan(0);
      expect(flags.every(f => f.escalated))
        .withContext('all flags must be escalated in bear regime')
        .toBeTrue();
    });

    it('when regime is "neutral", no flags are escalated', () => {
      const flags: ViolationFlag[] = svc.computeViolations(VIOLATION_PROFILES, 'neutral');
      expect(flags.length).toBeGreaterThan(0);
      expect(flags.every(f => !f.escalated))
        .withContext('no flags must be escalated in neutral regime')
        .toBeTrue();
    });

    it('when regime is "bull", no flags are escalated', () => {
      const flags: ViolationFlag[] = svc.computeViolations(VIOLATION_PROFILES, 'bull');
      expect(flags.length).toBeGreaterThan(0);
      expect(flags.every(f => !f.escalated))
        .withContext('no flags must be escalated in bull regime')
        .toBeTrue();
    });

    // INV-4: boundary: stress and bear → escalated; neutral and bull → not escalated
    (['stress', 'bear'] as const).forEach(regime => {
      it(`INV-4: when regime is "${regime}", escalated is true for every returned flag`, () => {
        const flags: ViolationFlag[] = svc.computeViolations(VIOLATION_PROFILES, regime);
        flags.forEach(f => {
          expect(f.escalated)
            .withContext(`flag ${f.kind} must be escalated in ${regime} regime`)
            .toBeTrue();
        });
      });
    });

    (['neutral', 'bull'] as const).forEach(regime => {
      it(`INV-4: when regime is "${regime}", escalated is false for every returned flag`, () => {
        const flags: ViolationFlag[] = svc.computeViolations(VIOLATION_PROFILES, regime);
        flags.forEach(f => {
          expect(f.escalated)
            .withContext(`flag ${f.kind} must NOT be escalated in ${regime} regime`)
            .toBeFalse();
        });
      });
    });
  });

  // ======================================================================
  // Edge cases
  // ======================================================================

  describe('edge cases', () => {
    it('when given an empty profile list, returns an empty violations array', () => {
      const flags: ViolationFlag[] = svc.computeViolations([], 'neutral');
      expect(Array.isArray(flags)).toBeTrue();
      expect(flags.length).toBe(0);
    });
  });
});
