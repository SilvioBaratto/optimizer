/**
 * Source-blind contract tests for the diversification compute helper (issue #1046).
 *
 * Every test is derived solely from the acceptance criteria and the thresholds in
 * requirements.md — the implementation does not exist yet (Red phase of TDD).
 *
 * Thresholds (from requirements.md Diversification spec):
 *   region          : no region > 60 %
 *   sector_pct      : no individual sector > 15 %
 *   sector_hhi      : target HHI < 0.12
 *   top4            : top-4 holdings combined < 30 %
 *   healthcare      : 8 % – 12 %
 *   technology      : 10 % – 12 %
 *   absent_sector   : all major sectors must be represented
 *
 * The helper computes under an equal-weight assumption, so for N assets each
 * asset has weight 1/N and each sector's weight equals (count in sector) / N.
 */
import { computeDiversification } from './diversification.helper';
import type { AssetWithMetadata, DiversificationFlag } from './diversification.helper';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeAsset(sector: string, region: string, assetClass = 'equity'): AssetWithMetadata {
  return { sector, region, assetClass };
}

function findFlag(flags: DiversificationFlag[], rule: string): DiversificationFlag | undefined {
  return flags.find(f => f.rule === rule);
}

// ---------------------------------------------------------------------------
// Specs
// ---------------------------------------------------------------------------

describe('computeDiversification', () => {

  // ── Report structure ──────────────────────────────────────────────────────

  describe('report structure', () => {
    it('when assets are provided, hhi is a number', () => {
      const report = computeDiversification([makeAsset('Technology', 'North America')]);
      expect(typeof report.hhi).toBe('number');
    });

    it('when assets are provided, top4Weight is a number', () => {
      const report = computeDiversification([makeAsset('Technology', 'North America')]);
      expect(typeof report.top4Weight).toBe('number');
    });

    it('when assets are provided, flags is an array', () => {
      const report = computeDiversification([makeAsset('Technology', 'North America')]);
      expect(Array.isArray(report.flags)).toBe(true);
    });

    it('when assets are provided, flags array contains an entry for every checklist rule', () => {
      const report = computeDiversification([makeAsset('Technology', 'North America')]);
      const rules = report.flags.map(f => f.rule);
      expect(rules).toContain('region');
      expect(rules).toContain('sector_pct');
      expect(rules).toContain('sector_hhi');
      expect(rules).toContain('top4');
      expect(rules).toContain('healthcare');
      expect(rules).toContain('technology');
      expect(rules).toContain('absent_sector');
    });
  });

  // ── HHI computation ───────────────────────────────────────────────────────
  //
  // Under equal-weight with N assets across K sectors, each sector weight is
  // (count_k / N) and HHI = Σ weight_k².

  describe('hhi computation', () => {
    it('when all assets belong to one sector, hhi equals 1.0', () => {
      const assets = Array.from({ length: 8 }, () => makeAsset('Technology', 'North America'));
      const report = computeDiversification(assets);
      expect(report.hhi).toBeCloseTo(1.0, 5);
    });

    it('when assets split evenly across two sectors, hhi equals 0.5', () => {
      const assets = [
        ...Array.from({ length: 5 }, () => makeAsset('Technology', 'North America')),
        ...Array.from({ length: 5 }, () => makeAsset('Healthcare', 'Europe')),
      ];
      const report = computeDiversification(assets);
      expect(report.hhi).toBeCloseTo(0.5, 5);
    });

    it('when assets spread across ten unique sectors, hhi equals 0.1', () => {
      // 10 assets, each in a distinct sector → HHI = 10 × (0.1)² = 0.10
      const assets = Array.from({ length: 10 }, (_, i) => makeAsset(`Sector${i}`, 'North America'));
      const report = computeDiversification(assets);
      expect(report.hhi).toBeCloseTo(0.1, 5);
    });
  });

  // ── top4Weight computation ────────────────────────────────────────────────
  //
  // Under equal-weight, top-4 assets each have weight 1/N, so top4Weight = 4/N
  // (or 1.0 when N ≤ 4).

  describe('top4Weight computation', () => {
    it('when 4 assets are selected, top4Weight equals 1.0 under equal-weight', () => {
      const assets = ['Technology', 'Healthcare', 'Financials', 'Energy']
        .map(s => makeAsset(s, 'North America'));
      const report = computeDiversification(assets);
      expect(report.top4Weight).toBeCloseTo(1.0, 5);
    });

    it('when 10 assets are selected, top4Weight equals 0.4 under equal-weight', () => {
      const assets = Array.from({ length: 10 }, (_, i) => makeAsset(`Sector${i}`, 'North America'));
      const report = computeDiversification(assets);
      expect(report.top4Weight).toBeCloseTo(0.4, 5);
    });
  });

  // ── Region flag ───────────────────────────────────────────────────────────
  //
  // Rule: no single region should exceed 60 % of the portfolio.

  describe('region flag — threshold 60 %', () => {
    it('when a region holds 70 % of assets, region flag is breached', () => {
      const assets = [
        ...Array.from({ length: 7 }, () => makeAsset('Technology', 'North America')),
        ...Array.from({ length: 3 }, () => makeAsset('Financials', 'Europe')),
      ];
      const flag = findFlag(computeDiversification(assets).flags, 'region');
      expect(flag?.breached).toBe(true);
    });

    it('when no region holds more than 50 % of assets, region flag is not breached', () => {
      const assets = [
        ...Array.from({ length: 5 }, () => makeAsset('Technology', 'North America')),
        ...Array.from({ length: 5 }, () => makeAsset('Financials', 'Europe')),
      ];
      const flag = findFlag(computeDiversification(assets).flags, 'region');
      expect(flag?.breached).toBe(false);
    });
  });

  // ── Sector % flag ─────────────────────────────────────────────────────────
  //
  // Rule: no individual sector should exceed 15 % of the portfolio.

  describe('sector_pct flag — threshold 15 %', () => {
    it('when a sector holds 30 % of assets, sector_pct flag is breached', () => {
      // 3 Technology out of 10 total = 30 %
      const assets = [
        ...Array.from({ length: 3 }, () => makeAsset('Technology', 'North America')),
        ...Array.from({ length: 7 }, (_, i) => makeAsset(`Other${i}`, 'Europe')),
      ];
      const flag = findFlag(computeDiversification(assets).flags, 'sector_pct');
      expect(flag?.breached).toBe(true);
    });

    it('when every sector holds ≈14 % of assets, sector_pct flag is not breached', () => {
      // 7 distinct sectors → each 1/7 ≈ 14.3 % < 15 %
      const assets = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        .map(s => makeAsset(s, 'North America'));
      const flag = findFlag(computeDiversification(assets).flags, 'sector_pct');
      expect(flag?.breached).toBe(false);
    });
  });

  // ── Sector HHI flag ───────────────────────────────────────────────────────
  //
  // Rule: target HHI < 0.12.

  describe('sector_hhi flag — threshold 0.12', () => {
    it('when all assets are in one sector, sector_hhi flag is breached (HHI = 1.0)', () => {
      const assets = Array.from({ length: 5 }, () => makeAsset('Technology', 'North America'));
      const flag = findFlag(computeDiversification(assets).flags, 'sector_hhi');
      expect(flag?.breached).toBe(true);
    });

    it('when assets spread across 10 unique sectors, sector_hhi flag is not breached (HHI = 0.1)', () => {
      // HHI = 10 × (0.1)² = 0.10 < 0.12
      const assets = Array.from({ length: 10 }, (_, i) => makeAsset(`Sector${i}`, 'North America'));
      const flag = findFlag(computeDiversification(assets).flags, 'sector_hhi');
      expect(flag?.breached).toBe(false);
    });

    it('when 7 sectors split evenly, sector_hhi flag is breached (HHI ≈ 0.143)', () => {
      // 7 sectors → HHI = 7 × (1/7)² = 1/7 ≈ 0.143 > 0.12
      const assets = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        .map(s => makeAsset(s, 'North America'));
      const flag = findFlag(computeDiversification(assets).flags, 'sector_hhi');
      expect(flag?.breached).toBe(true);
    });
  });

  // ── Top-4 flag ────────────────────────────────────────────────────────────
  //
  // Rule: top-4 holdings combined must be < 30 %.
  // Under equal-weight, top4Weight = 4/N.
  // 4/13 ≈ 30.8 % → breached; 4/14 ≈ 28.6 % → not breached.

  describe('top4 flag — threshold 30 %', () => {
    it('when 4 assets are selected, top4 flag is breached (top-4 weight = 100 %)', () => {
      const assets = ['A', 'B', 'C', 'D'].map(s => makeAsset(s, 'North America'));
      const flag = findFlag(computeDiversification(assets).flags, 'top4');
      expect(flag?.breached).toBe(true);
    });

    it('when 13 assets are selected, top4 flag is breached (4/13 ≈ 30.8 %)', () => {
      const assets = Array.from({ length: 13 }, (_, i) => makeAsset(`S${i}`, 'North America'));
      const flag = findFlag(computeDiversification(assets).flags, 'top4');
      expect(flag?.breached).toBe(true);
    });

    it('when 14 assets are selected, top4 flag is not breached (4/14 ≈ 28.6 %)', () => {
      const assets = Array.from({ length: 14 }, (_, i) => makeAsset(`S${i}`, 'North America'));
      const flag = findFlag(computeDiversification(assets).flags, 'top4');
      expect(flag?.breached).toBe(false);
    });
  });

  // ── Healthcare flag ───────────────────────────────────────────────────────
  //
  // Rule: healthcare allocation must be in the range 8 %–12 %.
  // Tests pin the lower bound (< 8 % → breached), in-range (not breached),
  // and upper bound (> 12 % → breached).

  describe('healthcare flag — range 8 %–12 %', () => {
    it('when no healthcare assets are present, healthcare flag is breached (0 % < 8 %)', () => {
      const assets = Array.from({ length: 10 }, () => makeAsset('Technology', 'North America'));
      const flag = findFlag(computeDiversification(assets).flags, 'healthcare');
      expect(flag?.breached).toBe(true);
    });

    it('when healthcare weight is ≈4.8 %, healthcare flag is breached (below 8 %)', () => {
      // 1 Healthcare + 20 others → 1/21 ≈ 4.8 %
      const assets = [
        makeAsset('Healthcare', 'North America'),
        ...Array.from({ length: 20 }, () => makeAsset('Technology', 'North America')),
      ];
      const flag = findFlag(computeDiversification(assets).flags, 'healthcare');
      expect(flag?.breached).toBe(true);
    });

    it('when healthcare weight is ≈8.3 %, healthcare flag is not breached (in 8–12 % range)', () => {
      // 1 Healthcare + 11 others → 1/12 ≈ 8.33 %
      const assets = [
        makeAsset('Healthcare', 'North America'),
        ...Array.from({ length: 11 }, () => makeAsset('Technology', 'North America')),
      ];
      const flag = findFlag(computeDiversification(assets).flags, 'healthcare');
      expect(flag?.breached).toBe(false);
    });

    it('when healthcare weight is ≈13.3 %, healthcare flag is breached (above 12 %)', () => {
      // 2 Healthcare + 13 others → 2/15 ≈ 13.3 %
      const assets = [
        ...Array.from({ length: 2 }, () => makeAsset('Healthcare', 'North America')),
        ...Array.from({ length: 13 }, () => makeAsset('Technology', 'North America')),
      ];
      const flag = findFlag(computeDiversification(assets).flags, 'healthcare');
      expect(flag?.breached).toBe(true);
    });
  });

  // ── Technology flag ───────────────────────────────────────────────────────
  //
  // Rule: technology allocation must be in the range 10 %–12 %.

  describe('technology flag — range 10 %–12 %', () => {
    it('when no technology assets are present, technology flag is breached (0 % < 10 %)', () => {
      const assets = Array.from({ length: 10 }, () => makeAsset('Healthcare', 'North America'));
      const flag = findFlag(computeDiversification(assets).flags, 'technology');
      expect(flag?.breached).toBe(true);
    });

    it('when technology weight is ≈9.1 %, technology flag is breached (below 10 %)', () => {
      // 1 Technology + 10 others → 1/11 ≈ 9.1 %
      const assets = [
        makeAsset('Technology', 'North America'),
        ...Array.from({ length: 10 }, () => makeAsset('Healthcare', 'North America')),
      ];
      const flag = findFlag(computeDiversification(assets).flags, 'technology');
      expect(flag?.breached).toBe(true);
    });

    it('when technology weight is exactly 10 %, technology flag is not breached (in 10–12 % range)', () => {
      // 1 Technology + 9 others → 1/10 = 10 %
      const assets = [
        makeAsset('Technology', 'North America'),
        ...Array.from({ length: 9 }, () => makeAsset('Healthcare', 'North America')),
      ];
      const flag = findFlag(computeDiversification(assets).flags, 'technology');
      expect(flag?.breached).toBe(false);
    });

    it('when technology weight is ≈13.3 %, technology flag is breached (above 12 %)', () => {
      // 2 Technology + 13 others → 2/15 ≈ 13.3 %
      const assets = [
        ...Array.from({ length: 2 }, () => makeAsset('Technology', 'North America')),
        ...Array.from({ length: 13 }, () => makeAsset('Healthcare', 'North America')),
      ];
      const flag = findFlag(computeDiversification(assets).flags, 'technology');
      expect(flag?.breached).toBe(true);
    });
  });

  // ── Absent-sector flag ────────────────────────────────────────────────────
  //
  // Rule: all major sectors must be represented.
  // Major sectors (standard financial taxonomy): Technology, Healthcare,
  // Financials, Industrials, Consumer Discretionary, Consumer Staples,
  // Energy, Utilities.

  describe('absent_sector flag', () => {
    it('when all assets are in a single sector, absent_sector flag is breached', () => {
      const assets = Array.from({ length: 10 }, () => makeAsset('Technology', 'North America'));
      const flag = findFlag(computeDiversification(assets).flags, 'absent_sector');
      expect(flag?.breached).toBe(true);
    });

    it('when all major sectors are represented, absent_sector flag is not breached', () => {
      const majorSectors = [
        'Technology',
        'Healthcare',
        'Financials',
        'Industrials',
        'Consumer Discretionary',
        'Consumer Staples',
        'Energy',
        'Utilities',
      ];
      const assets = majorSectors.map(s => makeAsset(s, 'North America'));
      const flag = findFlag(computeDiversification(assets).flags, 'absent_sector');
      expect(flag?.breached).toBe(false);
    });
  });

});
