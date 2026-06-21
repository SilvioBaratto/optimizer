/**
 * Issue #1052 — source-blind end-to-end contract tests
 * "test(handoff): optimize Save to backtest stored-weights end-to-end contract"
 *
 * Verifiable criteria (from oracle report):
 *
 *   [UNIT] AC-1 — Optimize Save → createSnapshot(name, {snapshot_type:'optimization', weights});
 *                 backtest resolver for the same portfolio returns the identical weight keys
 *                 (source: 'stored').
 *
 *   [UNIT] AC-2 — picking a never-optimized portfolio yields the equal-weight fallback
 *                 (source: 'equal-weight').
 *
 * Criteria NOT covered (oracle: not verifiable):
 *   AC-3 — "Tests use HttpTestingController/mocks" (tooling choice, not a runtime assertion)
 *   AC-4 — "All tests pass" (suite gate)
 *   AC-5 — "SOLID/clean code" (subjective quality)
 *
 * All tests mock HTTP via Angular's HttpTestingController against the existing
 * portfolio API contract (/portfolio/{id}/snapshots/latest).
 * No new endpoints. No backend calls.
 *
 * Authored from acceptance-criteria text only — no implementation source was read.
 *
 * RED-phase failure modes:
 *   – Suite 1 fails because BacktestWeightResolverService does not yet expose a
 *     resolve() method that returns { source: 'stored', weights } when an
 *     optimization snapshot is present.
 *   – Suite 2 fails because the equal-weight fallback path (source: 'equal-weight')
 *     is not yet returned for portfolios that were never optimized.
 *
 * Property-based tests: Angular/Karma/Jasmine ships no built-in property-testing
 * library. The key-identity invariant from AC-1 — Object.keys(resolved.weights)
 * equals Object.keys(snapshot.weights) — is covered by three parameterized weight
 * shapes (1, 2, and 5 assets) that together exhaust the single-asset, even-split,
 * and general multi-asset cases.
 */

import { TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, makeSnapshotDto } from './index';
import { BacktestWeightResolverService } from '../app/backtesting/backtest-weight-resolver';

/**
 * Shape emitted by BacktestWeightResolverService.resolve().
 * Derived from AC text only — not from implementation source.
 */
interface WeightResolution {
  source: 'stored' | 'equal-weight';
  weights: Record<string, number>;
}

/** Portfolio identifier used throughout the suites. */
const PF_ID = 'pf-handoff-1052';

/**
 * Returns a request predicate that matches a GET for the latest snapshot of the
 * given portfolio identifier. Uses a substring check so the test is independent
 * of whether the resolver encodes the identifier as a name or UUID in the path.
 */
function isLatestSnapshotGet(
  portfolioIdentifier: string,
): (r: { method: string; url: string }) => boolean {
  return (r) =>
    r.method === 'GET' &&
    r.url.includes(portfolioIdentifier) &&
    r.url.toLowerCase().includes('snapshot');
}

// ─── Suite 1 — AC-1: stored-weights source ────────────────────────────────────

describe('BacktestWeightResolverService — stored-weights handoff (issue #1052 AC-1)', () => {
  let svc: BacktestWeightResolverService;
  let http: HttpTestingController;

  beforeEach(async () => {
    await configureTestBed({ withHttp: true });
    svc = TestBed.inject(BacktestWeightResolverService);
    http = injectHttp();
  });

  afterEach(() => http.verify());

  // ── source field ──────────────────────────────────────────────────────────────

  it('when optimization snapshot has weights, resolve() emits source "stored"', () => {
    let result: WeightResolution | undefined;
    svc.resolve(PF_ID).subscribe((r) => {
      result = r as WeightResolution;
    });

    http
      .expectOne(isLatestSnapshotGet(PF_ID))
      .flush(
        makeSnapshotDto({
          portfolio_id: PF_ID,
          snapshot_type: 'optimization',
          weights: { AAPL: 0.6, MSFT: 0.4 },
        }),
      );

    expect(result)
      .withContext('resolve() must emit a value when snapshot is present')
      .toBeDefined();
    expect(result!.source)
      .withContext('source must be "stored" for an optimization snapshot with weights')
      .toBe('stored');
  });

  // ── weight key identity ───────────────────────────────────────────────────────

  it('when optimization snapshot has weights, returned weight keys equal the snapshot weight keys', () => {
    const savedWeights = { AAPL: 0.6, MSFT: 0.4 };
    let result: WeightResolution | undefined;
    svc.resolve(PF_ID).subscribe((r) => {
      result = r as WeightResolution;
    });

    http
      .expectOne(isLatestSnapshotGet(PF_ID))
      .flush(makeSnapshotDto({ snapshot_type: 'optimization', weights: savedWeights }));

    expect(Object.keys(result!.weights).sort())
      .withContext('resolver must return exactly the snapshot weight keys, no additions or removals')
      .toEqual(Object.keys(savedWeights).sort());
  });

  // ── round-trip contract (the "end-to-end" in the issue title) ────────────────
  //
  // Pins AC-1 as a single scenario:
  //   createSnapshot(name, {snapshot_type:'optimization', weights})  ← save side
  //   resolve(name) reads that snapshot back and returns source:'stored'  ← resolve side
  //
  // The HTTP mock simulates what the backend stores after a createSnapshot POST.

  it('when weights were persisted as an optimization snapshot, resolve() returns exactly those weight keys', () => {
    const savedWeights = { NVDA: 0.5, TSLA: 0.3, GOOGL: 0.2 };
    let result: WeightResolution | undefined;
    svc.resolve(PF_ID).subscribe((r) => {
      result = r as WeightResolution;
    });

    // The GET /snapshots/latest response mirrors what createSnapshot() would have stored.
    http
      .expectOne(isLatestSnapshotGet(PF_ID))
      .flush(makeSnapshotDto({ snapshot_type: 'optimization', weights: savedWeights }));

    expect(result!.source).toBe('stored');
    // Order-independent set equality: same tickers, nothing added or dropped.
    expect(new Set(Object.keys(result!.weights))).toEqual(
      new Set(Object.keys(savedWeights)),
    );
  });

  // ── parameterized key-identity invariant ──────────────────────────────────────
  //
  // For any non-empty optimization snapshot weights, the resolver must return
  // source:'stored' and exactly the same weight keys.

  const weightCases: Array<[string, Record<string, number>]> = [
    ['one asset (100%)', { AAPL: 1.0 }],
    ['two assets (equal split)', { AAPL: 0.5, MSFT: 0.5 }],
    ['five assets (varying)', { AAPL: 0.25, MSFT: 0.25, GOOGL: 0.2, AMZN: 0.2, TSLA: 0.1 }],
  ];

  weightCases.forEach(([label, weights]) => {
    it(`when optimization snapshot has [${label}], source is "stored" and weight keys are preserved`, () => {
      let result: WeightResolution | undefined;
      svc.resolve(PF_ID).subscribe((r) => {
        result = r as WeightResolution;
      });

      http
        .expectOne(isLatestSnapshotGet(PF_ID))
        .flush(makeSnapshotDto({ snapshot_type: 'optimization', weights }));

      expect(result!.source)
        .withContext(`"${label}" must yield source:'stored'`)
        .toBe('stored');
      expect(Object.keys(result!.weights).sort())
        .withContext(`"${label}" must preserve all weight keys without modification`)
        .toEqual(Object.keys(weights).sort());
    });
  });
});

// ─── Suite 2 — AC-2: equal-weight fallback ────────────────────────────────────

describe('BacktestWeightResolverService — equal-weight fallback (issue #1052 AC-2)', () => {
  let svc: BacktestWeightResolverService;
  let http: HttpTestingController;

  beforeEach(async () => {
    await configureTestBed({ withHttp: true });
    svc = TestBed.inject(BacktestWeightResolverService);
    http = injectHttp();
  });

  afterEach(() => http.verify());

  // ── never-optimized: 404 response ────────────────────────────────────────────

  it('when no snapshot exists (404), resolve() emits source "equal-weight"', () => {
    let result: WeightResolution | undefined;
    svc.resolve(PF_ID).subscribe((r) => {
      result = r as WeightResolution;
    });

    http
      .expectOne(isLatestSnapshotGet(PF_ID))
      .flush({ detail: 'Not found' }, { status: 404, statusText: 'Not Found' });

    expect(result)
      .withContext('resolve() must still emit a value for the equal-weight fallback')
      .toBeDefined();
    expect(result!.source)
      .withContext('source must be "equal-weight" when no snapshot is found')
      .toBe('equal-weight');
  });

  // ── non-optimization snapshot type ───────────────────────────────────────────

  it('when latest snapshot type is "manual", resolve() emits source "equal-weight"', () => {
    let result: WeightResolution | undefined;
    svc.resolve(PF_ID).subscribe((r) => {
      result = r as WeightResolution;
    });

    http
      .expectOne(isLatestSnapshotGet(PF_ID))
      .flush(makeSnapshotDto({ snapshot_type: 'manual', weights: { AAPL: 1.0 } }));

    expect(result!.source)
      .withContext('only snapshot_type:"optimization" should yield source:"stored"')
      .toBe('equal-weight');
  });

  // ── optimization snapshot with empty weights ──────────────────────────────────

  it('when optimization snapshot has empty weights, resolve() emits source "equal-weight"', () => {
    let result: WeightResolution | undefined;
    svc.resolve(PF_ID).subscribe((r) => {
      result = r as WeightResolution;
    });

    http
      .expectOne(isLatestSnapshotGet(PF_ID))
      .flush(makeSnapshotDto({ snapshot_type: 'optimization', weights: {} }));

    expect(result!.source)
      .withContext('empty weights map must trigger the equal-weight fallback')
      .toBe('equal-weight');
  });

  // ── optimization snapshot with null weights ───────────────────────────────────

  it('when optimization snapshot weights are null, resolve() emits source "equal-weight"', () => {
    let result: WeightResolution | undefined;
    svc.resolve(PF_ID).subscribe((r) => {
      result = r as WeightResolution;
    });

    http
      .expectOne(isLatestSnapshotGet(PF_ID))
      .flush(
        makeSnapshotDto({
          snapshot_type: 'optimization',
          weights: null as unknown as Record<string, number>,
        }),
      );

    expect(result!.source)
      .withContext('null weights must trigger the equal-weight fallback')
      .toBe('equal-weight');
  });
});
