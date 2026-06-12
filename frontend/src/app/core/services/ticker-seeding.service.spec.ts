/**
 * Source-blind spec for TickerSeedingService (issue #949).
 *
 * Authored directly from the acceptance criteria — no implementation was read.
 * Covers the four branches required by the spec:
 *   (a) null / empty portfolioName  → defaultTickers, no HTTP call
 *   (b) non-null name + weights     → Object.keys(snapshot.weights)
 *   (c) non-null name + empty wts   → defaultTickers
 *   (d) HTTP error                  → defaultTickers silently, never errors
 *
 * Every branch also verifies the "emits exactly one value then completes"
 * contract stated in the acceptance criteria.
 */

import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { TickerSeedingService } from './ticker-seeding.service';
import { PortfolioApiService } from './portfolio-api.service';

import type { SnapshotDto } from '../models/portfolio-api.model';

function makeSnapshot(weights: Record<string, number>): SnapshotDto {
  return {
    id: 'snap-test',
    portfolio_id: 'port-test',
    snapshot_date: '2026-01-01',
    snapshot_type: 'manual',
    weights,
    sector_mapping: null,
    summary: null,
    optimizer_config: null,
    turnover: null,
    holding_count: Object.keys(weights).length,
    created_at: '2026-01-01T00:00:00Z',
  };
}

describe('TickerSeedingService', () => {
  let service: TickerSeedingService;
  let portfolioApi: jasmine.SpyObj<PortfolioApiService>;

  const DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL'];

  beforeEach(() => {
    portfolioApi = jasmine.createSpyObj('PortfolioApiService', ['getLatestSnapshot']);

    TestBed.configureTestingModule({
      providers: [
        TickerSeedingService,
        { provide: PortfolioApiService, useValue: portfolioApi },
      ],
    });

    service = TestBed.inject(TickerSeedingService);
  });

  // ── Branch (a): null portfolioName ──────────────────────────────────────────

  describe('when portfolioName is null', () => {
    it('when null name given, defaultTickers are emitted immediately', (done) => {
      service.seedFromPortfolio(null, DEFAULT_TICKERS).subscribe({
        next: (tickers) => {
          expect(tickers).toEqual(DEFAULT_TICKERS);
          done();
        },
        error: done.fail,
      });
    });

    it('when null name given, no HTTP call is made', (done) => {
      service.seedFromPortfolio(null, DEFAULT_TICKERS).subscribe({
        next: () => {
          expect(portfolioApi.getLatestSnapshot).not.toHaveBeenCalled();
          done();
        },
        error: done.fail,
      });
    });

    it('when null name given, observable emits exactly one value then completes', (done) => {
      let emitCount = 0;
      service.seedFromPortfolio(null, DEFAULT_TICKERS).subscribe({
        next: () => emitCount++,
        complete: () => {
          expect(emitCount).toBe(1);
          done();
        },
        error: done.fail,
      });
    });
  });

  // ── Branch (b): non-null name + non-empty weights ───────────────────────────

  describe('when portfolioName is non-null and snapshot.weights is non-empty', () => {
    it('when valid name given, getLatestSnapshot is called with that name', (done) => {
      portfolioApi.getLatestSnapshot.and.returnValue(
        of(makeSnapshot({ TSLA: 0.6, NVDA: 0.4 })),
      );

      service.seedFromPortfolio('growth-fund', DEFAULT_TICKERS).subscribe({
        next: () => {
          expect(portfolioApi.getLatestSnapshot).toHaveBeenCalledWith('growth-fund');
          done();
        },
        error: done.fail,
      });
    });

    it('when snapshot has weights, Object.keys(snapshot.weights) is emitted', (done) => {
      portfolioApi.getLatestSnapshot.and.returnValue(
        of(makeSnapshot({ TSLA: 0.5, NVDA: 0.3, MSFT: 0.2 })),
      );

      service.seedFromPortfolio('growth-fund', DEFAULT_TICKERS).subscribe({
        next: (tickers) => {
          expect(tickers).toEqual(jasmine.arrayContaining(['TSLA', 'NVDA', 'MSFT']));
          expect(tickers.length).toBe(3);
          done();
        },
        error: done.fail,
      });
    });

    it('when snapshot has weights, defaultTickers are NOT returned', (done) => {
      portfolioApi.getLatestSnapshot.and.returnValue(
        of(makeSnapshot({ XYZ: 1.0 })),
      );

      service.seedFromPortfolio('growth-fund', DEFAULT_TICKERS).subscribe({
        next: (tickers) => {
          expect(tickers).not.toEqual(DEFAULT_TICKERS);
          done();
        },
        error: done.fail,
      });
    });

    it('when snapshot has weights, observable emits exactly one value then completes', (done) => {
      portfolioApi.getLatestSnapshot.and.returnValue(of(makeSnapshot({ SPY: 1.0 })));

      let emitCount = 0;
      service.seedFromPortfolio('any-fund', DEFAULT_TICKERS).subscribe({
        next: () => emitCount++,
        complete: () => {
          expect(emitCount).toBe(1);
          done();
        },
        error: done.fail,
      });
    });
  });

  // ── Branch (c): non-null name + empty weights ───────────────────────────────

  describe('when portfolioName is non-null but snapshot.weights is empty', () => {
    it('when snapshot has no weights, defaultTickers are emitted', (done) => {
      portfolioApi.getLatestSnapshot.and.returnValue(of(makeSnapshot({})));

      service.seedFromPortfolio('empty-fund', DEFAULT_TICKERS).subscribe({
        next: (tickers) => {
          expect(tickers).toEqual(DEFAULT_TICKERS);
          done();
        },
        error: done.fail,
      });
    });

    it('when snapshot has no weights, observable emits exactly one value then completes', (done) => {
      portfolioApi.getLatestSnapshot.and.returnValue(of(makeSnapshot({})));

      let emitCount = 0;
      service.seedFromPortfolio('empty-fund', DEFAULT_TICKERS).subscribe({
        next: () => emitCount++,
        complete: () => {
          expect(emitCount).toBe(1);
          done();
        },
        error: done.fail,
      });
    });
  });

  // ── Branch (d): HTTP error → silent fallback, never errors ──────────────────

  describe('when getLatestSnapshot returns an HTTP error', () => {
    it('when API errors, defaultTickers are emitted', (done) => {
      portfolioApi.getLatestSnapshot.and.returnValue(
        throwError(() => new Error('500 Internal Server Error')),
      );

      service.seedFromPortfolio('broken-fund', DEFAULT_TICKERS).subscribe({
        next: (tickers) => {
          expect(tickers).toEqual(DEFAULT_TICKERS);
          done();
        },
        error: () => done.fail('Observable must not emit an error — silent fallback required'),
      });
    });

    it('when API errors, the observable does not error', (done) => {
      portfolioApi.getLatestSnapshot.and.returnValue(
        throwError(() => new Error('Network timeout')),
      );

      service.seedFromPortfolio('broken-fund', DEFAULT_TICKERS).subscribe({
        next: (tickers) => expect(tickers).toEqual(DEFAULT_TICKERS),
        complete: done,
        error: () => done.fail('Observable must not error on HTTP failure'),
      });
    });

    it('when API errors, observable emits exactly one value then completes', (done) => {
      portfolioApi.getLatestSnapshot.and.returnValue(
        throwError(() => new Error('401 Unauthorized')),
      );

      let emitCount = 0;
      service.seedFromPortfolio('broken-fund', DEFAULT_TICKERS).subscribe({
        next: () => emitCount++,
        complete: () => {
          expect(emitCount).toBe(1);
          done();
        },
        error: () => done.fail('Observable must not error'),
      });
    });
  });

  // ── Criterion (c): reacts to portfolio context change ──────────────────────
  // Verifies that calling with a different name correctly routes to the new
  // portfolio — the service itself must not cache or memoize portfolio identity.

  describe('when called sequentially with different portfolio names', () => {
    it('when portfolio name changes, getLatestSnapshot is called with the new name', (done) => {
      portfolioApi.getLatestSnapshot.and.returnValue(of(makeSnapshot({ BRK: 1.0 })));

      service.seedFromPortfolio('portfolio-b', DEFAULT_TICKERS).subscribe({
        next: () => {
          expect(portfolioApi.getLatestSnapshot).toHaveBeenCalledWith('portfolio-b');
          done();
        },
        error: done.fail,
      });
    });

    it('when switched from null to a named portfolio, getLatestSnapshot is called for the named portfolio', (done) => {
      portfolioApi.getLatestSnapshot.and.returnValue(
        of(makeSnapshot({ AMZN: 0.7, META: 0.3 })),
      );

      // First call with null (no HTTP)
      service.seedFromPortfolio(null, DEFAULT_TICKERS).subscribe();

      // Second call with a real name
      service.seedFromPortfolio('real-portfolio', DEFAULT_TICKERS).subscribe({
        next: (tickers) => {
          expect(portfolioApi.getLatestSnapshot).toHaveBeenCalledWith('real-portfolio');
          expect(tickers).toEqual(jasmine.arrayContaining(['AMZN', 'META']));
          done();
        },
        error: done.fail,
      });
    });
  });
});
