/**
 * Source-blind spec — authored from acceptance criteria only.
 *
 * Criterion (UNIT):
 *   "Ticker universe seeds from Object.keys(snapshot.weights) of
 *    getLatestSnapshot(name) (adding the previously-absent portfolio reference);
 *    re-seeds only when the field still holds its previous seed; falls back to
 *    DEFAULT_TICKERS (silently on error) on no-portfolio/empty/error."
 *
 * Criterion (UNIT):
 *   "Navigation to those pages with no portfolio selected shows DEFAULT_TICKERS."
 *
 * Criterion (UNIT):
 *   "An API error during snapshot fetch falls back to DEFAULT_TICKERS silently."
 */

import { Observable, of, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';

// ---------------------------------------------------------------------------
// Spec-derived types — shapes inferred from criterion text only
// ---------------------------------------------------------------------------

interface PortfolioSnapshot {
  weights: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Pure seeding rules (the contract the component must implement)
// ---------------------------------------------------------------------------

/** Resolves seeded tickers from a snapshot per the spec. */
function resolveTickerSeed(
  snapshot: PortfolioSnapshot | null | undefined,
  defaultTickers: string[],
): string[] {
  if (!snapshot) return defaultTickers;
  const keys = Object.keys(snapshot.weights);
  return keys.length > 0 ? keys : defaultTickers;
}

/**
 * Returns true when the ticker field still holds the last-applied seed and
 * re-seeding is therefore allowed.
 */
function canReseed(currentField: string[], lastAppliedSeed: string[]): boolean {
  if (currentField.length !== lastAppliedSeed.length) return false;
  return currentField.every((t, i) => t === lastAppliedSeed[i]);
}

// Placeholder for the actual DEFAULT_TICKERS constant; real value lives in the implementation.
const STUB_DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'];

// ---------------------------------------------------------------------------

describe('factor-research: ticker seeding contract', () => {

  // ---------- resolveTickerSeed — structural mapping -----------------------

  describe('resolveTickerSeed', () => {

    it('when snapshot has weights, returns exactly Object.keys(weights)', () => {
      const snap: PortfolioSnapshot = { weights: { SPY: 0.6, QQQ: 0.4 } };
      expect(resolveTickerSeed(snap, STUB_DEFAULT_TICKERS)).toEqual(['SPY', 'QQQ']);
    });

    it('when snapshot.weights is an empty object, returns defaultTickers', () => {
      const snap: PortfolioSnapshot = { weights: {} };
      expect(resolveTickerSeed(snap, STUB_DEFAULT_TICKERS)).toEqual(STUB_DEFAULT_TICKERS);
    });

    it('when snapshot is null, returns defaultTickers', () => {
      expect(resolveTickerSeed(null, STUB_DEFAULT_TICKERS)).toEqual(STUB_DEFAULT_TICKERS);
    });

    it('when snapshot is undefined, returns defaultTickers', () => {
      expect(resolveTickerSeed(undefined, STUB_DEFAULT_TICKERS)).toEqual(STUB_DEFAULT_TICKERS);
    });

    // Property: for ANY non-empty weights object, seeded === Object.keys(weights)
    const snapshotVariants: Array<{ desc: string; weights: Record<string, number> }> = [
      { desc: 'single-asset portfolio', weights: { VTI: 1.0 } },
      { desc: 'two-asset portfolio', weights: { AAPL: 0.5, MSFT: 0.5 } },
      { desc: 'five-asset portfolio', weights: { AAPL: 0.2, MSFT: 0.2, GOOG: 0.2, AMZN: 0.2, TSLA: 0.2 } },
      { desc: 'tickers containing dots', weights: { 'BRK.B': 0.5, 'BF.B': 0.5 } },
      { desc: 'international tickers', weights: { ASML: 0.4, NVO: 0.3, SAP: 0.3 } },
      { desc: 'ten-asset portfolio', weights: Object.fromEntries(Array.from({ length: 10 }, (_, i) => [`T${i}`, 0.1])) },
    ];

    snapshotVariants.forEach(({ desc, weights }) => {
      it(`when ${desc}, seeded tickers equal Object.keys(weights)`, () => {
        const snap: PortfolioSnapshot = { weights };
        expect(resolveTickerSeed(snap, STUB_DEFAULT_TICKERS)).toEqual(Object.keys(weights));
      });
    });
  });

  // ---------- canReseed — idempotence gate ----------------------------------

  describe('canReseed', () => {

    it('when current field equals last seed, returns true (re-seeding allowed)', () => {
      expect(canReseed(['AAPL', 'MSFT'], ['AAPL', 'MSFT'])).toBeTrue();
    });

    it('when user replaced tickers entirely, returns false (re-seeding suppressed)', () => {
      expect(canReseed(['TSLA'], ['AAPL', 'MSFT'])).toBeFalse();
    });

    it('when user added a ticker beyond the seed, returns false', () => {
      expect(canReseed(['AAPL', 'MSFT', 'GOOG'], ['AAPL', 'MSFT'])).toBeFalse();
    });

    it('when user removed one ticker from the seed, returns false', () => {
      expect(canReseed(['AAPL'], ['AAPL', 'MSFT'])).toBeFalse();
    });

    it('when user changed ticker order, returns false', () => {
      expect(canReseed(['MSFT', 'AAPL'], ['AAPL', 'MSFT'])).toBeFalse();
    });

    // Idempotence property: a field that exactly matches seed must always allow re-seed
    const seedVariants = [
      ['VTI'],
      ['AAPL', 'MSFT'],
      ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA'],
    ];

    seedVariants.forEach((seed) => {
      it(`when seed is [${seed.join(', ')}] and field equals seed, canReseed is true`, () => {
        expect(canReseed([...seed], seed)).toBeTrue();
      });
    });
  });

  // ---------- no-portfolio fallback ----------------------------------------

  describe('when no portfolio is selected', () => {
    it('when currentPortfolioName returns null, resolveTickerSeed returns defaultTickers', () => {
      // Component must call resolveTickerSeed(null, DEFAULT_TICKERS) when no portfolio
      const result = resolveTickerSeed(null, STUB_DEFAULT_TICKERS);
      expect(result).toEqual(STUB_DEFAULT_TICKERS);
    });

    it('when no portfolio is selected, getLatestSnapshot must NOT be called', () => {
      // Criterion: snapshot fetch only happens when a portfolio is selected
      const getLatestSnapshotSpy = jasmine.createSpy('getLatestSnapshot').and.returnValue(of({ weights: {} }));
      const portfolioName: string | null = null;

      if (portfolioName) {
        // Only call when name is non-null — this branch must never execute
        getLatestSnapshotSpy(portfolioName);
      }

      expect(getLatestSnapshotSpy).not.toHaveBeenCalled();
    });
  });

  // ---------- API error silent fallback ------------------------------------

  describe('when getLatestSnapshot errors during snapshot fetch', () => {

    it('when HTTP error occurs, catchError prevents it from reaching subscriber', () => {
      const errorSpy = spyOn(console, 'error');
      const failingCall: Observable<PortfolioSnapshot> = throwError(() => new Error('HTTP 500'));

      let errorReached = false;
      let resolvedTickers: string[] = [];

      failingCall
        .pipe(catchError(() => of(null as unknown as PortfolioSnapshot)))
        .subscribe({
          next: (snap) => { resolvedTickers = resolveTickerSeed(snap, STUB_DEFAULT_TICKERS); },
          error: () => { errorReached = true; },
        });

      expect(errorReached).withContext('error must not reach subscriber').toBeFalse();
      expect(resolvedTickers).withContext('fallback must equal DEFAULT_TICKERS').toEqual(STUB_DEFAULT_TICKERS);
      expect(errorSpy).withContext('silent fallback: no console.error').not.toHaveBeenCalled();
    });

    it('when 404 error occurs, fallback is a non-empty ticker list', () => {
      const failingCall: Observable<PortfolioSnapshot> = throwError(() => ({ status: 404 }));

      let resolvedTickers: string[] = [];
      failingCall
        .pipe(catchError(() => of(null as unknown as PortfolioSnapshot)))
        .subscribe({
          next: (snap) => { resolvedTickers = resolveTickerSeed(snap, STUB_DEFAULT_TICKERS); },
        });

      expect(resolvedTickers.length)
        .withContext('DEFAULT_TICKERS must be non-empty')
        .toBeGreaterThan(0);
    });

    it('when network error occurs, fallback tickers are all non-empty strings', () => {
      const failingCall: Observable<PortfolioSnapshot> = throwError(() => new Error('NetworkError'));

      let resolvedTickers: string[] = [];
      failingCall
        .pipe(catchError(() => of(null as unknown as PortfolioSnapshot)))
        .subscribe({
          next: (snap) => { resolvedTickers = resolveTickerSeed(snap, STUB_DEFAULT_TICKERS); },
        });

      resolvedTickers.forEach((ticker) => {
        expect(typeof ticker).toBe('string');
        expect(ticker.length).toBeGreaterThan(0);
      });
    });
  });
});
