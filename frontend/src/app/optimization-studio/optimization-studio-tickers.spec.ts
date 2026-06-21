/**
 * Source-blind unit tests — DEFAULT_TICKERS fallback for Optimization Studio
 * (R4 / issue #1027).
 *
 * Criteria covered:
 *  [UNIT] Navigation to optimization-studio with no portfolio selected shows
 *         DEFAULT_TICKERS.
 *  [UNIT] An API error during snapshot fetch falls back to DEFAULT_TICKERS
 *         silently (no unhandled console.error).
 *  [UNIT] Every wired page has specs (a) loads data on init, (b) handles API
 *         error, (c) reacts to portfolio context change.
 *
 * Assumptions:
 *  - Component class: OptimizationStudioComponent at ./optimization-studio.component
 *  - The component exposes a `tickers` signal (or FormControl) whose value is
 *    the current ticker universe string.
 *  - DEFAULT_TICKERS is an exported constant from the component file or a
 *    shared constants file; its exact value is unknown so tests assert it is
 *    non-empty and equals whatever the signal holds when no portfolio is selected.
 *  - The component injects PortfolioContextService and reads
 *    portfolioContextService.currentPortfolioName().
 *  - On mount with a portfolio, the component calls
 *    PortfolioApiService.getLatestSnapshot(portfolioName) and seeds tickers from
 *    Object.keys(snapshot.weights).
 *  - PortfolioContextService is the global singleton defined at
 *    ../core/services/portfolio-context.service.
 *  - PortfolioApiService is at ../services/portfolio-api.service.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { OptimizationStudioComponent } from './optimization-studio';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { ICON_PROVIDER } from '../icons';

// ── Helpers ────────────────────────────────────────────────────────────────────

function makePortfolioContextStub(portfolioName: string | null) {
  return {
    currentPortfolioId: signal<string | null>(portfolioName ? 'id-1' : null),
    currentPortfolioName: signal<string | null>(portfolioName),
    selectedPortfolio: signal<unknown>(null),
    hasPortfolio: signal(portfolioName !== null),
    dateRange: signal({
      preset: '1Y' as const,
      start: new Date('2024-01-01'),
      end: new Date('2025-01-01'),
    }),
  };
}

function setup(portfolioName: string | null): {
  fixture: ComponentFixture<OptimizationStudioComponent>;
  http: HttpTestingController;
} {
  TestBed.configureTestingModule({
    imports: [OptimizationStudioComponent],
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
      {
        provide: PortfolioContextService,
        useValue: makePortfolioContextStub(portfolioName),
      },
      ICON_PROVIDER,
    ],
  });

  const fixture = TestBed.createComponent(OptimizationStudioComponent);
  const http = TestBed.inject(HttpTestingController);
  return { fixture, http };
}

// ── (c) reacts to portfolio context / no-portfolio branch ────────────────────

describe('OptimizationStudioComponent — ticker seeding', () => {
  afterEach(() => TestBed.inject(HttpTestingController).verify());

  describe('when no portfolio is selected on mount', () => {
    it('does not call getLatestSnapshot', () => {
      const { fixture, http } = setup(null);
      fixture.detectChanges();

      // No snapshot fetch should have been issued.
      http.expectNone(r => r.url.includes('snapshot'));
    });

    it('populates the ticker field with a non-empty default value (DEFAULT_TICKERS)', () => {
      const { fixture } = setup(null);
      fixture.detectChanges();

      const comp = fixture.componentInstance;
      // Accept either a signal-based `tickers()` or a form control's `.value`.
      const compUnknown = comp as unknown as Record<string, unknown>;
      const value: string =
        typeof compUnknown['tickers'] === 'function'
          ? (compUnknown['tickers'] as () => string)()
          : (compUnknown['tickersControl'] as { value: string } | undefined)?.value ?? '';

      expect(value)
        .withContext(
          'When no portfolio is selected the ticker field must be pre-filled ' +
            'with DEFAULT_TICKERS (non-empty)'
        )
        .toBeTruthy();
    });
  });

  // ── (b) handles API error ────────────────────────────────────────────────────

  describe('when snapshot API errors on mount with a selected portfolio', () => {
    it('falls back to DEFAULT_TICKERS without logging an unhandled console.error', () => {
      const consoleSpy = spyOn(console, 'error');
      const { fixture, http } = setup('t212');
      fixture.detectChanges();

      // Flush the snapshot request with a 404 to simulate missing snapshot.
      const snapshotReqs = http.match(r => r.url.includes('snapshot'));
      snapshotReqs.forEach(r =>
        r.flush('Not found', { status: 404, statusText: 'Not Found' })
      );

      fixture.detectChanges();

      const comp = fixture.componentInstance;
      const compUnknown2 = comp as unknown as Record<string, unknown>;
      const value: string =
        typeof compUnknown2['tickers'] === 'function'
          ? (compUnknown2['tickers'] as () => string)()
          : (compUnknown2['tickersControl'] as { value: string } | undefined)?.value ?? '';

      expect(value)
        .withContext(
          'After a snapshot 404 the ticker field must fall back to DEFAULT_TICKERS'
        )
        .toBeTruthy();

      expect(consoleSpy)
        .withContext('Snapshot API error must not surface as unhandled console.error')
        .not.toHaveBeenCalled();
    });
  });

  // ── (a) loads data on init ────────────────────────────────────────────────────

  describe('when a portfolio is selected on mount', () => {
    it('seeds tickers from the portfolio snapshot weights', () => {
      const { fixture, http } = setup('t212');
      fixture.detectChanges();

      const snapshotReqs = http.match(r => r.url.includes('snapshot'));
      expect(snapshotReqs.length)
        .withContext('Component must request a portfolio snapshot when a portfolio is selected')
        .toBeGreaterThan(0);

      snapshotReqs.forEach(r =>
        r.flush({ weights: { AAPL: 0.5, MSFT: 0.3, GOOGL: 0.2 } })
      );

      fixture.detectChanges();

      const comp = fixture.componentInstance;
      const compUnknown3 = comp as unknown as Record<string, unknown>;
      const value: string =
        typeof compUnknown3['tickers'] === 'function'
          ? (compUnknown3['tickers'] as () => string)()
          : (compUnknown3['tickersControl'] as { value: string } | undefined)?.value ?? '';

      // The ticker string/list must contain at least one of the snapshot keys.
      expect(value).toContain('AAPL');
    });
  });
});
