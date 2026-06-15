/**
 * Source-blind unit tests for issue #1015
 * feat(dashboard): read portfolio name from context and drop redundant portfolio-list refetch
 *
 * Criteria covered (oracle tier in parentheses):
 *   [UNIT] dashboard reads the portfolio name from PortfolioContextService.currentPortfolioName()
 *   [UNIT] dashboard does not inject PortfolioApiService and issues no portfolio list() request on load
 *   [UNIT] no unhandled observable errors reach console.error
 *   [UNIT] every wired page has specs for:
 *            (a) loads data on init
 *            (b) handles API error — graceful degradation (errors absorbed by service catchError)
 *            (c) reacts to portfolio context change
 *
 * Criteria skipped (oracle: NOT VERIFIABLE or T3):
 *   – Dashboard charts populate within 3 seconds on a local API connection (T3 benchmark)
 *   – Analytics endpoints use the active context key name (covered by dashboard-context-wiring.spec.ts)
 *
 * Note on error-state criterion:
 *   The DashboardService wraps each analytics endpoint with catchError(() => of(null)), so
 *   individual 500 responses are absorbed gracefully — the component never enters hasError=true.
 *   The visible error banner is only triggered when the outer forkJoin itself throws, which is
 *   tested via a mocked DashboardService in dashboard-context-wiring.spec.ts.
 *   This file verifies graceful degradation: no crash, no console.error on any 500.
 *
 * Import-path assumptions:
 *   DashboardComponent      → ./dashboard
 *   PortfolioContextService → ../../core/services/portfolio-context.service
 */
import { TestBed } from '@angular/core/testing';
import { ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection, signal } from '@angular/core';
import { By } from '@angular/platform-browser';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import { DashboardComponent } from './dashboard';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { ICON_PROVIDER } from '../icons';

/** Returns an appropriate stub body for each analytics endpoint URL. */
function stubBody(url: string): Record<string, unknown> {
  if (url.includes('/market/indices'))        return { indices: [], total: 0 };
  if (url.includes('/market/snapshot'))       return { vix: 0, vixChange: 0, sp500Return: 0, tenYearYield: 0, yieldChange: 0, usdIndex: 0, usdChange: 0, asOf: '2026-01-01T00:00:00Z' };
  if (url.includes('/equity-curve'))          return { points: [], portfolioTotalReturn: 0, benchmarkTotalReturn: 0 };
  if (url.includes('/performance-metrics'))   return { kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' };
  if (url.includes('/allocation'))            return { nodes: [], totalPositions: 0, totalSectors: 0 };
  if (url.includes('/asset-class-returns'))   return { returns: [] };
  if (url.includes('/rolling-metrics'))       return { window: 63, sharpe: [], volatility: [], beta: [] };
  return {};
}

/** Flush every outstanding HTTP request with a shape-correct stub response. */
function flushAll(ctrl: HttpTestingController): void {
  ctrl.match(() => true).forEach(r => r.flush(stubBody(r.request.url)));
}

/** Returns true when the URL looks like a portfolio-list endpoint. */
function looksLikeListEndpoint(url: string, method: string): boolean {
  // Matches GET /…/portfolio or /…/portfolios but NOT /…/portfolio-analytics/…
  return method === 'GET' && /\/portfolios?\b(?!\/)(?!-analytics)/.test(url);
}

describe('DashboardComponent — context-wiring spec (#1015)', () => {
  let fixture: ComponentFixture<DashboardComponent>;
  let http: HttpTestingController;
  let consoleErrorSpy: jasmine.Spy;

  // Shared writable signals — reset in beforeEach so each test starts clean.
  const nameSignal = signal<string | null>(null);
  const idSignal   = signal<string | null>(null);

  const mockContext = {
    currentPortfolioId:   idSignal,
    currentPortfolioName: nameSignal,
    selectedPortfolio:    signal<unknown>(null),
    dateRange:            signal({
      preset: '1Y' as const,
      start:  new Date('2024-01-01'),
      end:    new Date('2025-01-01'),
    }),
  };

  beforeEach(async () => {
    nameSignal.set(null);
    idSignal.set(null);
    consoleErrorSpy = spyOn(console, 'error');

    await TestBed.configureTestingModule({
      imports:   [DashboardComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: mockContext },
        // PortfolioApiService is intentionally absent.
        // If DashboardComponent injects PortfolioApiService, Angular will throw
        // NullInjectorError at createComponent time — the expected red state.
      ],
    }).compileComponents();

    http    = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(DashboardComponent);
  });

  afterEach(() => {
    flushAll(http);
    http.verify();
  });

  // ── scope: PortfolioApiService must not be a declared dependency ───────────

  it('when PortfolioApiService is absent from the injector, the component constructs without error', () => {
    expect(fixture.componentInstance).withContext(
      'DashboardComponent must not require PortfolioApiService'
    ).toBeTruthy();
  });

  // ── (a) loads data on init — reads name from context, not a local API call ─

  it('when context name is set before first detectChanges, the dashboard renders that name', async () => {
    nameSignal.set('ACME Fund');
    idSignal.set('port-1');

    fixture.detectChanges();
    await fixture.whenStable();
    flushAll(http);
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent as string)
      .withContext('portfolio name from PortfolioContextService must appear in the rendered HTML')
      .toContain('ACME Fund');
  });

  it('when component initialises, no HTTP GET request targets a portfolio-list endpoint', async () => {
    nameSignal.set('ACME Fund');
    idSignal.set('port-1');
    fixture.detectChanges();
    await fixture.whenStable();

    const allReqs  = http.match(() => true);
    const listReqs = allReqs.filter(r =>
      looksLikeListEndpoint(r.request.url, r.request.method)
    );
    allReqs.forEach(r => r.flush(stubBody(r.request.url)));

    expect(listReqs.length)
      .withContext('dashboard must not call a portfolio-list endpoint on load')
      .toBe(0);
  });

  // ── Example-based name invariant ──────────────────────────────────────────
  // Criterion: for ANY non-empty portfolio name set in context that name must
  // appear in the dashboard's rendered output (NEVER-RAISES invariant).

  it('when a variety of non-null portfolio names are provided by context, each appears in rendered output', async () => {
    const testNames = ['Alpha Fund', 'Beta-Corp-2024', 'My Portfolio', 'Ω Holdings'];
    for (const name of testNames) {
      nameSignal.set(name);
      idSignal.set('prop-test-id');
      fixture.detectChanges();
      await fixture.whenStable();
      flushAll(http);
      fixture.detectChanges();

      expect(fixture.nativeElement.textContent as string)
        .withContext(`portfolio name "${name}" must appear in the rendered output`)
        .toContain(name);
    }
  });

  // ── (b) handles API error — graceful degradation ──────────────────────────

  it('when analytics endpoints return 500, the component renders stably without throwing', async () => {
    nameSignal.set('ACME Fund');
    idSignal.set('port-1');
    fixture.detectChanges();
    await fixture.whenStable();

    expect(() => {
      http.match(() => true).forEach(r =>
        r.flush('Internal Server Error', { status: 500, statusText: 'Server Error' })
      );
      fixture.detectChanges();
    }).not.toThrow();
  });

  it('when analytics endpoints return 500, no unhandled observable error reaches console.error', async () => {
    nameSignal.set('ACME Fund');
    idSignal.set('port-1');
    fixture.detectChanges();
    await fixture.whenStable();

    http.match(() => true).forEach(r =>
      r.flush('Server Error', { status: 500, statusText: 'Server Error' })
    );
    fixture.detectChanges();

    expect(consoleErrorSpy)
      .withContext('catchError must swallow HTTP errors — nothing must reach console.error')
      .not.toHaveBeenCalled();
  });

  // ── (c) reacts to portfolio context change ────────────────────────────────

  it('when the portfolio context name changes, the dashboard renders the new name', async () => {
    nameSignal.set('AlphaFund');
    idSignal.set('port-a');
    fixture.detectChanges();
    await fixture.whenStable();
    flushAll(http);
    fixture.detectChanges();

    nameSignal.set('BetaCorpPortfolio');
    idSignal.set('port-b');
    fixture.detectChanges();
    await fixture.whenStable();
    flushAll(http);
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent as string)
      .withContext('the new portfolio name must appear after context change')
      .toContain('BetaCorpPortfolio');
  });

  it('when context changes to a new portfolio name, HTTP requests use the updated name', async () => {
    nameSignal.set('Alpha');
    idSignal.set('port-a');
    fixture.detectChanges();
    await fixture.whenStable();
    flushAll(http);
    fixture.detectChanges();

    nameSignal.set('Beta');
    idSignal.set('port-b');
    fixture.detectChanges();
    await fixture.whenStable();

    const newReqs  = http.match(() => true);
    const usesBeta = newReqs.some(r => r.request.url.includes('Beta'));
    newReqs.forEach(r => r.flush(stubBody(r.request.url)));

    expect(usesBeta)
      .withContext('HTTP requests after a context change must reference the new portfolio name')
      .toBeTrue();
  });
});
