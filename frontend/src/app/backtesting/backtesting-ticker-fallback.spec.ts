/**
 * Source-blind unit tests for issue #1015 — ticker-universe fallback behaviour.
 *
 * Criteria covered (oracle tier in parentheses):
 *   [UNIT] Navigation to optimization-studio, backtesting, or factor-research with no portfolio
 *          selected shows DEFAULT_TICKERS.
 *   [UNIT] An API error during snapshot fetch falls back to DEFAULT_TICKERS silently.
 *
 * Criteria skipped (oracle: NOT VERIFIABLE by unit test):
 *   – Navigation with a portfolio selected pre-fills tickers from that portfolio's latest snapshot
 *     (requires the full app + router integration, which is out of scope for this unit suite).
 *
 * Assumption (recorded per the test-designer mandate):
 *   "shows DEFAULT_TICKERS" is interpreted as: the component's tickersRaw() signal (which drives
 *   the ticker input) equals DEFAULT_TICKERS joined by ", ". DEFAULT_TICKERS is verified against
 *   the exported constant from the component module.
 *
 *   "silently" is interpreted as: the component does NOT call console.error when the snapshot
 *   API returns an error, and the error is not surfaced as a toast or alert in the DOM.
 *
 * Import-path assumptions:
 *   BacktestingComponent, DEFAULT_TICKERS → ./backtesting
 *   PortfolioContextService               → ../../core/services/portfolio-context.service
 */
import { TestBed } from '@angular/core/testing';
import { ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection, signal } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import { BacktestingComponent, DEFAULT_TICKERS } from './backtesting';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { ICON_PROVIDER } from '../icons';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;

describe('BacktestingComponent — ticker-universe fallback (#1015)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;
  let http: HttpTestingController;
  let consoleErrorSpy: jasmine.Spy;

  const nameSignal = signal<string | null>(null);
  const idSignal   = signal<string | null>(null);

  const noPortfolioContext = {
    currentPortfolioId:   idSignal,
    currentPortfolioName: nameSignal,
    selectedPortfolio:    signal<unknown>(null),
    dateRange:            signal({
      preset: '1Y' as const,
      start:  new Date('2024-01-01'),
      end:    new Date('2025-01-01'),
    }),
  };

  const DEFAULT_TICKERS_STR = [...DEFAULT_TICKERS].join(', ');

  async function setup(portfolioName: string | null = null): Promise<void> {
    nameSignal.set(portfolioName);
    idSignal.set(portfolioName ? 'port-test' : null);
    consoleErrorSpy = spyOn(console, 'error');

    await TestBed.configureTestingModule({
      imports:   [BacktestingComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: noPortfolioContext },
      ],
    }).compileComponents();

    http    = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(BacktestingComponent);
    fixture.detectChanges();
    await fixture.whenStable();
  }

  afterEach(() => {
    http.match(() => true).forEach(r => r.flush({}));
    http.verify();
    TestBed.resetTestingModule();
  });

  // ── No portfolio selected → DEFAULT_TICKERS ───────────────────────────────

  it('when no portfolio is selected in context, tickersRaw() equals DEFAULT_TICKERS joined by ", "', async () => {
    await setup(null);

    expect(fixture.componentInstance.tickersRaw())
      .withContext('tickersRaw must equal DEFAULT_TICKERS when no portfolio is selected')
      .toBe(DEFAULT_TICKERS_STR);
  });

  it('when no portfolio is selected, the component does not attempt a snapshot HTTP request', async () => {
    await setup(null);

    const snapshotReqs = http.match(r =>
      r.url.includes('snapshot') || r.url.includes('latest')
    );
    snapshotReqs.forEach(r => r.flush({}));

    expect(snapshotReqs.length)
      .withContext('no snapshot fetch should happen when there is no selected portfolio')
      .toBe(0);
  });

  // ── API error during snapshot fetch → DEFAULT_TICKERS, silently ───────────

  it('when a portfolio is selected but snapshot fetch returns 500, tickersRaw() falls back to DEFAULT_TICKERS', async () => {
    await setup('MyPortfolio');

    http.match(r => r.url.includes('snapshot') || r.url.includes('latest')).forEach(r =>
      r.flush('Server Error', { status: 500, statusText: 'Server Error' })
    );
    fixture.detectChanges();

    expect(fixture.componentInstance.tickersRaw())
      .withContext('tickersRaw must equal DEFAULT_TICKERS after a snapshot fetch error')
      .toBe(DEFAULT_TICKERS_STR);
  });

  it('when snapshot fetch errors, the error is swallowed silently — console.error is not called', async () => {
    await setup('MyPortfolio');

    http.match(r => r.url.includes('snapshot') || r.url.includes('latest')).forEach(r =>
      r.flush('Not Found', { status: 404, statusText: 'Not Found' })
    );
    fixture.detectChanges();

    expect(consoleErrorSpy)
      .withContext('snapshot error must be caught silently, never surfaced via console.error')
      .not.toHaveBeenCalled();
  });

  it('when snapshot fetch errors, no error alert or toast appears in the DOM', async () => {
    await setup('MyPortfolio');

    http.match(r => r.url.includes('snapshot') || r.url.includes('latest')).forEach(r =>
      r.flush('Not Found', { status: 404, statusText: 'Not Found' })
    );
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    const alertEl =
      el.querySelector('[role="alert"]') ??
      el.querySelector('.toast') ??
      el.querySelector('.error-banner');

    expect(alertEl)
      .withContext('snapshot API errors must not surface an alert or toast — fallback is silent')
      .toBeNull();
  });
});
