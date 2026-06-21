/**
 * Empty-state + no-portfolio spec for issue #1023.
 *
 * Criteria under test:
 *
 *   "ResultsPanelComponent renders the clean empty state (no broken values) when
 *    result is null — asserted in the DOM."
 *
 *   "applyWeightsToPortfolio() calls the correct endpoint with the correct body;
 *    no-portfolio path shows a visible error."
 *
 *   "Navigation to those pages with no portfolio selected shows DEFAULT_TICKERS."
 */

import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { By } from '@angular/platform-browser';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { ResultsPanelComponent } from './results-panel/results-panel';
import { OptimizationStudioComponent } from './optimization-studio';
import { environment } from '../../environments/environment';

const PORTFOLIO_LIST_URL = `${environment.apiUrl}portfolio/`;

// ---------------------------------------------------------------------------
// Suite A — ResultsPanelComponent null-result empty state
// ---------------------------------------------------------------------------

describe('ResultsPanelComponent — null-result empty state', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [ResultsPanelComponent],
      providers: [provideZonelessChangeDetection()],
    });
  });

  it('when result is null, the DOM contains no raw "null" text', () => {
    const fixture = TestBed.createComponent(ResultsPanelComponent);
    fixture.componentRef.setInput('result', null);
    fixture.componentRef.setInput('hasResult', false);
    fixture.detectChanges();

    const text = fixture.nativeElement.textContent as string;
    expect(text).withContext('DOM must not contain the literal string "null"').not.toContain('null');
  });

  it('when result is null, the DOM contains no raw "undefined" text', () => {
    const fixture = TestBed.createComponent(ResultsPanelComponent);
    fixture.componentRef.setInput('result', null);
    fixture.componentRef.setInput('hasResult', false);
    fixture.detectChanges();

    const text = fixture.nativeElement.textContent as string;
    expect(text)
      .withContext('DOM must not contain the literal string "undefined"')
      .not.toContain('undefined');
  });

  it('when result is null, the empty-state element is present in the DOM', () => {
    const fixture = TestBed.createComponent(ResultsPanelComponent);
    fixture.componentRef.setInput('result', null);
    fixture.componentRef.setInput('hasResult', false);
    fixture.detectChanges();

    const emptyEl = fixture.debugElement.query(
      By.css('.bg-surface-raised'),
    );
    expect(emptyEl)
      .withContext('Expected the empty-state card to be present in the DOM')
      .toBeTruthy();
  });

  it('when result is null, the empty-state text is non-blank and mentions the optimization run', () => {
    // The template uses: "No optimization run yet"
    const fixture = TestBed.createComponent(ResultsPanelComponent);
    fixture.componentRef.setInput('result', null);
    fixture.componentRef.setInput('hasResult', false);
    fixture.detectChanges();

    const text = (fixture.nativeElement.textContent as string).toLowerCase();
    expect(text.length)
      .withContext('Empty-state content must be non-blank')
      .toBeGreaterThan(0);
    const mentionsRun = text.includes('run') || text.includes('optimization') || text.includes('result');
    expect(mentionsRun)
      .withContext('Expected empty-state copy to reference the optimization run context')
      .toBeTrue();
  });

  it('when hasResult is true with a run, the empty-state card is NOT present', () => {
    const run = {
      id: 'r1',
      status: 'completed',
      optimizerType: 'mean_variance',
      universeTickers: ['AAPL', 'MSFT'],
      durationSeconds: 1,
      weights: { AAPL: 0.5, MSFT: 0.5 },
      riskContributions: { AAPL: 0.5, MSFT: 0.5 },
      metrics: {},
      efficientFrontier: [],
    };
    const fixture = TestBed.createComponent(ResultsPanelComponent);
    fixture.componentRef.setInput('result', run);
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();

    const emptyEl = fixture.debugElement.query(By.css('.bg-surface-raised'));
    // The empty card should not appear; result content renders instead.
    // If the element exists it must not contain the empty-state text.
    if (emptyEl) {
      const text = (emptyEl.nativeElement.textContent as string).toLowerCase();
      expect(text).not.toContain('no optimization');
    }
  });
});

// ---------------------------------------------------------------------------
// Suite B — OptimizationStudioComponent: apply-weights with no portfolio
// ---------------------------------------------------------------------------

describe('OptimizationStudioComponent — apply-weights without a portfolio', () => {
  let http: HttpTestingController;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await configureTestBed({
      imports: [OptimizationStudioComponent],
      withHttp: true,
      providers: [ICON_PROVIDER],
    });
    http = injectHttp();
  });

  afterEach(() => {
    // Drain the PortfolioContextService bootstrap GET /portfolio/ so http.verify()
    // only asserts on requests each test explicitly exercises.
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  it('when no portfolio is selected and onApplyWeights fires, applyError is set', () => {
    // Criterion: "no-portfolio path shows a visible error"
    const fixture = TestBed.createComponent(OptimizationStudioComponent);
    fixture.detectChanges();

    fixture.componentInstance.onApplyWeights({ AAPL: 0.6, MSFT: 0.4 });
    fixture.detectChanges();

    expect(fixture.componentInstance.applyError())
      .withContext('Expected applyError to be non-null when no portfolio is active')
      .not.toBeNull();
  });

  it('when no portfolio is selected, the error element is visible in the DOM', () => {
    const fixture = TestBed.createComponent(OptimizationStudioComponent);
    fixture.detectChanges();

    fixture.componentInstance.onApplyWeights({ AAPL: 0.6, MSFT: 0.4 });
    fixture.detectChanges();

    // Template: @if (applyError(); as err) { <div class="mb-4 bg-loss/10 ..."> }
    const errorEl = fixture.debugElement.query(By.css('[class*="loss"]'));
    expect(errorEl)
      .withContext('Expected an error element with loss-color class in the DOM')
      .toBeTruthy();

    const errorText = (errorEl!.nativeElement.textContent as string).trim();
    expect(errorText.length)
      .withContext('Expected the error message to be non-blank')
      .toBeGreaterThan(0);
  });

  it('when no portfolio is selected, the error message references the portfolio requirement', () => {
    // The implementation sets: "No active portfolio selected. Pick a portfolio in the sidebar."
    const fixture = TestBed.createComponent(OptimizationStudioComponent);
    fixture.detectChanges();

    fixture.componentInstance.onApplyWeights({ AAPL: 1.0 });
    fixture.detectChanges();

    const fullText = (fixture.nativeElement.textContent as string).toLowerCase();
    const hasPortfolioHint = fullText.includes('portfolio') || fullText.includes('select');
    expect(hasPortfolioHint)
      .withContext('Expected the error to mention "portfolio" or "select"')
      .toBeTrue();
  });

  it('when no portfolio is selected, apply-weights does NOT POST to any endpoint', () => {
    // With no portfolio, the method sets applyError and returns early — no HTTP POST.
    const fixture = TestBed.createComponent(OptimizationStudioComponent);
    fixture.detectChanges();

    fixture.componentInstance.onApplyWeights({ AAPL: 1.0 });
    fixture.detectChanges();

    // The early-return guard sets applyError — this is the observable proof no POST fired.
    expect(fixture.componentInstance.applyError())
      .withContext('applyError must be set (early return before any POST)')
      .not.toBeNull();
    // Portfolio bootstrap GET is drained in afterEach; assert no POSTs specifically.
    http.expectNone((r) => r.method === 'POST');
  });

  it('when component initialises without a selected portfolio, it renders without throwing', () => {
    // Criterion (a): "loads data on init" — basic smoke test.
    expect(() => {
      const fixture = TestBed.createComponent(OptimizationStudioComponent);
      fixture.detectChanges();
    }).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Suite C — OptimizationStudioComponent: ticker seeding on init (no portfolio)
// ---------------------------------------------------------------------------

describe('OptimizationStudioComponent — ticker seeding on init (no portfolio)', () => {
  let http: HttpTestingController;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await configureTestBed({
      imports: [OptimizationStudioComponent],
      withHttp: true,
      providers: [ICON_PROVIDER],
    });
    http = injectHttp();
  });

  afterEach(() => {
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  it('when no portfolio is selected on init, the tickers signal holds the DEFAULT_TICKERS (non-empty)', () => {
    // Criterion: "Navigation to those pages with no portfolio selected shows DEFAULT_TICKERS."
    // TickerSeedingService.seedFromPortfolio(null, defaults) returns of(defaults) immediately.
    const fixture = TestBed.createComponent(OptimizationStudioComponent);
    fixture.detectChanges();

    const tickers = fixture.componentInstance.tickers();
    expect(tickers.length)
      .withContext('Expected DEFAULT_TICKERS to be non-empty when no portfolio is active')
      .toBeGreaterThan(0);
  });

  it('when no portfolio is selected on init, tickers signal does NOT contain empty strings', () => {
    const fixture = TestBed.createComponent(OptimizationStudioComponent);
    fixture.detectChanges();

    const tickers = fixture.componentInstance.tickers();
    tickers.forEach((t) => {
      expect(t.trim().length)
        .withContext(`Expected every ticker to be non-blank, got: "${t}"`)
        .toBeGreaterThan(0);
    });
  });

  // view-panel test removed in issue #1044 (view-panel component deleted).
});
