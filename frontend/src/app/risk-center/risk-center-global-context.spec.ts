/**
 * Source-blind example tests for issue #1014.
 *
 * Derives each assertion from the acceptance criteria text and oracle
 * classification only.  No implementation source was read while authoring
 * this file.
 *
 * Criteria covered here (oracle: UNIT / T3):
 *   [UNIT] risk-center reads portfolioContextService signals; no local
 *          writable selectedPortfolio signal.
 *   [UNIT] No portfolio <select>/dropdown in risk-center.html.
 *   [UNIT] Every panel shows a visible, non-blank error state when its
 *          primary API call returns an error.
 *   [UNIT] No unhandled observable errors reach console.error.
 *   [UNIT] Every wired page: (a) loads data on init, (b) handles API error,
 *          (c) reacts to portfolio context change.
 *   [T3]   Switching the portfolio refetches risk panels with the new name
 *          within one render cycle; clearing fires zero new risk calls.
 *   [T3]   Selecting a portfolio propagates and currentPortfolioName()
 *          returns the correct name within one render cycle.
 *
 * Criteria skipped (oracle: NOT VERIFIABLE):
 *   - Each risk endpoint invoked with NAME not UUID (oracle: NOT VERIFIABLE)
 *   - All tests pass gate (NOT VERIFIABLE: prose)
 *   - SOLID code quality (NOT VERIFIABLE: subjective)
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import {
  provideZonelessChangeDetection,
  signal,
  computed,
  NO_ERRORS_SCHEMA,
} from '@angular/core';
import { of, throwError } from 'rxjs';

import { RiskCenterComponent } from './risk-center';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { RiskService } from './risk.service';
import { ICON_PROVIDER } from '../icons';

// ─── Fixtures ────────────────────────────────────────────────────────────────

const PORTFOLIO_ID = 'f47ac10b-58cc-4372-a567-0e02b2c3d479';
const PORTFOLIO_NAME = 'Global Equity Fund';
const SECOND_NAME = 'European Bonds';
const SECOND_ID = 'b2c3d4e5-0000-0000-0000-000000000002';

const VAR_RESPONSE = {
  var: { '95': 0.023, '99': 0.031 },
  cvar: { '95': 0.031, '99': 0.042 },
  method: 'historical',
  lookback: 252,
  nObservations: 252,
};
const CORRELATION_RESPONSE = {
  assets: ['AAPL', 'MSFT'],
  matrix: [[1, 0.75], [0.75, 1]],
  clusterLabels: [0, 0],
};
const FACTOR_RESPONSE = { exposures: { Momentum: 0.42 }, assetExposures: {} };
const CONCENTRATION_RESPONSE = {
  assets: [],
  summary: { hhi: 0, effectiveN: 0, topNRatio: 0 },
};
const LIQUIDITY_RESPONSE = {
  assets: [],
  summary: { weightedAvgDaysToLiquidate: 0 },
};
const LIMITS_RESPONSE = { items: [], breachCount: 0 };

// ─── Mock factories ───────────────────────────────────────────────────────────

function makePortfolioCtxMock(
  initialId: string | null = PORTFOLIO_ID,
  initialName: string | null = PORTFOLIO_NAME,
) {
  const idSig = signal<string | null>(initialId);
  const nameSig = signal<string | null>(initialName);
  return {
    currentPortfolioId: idSig,
    currentPortfolioName: computed(() => nameSig()),
    selectedPortfolio: computed(() => {
      const id = idSig();
      const name = nameSig();
      return id ? { id, name: name ?? '', benchmark_ticker: 'SPY' } : null;
    }),
    dateRange: signal({
      preset: '1Y' as const,
      start: new Date('2024-01-01'),
      end: new Date('2025-01-01'),
    }),
    benchmark: signal('SPY'),
    hasPortfolio: computed(() => idSig() !== null),
    activeMode: signal('backtest' as const),
    isLive: computed(() => false),
    isBacktest: computed(() => true),
    isPaper: computed(() => false),
    dateRangeLabel: computed(() => '1Y'),
    dateRangeDays: computed(() => 365),
    setPortfolio: jasmine.createSpy('setPortfolio'),
    setMode: jasmine.createSpy('setMode'),
    setPreset: jasmine.createSpy('setPreset'),
    setCustomRange: jasmine.createSpy('setCustomRange'),
    setBenchmark: jasmine.createSpy('setBenchmark'),
    reset: jasmine.createSpy('reset'),
    _nameSig: nameSig,
    _idSig: idSig,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function makeRiskSvcMock(overrides: Partial<Record<string, any>> = {}) {
  return {
    getVar: jasmine.createSpy('getVar').and.returnValue(of(VAR_RESPONSE)),
    getCorrelation: jasmine.createSpy('getCorrelation').and.returnValue(of(CORRELATION_RESPONSE)),
    getFactorExposure: jasmine.createSpy('getFactorExposure').and.returnValue(of(FACTOR_RESPONSE)),
    getConcentration: jasmine.createSpy('getConcentration').and.returnValue(of(CONCENTRATION_RESPONSE)),
    getLiquidity: jasmine.createSpy('getLiquidity').and.returnValue(of(LIQUIDITY_RESPONSE)),
    generateStressScenarios: jasmine.createSpy('generateStressScenarios').and.returnValue(of({ scenarios: [] })),
    listLimits: jasmine.createSpy('listLimits').and.returnValue(of(LIMITS_RESPONSE)),
    createLimit: jasmine.createSpy('createLimit').and.returnValue(of({})),
    updateLimit: jasmine.createSpy('updateLimit').and.returnValue(of({})),
    deleteLimit: jasmine.createSpy('deleteLimit').and.returnValue(of({})),
    ...overrides,
  };
}

function makeProviders(
  ctx: ReturnType<typeof makePortfolioCtxMock>,
  svc: ReturnType<typeof makeRiskSvcMock>,
) {
  return [
    provideZonelessChangeDetection(),
    ICON_PROVIDER,
    { provide: PortfolioContextService, useValue: ctx },
    { provide: RiskService, useValue: svc },
  ];
}

/** Returns all first-argument values from every primary risk API call. */
function collectPrimaryCallFirstArgs(svc: ReturnType<typeof makeRiskSvcMock>): unknown[] {
  return [
    ...svc.getVar.calls.allArgs(),
    ...svc.getCorrelation.calls.allArgs(),
    ...svc.getFactorExposure.calls.allArgs(),
    ...svc.getConcentration.calls.allArgs(),
    ...svc.getLiquidity.calls.allArgs(),
  ].map((args) => args[0]);
}

function totalPrimaryCallCount(svc: ReturnType<typeof makeRiskSvcMock>): number {
  return (
    svc.getVar.calls.count() +
    svc.getCorrelation.calls.count() +
    svc.getFactorExposure.calls.count() +
    svc.getConcentration.calls.count() +
    svc.getLiquidity.calls.count()
  );
}

function resetPrimaryCallCounts(svc: ReturnType<typeof makeRiskSvcMock>): void {
  svc.getVar.calls.reset();
  svc.getCorrelation.calls.reset();
  svc.getFactorExposure.calls.reset();
  svc.getConcentration.calls.reset();
  svc.getLiquidity.calls.reset();
  svc.listLimits.calls.reset();
}

// ─── Suite: no portfolio dropdown in template ──────────────────────────────

describe('RiskCenterComponent — no portfolio dropdown in template (issue #1014 [UNIT])', () => {
  /**
   * Criterion: "No portfolio <select>/dropdown element exists in risk-center.html;
   * selection happens only via the global picker."
   *
   * Approach: render the component with NO_ERRORS_SCHEMA and inspect the live DOM
   * for any <select> whose accessible label indicates portfolio selection.
   * Non-portfolio controls (e.g. VaR method, confidence level) are permitted.
   */
  let fixture: ComponentFixture<RiskCenterComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(makePortfolioCtxMock(), makeRiskSvcMock()),
    }).compileComponents();

    fixture = TestBed.createComponent(RiskCenterComponent);
    fixture.detectChanges();
  });

  it('when risk-center renders, no <select> element is labelled as a portfolio selector', () => {
    const el = fixture.nativeElement as HTMLElement;
    const selects = Array.from(el.querySelectorAll('select'));
    const portfolioSelect = selects.find((s) => {
      const ariaLabel = (s.getAttribute('aria-label') ?? '').toLowerCase();
      const name = (s.getAttribute('name') ?? '').toLowerCase();
      const id = (s.getAttribute('id') ?? '').toLowerCase();
      return (
        ariaLabel.includes('portfolio') ||
        name.includes('portfolio') ||
        id.includes('portfolio')
      );
    });
    expect(portfolioSelect)
      .withContext('Found a portfolio-labelled <select>; must use the global picker instead')
      .toBeUndefined();
  });
});

// ─── Suite: loads data on init (T3 criterion a) ───────────────────────────────

describe('RiskCenterComponent — loads data on init (issue #1014 [UNIT])', () => {
  /**
   * Criterion (T3 / [UNIT]):
   * "Every wired page has specs covering: (a) loads data on init."
   * "Risk-center: VaR, correlation, factor-exposure, concentration, and liquidity
   * panels all render when the portfolio has data."
   */

  it('when risk-center initialises with a portfolio in context, risk API calls are made', async () => {
    const svc = makeRiskSvcMock();

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(makePortfolioCtxMock(), svc),
    }).compileComponents();

    TestBed.createComponent(RiskCenterComponent).detectChanges();

    expect(totalPrimaryCallCount(svc))
      .withContext('Expected risk API calls to fire on init when portfolio is present')
      .toBeGreaterThan(0);
  });

  it('when risk-center initialises with no portfolio in context, no risk API calls are made', async () => {
    const emptyCtx = makePortfolioCtxMock(null, null);
    const svc = makeRiskSvcMock();

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(emptyCtx, svc),
    }).compileComponents();

    TestBed.createComponent(RiskCenterComponent).detectChanges();

    expect(totalPrimaryCallCount(svc))
      .withContext('Expected zero risk calls when no portfolio is selected')
      .toBe(0);
  });

  it('when risk-center initialises, getVar is among the API calls made', async () => {
    const svc = makeRiskSvcMock();

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(makePortfolioCtxMock(), svc),
    }).compileComponents();

    TestBed.createComponent(RiskCenterComponent).detectChanges();

    expect(svc.getVar.calls.count())
      .withContext('Expected getVar to be called during init')
      .toBeGreaterThan(0);
  });

  it('when risk-center initialises, getCorrelation is among the API calls made', async () => {
    const svc = makeRiskSvcMock();

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(makePortfolioCtxMock(), svc),
    }).compileComponents();

    TestBed.createComponent(RiskCenterComponent).detectChanges();

    expect(svc.getCorrelation.calls.count())
      .withContext('Expected getCorrelation to be called during init')
      .toBeGreaterThan(0);
  });

  it('when risk-center initialises, getFactorExposure is among the API calls made', async () => {
    const svc = makeRiskSvcMock();

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(makePortfolioCtxMock(), svc),
    }).compileComponents();

    TestBed.createComponent(RiskCenterComponent).detectChanges();

    expect(svc.getFactorExposure.calls.count())
      .withContext('Expected getFactorExposure to be called during init')
      .toBeGreaterThan(0);
  });

  it('when risk-center initialises, getConcentration is among the API calls made', async () => {
    const svc = makeRiskSvcMock();

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(makePortfolioCtxMock(), svc),
    }).compileComponents();

    TestBed.createComponent(RiskCenterComponent).detectChanges();

    expect(svc.getConcentration.calls.count())
      .withContext('Expected getConcentration to be called during init')
      .toBeGreaterThan(0);
  });

  it('when risk-center initialises, getLiquidity is among the API calls made', async () => {
    const svc = makeRiskSvcMock();

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(makePortfolioCtxMock(), svc),
    }).compileComponents();

    TestBed.createComponent(RiskCenterComponent).detectChanges();

    expect(svc.getLiquidity.calls.count())
      .withContext('Expected getLiquidity to be called during init')
      .toBeGreaterThan(0);
  });
});

// ─── Suite: error state when API fails (criterion b) ─────────────────────────

describe('RiskCenterComponent — error state when primary API fails (issue #1014 [UNIT])', () => {
  /**
   * Criterion ([UNIT]):
   * "Every page/panel shows a visible, non-blank error state (message or alert
   * component) when its primary API call returns an error."
   * "Every wired page has specs covering: (b) handles API error."
   */

  let fixture: ComponentFixture<RiskCenterComponent>;

  beforeEach(async () => {
    const errorSvc = makeRiskSvcMock({
      getVar: jasmine.createSpy('getVar').and.returnValue(
        throwError(() => new Error('500 Internal Server Error')),
      ),
      getCorrelation: jasmine.createSpy('getCorrelation').and.returnValue(
        throwError(() => new Error('500 Internal Server Error')),
      ),
      getFactorExposure: jasmine.createSpy('getFactorExposure').and.returnValue(
        throwError(() => new Error('500 Internal Server Error')),
      ),
      getConcentration: jasmine.createSpy('getConcentration').and.returnValue(
        throwError(() => new Error('500 Internal Server Error')),
      ),
      getLiquidity: jasmine.createSpy('getLiquidity').and.returnValue(
        throwError(() => new Error('500 Internal Server Error')),
      ),
      listLimits: jasmine.createSpy('listLimits').and.returnValue(
        throwError(() => new Error('500 Internal Server Error')),
      ),
    });

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(makePortfolioCtxMock(), errorSvc),
    }).compileComponents();

    fixture = TestBed.createComponent(RiskCenterComponent);
    fixture.detectChanges();
  });

  it('when primary risk APIs return errors, a role="alert" element is present in the DOM', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    const alertEl =
      el.querySelector('[role="alert"]') ??
      el.querySelector('app-page-error-banner') ??
      el.querySelector('app-alert-banner');

    expect(alertEl)
      .withContext('Expected a visible error indicator in the DOM')
      .not.toBeNull();
  });

  it('when primary risk APIs return errors, the error element contains non-blank text', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    const alertEl =
      el.querySelector('[role="alert"]') ??
      el.querySelector('app-page-error-banner') ??
      el.querySelector('app-alert-banner');

    expect(alertEl).not.toBeNull();
    const text = alertEl?.textContent?.trim() ?? '';
    expect(text.length)
      .withContext('Error element must contain non-blank text')
      .toBeGreaterThan(0);
  });
});

// ─── Suite: no unhandled observable errors to console.error ──────────────────

describe('RiskCenterComponent — no unhandled observable errors reach console.error (issue #1014 [UNIT])', () => {
  /**
   * Criterion ([UNIT]):
   * "No unhandled observable errors reach console.error."
   */

  it('when risk APIs return errors, console.error is not called', async () => {
    const consoleErrorSpy = spyOn(console, 'error');

    const errorSvc = makeRiskSvcMock({
      getVar: jasmine.createSpy('getVar').and.returnValue(
        throwError(() => new Error('500 Internal Server Error')),
      ),
      getCorrelation: jasmine.createSpy('getCorrelation').and.returnValue(
        throwError(() => new Error('500 Internal Server Error')),
      ),
      getFactorExposure: jasmine.createSpy('getFactorExposure').and.returnValue(
        throwError(() => new Error('500 Internal Server Error')),
      ),
      getConcentration: jasmine.createSpy('getConcentration').and.returnValue(
        throwError(() => new Error('500 Internal Server Error')),
      ),
      getLiquidity: jasmine.createSpy('getLiquidity').and.returnValue(
        throwError(() => new Error('500 Internal Server Error')),
      ),
      listLimits: jasmine.createSpy('listLimits').and.returnValue(
        throwError(() => new Error('500 Internal Server Error')),
      ),
    });

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(makePortfolioCtxMock(), errorSvc),
    }).compileComponents();

    const fixture = TestBed.createComponent(RiskCenterComponent);
    fixture.detectChanges();

    expect(consoleErrorSpy)
      .withContext('Unhandled observable error reached console.error')
      .not.toHaveBeenCalled();
  });
});

// ─── Suite: portfolio context change refetches panels (criterion c) ───────────

describe('RiskCenterComponent — reacts to portfolio context change (issue #1014 [T3])', () => {
  /**
   * Criterion ([T3]):
   * "Switching the portfolio in the global nav refetches the risk panels with
   * the new name within one render cycle; clearing it fires zero new risk calls."
   *
   * Criterion ([T3]):
   * "Selecting a portfolio in the header propagates to all pages without
   * additional selection steps."
   *
   * Criterion ([UNIT]):
   * "Every wired page has specs covering: (c) reacts to portfolio context change."
   */

  let ctx: ReturnType<typeof makePortfolioCtxMock>;
  let svc: ReturnType<typeof makeRiskSvcMock>;
  let fixture: ComponentFixture<RiskCenterComponent>;

  beforeEach(async () => {
    ctx = makePortfolioCtxMock();
    svc = makeRiskSvcMock();

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(ctx, svc),
    }).compileComponents();

    fixture = TestBed.createComponent(RiskCenterComponent);
    fixture.detectChanges();

    // Isolate: reset counters so only post-switch calls are measured.
    resetPrimaryCallCounts(svc);
  });

  it('when the portfolio name changes, risk APIs are called again within one render cycle', () => {
    ctx._nameSig.set(SECOND_NAME);
    ctx._idSig.set(SECOND_ID);
    fixture.detectChanges();

    expect(totalPrimaryCallCount(svc))
      .withContext('Expected risk API calls after portfolio switch')
      .toBeGreaterThan(0);
  });

  it('when the portfolio name changes, the refetch uses the new name', () => {
    ctx._nameSig.set(SECOND_NAME);
    ctx._idSig.set(SECOND_ID);
    fixture.detectChanges();

    const firstArgs = collectPrimaryCallFirstArgs(svc);
    expect(firstArgs.some((arg) => arg === SECOND_NAME))
      .withContext(`Expected at least one call with '${SECOND_NAME}'`)
      .toBeTrue();
  });

  it('when the portfolio name changes, the refetch does not use the old name', () => {
    ctx._nameSig.set(SECOND_NAME);
    ctx._idSig.set(SECOND_ID);
    fixture.detectChanges();

    const firstArgs = collectPrimaryCallFirstArgs(svc);
    expect(firstArgs.some((arg) => arg === PORTFOLIO_NAME))
      .withContext('Expected no calls with the old portfolio name after switch')
      .toBeFalse();
  });

  it('when the portfolio is cleared, zero new risk API calls are fired', () => {
    ctx._nameSig.set(null);
    ctx._idSig.set(null);
    fixture.detectChanges();

    expect(totalPrimaryCallCount(svc))
      .withContext('Expected zero risk calls after portfolio is cleared')
      .toBe(0);
  });
});

// ─── Suite: context service drives portfolio identity (no local override) ─────

describe('RiskCenterComponent — context service is the sole source of portfolio identity (issue #1014 [UNIT])', () => {
  /**
   * Criterion ([UNIT]):
   * "risk-center reads portfolioContextService.selectedPortfolio() /
   * currentPortfolioName() and holds no local writable selectedPortfolio signal."
   *
   * Behavioural proxy: changing only the context service signals updates the
   * component's behaviour (i.e. triggers a fresh fetch with the new name) without
   * any additional local-state-setting step.  If the component maintained a
   * divergent local writable signal, this test would be the regression gate for
   * detecting that divergence.
   */

  it('when context name changes without any local-state mutation, the component fetches with the new name', async () => {
    const ctx = makePortfolioCtxMock();
    const svc = makeRiskSvcMock();

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(ctx, svc),
    }).compileComponents();

    const fixture = TestBed.createComponent(RiskCenterComponent);
    fixture.detectChanges();

    resetPrimaryCallCounts(svc);

    // Change ONLY via context service signals — no local component method called.
    ctx._nameSig.set(SECOND_NAME);
    ctx._idSig.set(SECOND_ID);
    fixture.detectChanges();

    const firstArgs = collectPrimaryCallFirstArgs(svc);
    expect(firstArgs.some((arg) => arg === SECOND_NAME))
      .withContext('Component must react to context signal change with no local-state shim')
      .toBeTrue();
  });

  it('when portfolio context resolves name to null, the component makes no risk calls regardless of previous state', async () => {
    const ctx = makePortfolioCtxMock();
    const svc = makeRiskSvcMock();

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(ctx, svc),
    }).compileComponents();

    const fixture = TestBed.createComponent(RiskCenterComponent);
    fixture.detectChanges();

    resetPrimaryCallCounts(svc);

    ctx._nameSig.set(null);
    ctx._idSig.set(null);
    fixture.detectChanges();

    expect(totalPrimaryCallCount(svc))
      .withContext('Clearing context must not trigger any risk API call')
      .toBe(0);
  });
});

// ─── Suite: null-safety — empty API responses must not throw ─────────────────

describe('RiskCenterComponent — empty API responses do not throw (issue #1014 [UNIT])', () => {
  /**
   * Criterion ([UNIT]):
   * "Risk-center: VaR, correlation heatmap, factor exposure, concentration, and
   * liquidity panels all render when the portfolio has data."
   *
   * Complementary guard: the component must also not crash when responses
   * contain empty arrays/objects.
   */

  it('when all risk APIs return empty payloads, the component renders without throwing', async () => {
    const emptySvc = makeRiskSvcMock({
      getVar: jasmine.createSpy('getVar').and.returnValue(
        of({ var: {}, cvar: {}, method: 'historical', lookback: 252, nObservations: 0 }),
      ),
      getCorrelation: jasmine.createSpy('getCorrelation').and.returnValue(
        of({ assets: [], matrix: [], clusterLabels: [] }),
      ),
      getFactorExposure: jasmine.createSpy('getFactorExposure').and.returnValue(
        of({ exposures: {}, assetExposures: {} }),
      ),
      getConcentration: jasmine.createSpy('getConcentration').and.returnValue(
        of({ assets: [], summary: { hhi: 0, effectiveN: 0, topNRatio: 0 } }),
      ),
      getLiquidity: jasmine.createSpy('getLiquidity').and.returnValue(
        of({ assets: [], summary: { weightedAvgDaysToLiquidate: 0 } }),
      ),
      listLimits: jasmine.createSpy('listLimits').and.returnValue(
        of({ items: [], breachCount: 0 }),
      ),
    });

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(makePortfolioCtxMock(), emptySvc),
    }).compileComponents();

    const fixture = TestBed.createComponent(RiskCenterComponent);
    expect(() => fixture.detectChanges())
      .withContext('Component must not throw when risk APIs return empty payloads')
      .not.toThrow();
  });
});

// ─── Suite: cold-boot guard — name-only fetch gate ───────────────────────────

describe('RiskCenterComponent — cold-boot: no risk calls fire before portfolio name resolves (issue #1014 [UNIT])', () => {
  /**
   * Cold-boot scenario described in the issue context:
   * On a returning user load, currentPortfolioId() is restored from localStorage
   * synchronously, but currentPortfolioName() only resolves after the portfolio
   * list is fetched.  During that window (id known, name still null), the
   * component must fire ZERO risk endpoint calls to avoid 404s on UUID-based paths.
   *
   * This criterion was surfaced in the Phase 2 review as a potential bug:
   * if the component uses `name ?? id` as a trigger, the first effect run leaks
   * the raw UUID to name-keyed risk endpoints.  This test gates that path.
   *
   * Source of derivation: issue body "Shared cold-boot risk to verify: because
   * the list resolves lazily, on a cold load the id is known before the name."
   */

  it('when portfolio id is set but name has not resolved yet, no risk API calls are made', async () => {
    // id present, name still null — simulates the cold-boot window
    const coldBootCtx = makePortfolioCtxMock(PORTFOLIO_ID, null);
    const svc = makeRiskSvcMock();

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(coldBootCtx, svc),
    }).compileComponents();

    TestBed.createComponent(RiskCenterComponent).detectChanges();

    expect(totalPrimaryCallCount(svc))
      .withContext(
        'Risk APIs must not fire during the cold-boot window where id is known but name has not resolved',
      )
      .toBe(0);
  });

  it('when the portfolio name resolves after a cold-boot delay, risk APIs fire with the name (not the UUID)', async () => {
    // Start with id only (cold boot)
    const coldBootCtx = makePortfolioCtxMock(PORTFOLIO_ID, null);
    const svc = makeRiskSvcMock();

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: makeProviders(coldBootCtx, svc),
    }).compileComponents();

    const fixture = TestBed.createComponent(RiskCenterComponent);
    fixture.detectChanges(); // id known, name null — must not fetch

    // Name resolves (portfolio list arrives)
    coldBootCtx._nameSig.set(PORTFOLIO_NAME);
    fixture.detectChanges();

    const firstArgs = collectPrimaryCallFirstArgs(svc);
    expect(firstArgs.length)
      .withContext('Risk APIs must fire once name resolves')
      .toBeGreaterThan(0);
    expect(firstArgs.every((arg) => arg === PORTFOLIO_NAME))
      .withContext('All risk API calls must use the resolved name, never the raw UUID')
      .toBeTrue();
  });
});
