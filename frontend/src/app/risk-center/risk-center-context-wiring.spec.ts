/**
 * Source-blind contract tests for issue #959.
 *
 * T3 criteria pinned:
 *   (a) loads data on init — risk API calls use the portfolio NAME, not the raw UUID
 *   (b) handles API error — a visible, non-blank error state is shown when the
 *       primary API call returns an error
 *   (c) reacts to portfolio context change — switching portfolio refetches with the
 *       new portfolio name
 *
 * Panels in scope: VaR, correlation heatmap, factor exposure, concentration, liquidity.
 *
 * NOT-VERIFIABLE criteria intentionally omitted:
 *   - 3-second SLA for chart rendering
 *   - individual chart sub-renders (visual)
 *   - console.error suppression
 *   - navigation pre-fill of tickers
 *
 * Assumptions recorded (source-blind guesses that can be corrected without
 * changing the test intent):
 *   - Component class: RiskCenterComponent in ./risk-center
 *   - Primary service: RiskService (injected by the component)
 *   - Primary load method: loadRiskData(portfolioName, ...) — may be split into
 *     per-panel calls; mock pattern covers both because all spies share the same
 *     portfolio-name assertion
 *   - Error state uses <app-page-error-banner> or any element with role="alert"
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection, signal, computed, NO_ERRORS_SCHEMA } from '@angular/core';
import { of, throwError } from 'rxjs';
import { By } from '@angular/platform-browser';

import { RiskCenterComponent } from './risk-center';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { RiskService } from './risk.service';
import { ICON_PROVIDER } from '../icons';

// ─── Test constants ──────────────────────────────────────────────────────────

const PORTFOLIO_ID   = 'f47ac10b-58cc-4372-a567-0e02b2c3d479';
const PORTFOLIO_NAME = 'Global Equity Fund';

// Minimal stub responses — corrected to match actual VarApiResponse / service shapes.
const VAR_RESPONSE = {
  var: { '95': 0.023, '99': 0.031 },
  cvar: { '95': 0.031, '99': 0.042 },
  method: 'historical',
  lookback: 252,
  nObservations: 252,
};

const CORRELATION_RESPONSE = {
  assets: ['AAPL', 'MSFT', 'GOOGL'],
  matrix: [[1, 0.75, 0.68], [0.75, 1, 0.72], [0.68, 0.72, 1]],
  clusterLabels: [0, 0, 1],
};

const FACTOR_EXPOSURE_RESPONSE = {
  exposures: { Momentum: 0.42, Value: -0.18, Quality: 0.31 },
  assetExposures: {},
};

const CONCENTRATION_RESPONSE = {
  assets: [],
  summary: { hhi: 0.12, effectiveN: 8, topNRatio: 0.48 },
};

const LIQUIDITY_RESPONSE = {
  assets: [],
  summary: { weightedAvgDaysToLiquidate: 1.2 },
};

// ─── Mock factories ───────────────────────────────────────────────────────────

function makePortfolioCtxMock(
  id: string | null = PORTFOLIO_ID,
  name: string | null = PORTFOLIO_NAME,
) {
  const idSig = signal<string | null>(id);
  const nameSig = signal<string | null>(name);
  return {
    currentPortfolioId:   idSig,
    currentPortfolioName: computed(() => nameSig()),
    selectedPortfolio:    computed(() => (id ? { id, name: name ?? '' } : null)),
    dateRange: signal({
      preset: '1Y' as const,
      start: new Date('2024-01-01'),
      end:   new Date('2025-01-01'),
    }),
    benchmark:      signal('SPY'),
    hasPortfolio:   computed(() => id !== null),
    activeMode:     signal('backtest' as const),
    isLive:         computed(() => false),
    isBacktest:     computed(() => true),
    isPaper:        computed(() => false),
    dateRangeLabel: computed(() => '1Y'),
    dateRangeDays:  computed(() => 365),
    setPortfolio:   jasmine.createSpy('setPortfolio'),
    setMode:        jasmine.createSpy('setMode'),
    setPreset:      jasmine.createSpy('setPreset'),
    setCustomRange: jasmine.createSpy('setCustomRange'),
    setBenchmark:   jasmine.createSpy('setBenchmark'),
    reset:          jasmine.createSpy('reset'),
    // writable signal exposed so tests can switch portfolio
    _nameSig: nameSig,
    _idSig:   idSig,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function makeRiskSvcMock(
  varReturn:       any = of(VAR_RESPONSE),
  corrReturn:      any = of(CORRELATION_RESPONSE),
  factorReturn:    any = of(FACTOR_EXPOSURE_RESPONSE),
  concReturn:      any = of(CONCENTRATION_RESPONSE),
  liquidityReturn: any = of(LIQUIDITY_RESPONSE),
) {
  return {
    // Actual RiskService method names (source-blind guesses corrected to match impl)
    getVar:            jasmine.createSpy('getVar').and.returnValue(varReturn),
    getCorrelation:    jasmine.createSpy('getCorrelation').and.returnValue(corrReturn),
    getFactorExposure: jasmine.createSpy('getFactorExposure').and.returnValue(factorReturn),
    getConcentration:  jasmine.createSpy('getConcentration').and.returnValue(concReturn),
    getLiquidity:      jasmine.createSpy('getLiquidity').and.returnValue(liquidityReturn),
    listLimits:        jasmine.createSpy('listLimits').and.returnValue(of({ items: [], breachCount: 0 })),
    generateStressScenarios: jasmine.createSpy('generateStressScenarios').and.returnValue(of({})),
    createLimit:       jasmine.createSpy('createLimit').and.returnValue(of({})),
    updateLimit:       jasmine.createSpy('updateLimit').and.returnValue(of({})),
    deleteLimit:       jasmine.createSpy('deleteLimit').and.returnValue(of({})),
  };
}

// Common providers array reused across suites.
function riskProviders(
  ctx: ReturnType<typeof makePortfolioCtxMock>,
  svc: ReturnType<typeof makeRiskSvcMock>,
) {
  return [
    provideZonelessChangeDetection(),
    ICON_PROVIDER,
    { provide: PortfolioContextService, useValue: ctx },
    { provide: RiskService,             useValue: svc },
  ];
}

// ─── Suite: happy-path name wiring ───────────────────────────────────────────

describe('RiskCenterComponent — analytics use portfolio NAME, not UUID (issue #959)', () => {
  let portfolioCtxMock: ReturnType<typeof makePortfolioCtxMock>;
  let riskSvcMock: ReturnType<typeof makeRiskSvcMock>;
  let fixture: ComponentFixture<RiskCenterComponent>;

  beforeEach(async () => {
    portfolioCtxMock = makePortfolioCtxMock();
    riskSvcMock      = makeRiskSvcMock();

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: riskProviders(portfolioCtxMock, riskSvcMock),
    }).compileComponents();

    fixture = TestBed.createComponent(RiskCenterComponent);
    fixture.detectChanges();
  });

  // ── T3(a): loads data on init using the NAME ────────────────────────────

  it('when portfolio context has a resolved name, risk API is not called with the raw UUID', () => {
    // At least one risk service method must have been called;
    // none of them should receive the raw UUID as portfolio identifier.
    const allCalls = [
      ...riskSvcMock.getVar.calls.allArgs(),
      ...riskSvcMock.getCorrelation.calls.allArgs(),
      ...riskSvcMock.getFactorExposure.calls.allArgs(),
      ...riskSvcMock.getConcentration.calls.allArgs(),
      ...riskSvcMock.getLiquidity.calls.allArgs(),
    ] as unknown[][];

    // At least one service method must have been invoked.
    expect(allCalls.length).toBeGreaterThan(0);

    // The first argument to every call must not be the raw UUID.
    for (const args of allCalls) {
      expect(args[0]).not.toBe(PORTFOLIO_ID);
    }
  });

  it('when portfolio context has a resolved name, risk API is called with the portfolio name', () => {
    const nameCalls = [
      ...riskSvcMock.getVar.calls.allArgs(),
      ...riskSvcMock.getCorrelation.calls.allArgs(),
      ...riskSvcMock.getFactorExposure.calls.allArgs(),
      ...riskSvcMock.getConcentration.calls.allArgs(),
      ...riskSvcMock.getLiquidity.calls.allArgs(),
    ] as unknown[][];

    const calledWithName = nameCalls.some((args) => args[0] === PORTFOLIO_NAME);
    expect(calledWithName).toBeTrue();
  });

  it('when portfolio context has a resolved name, listLimits is called with the portfolio name', () => {
    // The limits endpoint is the 6th risk endpoint; it must be scoped to the
    // context name like the five analytics endpoints. (Stress, the 7th, is
    // user-triggered and scopes via its request body, not a name in the path.)
    expect(riskSvcMock.listLimits).toHaveBeenCalledWith(PORTFOLIO_NAME);
  });

});

// ─── Suite: null-safety — empty API responses must not throw ─────────────────

describe('RiskCenterComponent — empty API responses do not throw (issue #959)', () => {
  it('when risk API returns empty collections, component renders without throwing', async () => {
    const emptyCtx = makePortfolioCtxMock();
    const emptySvc = makeRiskSvcMock(
      of({ var: {}, cvar: {}, method: 'historical', lookback: 252, nObservations: 0 }),
      of({ assets: [], matrix: [], clusterLabels: [] }),
      of({ exposures: {}, assetExposures: {} }),
      of({ assets: [], summary: { hhi: 0, effectiveN: 0, topNRatio: 0 } }),
      of({ assets: [], summary: { weightedAvgDaysToLiquidate: 0 } }),
    );

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: riskProviders(emptyCtx, emptySvc),
    }).compileComponents();

    const f = TestBed.createComponent(RiskCenterComponent);
    expect(() => f.detectChanges()).not.toThrow();
  });
});

// ─── Suite: API error → visible error state ───────────────────────────────────

describe('RiskCenterComponent — error state when primary risk API fails (issue #959)', () => {
  let fixture: ComponentFixture<RiskCenterComponent>;

  beforeEach(async () => {
    const errorCtx = makePortfolioCtxMock();
    const errorSvc = makeRiskSvcMock(
      throwError(() => new Error('500 Internal Server Error')),
      throwError(() => new Error('500 Internal Server Error')),
      throwError(() => new Error('500 Internal Server Error')),
      throwError(() => new Error('500 Internal Server Error')),
      throwError(() => new Error('500 Internal Server Error')),
    );

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: riskProviders(errorCtx, errorSvc),
    }).compileComponents();

    fixture = TestBed.createComponent(RiskCenterComponent);
    fixture.detectChanges();
  });

  // ── T3(b): visible non-blank error state ────────────────────────────────

  it('when primary risk API returns an error, a role="alert" element is present in the DOM', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    const alertEl =
      el.querySelector('[role="alert"]') ??
      el.querySelector('app-page-error-banner') ??
      el.querySelector('app-alert-banner');

    expect(alertEl).not.toBeNull();
  });

  it('when primary risk API returns an error, the error region contains non-blank text', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    const alertEl =
      el.querySelector('[role="alert"]') ??
      el.querySelector('app-page-error-banner') ??
      el.querySelector('app-alert-banner');

    expect(alertEl).not.toBeNull();
    const text = alertEl?.textContent?.trim() ?? '';
    expect(text.length).toBeGreaterThan(0);
  });
});

// ─── Suite: portfolio context change → refetch ───────────────────────────────

describe('RiskCenterComponent — reacts to portfolio context change (issue #959)', () => {
  let portfolioCtxMock: ReturnType<typeof makePortfolioCtxMock>;
  let riskSvcMock: ReturnType<typeof makeRiskSvcMock>;
  let fixture: ComponentFixture<RiskCenterComponent>;

  const SECOND_PORTFOLIO_NAME = 'European Bonds';
  const SECOND_PORTFOLIO_ID   = 'a1b2c3d4-0000-0000-0000-000000000001';

  beforeEach(async () => {
    portfolioCtxMock = makePortfolioCtxMock();
    riskSvcMock      = makeRiskSvcMock();

    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: riskProviders(portfolioCtxMock, riskSvcMock),
    }).compileComponents();

    fixture = TestBed.createComponent(RiskCenterComponent);
    fixture.detectChanges();

    // Drain initial calls so assertion counts are clear.
    riskSvcMock.getVar.calls.reset();
    riskSvcMock.getCorrelation.calls.reset();
    riskSvcMock.getFactorExposure.calls.reset();
    riskSvcMock.getConcentration.calls.reset();
    riskSvcMock.getLiquidity.calls.reset();
  });

  // ── T3(c): reacts to portfolio context change ───────────────────────────

  it('when portfolio name changes, risk API is called again', () => {
    // Switch to second portfolio.
    portfolioCtxMock._nameSig.set(SECOND_PORTFOLIO_NAME);
    portfolioCtxMock._idSig.set(SECOND_PORTFOLIO_ID);
    fixture.detectChanges();

    const totalCalls =
      riskSvcMock.getVar.calls.count() +
      riskSvcMock.getCorrelation.calls.count() +
      riskSvcMock.getFactorExposure.calls.count() +
      riskSvcMock.getConcentration.calls.count() +
      riskSvcMock.getLiquidity.calls.count();

    expect(totalCalls).toBeGreaterThan(0);
  });

  it('when portfolio name changes, the new API call uses the new name, not the old one', () => {
    portfolioCtxMock._nameSig.set(SECOND_PORTFOLIO_NAME);
    portfolioCtxMock._idSig.set(SECOND_PORTFOLIO_ID);
    fixture.detectChanges();

    const allNewArgs = [
      ...riskSvcMock.getVar.calls.allArgs(),
      ...riskSvcMock.getCorrelation.calls.allArgs(),
      ...riskSvcMock.getFactorExposure.calls.allArgs(),
      ...riskSvcMock.getConcentration.calls.allArgs(),
      ...riskSvcMock.getLiquidity.calls.allArgs(),
    ] as unknown[][];

    const calledWithNew = allNewArgs.some((args) => args[0] === SECOND_PORTFOLIO_NAME);
    expect(calledWithNew).toBeTrue();
  });

  it('when portfolio is cleared, risk API is not called again with any identifier', () => {
    portfolioCtxMock._nameSig.set(null);
    portfolioCtxMock._idSig.set(null);
    fixture.detectChanges();

    // Calls AFTER the reset: the component guards on null name, so no new calls fire.
    const totalCallsAfterClear =
      riskSvcMock.getVar.calls.count() +
      riskSvcMock.getCorrelation.calls.count() +
      riskSvcMock.getFactorExposure.calls.count() +
      riskSvcMock.getConcentration.calls.count() +
      riskSvcMock.getLiquidity.calls.count();

    // No new service calls should have been triggered after clearing the portfolio.
    expect(totalCallsAfterClear).toBe(0);
  });
});
