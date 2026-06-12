/**
 * Source-blind contract tests for issue #960.
 *
 * T3 criteria pinned:
 *   (a) loads data on init — attribution API calls use the portfolio NAME, not the raw UUID
 *   (b) handles API error — a visible, non-blank error state is shown when the
 *       primary API call returns an error
 *   (c) reacts to portfolio context change — switching portfolio refetches with the
 *       new portfolio name
 *
 * Panels in scope: Brinson waterfall, factor attribution bars.
 *
 * NOT-VERIFIABLE criteria intentionally omitted:
 *   - 3-second SLA for chart rendering
 *   - visual rendering of individual sub-charts
 *   - console.error suppression
 *   - navigation pre-fill of tickers
 *
 * Assumptions recorded (source-blind guesses correctable without changing test intent):
 *   - Component class: AttributionComponent in ./attribution
 *   - Primary service: AttributionService injected by the component
 *   - Load methods: getBrinsonAttribution and/or getFactorAttribution (or a single
 *     loadAttributionData bundle); mock covers both because all spies share the same
 *     portfolio-name assertion
 *   - Error state uses <app-page-error-banner> or any element with role="alert"
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection, signal, computed, NO_ERRORS_SCHEMA } from '@angular/core';
import { of, throwError } from 'rxjs';

import { AttributionComponent } from './attribution';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { AttributionService } from './attribution.service';
import { ICON_PROVIDER } from '../icons';

// ─── Test constants ──────────────────────────────────────────────────────────

const PORTFOLIO_ID   = 'f47ac10b-58cc-4372-a567-0e02b2c3d479';
const PORTFOLIO_NAME = 'Global Equity Fund';

// Minimal stub responses shaped after the panel names in the acceptance criteria.
const BRINSON_RESPONSE = {
  periods: [],
  summary: { allocationEffect: 0.012, selectionEffect: -0.004, totalEffect: 0.008 },
};

const FACTOR_RESPONSE = {
  factors: [
    { name: 'Momentum', contribution: 0.006 },
    { name: 'Value',    contribution: -0.002 },
    { name: 'Quality',  contribution: 0.004 },
  ],
  totalFactorReturn: 0.008,
};

// ─── Mock factories ───────────────────────────────────────────────────────────

function makePortfolioCtxMock(
  id: string | null = PORTFOLIO_ID,
  name: string | null = PORTFOLIO_NAME,
) {
  const idSig   = signal<string | null>(id);
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
    _nameSig: nameSig,
    _idSig:   idSig,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function makeAttrSvcMock(
  brinsonReturn: any = of(BRINSON_RESPONSE),
  factorReturn:  any = of(FACTOR_RESPONSE),
) {
  return {
    getBrinsonAttribution:  jasmine.createSpy('getBrinsonAttribution').and.returnValue(brinsonReturn),
    getFactorAttribution:   jasmine.createSpy('getFactorAttribution').and.returnValue(factorReturn),
    // Bundled variant some implementations use instead of per-panel calls
    loadAttributionData:    jasmine.createSpy('loadAttributionData').and.returnValue(of({
      brinson: BRINSON_RESPONSE,
      factor:  FACTOR_RESPONSE,
    })),
    getAttribution:         jasmine.createSpy('getAttribution').and.returnValue(of({
      brinson: BRINSON_RESPONSE,
      factor:  FACTOR_RESPONSE,
    })),
  };
}

function attrProviders(
  ctx: ReturnType<typeof makePortfolioCtxMock>,
  svc: ReturnType<typeof makeAttrSvcMock>,
) {
  return [
    provideZonelessChangeDetection(),
    ICON_PROVIDER,
    { provide: PortfolioContextService, useValue: ctx },
    { provide: AttributionService,      useValue: svc },
  ];
}

function allCallArgs(svc: ReturnType<typeof makeAttrSvcMock>): unknown[][] {
  return [
    ...svc.getBrinsonAttribution.calls.allArgs(),
    ...svc.getFactorAttribution.calls.allArgs(),
    ...svc.loadAttributionData.calls.allArgs(),
    ...svc.getAttribution.calls.allArgs(),
  ] as unknown[][];
}

// ─── Suite: happy-path name wiring ───────────────────────────────────────────

describe('AttributionComponent — analytics use portfolio NAME, not UUID (issue #960)', () => {
  let portfolioCtxMock: ReturnType<typeof makePortfolioCtxMock>;
  let attrSvcMock: ReturnType<typeof makeAttrSvcMock>;
  let fixture: ComponentFixture<AttributionComponent>;

  beforeEach(async () => {
    portfolioCtxMock = makePortfolioCtxMock();
    attrSvcMock      = makeAttrSvcMock();

    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: attrProviders(portfolioCtxMock, attrSvcMock),
    }).compileComponents();

    fixture = TestBed.createComponent(AttributionComponent);
    fixture.detectChanges();
  });

  // ── T3(a): loads data on init using the NAME ────────────────────────────

  it('when portfolio context has a resolved name, attribution API is not called with the raw UUID', () => {
    const calls = allCallArgs(attrSvcMock);

    expect(calls.length).toBeGreaterThan(0);

    for (const args of calls) {
      expect(args[0]).not.toBe(PORTFOLIO_ID);
    }
  });

  it('when portfolio context has a resolved name, attribution API is called with the portfolio name', () => {
    const calls = allCallArgs(attrSvcMock);

    const calledWithName = calls.some((args) => args[0] === PORTFOLIO_NAME);
    expect(calledWithName).toBeTrue();
  });

});

// ─── Suite: null-safety — empty API responses must not throw ─────────────────

describe('AttributionComponent — empty API responses do not throw (issue #960)', () => {
  it('when attribution API returns empty collections, component renders without throwing', async () => {
    const emptyCtx = makePortfolioCtxMock();
    const emptySvc = makeAttrSvcMock(
      of({ periods: [], summary: { allocationEffect: 0, selectionEffect: 0, totalEffect: 0 } }),
      of({ factors: [], totalFactorReturn: 0 }),
    );
    emptySvc.loadAttributionData.and.returnValue(of({
      brinson: { periods: [], summary: { allocationEffect: 0, selectionEffect: 0, totalEffect: 0 } },
      factor:  { factors: [], totalFactorReturn: 0 },
    }));
    emptySvc.getAttribution.and.returnValue(of({
      brinson: { periods: [], summary: { allocationEffect: 0, selectionEffect: 0, totalEffect: 0 } },
      factor:  { factors: [], totalFactorReturn: 0 },
    }));

    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: attrProviders(emptyCtx, emptySvc),
    }).compileComponents();

    const f = TestBed.createComponent(AttributionComponent);
    expect(() => f.detectChanges()).not.toThrow();
  });
});

// ─── Suite: API error → visible error state ───────────────────────────────────

describe('AttributionComponent — error state when primary attribution API fails (issue #960)', () => {
  let fixture: ComponentFixture<AttributionComponent>;

  beforeEach(async () => {
    const errorCtx = makePortfolioCtxMock();
    const errorSvc = makeAttrSvcMock(
      throwError(() => new Error('500 Internal Server Error')),
      throwError(() => new Error('500 Internal Server Error')),
    );
    errorSvc.loadAttributionData.and.returnValue(
      throwError(() => new Error('500 Internal Server Error')),
    );
    errorSvc.getAttribution.and.returnValue(
      throwError(() => new Error('500 Internal Server Error')),
    );

    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: attrProviders(errorCtx, errorSvc),
    }).compileComponents();

    fixture = TestBed.createComponent(AttributionComponent);
    fixture.detectChanges();
  });

  // ── T3(b): visible non-blank error state ────────────────────────────────

  it('when primary attribution API returns an error, a role="alert" element is present in the DOM', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    const alertEl =
      el.querySelector('[role="alert"]') ??
      el.querySelector('app-page-error-banner') ??
      el.querySelector('app-alert-banner');

    expect(alertEl).not.toBeNull();
  });

  it('when primary attribution API returns an error, the error region contains non-blank text', () => {
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

describe('AttributionComponent — reacts to portfolio context change (issue #960)', () => {
  let portfolioCtxMock: ReturnType<typeof makePortfolioCtxMock>;
  let attrSvcMock: ReturnType<typeof makeAttrSvcMock>;
  let fixture: ComponentFixture<AttributionComponent>;

  const SECOND_NAME = 'European Bonds';
  const SECOND_ID   = 'a1b2c3d4-0000-0000-0000-000000000002';

  beforeEach(async () => {
    portfolioCtxMock = makePortfolioCtxMock();
    attrSvcMock      = makeAttrSvcMock();

    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: attrProviders(portfolioCtxMock, attrSvcMock),
    }).compileComponents();

    fixture = TestBed.createComponent(AttributionComponent);
    fixture.detectChanges();

    // Reset call counts so only post-switch calls are counted.
    attrSvcMock.getBrinsonAttribution.calls.reset();
    attrSvcMock.getFactorAttribution.calls.reset();
    attrSvcMock.loadAttributionData.calls.reset();
    attrSvcMock.getAttribution.calls.reset();
  });

  // ── T3(c): reacts to portfolio context change ───────────────────────────

  it('when portfolio name changes, attribution API is called again', () => {
    portfolioCtxMock._nameSig.set(SECOND_NAME);
    portfolioCtxMock._idSig.set(SECOND_ID);
    fixture.detectChanges();

    const totalCalls = allCallArgs(attrSvcMock).length;
    expect(totalCalls).toBeGreaterThan(0);
  });

  it('when portfolio name changes, the new API call uses the new name, not the old one', () => {
    portfolioCtxMock._nameSig.set(SECOND_NAME);
    portfolioCtxMock._idSig.set(SECOND_ID);
    fixture.detectChanges();

    const calls = allCallArgs(attrSvcMock);
    const calledWithNew = calls.some((args) => args[0] === SECOND_NAME);
    expect(calledWithNew).toBeTrue();
  });

  it('when portfolio is cleared, attribution API is not called with any portfolio identifier', () => {
    portfolioCtxMock._nameSig.set(null);
    portfolioCtxMock._idSig.set(null);
    fixture.detectChanges();

    const calls = allCallArgs(attrSvcMock);
    // Component should guard on null name/id and skip the fetch entirely.
    const anyIdentifierCall = calls.some((args) => args[0] != null && args[0] !== '');
    expect(anyIdentifierCall).toBeFalse();
  });
});
