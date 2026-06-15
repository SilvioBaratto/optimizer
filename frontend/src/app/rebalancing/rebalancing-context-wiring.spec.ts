/**
 * Source-blind contract tests for issue #961.
 *
 * T3 criteria pinned:
 *   (a) loads data on init — rebalancing API calls use the portfolio NAME, not the raw UUID
 *   (b) handles API error — a visible, non-blank error state is shown when the
 *       primary API call returns an error
 *   (c) reacts to portfolio context change — switching portfolio refetches with the
 *       new portfolio name
 *
 * Primary service calls in scope: getDrift, listPolicies, getPreview, getSnapshots.
 *
 * NOT-VERIFIABLE criteria intentionally omitted:
 *   - 3-second SLA for chart rendering
 *   - individual panel sub-renders (visual)
 *   - panel-level error banner per-call
 *   - decide/what-if form submission logic
 *
 * Assumptions recorded (source-blind guesses correctable without changing test intent):
 *   - Component class: RebalancingComponent in ./rebalancing
 *   - Primary service: RebalancingService (injected by the component)
 *   - Error state uses <app-page-error-banner> or any element with role="alert"
 *   - After migration the local PortfolioApiService is removed; PortfolioContextService drives portfolio selection
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection, signal, computed, NO_ERRORS_SCHEMA } from '@angular/core';
import { of, throwError } from 'rxjs';

import { RebalancingComponent } from './rebalancing';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { RebalancingService } from './rebalancing.service';
import { ICON_PROVIDER } from '../icons';

// ─── Test constants ──────────────────────────────────────────────────────────

const PORTFOLIO_ID   = 'f47ac10b-58cc-4372-a567-0e02b2c3d479';
const PORTFOLIO_NAME = 'Global Equity Fund';

const DRIFT_RESPONSE = {
  entries: [
    { asset: 'AAPL', targetWeight: 0.1, currentWeight: 0.13, drift: 0.03, isBreached: false },
  ],
  breachedCount: 0,
  threshold: 0.05,
};

const POLICIES_RESPONSE = {
  items: [
    { id: 'p1', name: 'Monthly Calendar', isActive: true, type: 'calendar', params: {} },
  ],
};

const PREVIEW_RESPONSE = {
  trades: [],
  estimatedCost: 0.002,
  totalTurnover: 0.12,
};

const SNAPSHOTS_RESPONSE = {
  items: [],
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
function makeRebalSvcMock(
  driftReturn:     any = of(DRIFT_RESPONSE),
  policiesReturn:  any = of(POLICIES_RESPONSE),
  previewReturn:   any = of(PREVIEW_RESPONSE),
  snapshotsReturn: any = of(SNAPSHOTS_RESPONSE),
) {
  return {
    getDrift:       jasmine.createSpy('getDrift').and.returnValue(driftReturn),
    listPolicies:   jasmine.createSpy('listPolicies').and.returnValue(policiesReturn),
    getPreview:     jasmine.createSpy('getPreview').and.returnValue(previewReturn),
    getSnapshots:   jasmine.createSpy('getSnapshots').and.returnValue(snapshotsReturn),
    activatePolicy: jasmine.createSpy('activatePolicy').and.returnValue(of(undefined)),
    createPolicy:   jasmine.createSpy('createPolicy').and.returnValue(of({})),
    decide:         jasmine.createSpy('decide').and.returnValue(of({})),
  };
}

function rebalProviders(
  ctx: ReturnType<typeof makePortfolioCtxMock>,
  svc: ReturnType<typeof makeRebalSvcMock>,
) {
  return [
    provideZonelessChangeDetection(),
    ICON_PROVIDER,
    { provide: PortfolioContextService, useValue: ctx },
    { provide: RebalancingService,      useValue: svc },
  ];
}

function primaryCallArgs(svc: ReturnType<typeof makeRebalSvcMock>): unknown[][] {
  return [
    ...svc.getDrift.calls.allArgs(),
    ...svc.listPolicies.calls.allArgs(),
    ...svc.getPreview.calls.allArgs(),
    ...svc.getSnapshots.calls.allArgs(),
  ] as unknown[][];
}

// ─── Suite: happy-path name wiring ───────────────────────────────────────────

describe('RebalancingComponent — analytics use portfolio NAME, not UUID (issue #961)', () => {
  let portfolioCtxMock: ReturnType<typeof makePortfolioCtxMock>;
  let rebalSvcMock: ReturnType<typeof makeRebalSvcMock>;
  let fixture: ComponentFixture<RebalancingComponent>;

  beforeEach(async () => {
    portfolioCtxMock = makePortfolioCtxMock();
    rebalSvcMock     = makeRebalSvcMock();

    await TestBed.configureTestingModule({
      imports: [RebalancingComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: rebalProviders(portfolioCtxMock, rebalSvcMock),
    }).compileComponents();

    fixture = TestBed.createComponent(RebalancingComponent);
    fixture.detectChanges();
  });

  // ── T3(a): loads data on init using the NAME ────────────────────────────

  it('when portfolio context has a resolved name, rebalancing API is not called with the raw UUID', () => {
    const calls = primaryCallArgs(rebalSvcMock);

    expect(calls.length).toBeGreaterThan(0);

    for (const args of calls) {
      expect(args[0]).not.toBe(PORTFOLIO_ID);
    }
  });

  it('when portfolio context has a resolved name, rebalancing API is called with the portfolio name', () => {
    const calls = primaryCallArgs(rebalSvcMock);

    const calledWithName = calls.some((args) => args[0] === PORTFOLIO_NAME);
    expect(calledWithName).toBeTrue();
  });
});

// ─── Suite: null-safety — empty API responses must not throw ─────────────────

describe('RebalancingComponent — empty API responses do not throw (issue #961)', () => {
  it('when rebalancing API returns empty collections, component renders without throwing', async () => {
    const emptyCtx = makePortfolioCtxMock();
    const emptySvc = makeRebalSvcMock(
      of({ entries: [], breachedCount: 0, threshold: 0.05 }),
      of({ items: [] }),
      of({ trades: [], estimatedCost: 0, totalTurnover: 0 }),
      of({ items: [] }),
    );

    await TestBed.configureTestingModule({
      imports: [RebalancingComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: rebalProviders(emptyCtx, emptySvc),
    }).compileComponents();

    const f = TestBed.createComponent(RebalancingComponent);
    expect(() => f.detectChanges()).not.toThrow();
  });
});

// ─── Suite: API error → visible error state ───────────────────────────────────

describe('RebalancingComponent — error state when primary rebalancing API fails (issue #961)', () => {
  let fixture: ComponentFixture<RebalancingComponent>;

  beforeEach(async () => {
    const errorCtx = makePortfolioCtxMock();
    const errorSvc = makeRebalSvcMock(
      throwError(() => new Error('500 Internal Server Error')),
      throwError(() => new Error('500 Internal Server Error')),
      throwError(() => new Error('500 Internal Server Error')),
      throwError(() => new Error('500 Internal Server Error')),
    );

    await TestBed.configureTestingModule({
      imports: [RebalancingComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: rebalProviders(errorCtx, errorSvc),
    }).compileComponents();

    fixture = TestBed.createComponent(RebalancingComponent);
    fixture.detectChanges();
  });

  // ── T3(b): visible non-blank error state ────────────────────────────────

  it('when primary rebalancing API returns an error, a role="alert" element is present in the DOM', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    const alertEl =
      el.querySelector('[role="alert"]') ??
      el.querySelector('app-page-error-banner') ??
      el.querySelector('app-alert-banner');

    expect(alertEl).not.toBeNull();
  });

  it('when primary rebalancing API returns an error, the error region contains non-blank text', () => {
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

describe('RebalancingComponent — reacts to portfolio context change (issue #961)', () => {
  let portfolioCtxMock: ReturnType<typeof makePortfolioCtxMock>;
  let rebalSvcMock: ReturnType<typeof makeRebalSvcMock>;
  let fixture: ComponentFixture<RebalancingComponent>;

  const SECOND_NAME = 'European Bonds';
  const SECOND_ID   = 'a1b2c3d4-0000-0000-0000-000000000002';

  beforeEach(async () => {
    portfolioCtxMock = makePortfolioCtxMock();
    rebalSvcMock     = makeRebalSvcMock();

    await TestBed.configureTestingModule({
      imports: [RebalancingComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: rebalProviders(portfolioCtxMock, rebalSvcMock),
    }).compileComponents();

    fixture = TestBed.createComponent(RebalancingComponent);
    fixture.detectChanges();

    // Reset call counts so only post-switch calls are counted.
    rebalSvcMock.getDrift.calls.reset();
    rebalSvcMock.listPolicies.calls.reset();
    rebalSvcMock.getPreview.calls.reset();
    rebalSvcMock.getSnapshots.calls.reset();
  });

  // ── T3(c): reacts to portfolio context change ───────────────────────────

  it('when portfolio name changes, rebalancing API is called again', () => {
    portfolioCtxMock._nameSig.set(SECOND_NAME);
    portfolioCtxMock._idSig.set(SECOND_ID);
    fixture.detectChanges();

    const totalCalls = primaryCallArgs(rebalSvcMock).length;
    expect(totalCalls).toBeGreaterThan(0);
  });

  it('when portfolio name changes, the new API call uses the new name, not the old one', () => {
    portfolioCtxMock._nameSig.set(SECOND_NAME);
    portfolioCtxMock._idSig.set(SECOND_ID);
    fixture.detectChanges();

    const calls = primaryCallArgs(rebalSvcMock);
    const calledWithNew = calls.some((args) => args[0] === SECOND_NAME);
    expect(calledWithNew).toBeTrue();
  });

  it('when portfolio is cleared, rebalancing API is not called with any portfolio identifier', () => {
    portfolioCtxMock._nameSig.set(null);
    portfolioCtxMock._idSig.set(null);
    fixture.detectChanges();

    const calls = primaryCallArgs(rebalSvcMock);
    const anyIdentifierCall = calls.some((args) => args[0] != null && args[0] !== '');
    expect(anyIdentifierCall).toBeFalse();
  });
});
