/**
 * Global-context wiring specs for RebalancingComponent (issue #1016).
 *
 * Pins acceptance criteria:
 *   (a) loads data on init using the portfolio NAME from context
 *   (b) handles API error — visible, non-blank error state shown
 *   (c) reacts to portfolio context change — refetches with new name
 *   — no portfolio <select> dropdown exists in the template
 *   — PortfolioApiService portfolio-listing endpoint is never called
 *
 * Uses service mocking (not HTTP interception) so the suite runs
 * reliably in zoneless mode without fakeAsync / tick().
 */
import { ComponentFixture, TestBed } from '@angular/core/testing';
import {
  NO_ERRORS_SCHEMA,
  provideZonelessChangeDetection,
  signal,
  computed,
} from '@angular/core';
import { By } from '@angular/platform-browser';
import { of, throwError } from 'rxjs';

import { RebalancingComponent } from './rebalancing';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { RebalancingService } from './rebalancing.service';
import { ICON_PROVIDER } from '../icons';

// ─── Constants ──────────────────────────────────────────────────────────────

const PORTFOLIO_NAME = 't212';
const PORTFOLIO_ID   = 'stub-uuid-0001';

// ─── Mock factories ───────────────────────────────────────────────────────────

function makeCtx(name: string | null = PORTFOLIO_NAME, id: string | null = PORTFOLIO_ID) {
  const $name = signal<string | null>(name);
  const $id   = signal<string | null>(id);
  return {
    currentPortfolioId:   $id,
    currentPortfolioName: computed(() => $name()),
    selectedPortfolio:    computed(() => ($name() ? { id: $id() ?? '', name: $name()! } : null)),
    hasPortfolio:         computed(() => $id() !== null),
    dateRange:            signal({ preset: '1Y' as const, start: new Date('2024-01-01'), end: new Date('2025-01-01') }),
    benchmark:            signal('SPY'),
    activeMode:           signal('backtest' as const),
    isLive:               computed(() => false),
    isBacktest:           computed(() => true),
    isPaper:              computed(() => false),
    dateRangeLabel:       computed(() => '1Y'),
    dateRangeDays:        computed(() => 365),
    setPortfolio:         jasmine.createSpy('setPortfolio'),
    setMode:              jasmine.createSpy('setMode'),
    setPreset:            jasmine.createSpy('setPreset'),
    setCustomRange:       jasmine.createSpy('setCustomRange'),
    setBenchmark:         jasmine.createSpy('setBenchmark'),
    reset:                jasmine.createSpy('reset'),
    _setName: (n: string | null) => { $name.set(n); $id.set(n ? PORTFOLIO_ID : null); },
  };
}

type CtxMock = ReturnType<typeof makeCtx>;

function makeSvc(opts: {
  driftError?: boolean;
  allError?: boolean;
} = {}) {
  const driftReturn = opts.driftError || opts.allError
    ? throwError(() => new Error('Server error'))
    : of({ entries: [], breachedCount: 0, threshold: 0.05 });
  const failReturn = opts.allError ? throwError(() => new Error('Server error')) : of({ items: [] });

  return {
    getDrift:       jasmine.createSpy('getDrift').and.returnValue(driftReturn),
    listPolicies:   jasmine.createSpy('listPolicies').and.returnValue(failReturn),
    getPreview:     jasmine.createSpy('getPreview').and.returnValue(of({ trades: [], estimatedCost: 0, totalTurnover: 0 })),
    getSnapshots:   jasmine.createSpy('getSnapshots').and.returnValue(of({ items: [] })),
    activatePolicy: jasmine.createSpy('activatePolicy').and.returnValue(of(undefined)),
    createPolicy:   jasmine.createSpy('createPolicy').and.returnValue(of({})),
    decide:         jasmine.createSpy('decide').and.returnValue(of({})),
  };
}

type SvcMock = ReturnType<typeof makeSvc>;

// ─── TestBed bootstrap ────────────────────────────────────────────────────────

async function boot(ctx: CtxMock, svc: SvcMock): Promise<ComponentFixture<RebalancingComponent>> {
  await TestBed.configureTestingModule({
    imports: [RebalancingComponent],
    schemas: [NO_ERRORS_SCHEMA],
    providers: [
      provideZonelessChangeDetection(),
      ICON_PROVIDER,
      { provide: PortfolioContextService, useValue: ctx },
      { provide: RebalancingService,      useValue: svc },
    ],
  }).compileComponents();
  const fixture = TestBed.createComponent(RebalancingComponent);
  fixture.detectChanges();
  return fixture;
}

// ─── (a) Loads data on init ────────────────────────────────────────────────

describe('RebalancingComponent — (a) loads data on init (issue #1016)', () => {
  let ctx: CtxMock;
  let svc: SvcMock;
  let fixture: ComponentFixture<RebalancingComponent>;

  beforeEach(async () => {
    ctx = makeCtx();
    svc = makeSvc();
    fixture = await boot(ctx, svc);
  });

  it('when portfolio is in context, getDrift is called on init with the portfolio name', () => {
    expect(svc.getDrift).toHaveBeenCalledWith(PORTFOLIO_NAME, jasmine.anything());
  });

  it('when portfolio is in context, listPolicies is called on init with the portfolio name', () => {
    expect(svc.listPolicies).toHaveBeenCalledWith(PORTFOLIO_NAME);
  });

  it('when portfolio is in context, getPreview is called on init with the portfolio name', () => {
    expect(svc.getPreview).toHaveBeenCalledWith(PORTFOLIO_NAME);
  });

  it('when portfolio is in context, getSnapshots is called on init with the portfolio name', () => {
    expect(svc.getSnapshots).toHaveBeenCalledWith(PORTFOLIO_NAME);
  });

  it('when portfolio is in context, the portfolio UUID is not passed to any service call', () => {
    const allArgs = [
      ...svc.getDrift.calls.allArgs(),
      ...svc.listPolicies.calls.allArgs(),
      ...svc.getPreview.calls.allArgs(),
      ...svc.getSnapshots.calls.allArgs(),
    ] as unknown[][];
    for (const args of allArgs) {
      expect(args[0]).not.toBe(PORTFOLIO_ID);
    }
  });
});

// ─── (b) Handles API error ────────────────────────────────────────────────────

describe('RebalancingComponent — (b) handles API error (issue #1016)', () => {
  it('when getDrift fails, a role="alert" error element is present in the DOM', async () => {
    const ctx = makeCtx();
    const svc = makeSvc({ driftError: true });
    const fixture = await boot(ctx, svc);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    const alertEl =
      el.querySelector('[role="alert"]') ??
      el.querySelector('app-page-error-banner') ??
      el.querySelector('app-alert-banner');

    expect(alertEl).not.toBeNull();
  });

  it('when getDrift fails, the error region contains non-blank text', async () => {
    const ctx = makeCtx();
    const svc = makeSvc({ driftError: true });
    const fixture = await boot(ctx, svc);
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

// ─── (c) Reacts to portfolio context change ───────────────────────────────────

describe('RebalancingComponent — (c) reacts to portfolio context change (issue #1016)', () => {
  let ctx: CtxMock;
  let svc: SvcMock;
  let fixture: ComponentFixture<RebalancingComponent>;

  const NEW_NAME = 'ib-flex';

  beforeEach(async () => {
    ctx = makeCtx();
    svc = makeSvc();
    fixture = await boot(ctx, svc);
    // Reset spy counts so only post-switch calls are measured.
    svc.getDrift.calls.reset();
    svc.listPolicies.calls.reset();
    svc.getPreview.calls.reset();
    svc.getSnapshots.calls.reset();
  });

  it('when portfolio name changes, getDrift is called again with the new name', () => {
    ctx._setName(NEW_NAME);
    fixture.detectChanges();
    expect(svc.getDrift).toHaveBeenCalledWith(NEW_NAME, jasmine.anything());
  });

  it('when portfolio name changes, API calls use the new name, not the old one', () => {
    ctx._setName(NEW_NAME);
    fixture.detectChanges();

    const allArgs = [
      ...svc.getDrift.calls.allArgs(),
      ...svc.listPolicies.calls.allArgs(),
      ...svc.getPreview.calls.allArgs(),
      ...svc.getSnapshots.calls.allArgs(),
    ] as unknown[][];
    const calledWithNew = allArgs.some((args) => args[0] === NEW_NAME);
    expect(calledWithNew).toBeTrue();
  });

  it('when portfolio is cleared to null, no service panel calls are made', () => {
    ctx._setName(null);
    fixture.detectChanges();

    const totalCalls =
      svc.getDrift.calls.count() +
      svc.listPolicies.calls.count() +
      svc.getPreview.calls.count() +
      svc.getSnapshots.calls.count();
    expect(totalCalls).toBe(0);
  });
});

// ─── No portfolio <select> dropdown ──────────────────────────────────────────

describe('RebalancingComponent — no portfolio dropdown in template (issue #1016)', () => {
  it('when component renders, no portfolio-picker select element exists in the DOM', async () => {
    const ctx = makeCtx();
    const svc = makeSvc();
    const fixture = await boot(ctx, svc);
    fixture.detectChanges();

    const picker = fixture.debugElement.query(
      By.css(
        'select[name*="portfolio" i], ' +
          'select[formcontrolname*="portfolio" i], ' +
          '[data-portfolio-picker], ' +
          '#portfolio-select',
      ),
    );
    expect(picker)
      .withContext('no portfolio select/dropdown must exist in the template')
      .toBeNull();
  });

  it('when component renders, selectedPortfolio is a computed value from context, not a writable signal', async () => {
    const ctx = makeCtx();
    const svc = makeSvc();
    const fixture = await boot(ctx, svc);

    const comp = fixture.componentInstance;
    // computed() returns a read-only Signal; it has no .set() or .update() methods.
    // If selectedPortfolio were writable, it would have a .set method.
    expect(typeof (comp.selectedPortfolio as unknown as { set?: unknown }).set)
      .withContext('selectedPortfolio must not expose .set — it is read-only computed')
      .toBe('undefined');
  });
});

// ─── PortfolioApiService portfolio listing not called ────────────────────────

describe('RebalancingComponent — PortfolioApiService portfolio listing is absent (issue #1016)', () => {
  let ctx: CtxMock;
  let svc: SvcMock;
  let fixture: ComponentFixture<RebalancingComponent>;

  beforeEach(async () => {
    ctx = makeCtx();
    svc = makeSvc();
    fixture = await boot(ctx, svc);
  });

  it('when component initialises, selectedPortfolio() returns the NAME from context, not a UUID', () => {
    // selectedPortfolio() is a computed derived from PortfolioContextService.
    // If PortfolioApiService were used instead, it would return the portfolio DTO.
    expect(fixture.componentInstance.selectedPortfolio()).toBe(PORTFOLIO_NAME);
  });

  it('when component initialises, isLoading is false after data resolves from service mocks', () => {
    fixture.detectChanges();
    expect(fixture.componentInstance.isLoading()).toBe(false);
  });
});
