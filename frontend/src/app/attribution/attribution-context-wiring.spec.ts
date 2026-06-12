/**
 * Context-wiring contract tests for attribution (issues #960 / #972).
 *
 * Criteria pinned (retargeted to the real, snapshot-based architecture per
 * requirements §5/§13 — the page resolves the portfolio NAME, fetches the
 * latest snapshot for that name, and POSTs the snapshot weights to
 * attribution.brinson/factor; the older GET /attribution/{brinson,factor}/{name}
 * methods hit non-existent endpoints and are not used):
 *   (a) loads data on init — the latest-snapshot fetch uses the portfolio NAME,
 *       not the raw UUID
 *   (b) handles API error — a visible, non-blank error state is shown when the
 *       primary (snapshot) call returns an error
 *   (c) reacts to portfolio context change — switching portfolio refetches the
 *       snapshot with the new portfolio name
 *
 * NOT-VERIFIABLE criteria intentionally omitted:
 *   - 3-second SLA for chart rendering
 *   - visual rendering of individual sub-charts
 *   - console.error suppression
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection, signal, computed, NO_ERRORS_SCHEMA } from '@angular/core';
import { of, throwError } from 'rxjs';

import { AttributionComponent } from './attribution';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { PortfolioApiService } from '../core/services/portfolio-api.service';
import { AttributionService } from './attribution.service';
import { ICON_PROVIDER } from '../icons';

import type { SnapshotDto } from '../core/models/portfolio-api.model';

// ─── Test constants ──────────────────────────────────────────────────────────

const PORTFOLIO_ID   = 'f47ac10b-58cc-4372-a567-0e02b2c3d479';
const PORTFOLIO_NAME = 'Global Equity Fund';
const SNAPSHOT_WEIGHTS: Record<string, number> = { AAPL: 0.5, MSFT: 0.5 };

const BRINSON_RESPONSE = {
  totalActiveReturn: 0.008,
  totalAllocation: 0.012,
  totalSelection: -0.004,
  sectors: [],
};

const FACTOR_RESPONSE = {
  residual: 0.001,
  factors: [],
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
    selectedPortfolio:    computed(() => (id ? { id, name: name ?? '', benchmark_ticker: 'SPY' } : null)),
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

function makeAttrSvcMock() {
  return {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    brinson: jasmine.createSpy('brinson').and.returnValue(of(BRINSON_RESPONSE as any)),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    factor:  jasmine.createSpy('factor').and.returnValue(of(FACTOR_RESPONSE as any)),
  };
}

function makePortfolioApiMock(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  snapshotReturn: any = of({ weights: SNAPSHOT_WEIGHTS } as unknown as SnapshotDto),
) {
  return {
    getLatestSnapshot: jasmine.createSpy('getLatestSnapshot').and.returnValue(snapshotReturn),
  };
}

function attrProviders(
  ctx: ReturnType<typeof makePortfolioCtxMock>,
  svc: ReturnType<typeof makeAttrSvcMock>,
  api: ReturnType<typeof makePortfolioApiMock>,
) {
  return [
    provideZonelessChangeDetection(),
    ICON_PROVIDER,
    { provide: PortfolioContextService, useValue: ctx },
    { provide: AttributionService,      useValue: svc },
    { provide: PortfolioApiService,     useValue: api },
  ];
}

// ─── Suite: happy-path name wiring ───────────────────────────────────────────

describe('AttributionComponent — snapshot fetch uses portfolio NAME, not UUID (issue #960)', () => {
  let portfolioApiMock: ReturnType<typeof makePortfolioApiMock>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: attrProviders(makePortfolioCtxMock(), makeAttrSvcMock(), (portfolioApiMock = makePortfolioApiMock())),
    }).compileComponents();

    const fixture = TestBed.createComponent(AttributionComponent);
    fixture.detectChanges();
  });

  it('when portfolio context has a resolved name, the snapshot is fetched with the portfolio name', () => {
    expect(portfolioApiMock.getLatestSnapshot).toHaveBeenCalledWith(PORTFOLIO_NAME);
  });

  it('when portfolio context has a resolved name, the snapshot is not fetched with the raw UUID', () => {
    expect(portfolioApiMock.getLatestSnapshot.calls.count()).toBeGreaterThan(0);
    for (const args of portfolioApiMock.getLatestSnapshot.calls.allArgs()) {
      expect(args[0]).not.toBe(PORTFOLIO_ID);
    }
  });
});

// ─── Suite: POST bodies use the snapshot weights (issue #983) ────────────────

describe('AttributionComponent — POST bodies use snapshot weights (issue #983)', () => {
  let svcMock: ReturnType<typeof makeAttrSvcMock>;

  beforeEach(async () => {
    svcMock = makeAttrSvcMock();
    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: attrProviders(makePortfolioCtxMock(), svcMock, makePortfolioApiMock()),
    }).compileComponents();

    TestBed.createComponent(AttributionComponent).detectChanges();
  });

  it('when the snapshot loads, the brinson() POST body carries the snapshot weights', () => {
    expect(svcMock.brinson).toHaveBeenCalledWith(
      jasmine.objectContaining({ portfolio_weights: SNAPSHOT_WEIGHTS }),
    );
  });

  it('when the snapshot loads, the factor() POST body carries the snapshot weights', () => {
    expect(svcMock.factor).toHaveBeenCalledWith(
      jasmine.objectContaining({ portfolio_weights: SNAPSHOT_WEIGHTS }),
    );
  });
});

// ─── Suite: null-safety — empty snapshot must not throw ──────────────────────

describe('AttributionComponent — empty snapshot does not throw (issue #960)', () => {
  it('when the snapshot has empty weights, the component renders without throwing', async () => {
    const emptyApi = makePortfolioApiMock(of({ weights: {} } as unknown as SnapshotDto));

    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: attrProviders(makePortfolioCtxMock(), makeAttrSvcMock(), emptyApi),
    }).compileComponents();

    const f = TestBed.createComponent(AttributionComponent);
    expect(() => f.detectChanges()).not.toThrow();
  });
});

// ─── Suite: API error → visible error state ───────────────────────────────────

describe('AttributionComponent — error state when the snapshot fetch fails (issue #960)', () => {
  let fixture: ComponentFixture<AttributionComponent>;

  beforeEach(async () => {
    const errorApi = makePortfolioApiMock(throwError(() => new Error('500 Internal Server Error')));

    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: attrProviders(makePortfolioCtxMock(), makeAttrSvcMock(), errorApi),
    }).compileComponents();

    fixture = TestBed.createComponent(AttributionComponent);
    fixture.detectChanges();
  });

  it('when the snapshot fetch returns an error, a role="alert" element is present in the DOM', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    const alertEl =
      el.querySelector('[role="alert"]') ??
      el.querySelector('app-page-error-banner') ??
      el.querySelector('app-alert-banner');

    expect(alertEl).not.toBeNull();
  });

  it('when the snapshot fetch returns an error, the error region contains non-blank text', () => {
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
  let portfolioApiMock: ReturnType<typeof makePortfolioApiMock>;
  let fixture: ComponentFixture<AttributionComponent>;

  const SECOND_NAME = 'European Bonds';
  const SECOND_ID   = 'a1b2c3d4-0000-0000-0000-000000000002';

  beforeEach(async () => {
    portfolioCtxMock = makePortfolioCtxMock();
    portfolioApiMock = makePortfolioApiMock();

    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: attrProviders(portfolioCtxMock, makeAttrSvcMock(), portfolioApiMock),
    }).compileComponents();

    fixture = TestBed.createComponent(AttributionComponent);
    fixture.detectChanges();

    portfolioApiMock.getLatestSnapshot.calls.reset();
  });

  it('when portfolio name changes, the snapshot is fetched again', () => {
    portfolioCtxMock._nameSig.set(SECOND_NAME);
    portfolioCtxMock._idSig.set(SECOND_ID);
    fixture.detectChanges();

    expect(portfolioApiMock.getLatestSnapshot.calls.count()).toBeGreaterThan(0);
  });

  it('when portfolio name changes, the snapshot fetch uses the new name, not the old one', () => {
    portfolioCtxMock._nameSig.set(SECOND_NAME);
    portfolioCtxMock._idSig.set(SECOND_ID);
    fixture.detectChanges();

    expect(portfolioApiMock.getLatestSnapshot).toHaveBeenCalledWith(SECOND_NAME);
    for (const args of portfolioApiMock.getLatestSnapshot.calls.allArgs()) {
      expect(args[0]).not.toBe(PORTFOLIO_NAME);
    }
  });

  it('when portfolio is cleared, the snapshot is not fetched again', () => {
    portfolioCtxMock._nameSig.set(null);
    portfolioCtxMock._idSig.set(null);
    fixture.detectChanges();

    expect(portfolioApiMock.getLatestSnapshot.calls.count()).toBe(0);
  });
});
