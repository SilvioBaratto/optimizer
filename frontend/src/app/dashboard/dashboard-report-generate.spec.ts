/**
 * Source-blind spec for issue #1018 — report generation wiring and
 * market/indices independence from portfolio selection.
 *
 * Criteria under test (oracle):
 *   [T2] `POST /reports/generate` passes the correct portfolio id;
 *        market snapshot + indices load regardless of selection.
 *
 * Design decisions (from acceptance criteria text, not implementation):
 *   - The report endpoint receives portfolio_id (a UUID), not the portfolio
 *     name.  This is consistent with the issue body: "passes portfolio_id
 *     (UUID, not name): dashboard.ts:327".
 *   - Market snapshot and indices fire independently — they must appear in
 *     requests even when no portfolio is selected (context name = null).
 *   - These are component-level integration tests using HttpTestingController
 *     so the real DashboardService URL-building logic is exercised, but the
 *     backend is replaced by the HTTP mock.
 */

import { TestBed } from '@angular/core/testing';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  NO_ERRORS_SCHEMA,
  provideZonelessChangeDetection,
  signal,
  computed,
} from '@angular/core';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { DashboardComponent } from './dashboard';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { DashboardService } from './dashboard.service';
import { MarketService } from '../core/services/market.service';
import { ReportsService } from '../core/services/reports.service';
import { FormatService } from '../core/services/format.service';
import { ModalService } from '../shared/modal/modal.service';
import { ICON_PROVIDER } from '../icons';
import { environment } from '../../environments/environment';

// ── Constants ────────────────────────────────────────────────────────────────

const PORTFOLIO_ID   = 'a1b2c3d4-0000-4000-8000-000000000001';
const PORTFOLIO_NAME = 'Tech Growth Fund';
const API = environment.apiUrl;

// ── Mock helpers ─────────────────────────────────────────────────────────────

function makeCtxMock(
  id: string | null = PORTFOLIO_ID,
  name: string | null = PORTFOLIO_NAME,
) {
  const idSig = signal<string | null>(id);
  return {
    currentPortfolioId:   idSig,
    currentPortfolioName: computed(() => name),
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
  };
}

const FORMAT_SERVICE_STUB = {
  formatCurrencyCompact: () => '€250k',
  formatPercent:         () => '1.23%',
  formatBps:             () => '123 bps',
  formatCurrency:        () => '€250,000',
  formatRatio:           () => '1.42',
  formatReturn:          () => '1.23%',
  formatDate:            () => '2024-01-01',
};

const DASH_SVC_STUB = {
  loadPortfolioData: jasmine.createSpy('loadPortfolioData').and.returnValue(
    of({
      kpis: [], nav: 0, navChangePct: 0, currency: 'EUR',
      equityCurvePoints: [], allocationNodes: [], assetClassReturns: [],
    }),
  ),
  getEquityCurve:        jasmine.createSpy().and.returnValue(of({ points: [] })),
  getPerformanceMetrics: jasmine.createSpy().and.returnValue(of({ kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' })),
  getRollingMetrics:     jasmine.createSpy().and.returnValue(of(null)),
  getMarketSnapshot:     jasmine.createSpy().and.returnValue(of({
    vix: 0, vixChange: 0, sp500Return: 0, tenYearYield: 0,
    yieldChange: 0, usdIndex: 0, usdChange: 0, asOf: '2026-01-01T00:00:00Z',
  })),
  formatKpiValue: () => '—',
  kpiTrend:       () => 'flat',
  kpiDelta:       () => 0,
  kpiDeltaFormat: () => 'percent',
};

// ── Suite: POST /reports/generate passes portfolio_id ─────────────────────

describe('DashboardComponent — POST /reports/generate passes portfolio_id (issue #1018)', () => {
  let reportsSpy: jasmine.SpyObj<{ generate: (body: unknown) => unknown }>;
  let capturedBody: Record<string, unknown> | undefined;

  beforeEach(async () => {
    capturedBody = undefined;
    reportsSpy = jasmine.createSpyObj('ReportsService', {
      generate:    of({ job_id: 'job-001' }),
      pollJob:     of({ result: null }),
      downloadUrl: '',
    });

    // Intercept the generate call to capture the request body
    (reportsSpy.generate as jasmine.Spy).and.callFake((body: unknown) => {
      capturedBody = body as Record<string, unknown>;
      return of({ job_id: 'job-001' });
    });

    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: makeCtxMock() },
        { provide: DashboardService,        useValue: DASH_SVC_STUB },
        { provide: MarketService, useValue: { getIndices: jasmine.createSpy().and.returnValue(of({ indices: [] })) } },
        { provide: ReportsService,          useValue: reportsSpy },
        { provide: FormatService,           useValue: FORMAT_SERVICE_STUB },
        { provide: ModalService,            useValue: { open: jasmine.createSpy('open') } },
      ],
    }).compileComponents();
  });

  it('when the user triggers report generation, the request body contains portfolio_id equal to the context portfolio UUID', () => {
    const fixture = TestBed.createComponent(DashboardComponent);
    fixture.detectChanges();

    // Trigger report generation via the component's public method
    fixture.componentInstance.onGenerateReport();

    expect(reportsSpy.generate).toHaveBeenCalled();
    // The body must pass the portfolio UUID, not the name
    expect(capturedBody).toBeDefined();
    expect(capturedBody?.['portfolio_id']).toBe(PORTFOLIO_ID);
  });

  it('when the user triggers report generation, the request body does NOT use the portfolio name in place of portfolio_id', () => {
    const fixture = TestBed.createComponent(DashboardComponent);
    fixture.detectChanges();

    fixture.componentInstance.onGenerateReport();

    expect(capturedBody?.['portfolio_id']).not.toBe(PORTFOLIO_NAME);
  });
});

// ── Suite: market snapshot + indices load regardless of portfolio selection ──

describe('DashboardComponent — market snapshot and indices load independently (issue #1018)', () => {
  async function buildWithContext(
    id: string | null,
    name: string | null,
  ) {
    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: makeCtxMock(id, name) },
        { provide: DashboardService,        useValue: { ...DASH_SVC_STUB } },
        {
          provide: MarketService,
          useValue: {
            getIndices: jasmine.createSpy('getIndices').and.returnValue(of({ indices: [{ ticker: 'SPY', name: 'S&P 500' }] })),
          },
        },
        {
          provide: ReportsService,
          useValue: { generate: jasmine.createSpy().and.returnValue(of({ job_id: 'j1' })), pollJob: jasmine.createSpy().and.returnValue(of({ result: null })), downloadUrl: jasmine.createSpy().and.returnValue('') },
        },
        { provide: FormatService, useValue: FORMAT_SERVICE_STUB },
        { provide: ModalService,  useValue: { open: jasmine.createSpy('open') } },
      ],
    }).compileComponents();
  }

  afterEach(() => TestBed.resetTestingModule());

  it('when no portfolio is selected (context null), getMarketSnapshot is still called', async () => {
    await buildWithContext(null, null);

    const dashSvc = TestBed.inject(DashboardService);
    const fixture = TestBed.createComponent(DashboardComponent);
    fixture.detectChanges();

    expect((dashSvc as unknown as { getMarketSnapshot: jasmine.Spy }).getMarketSnapshot)
      .toHaveBeenCalled();
  });

  it('when no portfolio is selected (context null), the market-service getIndices is still called', async () => {
    await buildWithContext(null, null);

    const mktSvc = TestBed.inject(MarketService);
    const fixture = TestBed.createComponent(DashboardComponent);
    fixture.detectChanges();

    expect((mktSvc as unknown as { getIndices: jasmine.Spy }).getIndices)
      .toHaveBeenCalled();
  });

  it('when a portfolio IS selected, getMarketSnapshot is also called (independent of portfolio state)', async () => {
    await buildWithContext(PORTFOLIO_ID, PORTFOLIO_NAME);

    const dashSvc = TestBed.inject(DashboardService);
    const fixture = TestBed.createComponent(DashboardComponent);
    fixture.detectChanges();

    expect((dashSvc as unknown as { getMarketSnapshot: jasmine.Spy }).getMarketSnapshot)
      .toHaveBeenCalled();
  });

  it('when a portfolio IS selected, getIndices is also called (independent of portfolio state)', async () => {
    await buildWithContext(PORTFOLIO_ID, PORTFOLIO_NAME);

    const mktSvc = TestBed.inject(MarketService);
    const fixture = TestBed.createComponent(DashboardComponent);
    fixture.detectChanges();

    expect((mktSvc as unknown as { getIndices: jasmine.Spy }).getIndices)
      .toHaveBeenCalled();
  });
});

// ── Suite: live-HTTP lock — indices and snapshot endpoints fire independently ─
// Uses the real DashboardService + HttpTestingController to assert URL patterns.

describe('DashboardComponent — live HTTP: market/indices independent of portfolio (issue #1018)', () => {
  let http: HttpTestingController;

  function stubAll(): void {
    for (const req of http.match(() => true)) {
      const url = req.request.url;
      if (url.includes('/market/snapshot')) {
        req.flush({ vix: 0, vixChange: 0, sp500Return: 0, tenYearYield: 0, yieldChange: 0, usdIndex: 0, usdChange: 0, asOf: '' });
      } else if (url.includes('/market/indices')) {
        req.flush({ indices: [], total: 0 });
      } else if (url.includes('/equity-curve')) {
        req.flush({ points: [] });
      } else if (url.includes('/performance-metrics')) {
        req.flush({ kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' });
      } else if (url.includes('/allocation')) {
        req.flush({ nodes: [], totalPositions: 0, totalSectors: 0 });
      } else if (url.includes('/asset-class-returns')) {
        req.flush({ returns: [] });
      } else if (url.includes('/rolling-metrics')) {
        req.flush({ window: 63, sharpe: [], volatility: [], beta: [] });
      } else {
        req.flush({});
      }
    }
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
        // No portfolio selected
        { provide: PortfolioContextService, useValue: makeCtxMock(null, null) },
      ],
    }).compileComponents();

    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    stubAll();
    http.verify();
  });

  it('when no portfolio is selected, a request to market/snapshot is fired', () => {
    TestBed.createComponent(DashboardComponent).detectChanges();

    const snapshots = http.match((r) => r.url === `${API}market/snapshot`);
    expect(snapshots.length).toBeGreaterThanOrEqual(1);
    stubAll();
  });

  it('when no portfolio is selected, a request to market/indices is fired', () => {
    TestBed.createComponent(DashboardComponent).detectChanges();

    const indices = http.match((r) => r.url === `${API}market/indices`);
    expect(indices.length).toBeGreaterThanOrEqual(1);
    stubAll();
  });
});
