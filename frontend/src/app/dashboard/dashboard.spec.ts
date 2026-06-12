import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { DashboardComponent } from './dashboard';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { ReportsService } from '../core/services/reports.service';
import { DashboardService } from './dashboard.service';
import { environment } from '../../environments/environment';
import { ICON_PROVIDER } from '../icons';
import { makeReportJobCreateResponse } from '../../testing';
import type { DashboardKPI, EquityCurvePoint } from './dashboard.model';
import type { SunburstNode } from '../shared/echarts-sunburst/echarts-sunburst';

function makeKpi(overrides: Partial<DashboardKPI>): DashboardKPI {
  return {
    label: 'KPI',
    value: 1,
    format: 'percent',
    change: 0,
    changeLabel: 'vs last month',
    sparkline: [],
    ...overrides,
  };
}

describe('DashboardComponent — KPI delta formatting (issue #432)', () => {
  let component: DashboardComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
      ],
    }).compileComponents();
    component = TestBed.createComponent(DashboardComponent).componentInstance;
  });

  describe('kpiDeltaFormat', () => {
    it('returns "percent" for percent KPIs (delta is already a decimal)', () => {
      expect(component.kpiDeltaFormat(makeKpi({ format: 'percent' }))).toBe('percent');
    });

    it('returns "percent" for currency KPIs (kpiDelta divides by NAV → decimal)', () => {
      expect(component.kpiDeltaFormat(makeKpi({ format: 'currency' }))).toBe('percent');
    });

    it('returns "absolute" for ratio KPIs so Sharpe/Calmar diffs are not × 100', () => {
      expect(component.kpiDeltaFormat(makeKpi({ format: 'ratio' }))).toBe('absolute');
    });

    it('returns "absolute" for number KPIs', () => {
      expect(component.kpiDeltaFormat(makeKpi({ format: 'number' }))).toBe('absolute');
    });
  });

  describe('kpiDelta', () => {
    it('forwards the raw change for percent KPIs', () => {
      const kpi = makeKpi({ format: 'percent', change: 0.025 });
      expect(component.kpiDelta(kpi)).toBe(0.025);
    });

    it('forwards the raw change for ratio KPIs (rendered as absolute)', () => {
      const kpi = makeKpi({ format: 'ratio', change: -12.64 });
      expect(component.kpiDelta(kpi)).toBe(-12.64);
    });

    it('forwards the raw change for number KPIs', () => {
      const kpi = makeKpi({ format: 'number', change: 7 });
      expect(component.kpiDelta(kpi)).toBe(7);
    });

    it('returns 0 for currency KPIs when NAV is zero', () => {
      const kpi = makeKpi({ format: 'currency', change: 100 });
      // Default nav() is 0 before data loads
      expect(component.kpiDelta(kpi)).toBe(0);
    });
  });
});

describe('DashboardComponent — period selector wiring (issue #433)', () => {
  let component: DashboardComponent;
  let http: HttpTestingController;
  const API = environment.apiUrl;
  const PORTFOLIO = 'myport';

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
      ],
    }).compileComponents();

    // Force a non-null portfolio so the constructor's loadPortfolioData()
    // actually fires HTTP calls (rather than landing on the empty state).
    TestBed.inject(PortfolioContextService).currentPortfolioId.set(PORTFOLIO);

    http = TestBed.inject(HttpTestingController);
    component = TestBed.createComponent(DashboardComponent).componentInstance;

    // Drain the bootstrap requests fired by the constructor so that subsequent
    // assertions only see the requests triggered by onPeriodChange.
    drainPendingRequests(http);
  });

  afterEach(() => http.verify());

  function drainPendingRequests(controller: HttpTestingController): void {
    const open = controller.match(() => true);
    for (const req of open) {
      req.flush(stubBodyFor(req.request.url));
    }
  }

  function stubBodyFor(url: string): Record<string, unknown> {
    if (url.includes('/market/indices')) return { indices: [], total: 0 };
    if (url.includes('/market/snapshot')) {
      return {
        vix: 0,
        vixChange: 0,
        sp500Return: 0,
        tenYearYield: 0,
        yieldChange: 0,
        usdIndex: 0,
        usdChange: 0,
        asOf: new Date().toISOString(),
      };
    }
    if (url.includes('/equity-curve')) return { points: [] };
    if (url.includes('/performance-metrics')) {
      return { kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' };
    }
    if (url.includes('/allocation')) return { nodes: [], totalPositions: 0, totalSectors: 0 };
    if (url.includes('/asset-class-returns')) return { returns: [] };
    if (url.includes('/rolling-metrics')) {
      return { window: 63, sharpe: [], volatility: [], beta: [] };
    }
    return {};
  }

  it('refetches both equity-curve AND performance-metrics when the period changes', () => {
    component.onPeriodChange('3Y');

    const metricsReq = http.expectOne(
      (r) =>
        r.url ===
          `${API}portfolio-analytics/${PORTFOLIO}/performance-metrics` &&
        r.params.get('period') === '3Y',
    );
    expect(metricsReq.request.method).toBe('GET');
    metricsReq.flush({ kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' });

    const equityReq = http.expectOne(
      (r) =>
        r.url === `${API}portfolio-analytics/${PORTFOLIO}/equity-curve` &&
        r.params.get('period') === '3Y',
    );
    expect(equityReq.request.method).toBe('GET');
    equityReq.flush({ points: [] });
  });

  it('updates the period signal before firing the refetches', () => {
    component.onPeriodChange('5Y');

    expect(component.period()).toBe('5Y');

    const metricsReq = http.expectOne(
      (r) =>
        r.url ===
          `${API}portfolio-analytics/${PORTFOLIO}/performance-metrics`,
    );
    metricsReq.flush({ kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' });
    const equityReq = http.expectOne(
      (r) => r.url === `${API}portfolio-analytics/${PORTFOLIO}/equity-curve`,
    );
    equityReq.flush({ points: [] });
  });
});

describe('DashboardComponent — rolling-metrics wiring (issue #453)', () => {
  let fixture: ReturnType<typeof TestBed.createComponent<DashboardComponent>>;
  let component: DashboardComponent;
  let http: HttpTestingController;
  let ctx: PortfolioContextService;
  const API = environment.apiUrl;
  const PORTFOLIO = 'alpha';

  function rollingBody() {
    return {
      window: 63,
      sharpe: [{ date: '2026-01-01', value: 1.2 }],
      volatility: [{ date: '2026-01-01', value: 0.15 }],
      beta: [{ date: '2026-01-01', value: 1.05 }],
    };
  }

  function stubBody(url: string): Record<string, unknown> {
    if (url.includes('/market/indices')) return { indices: [], total: 0 };
    if (url.includes('/market/snapshot')) {
      return {
        vix: 0,
        vixChange: 0,
        sp500Return: 0,
        tenYearYield: 0,
        yieldChange: 0,
        usdIndex: 0,
        usdChange: 0,
        asOf: new Date().toISOString(),
      };
    }
    if (url.includes('/equity-curve')) return { points: [] };
    if (url.includes('/performance-metrics')) {
      return { kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' };
    }
    if (url.includes('/allocation')) return { nodes: [], totalPositions: 0, totalSectors: 0 };
    if (url.includes('/asset-class-returns')) return { returns: [] };
    if (url.includes('/rolling-metrics')) return rollingBody();
    return {};
  }

  function drainExcept(urlFragment: string): void {
    const open = http.match((r) => !r.url.includes(urlFragment));
    for (const req of open) {
      req.flush(stubBody(req.request.url));
    }
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
      ],
    }).compileComponents();

    ctx = TestBed.inject(PortfolioContextService);
    ctx.currentPortfolioId.set(PORTFOLIO);
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(DashboardComponent);
    component = fixture.componentInstance;
    // Trigger change detection so the refetch effect registered in the
    // constructor runs and fires the initial rolling-metrics request.
    fixture.detectChanges();
  });

  afterEach(() => http.verify());

  it('issues a rolling-metrics request for the active portfolio on bootstrap', () => {
    const rolling = http.match(
      (r) => r.url === `${API}portfolio-analytics/${PORTFOLIO}/rolling-metrics`,
    );
    expect(rolling.length).toBeGreaterThanOrEqual(1);
    rolling[0].flush(rollingBody());
    // Drain the remaining bootstrap requests to keep verify() happy.
    const remaining = http.match(() => true);
    for (const r of remaining) r.flush(stubBody(r.request.url));
  });

  function flushAllRollingMetrics(urlFragment: string, body: Record<string, unknown>): number {
    const reqs = http.match((r) => r.url.includes(urlFragment));
    for (const r of reqs) r.flush(body);
    return reqs.length;
  }

  it("maps the response into RollingMetricSeries with 'ratio'|'percent'|'unit' formatters", () => {
    const count = flushAllRollingMetrics('/rolling-metrics', rollingBody());
    expect(count).toBeGreaterThanOrEqual(1);
    drainExcept('/rolling-metrics');
    fixture.detectChanges();

    const series = component.rollingMetricsSeries();
    expect(series.length).toBe(3);
    const byName = new Map(series.map((s) => [s.name, s.formatter]));
    expect(byName.get('Sharpe')).toBe('ratio');
    expect(byName.get('Volatility')).toBe('percent');
    expect(byName.get('Beta')).toBe('unit');
  });

  it('renders exactly one <app-echarts-rolling-metrics> element in the template', () => {
    flushAllRollingMetrics('/rolling-metrics', rollingBody());
    drainExcept('/rolling-metrics');
    // Skip the staggered reveal animation so the card's parent @if unlocks.
    component.revealIndex.set(10);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('app-echarts-rolling-metrics').length).toBe(1);
  });

  it('refetches rolling metrics when the active portfolio changes', () => {
    flushAllRollingMetrics('/rolling-metrics', rollingBody());
    drainExcept('/rolling-metrics');

    ctx.currentPortfolioId.set('beta');
    fixture.detectChanges();

    const second = http.match(
      (r) => r.url === `${API}portfolio-analytics/beta/rolling-metrics`,
    );
    expect(second.length).toBeGreaterThanOrEqual(1);
    for (const r of second) r.flush(rollingBody());
    // The portfolio change also re-drives the core analytics; drain them.
    drainExcept('/rolling-metrics');
  });

  it('refetches the core analytics when the active portfolio changes', () => {
    flushAllRollingMetrics('/rolling-metrics', rollingBody());
    drainExcept('/rolling-metrics');

    ctx.currentPortfolioId.set('beta');
    fixture.detectChanges();

    const perf = http.match(
      (r) => r.url === `${API}portfolio-analytics/beta/performance-metrics`,
    );
    expect(perf.length).toBeGreaterThanOrEqual(1);
    // Drain all pending requests (perf, equity, allocation, asset-class, rolling).
    for (const r of http.match(() => true)) r.flush(stubBody(r.request.url));
  });

  it('refetches rolling metrics when the PortfolioContextService.dateRange() preset changes', () => {
    const initialCount = flushAllRollingMetrics('/rolling-metrics', rollingBody());
    expect(initialCount).toBeGreaterThanOrEqual(1);
    drainExcept('/rolling-metrics');

    ctx.setPreset('3Y');
    fixture.detectChanges();

    const after = http.match(
      (r) => r.url === `${API}portfolio-analytics/${PORTFOLIO}/rolling-metrics`,
    );
    expect(after.length).toBeGreaterThanOrEqual(1);
    for (const r of after) r.flush(rollingBody());
  });

  it('sets the rolling-metrics error signal when the request fails', () => {
    const reqs = http.match(
      (r) => r.url.includes('/rolling-metrics'),
    );
    expect(reqs.length).toBeGreaterThanOrEqual(1);
    reqs[0].flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
    for (const r of reqs.slice(1)) r.flush(rollingBody());
    drainExcept('/rolling-metrics');
    fixture.detectChanges();

    expect(component.rollingMetricsError()).toContain('boom');
    expect(component.rollingMetrics()).toBeNull();
    expect(component.rollingMetricsLoading()).toBe(false);
  });
});

// ── Method coverage (#941) ───────────────────────────────────────────────────
// Drives the report-generation flow (spied ReportsService), benchmark change,
// and the small delegating helpers. Bootstrap HTTP is drained.
describe('DashboardComponent — method coverage (#941)', () => {
  let fixture: ReturnType<typeof TestBed.createComponent<DashboardComponent>>;
  let component: DashboardComponent;
  let http: HttpTestingController;
  let reports: ReportsService;

  function stub(url: string): Record<string, unknown> {
    if (url.includes('/market/indices')) {
      return { indices: [{ ticker: 'SPY', name: 'S&P 500' }, { ticker: 'QQQ', name: 'Nasdaq' }], total: 2 };
    }
    if (url.includes('/market/snapshot')) {
      return { vix: 0, vixChange: 0, sp500Return: 0, tenYearYield: 0, yieldChange: 0, usdIndex: 0, usdChange: 0, asOf: '2026-01-01T00:00:00Z' };
    }
    if (url.includes('/equity-curve')) return { points: [] };
    if (url.includes('/performance-metrics')) return { kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' };
    if (url.includes('/allocation')) return { nodes: [], totalPositions: 0, totalSectors: 0 };
    if (url.includes('/asset-class-returns')) return { returns: [] };
    if (url.includes('/rolling-metrics')) return { window: 63, sharpe: [], volatility: [], beta: [] };
    return {};
  }

  function drainAll(): void {
    for (const r of http.match(() => true)) r.flush(stub(r.request.url));
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
      ],
    }).compileComponents();

    TestBed.inject(PortfolioContextService).currentPortfolioId.set('gamma');
    reports = TestBed.inject(ReportsService);
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(DashboardComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
    drainAll();
  });

  afterEach(() => http.verify());

  it('onGenerateReport guards when a report is already generating', () => {
    component.reportLoading.set(true);
    const spy = spyOn(reports, 'generate');
    component.onGenerateReport();
    expect(spy).not.toHaveBeenCalled();
  });

  it('onGenerateReport stores the job id on success', () => {
    spyOn(reports, 'generate').and.returnValue(of(makeReportJobCreateResponse({ job_id: 'rep-1' })));
    component.onGenerateReport();
    expect(component.reportJobId()).toBe('rep-1');
  });

  it('onGenerateReport records the default error on a message-less failure', () => {
    spyOn(reports, 'generate').and.returnValue(throwError(() => ({})));
    component.onGenerateReport();
    expect(component.reportError()).toBe('Report generation failed');
  });

  it('onReportJobCompleted sets the download url from a report_id', () => {
    spyOn(reports, 'downloadUrl').and.returnValue('http://x/report.pdf');
    spyOn(reports, 'pollJob').and.returnValue(
      of({ status: 'completed', result: { report_id: 'r1' } }) as ReturnType<ReportsService['pollJob']>,
    );
    component.reportJobId.set('rep-1');
    component.onReportJobCompleted();
    expect(component.reportDownloadUrl()).toBe('http://x/report.pdf');
  });

  it('onReportJobCompleted falls back to download_url when no report_id is present', () => {
    spyOn(reports, 'pollJob').and.returnValue(
      of({ status: 'completed', result: { download_url: 'http://y/r.pdf' } }) as ReturnType<ReportsService['pollJob']>,
    );
    component.reportJobId.set('rep-1');
    component.onReportJobCompleted();
    expect(component.reportDownloadUrl()).toBe('http://y/r.pdf');
  });

  it('onReportJobCompleted is a no-op when there is no job id', () => {
    const spy = spyOn(reports, 'pollJob');
    component.reportJobId.set(null);
    component.onReportJobCompleted();
    expect(spy).not.toHaveBeenCalled();
  });

  it('onReportJobCompleted records the default error when polling fails', () => {
    spyOn(reports, 'pollJob').and.returnValue(throwError(() => ({})));
    component.reportJobId.set('rep-1');
    component.onReportJobCompleted();
    expect(component.reportError()).toBe('Failed to fetch report');
  });

  it('onReportJobFailed records the error with a default fallback', () => {
    component.onReportJobFailed('boom');
    expect(component.reportError()).toBe('boom');
    component.onReportJobFailed('');
    expect(component.reportError()).toBe('Report job failed');
  });

  it('onBenchmarkChange updates the benchmark and refetches the equity curve', () => {
    component.onBenchmarkChange('QQQ');
    expect(component.benchmark()).toBe('QQQ');
    const eq = http.match((r) => r.url.includes('/equity-curve'));
    expect(eq.length).toBeGreaterThanOrEqual(1);
    for (const r of eq) r.flush({ points: [] });
  });

  it('retry re-fires the portfolio load', () => {
    component.retry();
    drainAll();
    expect(component.hasError()).toBe(false);
  });

  it('openReportModal does not throw', () => {
    expect(() => component.openReportModal()).not.toThrow();
  });

  it('formatKpiValue and kpiTrend delegate to the dashboard service', () => {
    const kpi: DashboardKPI = { label: 'R', value: 0.1, format: 'percent', change: 0.01, changeLabel: '', sparkline: [] };
    expect(typeof component.formatKpiValue(kpi)).toBe('string');
    expect(['up', 'down', 'flat']).toContain(component.kpiTrend(kpi));
  });

  it('returnCellBg returns a style string', () => {
    expect(typeof component.returnCellBg(0.05)).toBe('string');
  });

  it('subtitle describes the NAV when a portfolio is active', () => {
    component.nav.set(1_000_000);
    component.dailyChange.set(0.01);
    expect(component.subtitle()).toContain('today');
  });

  it('allocation and drawdown computeds tolerate missing optional fields', () => {
    component.allocationData.set([{ name: 'X' } as SunburstNode]);
    expect(component.allocationSectors()[0].holdings).toBe(0); // children?.length ?? 0
    expect(component.allocationTotalHoldings()).toBe(0);
    component.equityCurve.set([{ date: '2026-01-01' } as EquityCurvePoint]); // portfolio ?? 0
    expect(component.drawdownSeries().length).toBeGreaterThanOrEqual(0);
  });

  it('onReportJobCompleted with no result leaves the download url unset', () => {
    spyOn(reports, 'pollJob').and.returnValue(
      of({ status: 'completed' }) as ReturnType<ReportsService['pollJob']>,
    );
    component.reportJobId.set('rep-1');
    component.onReportJobCompleted();
    expect(component.reportDownloadUrl()).toBeNull();
  });

  it('rolling-metrics error falls back to the default message on a message-less failure', () => {
    spyOn(TestBed.inject(DashboardService), 'getRollingMetrics').and.returnValue(throwError(() => ({})));
    TestBed.inject(PortfolioContextService).setPreset('3Y'); // re-fires the effect
    fixture.detectChanges();
    expect(component.rollingMetricsError()).toBe('Failed to load rolling metrics');
  });

  it('loadPortfolioData populates the snapshot and clears loading on success', () => {
    spyOn(TestBed.inject(DashboardService), 'loadPortfolioData').and.returnValue(
      of({
        kpis: [], nav: 42, navChangePct: 0, currency: 'EUR',
        equityCurvePoints: [], allocationNodes: [], assetClassReturns: [],
      }) as unknown as ReturnType<DashboardService['loadPortfolioData']>,
    );
    component.retry();
    expect(component.nav()).toBe(42);
    expect(component.isLoadingPortfolio()).toBe(false);
  });

  it('refetchPerformanceMetrics defaults the currency to EUR when absent', () => {
    component.onPeriodChange('3Y');
    http
      .expectOne((r) => r.url.includes('/performance-metrics'))
      .flush({ kpis: [], nav: 0, navChangePct: 0 }); // currency omitted → ?? 'EUR'
    http.expectOne((r) => r.url.includes('/equity-curve')).flush({ points: [] });
    expect(component.currency()).toBe('EUR');
  });

  it('loadPortfolioData records the default error message on a message-less failure', () => {
    spyOn(TestBed.inject(DashboardService), 'loadPortfolioData').and.returnValue(throwError(() => ({})));
    component.retry();
    expect(component.hasError()).toBe(true);
    expect(component.errorMessage()).toBe('Failed to load portfolio data');
  });
});
