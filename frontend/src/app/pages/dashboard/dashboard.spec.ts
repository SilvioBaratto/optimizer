import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { TestBed } from '@angular/core/testing';
import { DashboardComponent } from './dashboard';
import { PortfolioContextService } from '../../services/portfolio-context.service';
import { environment } from '../../../environments/environment';
import { ICON_PROVIDER } from '../../icons';
import type { DashboardKPI } from '../../models/dashboard.model';

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
