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
    if (url.includes('/market/regime')) {
      return {
        current: 'bull',
        probability: 1,
        since: new Date().toISOString(),
        hmmStates: [],
        modelInfo: { nStates: 4, lastFitted: new Date().toISOString() },
      };
    }
    if (url.includes('/equity-curve')) return { points: [] };
    if (url.includes('/performance-metrics')) {
      return { kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' };
    }
    if (url.includes('/allocation')) return { nodes: [], totalPositions: 0, totalSectors: 0 };
    if (url.includes('/drift')) return { entries: [], breachedCount: 0, threshold: 0.05 };
    if (url.includes('/activity')) return { items: [], total: 0 };
    if (url.includes('/asset-class-returns')) return { returns: [] };
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
