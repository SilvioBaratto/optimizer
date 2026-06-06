import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import type {
  ApiPerformanceMetricsResponse,
  ApiRollingMetricsResponse,
  ApiEquityCurveResponse,
  ApiAllocationResponse,
  ApiMarketSnapshotResponse,
  ApiAssetClassReturnsResponse,
} from '../core/models/dashboard-api.model';

import { DashboardService, returnCellBg } from './dashboard.service';
import { environment } from '../../environments/environment';
import type { DashboardKPI, PortfolioDataSnapshot } from './dashboard.model';

const API = environment.apiUrl;

describe('DashboardService', () => {
  let svc: DashboardService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        DashboardService,
      ],
    });
    svc = TestBed.inject(DashboardService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  describe('getPerformanceMetrics(name, period)', () => {
    it('encodes the period as a query parameter (issue #433)', () => {
      svc.getPerformanceMetrics('myport', '3Y').subscribe();

      const req = http.expectOne(
        (r) =>
          r.url ===
          `${API}portfolio-analytics/myport/performance-metrics`,
      );
      expect(req.request.method).toBe('GET');
      expect(req.request.params.get('period')).toBe('3Y');
      req.flush({ kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' });
    });

    it('defaults the period to 1Y when omitted', () => {
      svc.getPerformanceMetrics('myport').subscribe();

      const req = http.expectOne(
        (r) =>
          r.url ===
          `${API}portfolio-analytics/myport/performance-metrics`,
      );
      expect(req.request.params.get('period')).toBe('1Y');
      req.flush({ kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' });
    });

    it('round-trips every supported period', () => {
      for (const period of ['1Y', '3Y', '5Y', 'MAX'] as const) {
        svc.getPerformanceMetrics('myport', period).subscribe();
        const req = http.expectOne(
          (r) =>
            r.url ===
              `${API}portfolio-analytics/myport/performance-metrics` &&
            r.params.get('period') === period,
        );
        expect(req.request.params.get('period')).toBe(period);
        req.flush({ kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' });
      }
    });

    it('when portfolio not found, mapped error message is propagated', () => {
      let error: Error | undefined;
      svc.getPerformanceMetrics('ghost').subscribe({ error: (e) => (error = e) });

      http
        .expectOne(
          (r) => r.url === `${API}portfolio-analytics/ghost/performance-metrics`,
        )
        .flush({ detail: 'portfolio not found' }, { status: 404, statusText: 'Not Found' });

      expect(error?.message).toBe('portfolio not found');
    });

    it('when response mapping succeeds, kpis array and nav are returned', () => {
      const payload: ApiPerformanceMetricsResponse = {
        kpis: [{ label: 'Sharpe', value: 1.2, format: 'ratio', change: 0.1, changeLabel: '+0.1', sparkline: [1, 2, 3] }],
        nav: 105000,
        navChangePct: 2.5,
        currency: 'USD',
      };
      let result: ApiPerformanceMetricsResponse | undefined;
      svc.getPerformanceMetrics('myport', '1Y').subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url === `${API}portfolio-analytics/myport/performance-metrics`)
        .flush(payload);

      expect(result?.nav).toBe(105000);
      expect(result?.kpis[0].label).toBe('Sharpe');
    });
  });

  describe('getRollingMetrics(name, period, window)', () => {
    it('GETs /portfolio-analytics/{name}/rolling-metrics with period param', () => {
      svc.getRollingMetrics('myport', '3Y').subscribe();

      const req = http.expectOne(
        (r) => r.url === `${API}portfolio-analytics/myport/rolling-metrics`,
      );
      expect(req.request.method).toBe('GET');
      expect(req.request.params.get('period')).toBe('3Y');
      req.flush({ window: 63, sharpe: [], volatility: [], beta: [] });
    });

    it('defaults the period to 3Y when omitted', () => {
      svc.getRollingMetrics('myport').subscribe();

      const req = http.expectOne(
        (r) => r.url === `${API}portfolio-analytics/myport/rolling-metrics`,
      );
      expect(req.request.params.get('period')).toBe('3Y');
      req.flush({ window: 63, sharpe: [], volatility: [], beta: [] });
    });

    it('when window is provided, includes window query param', () => {
      svc.getRollingMetrics('myport', '1Y', 126).subscribe();

      const req = http.expectOne(
        (r) => r.url === `${API}portfolio-analytics/myport/rolling-metrics`,
      );
      expect(req.request.params.get('period')).toBe('1Y');
      expect(req.request.params.get('window')).toBe('126');
      req.flush({ window: 126, sharpe: [], volatility: [], beta: [] });
    });

    it('when window is omitted, window param is absent', () => {
      svc.getRollingMetrics('myport', '1Y').subscribe();

      const req = http.expectOne(
        (r) => r.url === `${API}portfolio-analytics/myport/rolling-metrics`,
      );
      expect(req.request.params.get('window')).toBeNull();
      req.flush({ window: 63, sharpe: [], volatility: [], beta: [] });
    });

    it('when response arrives, sharpe series is mapped correctly', () => {
      const payload: ApiRollingMetricsResponse = {
        window: 63,
        sharpe: [{ date: '2024-01-01', value: 1.1 }],
        volatility: [{ date: '2024-01-01', value: 0.15 }],
        beta: [{ date: '2024-01-01', value: 0.9 }],
      };
      let result: ApiRollingMetricsResponse | undefined;
      svc.getRollingMetrics('myport').subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url === `${API}portfolio-analytics/myport/rolling-metrics`)
        .flush(payload);

      expect(result?.window).toBe(63);
      expect(result?.sharpe[0].value).toBe(1.1);
    });

    it('when portfolio not found, mapped error message is propagated', () => {
      let error: Error | undefined;
      svc.getRollingMetrics('ghost').subscribe({ error: (e) => (error = e) });

      http
        .expectOne((r) => r.url === `${API}portfolio-analytics/ghost/rolling-metrics`)
        .flush({ detail: 'portfolio not found' }, { status: 404, statusText: 'Not Found' });

      expect(error?.message).toBe('portfolio not found');
    });
  });

  describe('getEquityCurve(name, benchmark, period)', () => {
    it('GETs /portfolio-analytics/{name}/equity-curve with benchmark and period params', () => {
      svc.getEquityCurve('myport', 'QQQ', '5Y').subscribe();

      const req = http.expectOne(
        (r) => r.url === `${API}portfolio-analytics/myport/equity-curve`,
      );
      expect(req.request.method).toBe('GET');
      expect(req.request.params.get('benchmark')).toBe('QQQ');
      expect(req.request.params.get('period')).toBe('5Y');
      req.flush({ points: [], portfolioTotalReturn: 0, benchmarkTotalReturn: 0 });
    });

    it('defaults benchmark to SPY and period to 3Y when omitted', () => {
      svc.getEquityCurve('myport').subscribe();

      const req = http.expectOne(
        (r) => r.url === `${API}portfolio-analytics/myport/equity-curve`,
      );
      expect(req.request.params.get('benchmark')).toBe('SPY');
      expect(req.request.params.get('period')).toBe('3Y');
      req.flush({ points: [], portfolioTotalReturn: 0, benchmarkTotalReturn: 0 });
    });

    it('when response arrives, portfolioTotalReturn and points are returned', () => {
      const payload: ApiEquityCurveResponse = {
        points: [{ date: '2024-01-02', portfolio: 1.0, benchmark: 1.0 }],
        portfolioTotalReturn: 0.12,
        benchmarkTotalReturn: 0.10,
      };
      let result: ApiEquityCurveResponse | undefined;
      svc.getEquityCurve('myport').subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url === `${API}portfolio-analytics/myport/equity-curve`)
        .flush(payload);

      expect(result?.portfolioTotalReturn).toBe(0.12);
      expect(result?.points.length).toBe(1);
    });

    it('URI-encodes the portfolio name', () => {
      svc.getEquityCurve('my port').subscribe();

      const req = http.expectOne(
        (r) => r.url === `${API}portfolio-analytics/my%20port/equity-curve`,
      );
      req.flush({ points: [], portfolioTotalReturn: 0, benchmarkTotalReturn: 0 });
    });

    it('when portfolio not found, mapped error message is propagated', () => {
      let error: Error | undefined;
      svc.getEquityCurve('ghost').subscribe({ error: (e) => (error = e) });

      http
        .expectOne((r) => r.url === `${API}portfolio-analytics/ghost/equity-curve`)
        .flush({ detail: 'portfolio not found' }, { status: 404, statusText: 'Not Found' });

      expect(error?.message).toBe('portfolio not found');
    });
  });

  describe('getAllocation(name)', () => {
    it('GETs /portfolio-analytics/{name}/allocation', () => {
      svc.getAllocation('myport').subscribe();

      const req = http.expectOne(
        (r) => r.url === `${API}portfolio-analytics/myport/allocation`,
      );
      expect(req.request.method).toBe('GET');
      req.flush({ nodes: [], totalPositions: 0, totalSectors: 0 });
    });

    it('URI-encodes the portfolio name', () => {
      svc.getAllocation('my port').subscribe();

      const req = http.expectOne(
        (r) => r.url === `${API}portfolio-analytics/my%20port/allocation`,
      );
      req.flush({ nodes: [], totalPositions: 0, totalSectors: 0 });
    });

    it('when response arrives, nodes array and totalPositions are returned', () => {
      const payload: ApiAllocationResponse = {
        nodes: [{ name: 'Technology', value: 0.45, children: [{ name: 'AAPL', value: 0.25 }] }],
        totalPositions: 10,
        totalSectors: 3,
      };
      let result: ApiAllocationResponse | undefined;
      svc.getAllocation('myport').subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url === `${API}portfolio-analytics/myport/allocation`)
        .flush(payload);

      expect(result?.totalPositions).toBe(10);
      expect(result?.nodes[0].name).toBe('Technology');
    });

    it('when portfolio not found, mapped error message is propagated', () => {
      let error: Error | undefined;
      svc.getAllocation('ghost').subscribe({ error: (e) => (error = e) });

      http
        .expectOne((r) => r.url === `${API}portfolio-analytics/ghost/allocation`)
        .flush({ detail: 'portfolio not found' }, { status: 404, statusText: 'Not Found' });

      expect(error?.message).toBe('portfolio not found');
    });
  });

  describe('getMarketSnapshot()', () => {
    it('GETs /market/snapshot', () => {
      svc.getMarketSnapshot().subscribe();

      const req = http.expectOne(`${API}market/snapshot`);
      expect(req.request.method).toBe('GET');
      req.flush({ vix: 18, vixChange: -0.5, sp500Return: 0.01, tenYearYield: 4.2, yieldChange: 0.02, usdIndex: 103, usdChange: 0.3, asOf: '2024-01-02' });
    });

    it('when response arrives, vix and sp500Return are returned', () => {
      const payload: ApiMarketSnapshotResponse = {
        vix: 22.5,
        vixChange: 1.3,
        sp500Return: -0.005,
        tenYearYield: 4.5,
        yieldChange: 0.05,
        usdIndex: 104.2,
        usdChange: 0.1,
        asOf: '2024-06-05',
      };
      let result: ApiMarketSnapshotResponse | undefined;
      svc.getMarketSnapshot().subscribe((r) => (result = r));

      http.expectOne(`${API}market/snapshot`).flush(payload);

      expect(result?.vix).toBe(22.5);
      expect(result?.sp500Return).toBe(-0.005);
    });

    it('when backend is unavailable, mapped error message is propagated', () => {
      let error: Error | undefined;
      svc.getMarketSnapshot().subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}market/snapshot`)
        .flush({ detail: 'market data unavailable' }, { status: 503, statusText: 'Service Unavailable' });

      expect(error?.message).toBe('market data unavailable');
    });
  });

  describe('getAssetClassReturns(name)', () => {
    it('GETs /portfolio-analytics/{name}/asset-class-returns', () => {
      svc.getAssetClassReturns('myport').subscribe();

      const req = http.expectOne(
        (r) => r.url === `${API}portfolio-analytics/myport/asset-class-returns`,
      );
      expect(req.request.method).toBe('GET');
      req.flush({ returns: [], asOf: '2024-06-05' });
    });

    it('URI-encodes the portfolio name', () => {
      svc.getAssetClassReturns('my port').subscribe();

      const req = http.expectOne(
        (r) => r.url === `${API}portfolio-analytics/my%20port/asset-class-returns`,
      );
      req.flush({ returns: [], asOf: '2024-06-05' });
    });

    it('when response arrives, returns rows and asOf are mapped', () => {
      const payload: ApiAssetClassReturnsResponse = {
        returns: [{ name: 'Equities', '1D': 0.01, '1W': 0.02, '1M': 0.05, 'YTD': 0.12 }],
        asOf: '2024-06-05',
      };
      let result: ApiAssetClassReturnsResponse | undefined;
      svc.getAssetClassReturns('myport').subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url === `${API}portfolio-analytics/myport/asset-class-returns`)
        .flush(payload);

      expect(result?.asOf).toBe('2024-06-05');
      expect(result?.returns[0].name).toBe('Equities');
    });

    it('when portfolio not found, mapped error message is propagated', () => {
      let error: Error | undefined;
      svc.getAssetClassReturns('ghost').subscribe({ error: (e) => (error = e) });

      http
        .expectOne((r) => r.url === `${API}portfolio-analytics/ghost/asset-class-returns`)
        .flush({ detail: 'portfolio not found' }, { status: 404, statusText: 'Not Found' });

      expect(error?.message).toBe('portfolio not found');
    });
  });

  describe('loadPortfolioData(name, period, benchmark)', () => {
    it('when all four calls succeed, emits a complete PortfolioDataSnapshot', () => {
      let result: PortfolioDataSnapshot | undefined;
      svc.loadPortfolioData('myport', '1Y', 'SPY').subscribe(r => (result = r));

      http.expectOne(r => r.url === `${API}portfolio-analytics/myport/performance-metrics`)
        .flush({ kpis: [{ label: 'Sharpe', value: 1.2, format: 'ratio', change: 0.1, changeLabel: '', sparkline: [] }], nav: 100000, navChangePct: 0.02, currency: 'USD' });
      http.expectOne(r => r.url === `${API}portfolio-analytics/myport/equity-curve`)
        .flush({ points: [{ date: '2024-01-01', portfolio: 1.0, benchmark: 1.0 }], portfolioTotalReturn: 0.1, benchmarkTotalReturn: 0.08 });
      http.expectOne(r => r.url === `${API}portfolio-analytics/myport/allocation`)
        .flush({ nodes: [{ name: 'Tech', value: 0.4, children: [] }], totalPositions: 5, totalSectors: 1 });
      http.expectOne(r => r.url === `${API}portfolio-analytics/myport/asset-class-returns`)
        .flush({ returns: [{ name: 'Equities', '1D': 0.01, '1W': 0.02, '1M': 0.05, 'YTD': 0.12 }], asOf: '2024-06-05' });

      expect(result?.kpis[0].label).toBe('Sharpe');
      expect(result?.nav).toBe(100000);
      expect(result?.currency).toBe('USD');
      expect(result?.equityCurvePoints[0].date).toBe('2024-01-01');
      expect(result?.allocationNodes[0].name).toBe('Tech');
      expect(result?.assetClassReturns[0].name).toBe('Equities');
    });

    it('when performance-metrics fails, snapshot has empty kpis and nav=0, other fields populated', () => {
      let result: PortfolioDataSnapshot | undefined;
      svc.loadPortfolioData('myport', '1Y', 'SPY').subscribe(r => (result = r));

      http.expectOne(r => r.url === `${API}portfolio-analytics/myport/performance-metrics`)
        .flush({ detail: 'not found' }, { status: 404, statusText: 'Not Found' });
      http.expectOne(r => r.url === `${API}portfolio-analytics/myport/equity-curve`)
        .flush({ points: [{ date: '2024-01-01', portfolio: 1.0, benchmark: 1.0 }], portfolioTotalReturn: 0, benchmarkTotalReturn: 0 });
      http.expectOne(r => r.url === `${API}portfolio-analytics/myport/allocation`)
        .flush({ nodes: [], totalPositions: 0, totalSectors: 0 });
      http.expectOne(r => r.url === `${API}portfolio-analytics/myport/asset-class-returns`)
        .flush({ returns: [], asOf: '' });

      expect(result?.kpis).toEqual([]);
      expect(result?.nav).toBe(0);
      expect(result?.equityCurvePoints[0].date).toBe('2024-01-01');
    });
  });

  describe('kpiTrend(kpi)', () => {
    it('when change is positive, returns "up"', () => {
      expect(svc.kpiTrend({ change: 0.1 } as DashboardKPI)).toBe('up');
    });
    it('when change is negative, returns "down"', () => {
      expect(svc.kpiTrend({ change: -0.1 } as DashboardKPI)).toBe('down');
    });
    it('when change is zero, returns "flat"', () => {
      expect(svc.kpiTrend({ change: 0 } as DashboardKPI)).toBe('flat');
    });
  });

  describe('kpiDelta(kpi, nav)', () => {
    it('when format is percent, returns raw change', () => {
      expect(svc.kpiDelta({ format: 'percent', change: 0.025 } as DashboardKPI, 100000)).toBe(0.025);
    });
    it('when format is ratio, returns raw change', () => {
      expect(svc.kpiDelta({ format: 'ratio', change: -12.64 } as DashboardKPI, 100000)).toBe(-12.64);
    });
    it('when format is currency, divides change by nav', () => {
      expect(svc.kpiDelta({ format: 'currency', change: 2000 } as DashboardKPI, 100000)).toBeCloseTo(0.02);
    });
    it('when format is currency and nav is zero, returns 0', () => {
      expect(svc.kpiDelta({ format: 'currency', change: 500 } as DashboardKPI, 0)).toBe(0);
    });
  });

  describe('kpiDeltaFormat(kpi)', () => {
    it('when format is percent, returns "percent"', () => {
      expect(svc.kpiDeltaFormat({ format: 'percent' } as DashboardKPI)).toBe('percent');
    });
    it('when format is currency, returns "percent"', () => {
      expect(svc.kpiDeltaFormat({ format: 'currency' } as DashboardKPI)).toBe('percent');
    });
    it('when format is ratio, returns "absolute" so Sharpe delta is not multiplied by 100', () => {
      expect(svc.kpiDeltaFormat({ format: 'ratio' } as DashboardKPI)).toBe('absolute');
    });
    it('when format is number, returns "absolute"', () => {
      expect(svc.kpiDeltaFormat({ format: 'number' } as DashboardKPI)).toBe('absolute');
    });
  });
});

describe('returnCellBg(value)', () => {
  it('when value > 0.05, returns heatmap-7 with white text', () => {
    const r = returnCellBg(0.06);
    expect(r).toContain('heatmap-7');
    expect(r).toContain('color: white');
  });
  it('when value is exactly 0.05, returns heatmap-6', () => {
    expect(returnCellBg(0.05)).toContain('heatmap-6');
  });
  it('when value > 0.02 and < 0.05, returns heatmap-6 with white text', () => {
    expect(returnCellBg(0.03)).toContain('heatmap-6');
    expect(returnCellBg(0.03)).toContain('color: white');
  });
  it('when value > 0 and <= 0.02, returns heatmap-5 without white text', () => {
    const r = returnCellBg(0.01);
    expect(r).toContain('heatmap-5');
    expect(r).not.toContain('color: white');
  });
  it('when value is 0, returns heatmap-4', () => {
    expect(returnCellBg(0)).toContain('heatmap-4');
  });
  it('when value > -0.02 and < 0, returns heatmap-3', () => {
    expect(returnCellBg(-0.01)).toContain('heatmap-3');
  });
  it('when value > -0.05 and <= -0.02, returns heatmap-2 with white text', () => {
    const r = returnCellBg(-0.03);
    expect(r).toContain('heatmap-2');
    expect(r).toContain('color: white');
  });
  it('when value <= -0.05, returns heatmap-1 with white text', () => {
    expect(returnCellBg(-0.05)).toContain('heatmap-1');
    expect(returnCellBg(-0.06)).toContain('heatmap-1');
  });
});
