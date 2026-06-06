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

import { DashboardService } from './dashboard.service';
import { environment } from '../../environments/environment';

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
});
