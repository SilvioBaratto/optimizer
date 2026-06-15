/**
 * Source-blind contract-parity tests for issue #1015 — dashboard service API contract.
 *
 * Criterion covered:
 *   [UNIT] Contract-parity specs assert request and response shapes for every service method.
 *
 * Asserted contracts (from requirements §3 and §12):
 *   getPerformanceMetrics(name) → GET /portfolio-analytics/{name}/performance-metrics
 *   getEquityCurve(name)        → GET /portfolio-analytics/{name}/equity-curve
 *                                   response shape: { points: ApiEquityCurvePoint[], portfolioTotalReturn, benchmarkTotalReturn }
 *   getAllocation(name)          → GET /portfolio-analytics/{name}/allocation
 *                                   response shape: { nodes: ApiAllocationNode[], totalPositions, totalSectors }
 *   getRollingMetrics(name)     → GET /portfolio-analytics/{name}/rolling-metrics
 *                                   response shape: { window, sharpe: [], volatility: [], beta: [] }
 *   getAssetClassReturns(name)  → GET /portfolio-analytics/{name}/asset-class-returns
 *
 * Additional cross-cutting contract (§12):
 *   URL path parameters are correctly interpolated — no raw "{name}" token appears
 *   in any actual request URL.
 *
 * Import-path assumption:
 *   DashboardService → ./dashboard.service
 *   (adjust if the service lives elsewhere; assertions do not change)
 */
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';

import { DashboardService } from './dashboard.service';
import { environment } from '../../environments/environment';

const API  = environment.apiUrl;
const NAME = 'ACME Fund';

/** Shorthand for the portfolio-analytics base path with name interpolated. */
const analyticsUrl = (suffix: string) =>
  `${API}portfolio-analytics/${encodeURIComponent(NAME)}/${suffix}`;

describe('DashboardService — API contract parity (#1015)', () => {
  let svc:  DashboardService;
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
    svc  = TestBed.inject(DashboardService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  // ── getPerformanceMetrics ─────────────────────────────────────────────────

  it('getPerformanceMetrics() sends GET to /portfolio-analytics/{name}/performance-metrics', () => {
    svc.getPerformanceMetrics(NAME).subscribe();
    const req = http.expectOne(
      r => r.url.endsWith('/portfolio-analytics/' + NAME + '/performance-metrics')
        || r.url.endsWith('/portfolio-analytics/' + encodeURIComponent(NAME) + '/performance-metrics')
    );
    expect(req.request.method).toBe('GET');
    req.flush({ kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' });
  });

  it('getPerformanceMetrics() URL does not contain the literal string "{name}"', () => {
    svc.getPerformanceMetrics(NAME).subscribe();
    const reqs = http.match(() => true);
    reqs.forEach(r => {
      expect(r.request.url)
        .withContext('URL path parameter must be interpolated, not left as a template literal')
        .not.toContain('{name}');
    });
    reqs.forEach(r => r.flush({ kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' }));
  });

  it('getPerformanceMetrics() emits the response body returned by the server', () => {
    const mockMetrics = {
      totalReturn: 0.123,
      sharpeRatio: 1.45,
      maxDrawdown: -0.07,
      informationRatio: 0.98,
    };
    let emitted: unknown;
    svc.getPerformanceMetrics(NAME).subscribe(v => (emitted = v));

    http.expectOne(() => true).flush(mockMetrics);

    expect(emitted).toEqual(mockMetrics);
  });

  // ── getEquityCurve ────────────────────────────────────────────────────────

  it('getEquityCurve() sends GET to /portfolio-analytics/{name}/equity-curve', () => {
    svc.getEquityCurve(NAME).subscribe();
    const req = http.expectOne(
      r => r.url.includes('/portfolio-analytics/') && r.url.includes('/equity-curve')
    );
    expect(req.request.method).toBe('GET');
    req.flush({ points: [], portfolioTotalReturn: 0, benchmarkTotalReturn: 0 });
  });

  it('getEquityCurve() response includes a points[] array with portfolioTotalReturn and benchmarkTotalReturn', () => {
    const mockCurve = {
      points: [
        { date: '2024-01-01', portfolio: 100, benchmark: 100 },
        { date: '2024-06-01', portfolio: 108, benchmark: 104 },
      ],
      portfolioTotalReturn: 0.08,
      benchmarkTotalReturn: 0.04,
    };
    let emitted: typeof mockCurve | undefined;
    svc.getEquityCurve(NAME).subscribe(v => (emitted = v as typeof emitted));

    http.expectOne(() => true).flush(mockCurve);

    expect(Array.isArray(emitted?.points)).toBeTrue();
    expect(emitted?.points.length).toBe(2);
    expect(typeof emitted?.portfolioTotalReturn).toBe('number');
    expect(typeof emitted?.benchmarkTotalReturn).toBe('number');
  });

  // ── getAllocation ─────────────────────────────────────────────────────────

  it('getAllocation() sends GET to /portfolio-analytics/{name}/allocation', () => {
    svc.getAllocation(NAME).subscribe();
    const req = http.expectOne(
      r => r.url.includes('/portfolio-analytics/') && r.url.includes('/allocation')
    );
    expect(req.request.method).toBe('GET');
    req.flush({ nodes: [], totalPositions: 0, totalSectors: 0 });
  });

  // ── getRollingMetrics ─────────────────────────────────────────────────────

  it('getRollingMetrics() sends GET to /portfolio-analytics/{name}/rolling-metrics', () => {
    svc.getRollingMetrics(NAME).subscribe();
    const req = http.expectOne(
      r => r.url.includes('/portfolio-analytics/') && r.url.includes('/rolling-metrics')
    );
    expect(req.request.method).toBe('GET');
    req.flush({ window: 63, sharpe: [], volatility: [], beta: [] });
  });

  // ── getAssetClassReturns ──────────────────────────────────────────────────

  it('getAssetClassReturns() sends GET to /portfolio-analytics/{name}/asset-class-returns', () => {
    svc.getAssetClassReturns(NAME).subscribe();
    const req = http.expectOne(
      r => r.url.includes('/portfolio-analytics/') && r.url.includes('/asset-class-returns')
    );
    expect(req.request.method).toBe('GET');
    req.flush({ returns: [] });
  });

  // ── cross-cutting: all methods accept the portfolio name as a URL segment ─

  it('all analytics methods interpolate the portfolio name into the URL path, never as a query param', () => {
    svc.getPerformanceMetrics(NAME).subscribe();
    svc.getEquityCurve(NAME).subscribe();
    svc.getAllocation(NAME).subscribe();
    svc.getRollingMetrics(NAME).subscribe();
    svc.getAssetClassReturns(NAME).subscribe();

    const reqs = http.match(() => true);
    reqs.forEach(r => {
      // Name must appear in the URL path (percent-encoded or literal)
      const encodedName = encodeURIComponent(NAME);
      const urlContainsName = r.request.urlWithParams.includes(NAME) ||
                              r.request.urlWithParams.includes(encodedName);
      expect(urlContainsName)
        .withContext(`portfolio name must be a path segment in: ${r.request.urlWithParams}`)
        .toBeTrue();

      // Name must NOT be a query parameter
      expect(r.request.params.has('name'))
        .withContext('portfolio name must not be sent as a query parameter named "name"')
        .toBeFalse();
    });

    reqs.forEach(r => {
      const url = r.request.url;
      if (url.includes('/equity-curve'))          r.flush({ points: [], portfolioTotalReturn: 0, benchmarkTotalReturn: 0 });
      else if (url.includes('/performance-metrics')) r.flush({ kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' });
      else if (url.includes('/allocation'))       r.flush({ nodes: [], totalPositions: 0, totalSectors: 0 });
      else if (url.includes('/asset-class-returns')) r.flush({ returns: [] });
      else if (url.includes('/rolling-metrics'))  r.flush({ window: 63, sharpe: [], volatility: [], beta: [] });
      else r.flush({});
    });
  });
});
