/**
 * Request-shape contract parity for DashboardService (issue #999, Scope 12).
 *
 * Asserts HTTP method, exact interpolated URL (no raw `{token}`, path params
 * encoded), query-parameter placement (in `req.request.params`, not the body),
 * and body shape for every DashboardService method — via the shared
 * request-assertion helper (#998), so each assertion stays one line.
 */
import { TestBed } from '@angular/core/testing';
import type { HttpTestingController, TestRequest } from '@angular/common/http/testing';

import {
  assertMethod,
  assertQueryParams,
  assertUrl,
  configureTestBed,
  injectHttp,
} from '../../testing';
import { DashboardService } from './dashboard.service';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;
const NAME = 'My Port'; // contains a space → proves encodeURIComponent interpolation
const ENC = encodeURIComponent(NAME);
const ANALYTICS = `${API}portfolio-analytics/${ENC}`;

describe('DashboardService — request contract parity (issue #999)', () => {
  let svc: DashboardService;
  let http: HttpTestingController;

  beforeEach(async () => {
    await configureTestBed({ withHttp: true });
    svc = TestBed.inject(DashboardService);
    http = injectHttp();
  });

  afterEach(() => http.verify());

  /** Capture the single pending request to `url`, flush it, return it for assertions. */
  function capture(url: string): TestRequest {
    const req = http.expectOne((r) => r.url === url);
    req.flush({});
    return req;
  }

  it('when getPerformanceMetrics is called, a GET to the interpolated URL carries period in params', () => {
    svc.getPerformanceMetrics(NAME, '1Y').subscribe({ error: () => {} });
    const req = capture(`${ANALYTICS}/performance-metrics`);
    assertMethod(req, 'GET');
    assertUrl(req, `${ANALYTICS}/performance-metrics`);
    assertQueryParams(req, { period: '1Y' });
  });

  it('when getRollingMetrics is called, a GET carries period and window in params', () => {
    svc.getRollingMetrics(NAME, '3Y', 60).subscribe({ error: () => {} });
    const req = capture(`${ANALYTICS}/rolling-metrics`);
    assertMethod(req, 'GET');
    assertUrl(req, `${ANALYTICS}/rolling-metrics`);
    assertQueryParams(req, { period: '3Y', window: 60 });
  });

  it('when getEquityCurve is called, a GET carries benchmark and period in params', () => {
    svc.getEquityCurve(NAME, 'URTH', '5Y').subscribe({ error: () => {} });
    const req = capture(`${ANALYTICS}/equity-curve`);
    assertMethod(req, 'GET');
    assertUrl(req, `${ANALYTICS}/equity-curve`);
    assertQueryParams(req, { benchmark: 'URTH', period: '5Y' });
  });

  it('when getAllocation is called, a GET to the interpolated allocation URL is issued', () => {
    svc.getAllocation(NAME).subscribe({ error: () => {} });
    const req = capture(`${ANALYTICS}/allocation`);
    assertMethod(req, 'GET');
    assertUrl(req, `${ANALYTICS}/allocation`);
  });

  it('when getMarketSnapshot is called, a GET to the portfolio-independent market URL is issued', () => {
    svc.getMarketSnapshot().subscribe({ error: () => {} });
    const req = capture(`${API}market/snapshot`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}market/snapshot`);
  });

  it('when getAssetClassReturns is called, a GET to the interpolated asset-class URL is issued', () => {
    svc.getAssetClassReturns(NAME).subscribe({ error: () => {} });
    const req = capture(`${ANALYTICS}/asset-class-returns`);
    assertMethod(req, 'GET');
    assertUrl(req, `${ANALYTICS}/asset-class-returns`);
  });
});
