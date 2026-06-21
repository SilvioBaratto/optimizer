/**
 * Request-shape contract parity for OptimizationService (issue #1000, Scope 12).
 *
 * Sibling of optimization-studio-contracts.spec.ts (which is component-level and
 * already near the 550-line budget): this file covers the SERVICE's HTTP methods
 * directly — method, exact interpolated URL, and body — via the #998 helper.
 *
 * Issue #1044: removed contract tests for calibrateDelta, adaptFactorWeights,
 * selectCovRegime, generateViews, opinionPool, entropyPooling — those service
 * methods were orphaned when their panel components were deleted.
 * Non-HTTP members (buildOptimizeBody) issue no request and are intentionally
 * excluded from request-contract assertions.
 */
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
  type TestRequest,
} from '@angular/common/http/testing';

import { assertBodyKeys, assertMethod, assertUrl } from '../../testing';
import { OptimizationService } from './optimization.service';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;
const RUN_ID = 'run 1';
const RUN_ENC = encodeURIComponent(RUN_ID);
const FWD = { tickers: ['AAPL'], start_date: '2024-01-01', end_date: '2024-12-31' };

describe('OptimizationService — request contract parity (issue #1000)', () => {
  let svc: OptimizationService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    svc = TestBed.inject(OptimizationService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function capture(url: string): TestRequest {
    const req = http.expectOne((r) => r.url === url);
    req.flush({});
    return req;
  }

  it('when optimize is called, a POST carries the optimize body field-by-field', () => {
    const body = svc.buildOptimizeBody(
      { optimizerType: 'mean_risk', config: {} },
      ['AAPL'],
      '2024-01-01',
      '2024-12-31',
    );
    svc.optimize(body).subscribe({ error: () => {} });
    const req = capture(`${API}optimize`);
    assertMethod(req, 'POST');
    assertUrl(req, `${API}optimize`);
    assertBodyKeys(req, ['tickers', 'start_date', 'end_date', 'optimizer_type', 'config']);
  });

  it('when getOptimizationRun is called, a GET to the interpolated run URL is issued', () => {
    svc.getOptimizationRun(RUN_ID).subscribe({ error: () => {} });
    const req = capture(`${API}optimize/${RUN_ENC}`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}optimize/${RUN_ENC}`);
  });

  it('when tune is called, a POST to /tune forwards the body verbatim', () => {
    svc.tune(FWD as unknown as Parameters<typeof svc.tune>[0]).subscribe({ error: () => {} });
    const req = capture(`${API}tune`);
    assertMethod(req, 'POST');
    assertUrl(req, `${API}tune`);
    expect(req.request.body).toEqual(FWD);
  });

  it('when applyWeightsToPortfolio is called, it POSTs a snapshot to the resolved portfolio', () => {
    svc.applyWeightsToPortfolio('My Port', { AAPL: 1 }, '2024-01-01').subscribe({ error: () => {} });
    // Step 1: it resolves the portfolio name via GET /portfolio/.
    const listReq = http.expectOne((r) => r.url === `${API}portfolio/`);
    listReq.flush({
      items: [{ id: 'pf', name: 'My Port', description: null, currency: 'USD', benchmark_ticker: 'SPY', is_active: true, created_at: '', updated_at: '' }],
      total: 1,
    });
    // Step 2: it POSTs the snapshot to the interpolated snapshots URL.
    const req = capture(`${API}portfolio/${encodeURIComponent('My Port')}/snapshots`);
    assertMethod(req, 'POST');
    assertUrl(req, `${API}portfolio/${encodeURIComponent('My Port')}/snapshots`);
    assertBodyKeys(req, ['snapshot_date', 'snapshot_type', 'weights']);
  });
});
