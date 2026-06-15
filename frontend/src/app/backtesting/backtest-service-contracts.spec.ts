/**
 * Request-shape contract parity for BacktestService (issue #1012).
 *
 * Distinct from the component-level backtesting-contracts.spec.ts. This file
 * covers the SERVICE layer only — no component rendering, no fixture.
 *
 * Six HTTP methods covered:
 *  1. runBacktest      → POST  /backtest
 *  2. pollBacktest     → GET   /backtest/{jobId}
 *  3. getBacktestRun   → GET   /backtest/runs/{runId}
 *  4. runWalkForward   → POST  /validate/walk-forward
 *  5. pollWalkForward  → GET   /validate/walk-forward/{jobId}
 *  6. getEquityCurve   → GET   /portfolio-analytics/{name}/equity-curve
 *
 * Assertions are request-side only (verb, exact interpolated URL, body key-set,
 * query-param placement). No response field-parity assertions.
 *
 * URL-encoding contract: every path-parameter segment is URL-encoded so that
 * names containing spaces, slashes, or ampersands cannot corrupt the path.
 */
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
  type TestRequest,
} from '@angular/common/http/testing';

import {
  assertBodyKeys,
  assertMethod,
  assertUrl,
} from '../../testing';
import { BacktestService } from './backtest.service';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;
const BACKTEST_URL    = `${API}backtest`;
const WALK_FWD_URL    = `${API}validate/walk-forward`;
const ANALYTICS_BASE  = `${API}portfolio-analytics`;

// Path-segment fixtures that contain characters requiring URL-encoding.
const JOB_ID      = 'job 1';
const JOB_ENC     = encodeURIComponent(JOB_ID);
const RUN_ID      = 'run 1';
const RUN_ENC     = encodeURIComponent(RUN_ID);
const PORT_NAME   = 'My Portfolio';
const PORT_ENC    = encodeURIComponent(PORT_NAME);

// Minimal valid BacktestRequest body (matches the four-key schema from §8).
const BACKTEST_BODY = {
  tickers: ['AAPL', 'MSFT'],
  start_date: '2024-01-01',
  end_date: '2024-12-31',
  pipeline_config: { benchmark: 'SPY' },
};

// Minimal valid ValidateRequest body for walk-forward.
const WALK_FWD_BODY = {
  tickers: ['AAPL', 'MSFT'],
  start_date: '2024-01-01',
  end_date: '2024-12-31',
  cv_type: 'walk_forward',
  cv_config: {},
  optimizer_type: 'mean_risk',
  optimizer_config: {},
};

describe('BacktestService — request contract parity (issue #1012)', () => {
  let svc: BacktestService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    svc  = TestBed.inject(BacktestService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function capture(url: string): TestRequest {
    const req = http.expectOne((r) => r.url === url);
    req.flush({});
    return req;
  }

  // ── 1. runBacktest — POST /backtest ──────────────────────────────────────────

  describe('runBacktest()', () => {
    it('when runBacktest is called, a POST is issued to /backtest', () => {
      svc.runBacktest(BACKTEST_BODY).subscribe({ error: () => {} });
      const req = capture(BACKTEST_URL);
      assertMethod(req, 'POST');
      assertUrl(req, BACKTEST_URL);
    });

    it('when runBacktest is called, the body key-set matches the BacktestRequest schema', () => {
      svc.runBacktest(BACKTEST_BODY).subscribe({ error: () => {} });
      const req = capture(BACKTEST_URL);
      assertBodyKeys(req, ['tickers', 'start_date', 'end_date', 'pipeline_config']);
    });

    it('when runBacktest is called, tickers is an array in the body', () => {
      svc.runBacktest(BACKTEST_BODY).subscribe({ error: () => {} });
      const req = capture(BACKTEST_URL);
      expect(Array.isArray((req.request.body as Record<string, unknown>)['tickers'])).toBe(true);
    });

    it('when runBacktest is called, start_date is an ISO date string in the body', () => {
      svc.runBacktest(BACKTEST_BODY).subscribe({ error: () => {} });
      const req = capture(BACKTEST_URL);
      const body = req.request.body as Record<string, unknown>;
      expect(typeof body['start_date']).toBe('string');
      expect(body['start_date'] as string).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    });

    it('when runBacktest is called, end_date is an ISO date string in the body', () => {
      svc.runBacktest(BACKTEST_BODY).subscribe({ error: () => {} });
      const req = capture(BACKTEST_URL);
      const body = req.request.body as Record<string, unknown>;
      expect(typeof body['end_date']).toBe('string');
      expect(body['end_date'] as string).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    });

    it('when runBacktest is called, pipeline_config is an object in the body', () => {
      svc.runBacktest(BACKTEST_BODY).subscribe({ error: () => {} });
      const req = capture(BACKTEST_URL);
      const body = req.request.body as Record<string, unknown>;
      expect(typeof body['pipeline_config']).toBe('object');
      expect(body['pipeline_config']).not.toBeNull();
    });

    it('when runBacktest is called, no camelCase key appears in the request body', () => {
      svc.runBacktest(BACKTEST_BODY).subscribe({ error: () => {} });
      const req = capture(BACKTEST_URL);
      const camelKeys = Object.keys(req.request.body as object).filter((k) => /[A-Z]/.test(k));
      expect(camelKeys).toEqual([]);
    });

    it('when runBacktest returns 4xx, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.runBacktest(BACKTEST_BODY).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(BACKTEST_URL).flush(
        { detail: 'bad request' },
        { status: 400, statusText: 'Bad Request' },
      );
      expect(err?.status).toBe(400);
    });

    it('when runBacktest returns 5xx, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.runBacktest(BACKTEST_BODY).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(BACKTEST_URL).flush(
        { detail: 'server error' },
        { status: 500, statusText: 'Internal Server Error' },
      );
      expect(err?.status).toBe(500);
    });
  });

  // ── 2. pollBacktest — GET /backtest/{jobId} ──────────────────────────────────

  describe('pollBacktest()', () => {
    it('when pollBacktest is called, a GET is issued to the interpolated /backtest/{jobId} URL', () => {
      svc.pollBacktest(JOB_ID).subscribe({ error: () => {} });
      const req = capture(`${BACKTEST_URL}/${JOB_ENC}`);
      assertMethod(req, 'GET');
      assertUrl(req, `${BACKTEST_URL}/${JOB_ENC}`);
    });

    it('when pollBacktest is called, the URL contains no raw {token} placeholder', () => {
      svc.pollBacktest('job-abc').subscribe({ error: () => {} });
      const req = http.expectOne(`${BACKTEST_URL}/job-abc`);
      expect(req.request.url).not.toMatch(/\{[^}]*\}/);
      req.flush({});
    });

    it('when pollBacktest is called with a job ID containing spaces, the ID is URL-encoded', () => {
      svc.pollBacktest(JOB_ID).subscribe({ error: () => {} });
      const req = http.expectOne(`${BACKTEST_URL}/${JOB_ENC}`);
      expect(req.request.url).toContain(JOB_ENC);
      req.flush({});
    });

    it('when pollBacktest is called with a job ID containing a slash, the slash is encoded', () => {
      const id = 'job/2';
      svc.pollBacktest(id).subscribe({ error: () => {} });
      const req = http.expectOne(`${BACKTEST_URL}/${encodeURIComponent(id)}`);
      expect(req.request.url).toContain(encodeURIComponent(id));
      req.flush({});
    });

    it('when pollBacktest returns 404, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.pollBacktest('gone').subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${BACKTEST_URL}/gone`).flush(
        { detail: 'not found' },
        { status: 404, statusText: 'Not Found' },
      );
      expect(err?.status).toBe(404);
    });
  });

  // ── 3. getBacktestRun — GET /backtest/runs/{runId} ───────────────────────────

  describe('getBacktestRun()', () => {
    it('when getBacktestRun is called, a GET is issued to the interpolated /backtest/runs/{runId} URL', () => {
      svc.getBacktestRun(RUN_ID).subscribe({ error: () => {} });
      const req = capture(`${BACKTEST_URL}/runs/${RUN_ENC}`);
      assertMethod(req, 'GET');
      assertUrl(req, `${BACKTEST_URL}/runs/${RUN_ENC}`);
    });

    it('when getBacktestRun is called, the URL contains no raw {token} placeholder', () => {
      svc.getBacktestRun('run-xyz').subscribe({ error: () => {} });
      const req = http.expectOne(`${BACKTEST_URL}/runs/run-xyz`);
      expect(req.request.url).not.toMatch(/\{[^}]*\}/);
      req.flush({});
    });

    it('when getBacktestRun is called with a run ID containing spaces, the ID is URL-encoded', () => {
      svc.getBacktestRun(RUN_ID).subscribe({ error: () => {} });
      const req = http.expectOne(`${BACKTEST_URL}/runs/${RUN_ENC}`);
      expect(req.request.url).toContain(RUN_ENC);
      req.flush({});
    });

    it('when getBacktestRun is called with a run ID containing ampersands, the ID is URL-encoded', () => {
      const id = 'run&v2';
      svc.getBacktestRun(id).subscribe({ error: () => {} });
      const req = http.expectOne(`${BACKTEST_URL}/runs/${encodeURIComponent(id)}`);
      expect(req.request.url).not.toContain('run&v2');
      req.flush({});
    });

    it('when getBacktestRun returns 404, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.getBacktestRun('no-run').subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${BACKTEST_URL}/runs/no-run`).flush(
        { detail: 'not found' },
        { status: 404, statusText: 'Not Found' },
      );
      expect(err?.status).toBe(404);
    });

    it('when getBacktestRun returns 5xx, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.getBacktestRun('run-err').subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${BACKTEST_URL}/runs/run-err`).flush(
        { detail: 'db error' },
        { status: 503, statusText: 'Service Unavailable' },
      );
      expect(err?.status).toBe(503);
    });
  });

  // ── 4. runWalkForward — POST /validate/walk-forward ─────────────────────────

  describe('runWalkForward()', () => {
    it('when runWalkForward is called, a POST is issued to /validate/walk-forward', () => {
      svc.runWalkForward(WALK_FWD_BODY as Parameters<typeof svc.runWalkForward>[0]).subscribe({ error: () => {} });
      const req = capture(WALK_FWD_URL);
      assertMethod(req, 'POST');
      assertUrl(req, WALK_FWD_URL);
    });

    it('when runWalkForward is called, the body key-set matches the ValidateRequest schema', () => {
      svc.runWalkForward(WALK_FWD_BODY as Parameters<typeof svc.runWalkForward>[0]).subscribe({ error: () => {} });
      const req = capture(WALK_FWD_URL);
      assertBodyKeys(req, [
        'tickers',
        'start_date',
        'end_date',
        'cv_type',
        'cv_config',
        'optimizer_type',
        'optimizer_config',
      ]);
    });

    it('when runWalkForward is called, tickers is an array in the body', () => {
      svc.runWalkForward(WALK_FWD_BODY as Parameters<typeof svc.runWalkForward>[0]).subscribe({ error: () => {} });
      const req = capture(WALK_FWD_URL);
      expect(Array.isArray((req.request.body as Record<string, unknown>)['tickers'])).toBe(true);
    });

    it('when runWalkForward is called, start_date and end_date are ISO date strings in the body', () => {
      svc.runWalkForward(WALK_FWD_BODY as Parameters<typeof svc.runWalkForward>[0]).subscribe({ error: () => {} });
      const req = capture(WALK_FWD_URL);
      const body = req.request.body as Record<string, unknown>;
      expect(body['start_date'] as string).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(body['end_date'] as string).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    });

    it('when runWalkForward is called, cv_type is a string in the body', () => {
      svc.runWalkForward(WALK_FWD_BODY as Parameters<typeof svc.runWalkForward>[0]).subscribe({ error: () => {} });
      const req = capture(WALK_FWD_URL);
      expect(typeof (req.request.body as Record<string, unknown>)['cv_type']).toBe('string');
    });

    it('when runWalkForward is called, no camelCase key appears in the request body', () => {
      svc.runWalkForward(WALK_FWD_BODY as Parameters<typeof svc.runWalkForward>[0]).subscribe({ error: () => {} });
      const req = capture(WALK_FWD_URL);
      const camelKeys = Object.keys(req.request.body as object).filter((k) => /[A-Z]/.test(k));
      expect(camelKeys).toEqual([]);
    });

    it('when runWalkForward returns 5xx, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.runWalkForward(WALK_FWD_BODY as Parameters<typeof svc.runWalkForward>[0]).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(WALK_FWD_URL).flush(
        { detail: 'server error' },
        { status: 500, statusText: 'Internal Server Error' },
      );
      expect(err?.status).toBe(500);
    });
  });

  // ── 5. pollWalkForward — GET /validate/walk-forward/{jobId} ─────────────────

  describe('pollWalkForward()', () => {
    it('when pollWalkForward is called, a GET is issued to the interpolated walk-forward job URL', () => {
      svc.pollWalkForward(JOB_ID).subscribe({ error: () => {} });
      const req = capture(`${WALK_FWD_URL}/${JOB_ENC}`);
      assertMethod(req, 'GET');
      assertUrl(req, `${WALK_FWD_URL}/${JOB_ENC}`);
    });

    it('when pollWalkForward is called, the URL contains no raw {token} placeholder', () => {
      svc.pollWalkForward('wf-job').subscribe({ error: () => {} });
      const req = http.expectOne(`${WALK_FWD_URL}/wf-job`);
      expect(req.request.url).not.toMatch(/\{[^}]*\}/);
      req.flush({});
    });

    it('when pollWalkForward is called with a job ID containing spaces, the ID is URL-encoded', () => {
      svc.pollWalkForward(JOB_ID).subscribe({ error: () => {} });
      const req = http.expectOne(`${WALK_FWD_URL}/${JOB_ENC}`);
      expect(req.request.url).toContain(JOB_ENC);
      req.flush({});
    });

    it('when pollWalkForward is called with a job ID containing a slash, the slash is encoded', () => {
      const id = 'wf/job-2';
      svc.pollWalkForward(id).subscribe({ error: () => {} });
      const req = http.expectOne(`${WALK_FWD_URL}/${encodeURIComponent(id)}`);
      expect(req.request.url).toContain(encodeURIComponent(id));
      req.flush({});
    });

    it('when pollWalkForward returns 404, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.pollWalkForward('gone').subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${WALK_FWD_URL}/gone`).flush(
        { detail: 'not found' },
        { status: 404, statusText: 'Not Found' },
      );
      expect(err?.status).toBe(404);
    });
  });

  // ── 6. getEquityCurve — GET /portfolio-analytics/{name}/equity-curve ────────

  describe('getEquityCurve()', () => {
    it('when getEquityCurve is called, a GET is issued to the interpolated equity-curve URL', () => {
      svc.getEquityCurve(PORT_NAME).subscribe({ error: () => {} });
      const req = capture(`${ANALYTICS_BASE}/${PORT_ENC}/equity-curve`);
      assertMethod(req, 'GET');
      assertUrl(req, `${ANALYTICS_BASE}/${PORT_ENC}/equity-curve`);
    });

    it('when getEquityCurve is called, the portfolio name is URL-encoded in the path', () => {
      svc.getEquityCurve(PORT_NAME).subscribe({ error: () => {} });
      const req = http.expectOne(`${ANALYTICS_BASE}/${PORT_ENC}/equity-curve`);
      expect(req.request.url).toContain(PORT_ENC);
      req.flush({});
    });

    it('when getEquityCurve is called, the URL contains no raw {token} placeholder', () => {
      svc.getEquityCurve('myport').subscribe({ error: () => {} });
      const req = http.expectOne(`${ANALYTICS_BASE}/myport/equity-curve`);
      expect(req.request.url).not.toMatch(/\{[^}]*\}/);
      req.flush({});
    });

    it('when getEquityCurve is called with a name containing a slash, the slash is encoded (no path traversal)', () => {
      const name = 'port/v2';
      svc.getEquityCurve(name).subscribe({ error: () => {} });
      const req = http.expectOne(`${ANALYTICS_BASE}/${encodeURIComponent(name)}/equity-curve`);
      expect(req.request.url).not.toContain('/portfolio-analytics/port/v2/equity-curve');
      req.flush({});
    });

    it('when getEquityCurve is called with a name containing ampersands, the ampersand is encoded', () => {
      const name = 'port&v2';
      svc.getEquityCurve(name).subscribe({ error: () => {} });
      const req = http.expectOne(`${ANALYTICS_BASE}/${encodeURIComponent(name)}/equity-curve`);
      expect(req.request.url).toContain(encodeURIComponent(name));
      req.flush({});
    });

    it('when getEquityCurve returns 404, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.getEquityCurve('ghost').subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${ANALYTICS_BASE}/ghost/equity-curve`).flush(
        { detail: 'portfolio not found' },
        { status: 404, statusText: 'Not Found' },
      );
      expect(err?.status).toBe(404);
    });

    it('when getEquityCurve returns 5xx, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.getEquityCurve('myport').subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${ANALYTICS_BASE}/myport/equity-curve`).flush(
        { detail: 'db error' },
        { status: 503, statusText: 'Service Unavailable' },
      );
      expect(err?.status).toBe(503);
    });
  });
});
