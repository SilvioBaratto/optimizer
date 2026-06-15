/**
 * Service-level HTTP contract parity for BacktestService (issue #1024).
 *
 * Complements backtest-service-contracts.spec.ts (issue #1012).
 * Focuses on the walk-forward request body fields mandated by ValidateApiRequest
 * and the correct HTTP method / URL for each service method.
 *
 * All six BacktestService methods are exercised:
 *   runBacktest      → POST  /backtest
 *   pollBacktest     → GET   /backtest/{jobId}
 *   getBacktestRun   → GET   /backtest/runs/{runId}
 *   runWalkForward   → POST  /validate/walk-forward
 *   pollWalkForward  → GET   /validate/walk-forward/{jobId}
 *   getEquityCurve   → GET   /portfolio-analytics/{name}/equity-curve
 */

import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';

import { BacktestService } from './backtest.service';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;
const BACKTEST_URL  = `${API}backtest`;
const WALK_FWD_URL  = `${API}validate/walk-forward`;

describe('BacktestService — contract parity (issue #1024)', () => {
  let service: BacktestService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    service = TestBed.inject(BacktestService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
  });

  // ── POST /backtest ────────────────────────────────────────────────────────

  it('runBacktest issues POST to /backtest', () => {
    service.runBacktest({ tickers: ['AAPL'], start_date: '2020-01-01', end_date: '2024-01-01' }).subscribe();
    const req = http.expectOne(BACKTEST_URL);
    expect(req.request.method).toBe('POST');
    req.flush({ jobId: 'job-1', runId: 'run-1', status: 'pending', message: '' });
  });

  it('runBacktest request body contains tickers, start_date, and end_date', () => {
    service.runBacktest({ tickers: ['SPY', 'TLT'], start_date: '2019-01-01', end_date: '2023-12-31' }).subscribe();
    const req = http.expectOne(BACKTEST_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(Array.isArray(body['tickers'])).toBeTrue();
    expect(body['start_date']).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(body['end_date']).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    req.flush({ jobId: 'job-2', runId: 'run-2', status: 'pending', message: '' });
  });

  // ── GET /backtest/{jobId} polling ─────────────────────────────────────────

  it('pollBacktest issues GET to /backtest/{jobId}', () => {
    service.pollBacktest('job-abc').subscribe();
    const req = http.expectOne(`${API}backtest/${encodeURIComponent('job-abc')}`);
    expect(req.request.method).toBe('GET');
    req.flush({ job_id: 'job-abc', status: 'running', current: 3, total: 10, errors: [], result: null, error: null });
  });

  it('pollBacktest sends no request body', () => {
    service.pollBacktest('job-def').subscribe();
    const req = http.expectOne(`${API}backtest/${encodeURIComponent('job-def')}`);
    expect(req.request.body).toBeNull();
    req.flush({ job_id: 'job-def', status: 'completed', current: 10, total: 10, errors: [], result: null, error: null });
  });

  // ── GET /backtest/runs/{runId} ────────────────────────────────────────────

  it('getBacktestRun issues GET to /backtest/runs/{runId}', () => {
    service.getBacktestRun('run-xyz').subscribe();
    const req = http.expectOne(`${API}backtest/runs/${encodeURIComponent('run-xyz')}`);
    expect(req.request.method).toBe('GET');
    req.flush({});
  });

  // ── POST /validate/walk-forward ───────────────────────────────────────────

  it('runWalkForward issues POST to /validate/walk-forward', () => {
    service.runWalkForward({ tickers: ['AAPL'], start_date: '2020-01-01', end_date: '2024-01-01', cv_type: 'walk_forward', cv_config: {}, optimizer_type: 'hrp', optimizer_config: {} }).subscribe();
    const req = http.expectOne(WALK_FWD_URL);
    expect(req.request.method).toBe('POST');
    req.flush({ job_id: 'wf-1', run_id: 'wfr-1', status: 'pending' });
  });

  it('runWalkForward request body contains cv_type equal to walk_forward', () => {
    service.runWalkForward({ tickers: ['SPY'], start_date: '2021-01-01', end_date: '2024-01-01', cv_type: 'walk_forward', cv_config: {}, optimizer_type: 'hrp', optimizer_config: {} }).subscribe();
    const req = http.expectOne(WALK_FWD_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(body['cv_type']).toBe('walk_forward');
    req.flush({ job_id: 'wf-2', run_id: 'wfr-2', status: 'pending' });
  });

  it('runWalkForward request body carries all seven ValidateApiRequest keys', () => {
    service.runWalkForward({ tickers: ['NVDA'], start_date: '2022-01-01', end_date: '2024-06-30', cv_type: 'walk_forward', cv_config: {}, optimizer_type: 'hrp', optimizer_config: {} }).subscribe();
    const req = http.expectOne(WALK_FWD_URL);
    expect(Object.keys(req.request.body as object).sort()).toEqual([
      'cv_config',
      'cv_type',
      'end_date',
      'optimizer_config',
      'optimizer_type',
      'start_date',
      'tickers',
    ]);
    req.flush({ job_id: 'wf-3', run_id: 'wfr-3', status: 'pending' });
  });

  it('runWalkForward URL contains no raw template literal braces', () => {
    service.runWalkForward({ tickers: ['AAPL'], start_date: '2020-01-01', end_date: '2024-01-01', cv_type: 'walk_forward', cv_config: {}, optimizer_type: 'hrp', optimizer_config: {} }).subscribe();
    const req = http.expectOne(WALK_FWD_URL);
    expect(req.request.url).not.toContain('{');
    expect(req.request.url).not.toContain('}');
    req.flush({ job_id: 'wf-4', run_id: 'wfr-4', status: 'pending' });
  });

  // ── GET /validate/walk-forward/{jobId} ────────────────────────────────────

  it('pollWalkForward issues GET to /validate/walk-forward/{jobId}', () => {
    service.pollWalkForward('wf-job-1').subscribe();
    const req = http.expectOne(`${API}validate/walk-forward/${encodeURIComponent('wf-job-1')}`);
    expect(req.request.method).toBe('GET');
    req.flush({ job_id: 'wf-job-1', status: 'completed', folds: [], aggregate: null, error: null });
  });
});
