import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { BacktestService } from './backtest.service';
import { environment } from '../../environments/environment';
import type {
  BacktestApiRequest,
  BacktestAsyncResponse,
  BacktestProgressResponse,
  EquityCurvePoint,
  ValidateApiRequest,
  ValidateAsyncResponse,
  ValidateProgressResponse,
} from '../models/backtest.model';

const API = environment.apiUrl;

function backtestRequest(): BacktestApiRequest {
  return {
    tickers: ['AAPL', 'MSFT'],
    start_date: '2024-01-01',
    end_date: '2024-12-31',
    pipeline_config: {},
  };
}

describe('BacktestService', () => {
  let svc: BacktestService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        BacktestService,
      ],
    });
    svc = TestBed.inject(BacktestService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  describe('runBacktest()', () => {
    it('POSTs to /backtest and returns the 202 + jobId + runId payload', () => {
      let result: BacktestAsyncResponse | undefined;
      svc.runBacktest(backtestRequest()).subscribe((r) => (result = r));

      const req = http.expectOne(`${API}backtest`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(backtestRequest());

      req.flush(
        { jobId: 'j1', runId: 'r1', status: 'pending', message: 'Backtest started' },
        { status: 202, statusText: 'Accepted' },
      );

      expect(result).toEqual({
        jobId: 'j1',
        runId: 'r1',
        status: 'pending',
        message: 'Backtest started',
      });
    });

    it('propagates 409 when a backtest job is already running', () => {
      let error: HttpErrorResponse | undefined;
      svc.runBacktest(backtestRequest()).subscribe({
        error: (e) => (error = e),
      });

      http
        .expectOne(`${API}backtest`)
        .flush({ detail: 'already running' }, { status: 409, statusText: 'Conflict' });

      expect(error?.status).toBe(409);
    });
  });

  describe('pollBacktest()', () => {
    it('GETs /backtest/{job_id} and returns progress', () => {
      let result: BacktestProgressResponse | undefined;
      svc.pollBacktest('j1').subscribe((r) => (result = r));

      const req = http.expectOne(`${API}backtest/j1`);
      expect(req.request.method).toBe('GET');
      req.flush({
        job_id: 'j1',
        status: 'completed',
        current: 100,
        total: 100,
        errors: [],
        result: { equity_curve: {} },
        error: null,
      });
      expect(result?.status).toBe('completed');
    });

    it('URI-encodes the job_id segment', () => {
      let result: BacktestProgressResponse | undefined;
      svc.pollBacktest('a b/c').subscribe((r) => (result = r));

      http.expectOne(`${API}backtest/a%20b%2Fc`).flush({
        job_id: 'a b/c',
        status: 'running',
        current: 1,
        total: 10,
        errors: [],
        result: null,
        error: null,
      });
      expect(result?.job_id).toBe('a b/c');
    });
  });

  describe('runCrossValidation()', () => {
    it('POSTs to /validate/cross-validation with cv_type and returns 202 payload', () => {
      const body: ValidateApiRequest = {
        tickers: ['AAPL'],
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        cv_type: 'walk_forward',
      };
      let result: ValidateAsyncResponse | undefined;
      svc.runCrossValidation(body).subscribe((r) => (result = r));

      const req = http.expectOne(`${API}validate/cross-validation`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body.cv_type).toBe('walk_forward');
      req.flush(
        { job_id: 'v1', status: 'pending', message: 'CV started' },
        { status: 202, statusText: 'Accepted' },
      );
      expect(result?.job_id).toBe('v1');
    });

    it('propagates 409 when a CV job is already running', () => {
      let error: HttpErrorResponse | undefined;
      svc
        .runCrossValidation({
          tickers: ['AAPL'],
          start_date: '2024-01-01',
          end_date: '2024-12-31',
        })
        .subscribe({
          error: (e) => (error = e),
        });

      http
        .expectOne(`${API}validate/cross-validation`)
        .flush({ detail: 'busy' }, { status: 409, statusText: 'Conflict' });

      expect(error?.status).toBe(409);
    });
  });

  describe('pollCrossValidation()', () => {
    it('GETs /validate/cross-validation/{job_id} and returns progress', () => {
      let result: ValidateProgressResponse | undefined;
      svc.pollCrossValidation('v1').subscribe((r) => (result = r));

      const req = http.expectOne(`${API}validate/cross-validation/v1`);
      expect(req.request.method).toBe('GET');
      req.flush({
        job_id: 'v1',
        status: 'completed',
        current: 10,
        total: 10,
        current_fold: 10,
        total_folds: 10,
        errors: [],
        result: {
          folds: [{ weights: { AAPL: 1 }, measures: { sharpe: 1.2 } }],
          aggregate: { sharpe: 1.2 },
        },
        error: null,
      });
      expect(result?.status).toBe('completed');
      expect(result?.result?.folds?.length).toBe(1);
    });
  });

  describe('getEquityCurve()', () => {
    it('GETs /portfolio-analytics/{name}/equity-curve and returns historical points', () => {
      const payload: EquityCurvePoint[] = [
        { date: '2024-01-01', value: 100 },
        { date: '2024-01-02', value: 101 },
      ];
      let result: EquityCurvePoint[] | undefined;
      svc.getEquityCurve('myport').subscribe((r) => (result = r));

      const req = http.expectOne(`${API}portfolio-analytics/myport/equity-curve`);
      expect(req.request.method).toBe('GET');
      req.flush(payload);
      expect(result).toEqual(payload);
    });

    it('URI-encodes the portfolio name', () => {
      let result: EquityCurvePoint[] | undefined;
      svc.getEquityCurve('my port').subscribe((r) => (result = r));

      http
        .expectOne(`${API}portfolio-analytics/my%20port/equity-curve`)
        .flush([]);
      expect(result).toEqual([]);
    });
  });
});
