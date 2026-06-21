import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import {
  OptimizationService,
  type RunConfig,
} from './optimization.service';
import { environment } from '../../environments/environment';
import type {
  OptimizationRunResponse,
  OptimizeAsyncResponse,
  OptimizeRequest,
  TuneJobCreateResponse,
  TuneRequest,
} from '../core/models/optimization.model';

const API = environment.apiUrl;

function baseRequest(): OptimizeRequest {
  return {
    tickers: ['AAPL', 'MSFT'],
    start_date: '2024-01-01',
    end_date: '2024-12-31',
    optimizer_type: 'mean_risk',
    config: {},
  };
}

function syncResponse(overrides: Partial<OptimizationRunResponse> = {}): OptimizationRunResponse {
  return {
    id: 'run-1',
    portfolioId: null,
    jobId: null,
    status: 'completed',
    optimizerType: 'mean_risk',
    universeTickers: ['AAPL', 'MSFT'],
    config: {},
    weights: { AAPL: 0.6, MSFT: 0.4 },
    metrics: { sharpe: 1.2 },
    riskContributions: { AAPL: 0.55, MSFT: 0.45 },
    efficientFrontier: [{ risk: 0.1, return: 0.08, sharpe: 0.8 }],
    errorMessage: null,
    solverLog: null,
    durationSeconds: 1.2,
    createdAt: '2026-04-17T00:00:00Z',
    updatedAt: '2026-04-17T00:00:01Z',
    ...overrides,
  };
}

describe('OptimizationService', () => {
  let svc: OptimizationService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        OptimizationService,
      ],
    });
    svc = TestBed.inject(OptimizationService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  describe('optimize()', () => {
    it('POSTs the request body to /optimize and returns the sync result', () => {
      const body = baseRequest();
      let result: OptimizationRunResponse | OptimizeAsyncResponse | undefined;
      svc.optimize(body).subscribe((r) => (result = r));

      const req = http.expectOne(`${API}optimize`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(body);

      const payload = syncResponse();
      req.flush(payload);
      expect(result).toEqual(payload);
    });

    it('returns the async 202 payload when the backend enqueues a job', () => {
      const body: OptimizeRequest = { ...baseRequest(), tickers: new Array(60).fill('A') };
      let result: OptimizationRunResponse | OptimizeAsyncResponse | undefined;
      svc.optimize(body).subscribe((r) => (result = r));

      const req = http.expectOne(`${API}optimize`);
      const payload: OptimizeAsyncResponse = { job_id: 'j-1', run_id: 'r-1' };
      req.flush(payload, { status: 202, statusText: 'Accepted' });

      expect(result).toEqual(payload);
    });

    it('propagates 4xx errors from /optimize', () => {
      let error: HttpErrorResponse | undefined;
      svc.optimize(baseRequest()).subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}optimize`)
        .flush({ detail: 'bad ticker' }, { status: 422, statusText: 'Unprocessable' });

      expect(error?.status).toBe(422);
    });
  });

  describe('getOptimizationRun()', () => {
    it('GETs /optimize/{runId} and returns the run response', () => {
      let result: OptimizationRunResponse | undefined;
      svc.getOptimizationRun('run-1').subscribe((r) => (result = r));

      const req = http.expectOne(`${API}optimize/run-1`);
      expect(req.request.method).toBe('GET');
      req.flush(syncResponse());
      expect(result?.id).toBe('run-1');
    });

    it('URI-encodes the run_id path segment', () => {
      let result: OptimizationRunResponse | undefined;
      svc.getOptimizationRun('a b/c').subscribe((r) => (result = r));
      http.expectOne(`${API}optimize/a%20b%2Fc`).flush(syncResponse({ id: 'a b/c' }));
      expect(result?.id).toBe('a b/c');
    });

    it('propagates 404 when the run_id does not exist', () => {
      let error: HttpErrorResponse | undefined;
      svc.getOptimizationRun('gone').subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}optimize/gone`)
        .flush({ detail: 'run not found' }, { status: 404, statusText: 'Not Found' });
      expect(error?.status).toBe(404);
    });
  });

  describe('tune()', () => {
    it('POSTs to /tune and returns 202 + job_id', () => {
      const body: TuneRequest = {
        tickers: ['AAPL'],
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        optimizer_type: 'mean_risk',
        search_space: { l2_coef: [0.0, 0.1] },
      };
      let result: TuneJobCreateResponse | undefined;
      svc.tune(body).subscribe((r) => (result = r));

      const req = http.expectOne(`${API}tune`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(body);
      const payload: TuneJobCreateResponse = {
        job_id: 'tj-1',
        status: 'pending',
        message: '',
      };
      req.flush(payload, { status: 202, statusText: 'Accepted' });
      expect(result).toEqual(payload);
    });

    it('surfaces a 409 when a tune job is already running', () => {
      let error: HttpErrorResponse | undefined;
      svc
        .tune({
          tickers: ['AAPL'],
          start_date: '2024-01-01',
          end_date: '2024-12-31',
          optimizer_type: 'mean_risk',
          search_space: {},
        })
        .subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}tune`)
        .flush({ detail: 'already running' }, { status: 409, statusText: 'Conflict' });

      expect(error?.status).toBe(409);
    });
  });

  describe('isAsyncResponse()', () => {
    it('returns true for {job_id, run_id}', () => {
      expect(
        OptimizationService.isAsyncResponse({ job_id: 'j', run_id: 'r' }),
      ).toBe(true);
    });

    it('returns false for an OptimizationRunResponse', () => {
      expect(OptimizationService.isAsyncResponse(syncResponse())).toBe(false);
    });
  });

  describe('buildOptimizeBody()', () => {
    it('assembles an OptimizeRequest from parts', () => {
      const req: RunConfig = { optimizerType: 'mean_risk', config: { l2_coef: 0.1 } };
      const body = svc.buildOptimizeBody(req, ['AAPL', 'MSFT'], '2024-01-01', '2024-12-31');
      expect(body).toEqual({
        tickers: ['AAPL', 'MSFT'],
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        optimizer_type: 'mean_risk',
        config: { l2_coef: 0.1 },
      });
    });
  });

  describe('applyWeightsToPortfolio()', () => {
    const PORTFOLIO_URL = `${API}portfolio/`;
    const SNAPSHOT_URL = `${API}portfolio/my-portfolio/snapshots`;

    it('when portfolioRef not found in list, errors with "not found" message', () => {
      let error: Error | undefined;
      svc
        .applyWeightsToPortfolio('unknown-id', { AAPL: 1 }, '2026-01-01')
        .subscribe({ error: (e: Error) => (error = e) });

      http
        .expectOne(PORTFOLIO_URL)
        .flush({ items: [{ id: 'other', name: 'other' }] });

      expect(error?.message).toContain('not found');
    });

    it('when portfolioRef matches by id, creates snapshot and emits portfolio name', () => {
      let result: string | undefined;
      svc
        .applyWeightsToPortfolio('pid-1', { AAPL: 0.6, MSFT: 0.4 }, '2026-01-01')
        .subscribe((r) => (result = r));

      http
        .expectOne(PORTFOLIO_URL)
        .flush({ items: [{ id: 'pid-1', name: 'my-portfolio' }] });

      const snapReq = http.expectOne(SNAPSHOT_URL);
      expect(snapReq.request.method).toBe('POST');
      expect(snapReq.request.body).toEqual({
        snapshot_date: '2026-01-01',
        snapshot_type: 'optimization',
        weights: { AAPL: 0.6, MSFT: 0.4 },
      });
      snapReq.flush({ id: 'snap-1' });

      expect(result).toBe('my-portfolio');
    });

    it('when portfolioRef matches by name, creates snapshot and emits portfolio name', () => {
      let result: string | undefined;
      svc
        .applyWeightsToPortfolio('my-portfolio', { AAPL: 1 }, '2026-01-01')
        .subscribe((r) => (result = r));

      http
        .expectOne(PORTFOLIO_URL)
        .flush({ items: [{ id: 'pid-2', name: 'my-portfolio' }] });

      http.expectOne(SNAPSHOT_URL).flush({ id: 'snap-2' });

      expect(result).toBe('my-portfolio');
    });
  });

  describe('malformed success body — no schema validation (issue #911)', () => {
    it('when /optimize returns a body missing weights, the result is defined and weights is undefined', () => {
      let result: OptimizationRunResponse | undefined;
      svc
        .optimize(baseRequest())
        .subscribe((r) => (result = r as OptimizationRunResponse));

      http
        .expectOne(`${API}optimize`)
        .flush({ id: 'run-x', status: 'completed' });

      expect(result).toBeDefined();
      expect(result?.weights).toBeUndefined();
    });
  });
});
