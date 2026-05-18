import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { OptimizationService } from './optimization.service';
import { environment } from '../../environments/environment';
import type {
  AdaptFactorWeightsResponse,
  CalibrateDeltaResponse,
  EntropyPoolingResponse,
  GenerateViewsResponse,
  OpinionPoolResponse,
  OptimizationRunResponse,
  OptimizeAsyncResponse,
  OptimizeRequest,
  SelectCovRegimeResponse,
  TuneJobCreateResponse,
  TuneRequest,
} from '../models/optimization.model';

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

  describe('LLM moments', () => {
    it('POSTs /llm-moments/calibrate-delta', () => {
      let result: CalibrateDeltaResponse | undefined;
      svc.calibrateDelta({ macro_text: 'high vol regime' }).subscribe((r) => (result = r));

      const req = http.expectOne(`${API}llm-moments/calibrate-delta`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({ macro_text: 'high vol regime' });
      req.flush({ delta: 3.5, rationale: 'elevated VIX' });
      expect(result).toEqual({ delta: 3.5, rationale: 'elevated VIX' });
    });

    it('POSTs /llm-moments/adapt-factor-weights', () => {
      let result: AdaptFactorWeightsResponse | undefined;
      svc
        .adaptFactorWeights({
          macro_indicators: 'gdp 1.5',
          factor_groups: ['value', 'momentum'],
        })
        .subscribe((r) => (result = r));

      const req = http.expectOne(`${API}llm-moments/adapt-factor-weights`);
      req.flush({
        phase: 'expansion',
        weights: { value: 1.1, momentum: 0.9 },
        rationale: 'cycle',
      });
      expect(result?.phase).toBe('expansion');
    });

    it('POSTs /llm-moments/select-cov-regime', () => {
      let result: SelectCovRegimeResponse | undefined;
      svc
        .selectCovRegime({
          news_headlines: ['sell-off'],
          avg_sentiment_score: -0.4,
          realized_vol_30d: 0.22,
        })
        .subscribe((r) => (result = r));

      const req = http.expectOne(`${API}llm-moments/select-cov-regime`);
      req.flush({ estimator_type: 'ledoit_wolf', rationale: 'elevated vol' });
      expect(result?.estimator_type).toBe('ledoit_wolf');
    });

    it('propagates 502 errors from LLM moment endpoints', () => {
      let error: HttpErrorResponse | undefined;
      svc.calibrateDelta({ macro_text: 'x' }).subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}llm-moments/calibrate-delta`)
        .flush({ detail: 'llm down' }, { status: 502, statusText: 'Bad Gateway' });
      expect(error?.status).toBe(502);
    });
  });

  describe('Views endpoints', () => {
    it('POSTs /views/generate for Black-Litterman views', () => {
      let result: GenerateViewsResponse | undefined;
      svc.generateViews({ tickers: ['AAPL', 'MSFT'] }).subscribe((r) => (result = r));

      const req = http.expectOne(`${API}views/generate`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({ tickers: ['AAPL', 'MSFT'] });
      req.flush({
        P: [[1, 0]],
        Q: [0.05],
        view_confidences: [0.8],
        idzorek_alphas: [0.5],
        tickers: ['AAPL', 'MSFT'],
        tickers_missing: [],
      });
      expect(result?.Q).toEqual([0.05]);
    });

    it('POSTs /views/opinion-pool with personas', () => {
      let result: OpinionPoolResponse | undefined;
      svc
        .opinionPool({
          tickers: ['AAPL'],
          personas: ['value', 'momentum'],
        })
        .subscribe((r) => (result = r));

      const req = http.expectOne(`${API}views/opinion-pool`);
      expect(req.request.method).toBe('POST');
      req.flush({
        P: [[1]],
        Q: [0.04],
        view_confidences: [0.7],
        persona_weights: { value: 0.5, momentum: 0.5 },
      });
      expect(result?.persona_weights).toEqual({ value: 0.5, momentum: 0.5 });
    });

    it('POSTs /views/entropy-pooling with date range + tickers', () => {
      let result: EntropyPoolingResponse | undefined;
      svc
        .entropyPooling({
          tickers: ['AAPL', 'MSFT'],
          start_date: '2024-01-01',
          end_date: '2024-12-31',
          mean_views: ['AAPL == 0.10'],
        })
        .subscribe((r) => (result = r));

      const req = http.expectOne(`${API}views/entropy-pooling`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body.mean_views).toEqual(['AAPL == 0.10']);
      req.flush({
        tickers: ['AAPL', 'MSFT'],
        mu: [0.1, 0.05],
        covariance: [
          [0.04, 0.01],
          [0.01, 0.03],
        ],
        effective_sample_size: 123,
      });
      expect(result?.effective_sample_size).toBe(123);
    });

    it('propagates 422 from /views/entropy-pooling', () => {
      let error: HttpErrorResponse | undefined;
      svc
        .entropyPooling({
          tickers: ['AAPL'],
          start_date: '2024-01-01',
          end_date: '2024-12-31',
        })
        .subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}views/entropy-pooling`)
        .flush({ detail: 'no price data' }, { status: 422, statusText: 'Unprocessable' });
      expect(error?.status).toBe(422);
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
});
