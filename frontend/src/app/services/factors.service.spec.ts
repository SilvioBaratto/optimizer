import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { FactorsService } from './factors.service';
import { environment } from '../../environments/environment';
import type {
  FactorComputeRequest,
  FactorQuintileSpreadApiResponse,
  FactorRegimeTiltApiResponse,
  FactorValidateResponse,
} from '../models/factor.model';

const API = environment.apiUrl;

function computeRequest(): FactorComputeRequest {
  return {
    tickers: ['AAPL', 'MSFT'],
    start_date: '2024-01-01',
    end_date: '2024-12-31',
  };
}

describe('FactorsService', () => {
  let svc: FactorsService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        FactorsService,
      ],
    });
    svc = TestBed.inject(FactorsService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  describe('compute() (async)', () => {
    it('POSTs /factors/compute and returns 202 + job_id payload', () => {
      svc.compute(computeRequest()).subscribe();
      const req = http.expectOne(`${API}factors/compute`);
      expect(req.request.method).toBe('POST');
      req.flush(
        { job_id: 'j1', status: 'pending', message: 'Factor compute started.' },
        { status: 202, statusText: 'Accepted' },
      );
    });

    it('when body is sent, sends the full request body', () => {
      const body = computeRequest();
      svc.compute(body).subscribe();
      const req = http.expectOne(`${API}factors/compute`);
      expect(req.request.body).toEqual(body);
      req.flush({ job_id: 'j2', status: 'pending', message: '' }, { status: 202, statusText: 'Accepted' });
    });

    it('propagates 409 when a factor compute job is already running', () => {
      let error: HttpErrorResponse | undefined;
      svc.compute(computeRequest()).subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}factors/compute`)
        .flush({ detail: 'already running' }, { status: 409, statusText: 'Conflict' });
      expect(error?.status).toBe(409);
    });
  });

  describe('pollCompute()', () => {
    it('GETs /factors/compute/{job_id}', () => {
      svc.pollCompute('j1').subscribe();
      const req = http.expectOne(`${API}factors/compute/j1`);
      expect(req.request.method).toBe('GET');
      req.flush({
        job_id: 'j1', status: 'completed', current: 100, total: 100,
        errors: [], result: {}, error: null,
      });
    });

    it('when job_id contains spaces, URI-encodes the path segment', () => {
      svc.pollCompute('job id 1').subscribe();
      const req = http.expectOne(`${API}factors/compute/job%20id%201`);
      expect(req.request.method).toBe('GET');
      req.flush({ job_id: 'job id 1', status: 'running', current: 0, total: 100, errors: [], result: null, error: null });
    });

    it('when poll returns completed, response mapping exposes result', () => {
      let result: import('../models/factor.model').FactorComputeProgress | undefined;
      svc.pollCompute('j2').subscribe((r) => (result = r));
      http.expectOne(`${API}factors/compute/j2`).flush({
        job_id: 'j2', status: 'completed', current: 50, total: 50,
        errors: ['warn'], result: { factor_scores: {} }, error: null,
      });
      expect(result?.status).toBe('completed');
      expect(result?.errors).toEqual(['warn']);
    });

    it('propagates 404 when polling a non-existent job', () => {
      let error: HttpErrorResponse | undefined;
      svc.pollCompute('gone').subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}factors/compute/gone`)
        .flush({ detail: 'not found' }, { status: 404, statusText: 'Not Found' });
      expect(error?.status).toBe(404);
    });
  });

  describe('validate() (synchronous)', () => {
    it('POSTs /factors/validate and returns the IC report', () => {
      let result: FactorValidateResponse | undefined;
      svc.validate({
        tickers: ['AAPL'],
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        factor_type: 'momentum_12_1',
      }).subscribe((r) => (result = r));

      const req = http.expectOne(`${API}factors/validate`);
      expect(req.request.method).toBe('POST');
      req.flush({
        report_date: '2024-12-31',
        factor_type: 'momentum_12_1',
        validation_type: 'in_sample',
        ic_mean: 0.05,
        ic_std: 0.12,
        icir: 0.42,
        t_stat: 2.1,
        p_value: 0.035,
        vif: 1.8,
        details: {},
      });
      expect(result?.ic_mean).toBe(0.05);
    });

    it('when validation_type is out_of_sample, sends it in the request body', () => {
      const body = {
        tickers: ['MSFT'],
        start_date: '2023-01-01',
        end_date: '2023-12-31',
        factor_type: 'value',
        validation_type: 'out_of_sample' as const,
      };
      svc.validate(body).subscribe();
      const req = http.expectOne(`${API}factors/validate`);
      expect(req.request.body.validation_type).toBe('out_of_sample');
      req.flush({ report_date: '2023-12-31', factor_type: 'value', validation_type: 'out_of_sample', ic_mean: null, ic_std: null, icir: null, t_stat: null, p_value: null, vif: null, details: null });
    });

    it('propagates 422 on FactorDataError', () => {
      let error: HttpErrorResponse | undefined;
      svc.validate({
        tickers: ['AAPL'],
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        factor_type: 'momentum_12_1',
      }).subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}factors/validate`)
        .flush({ detail: 'no factor data' }, { status: 422, statusText: 'Unprocessable' });
      expect(error?.status).toBe(422);
    });
  });

  describe('score()', () => {
    it('POSTs /factors/score with composite_method and returns scores + group contributions', () => {
      svc.score({
        tickers: ['AAPL', 'MSFT'],
        score_date: '2024-12-31',
        composite_method: 'ic_weighted',
      }).subscribe();
      const req = http.expectOne(`${API}factors/score`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body.composite_method).toBe('ic_weighted');
      req.flush({
        score_date: '2024-12-31',
        scores: { AAPL: 0.4, MSFT: 0.2 },
        group_contributions: { value: 0.3 },
      });
    });

    it('when score response is returned, maps scores and group_contributions', () => {
      let result: import('../models/factor.model').FactorScoreApiResponse | undefined;
      svc.score({ tickers: ['AAPL', 'MSFT'], score_date: '2024-12-31', composite_method: 'equal_weight' }).subscribe((r) => (result = r));
      http.expectOne(`${API}factors/score`).flush({
        score_date: '2024-12-31',
        scores: { AAPL: 0.7, MSFT: 0.3 },
        group_contributions: { momentum: 0.5 },
      });
      expect(result?.scores['AAPL']).toBe(0.7);
      expect(result?.group_contributions['momentum']).toBe(0.5);
    });

    it('propagates 500 from /factors/score', () => {
      let error: HttpErrorResponse | undefined;
      svc.score({ tickers: ['AAPL'], score_date: '2024-12-31', composite_method: 'equal_weight' }).subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}factors/score`)
        .flush({ detail: 'scoring failed' }, { status: 500, statusText: 'Server Error' });
      expect(error?.status).toBe(500);
    });
  });

  describe('select()', () => {
    it('POSTs /factors/select with method=fixed_count + target_count', () => {
      svc.select({
        tickers: ['AAPL', 'MSFT', 'GOOGL'],
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        method: 'fixed_count',
        target_count: 2,
      }).subscribe();
      const req = http.expectOne(`${API}factors/select`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body.method).toBe('fixed_count');
      req.flush({
        selected_tickers: ['AAPL', 'MSFT'],
        count: 2,
        turnover: null,
        buffer_zone: { entered: [], exited: [] },
      });
    });

    it('when select response is returned, maps selected_tickers and count', () => {
      let result: import('../models/factor.model').FactorSelectApiResponse | undefined;
      svc.select({ tickers: ['AAPL', 'MSFT', 'GOOGL'], start_date: '2024-01-01', end_date: '2024-12-31' }).subscribe((r) => (result = r));
      http.expectOne(`${API}factors/select`).flush({
        selected_tickers: ['AAPL'],
        count: 1,
        turnover: 0.5,
        buffer_zone: { entered: ['AAPL'], exited: ['GOOGL'] },
      });
      expect(result?.count).toBe(1);
      expect(result?.buffer_zone.exited).toEqual(['GOOGL']);
    });

    it('propagates 422 from /factors/select', () => {
      let error: HttpErrorResponse | undefined;
      svc.select({ tickers: ['AAPL'], start_date: '2024-01-01', end_date: '2024-12-31' }).subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}factors/select`)
        .flush({ detail: 'insufficient data' }, { status: 422, statusText: 'Unprocessable' });
      expect(error?.status).toBe(422);
    });
  });

  describe('exposureConstraints()', () => {
    it('POSTs /factors/exposure-constraints and returns constraint matrices', () => {
      svc.exposureConstraints({
        tickers: ['AAPL'],
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        bounds: [-0.5, 0.5],
      }).subscribe();
      const req = http.expectOne(`${API}factors/exposure-constraints`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body.bounds).toEqual([-0.5, 0.5]);
      req.flush({ left_inequality: [[0.1]], right_inequality: [0.5] });
    });

    it('when response is returned, maps left_inequality and right_inequality', () => {
      let result: import('../models/factor.model').FactorExposureConstraintsApiResponse | undefined;
      svc.exposureConstraints({ tickers: ['AAPL', 'MSFT'], start_date: '2024-01-01', end_date: '2024-12-31', bounds: [-1, 1] }).subscribe((r) => (result = r));
      http.expectOne(`${API}factors/exposure-constraints`).flush({
        left_inequality: [[0.2, 0.3]],
        right_inequality: [0.8],
      });
      expect(result?.left_inequality).toEqual([[0.2, 0.3]]);
      expect(result?.right_inequality).toEqual([0.8]);
    });

    it('propagates 422 from /factors/exposure-constraints', () => {
      let error: HttpErrorResponse | undefined;
      svc.exposureConstraints({ tickers: ['AAPL'], start_date: '2024-01-01', end_date: '2024-12-31', bounds: [-0.5, 0.5] }).subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}factors/exposure-constraints`)
        .flush({ detail: 'not enough data' }, { status: 422, statusText: 'Unprocessable' });
      expect(error?.status).toBe(422);
    });
  });

  describe('quintileSpread()', () => {
    it('POSTs /factors/quintile-spread and returns spread cumulative returns', () => {
      let result: FactorQuintileSpreadApiResponse | undefined;
      svc.quintileSpread({
        tickers: ['AAPL', 'MSFT'],
        factor_name: 'momentum_12_1',
        start_date: '2024-01-01',
        end_date: '2024-12-31',
      }).subscribe((r) => (result = r));
      const req = http.expectOne(`${API}factors/quintile-spread`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body.factor_name).toBe('momentum_12_1');
      req.flush({
        quintile_cumulative_returns: { Q1: [0, 0.01], Q5: [0, 0.03] },
        spread_cumulative_return: [0, 0.02],
        annualized_spread: 0.05,
      });
      expect(result?.annualized_spread).toBe(0.05);
    });

    it('propagates 422 from /factors/quintile-spread', () => {
      let error: HttpErrorResponse | undefined;
      svc.quintileSpread({ tickers: ['AAPL'], factor_name: 'beta', start_date: '2024-01-01', end_date: '2024-12-31' }).subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}factors/quintile-spread`)
        .flush({ detail: 'no price history' }, { status: 422, statusText: 'Unprocessable' });
      expect(error?.status).toBe(422);
    });
  });

  describe('regimeTilt()', () => {
    it('POSTs /factors/regime-tilt with group_weights and returns tilted weights', () => {
      let result: FactorRegimeTiltApiResponse | undefined;
      svc.regimeTilt({
        group_weights: { value: 1, momentum: 1 },
      }).subscribe((r) => (result = r));
      const req = http.expectOne(`${API}factors/regime-tilt`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body.group_weights).toEqual({ value: 1, momentum: 1 });
      req.flush({
        regime: 'expansion',
        tilted_weights: { value: 0.9, momentum: 1.1 },
        tilt_multipliers: { value: 0.9, momentum: 1.1 },
      });
      expect(result?.regime).toBe('expansion');
      expect(result?.tilted_weights['momentum']).toBe(1.1);
    });

    it('propagates 500 from /factors/regime-tilt', () => {
      let error: HttpErrorResponse | undefined;
      svc.regimeTilt({ group_weights: { value: 1 } }).subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}factors/regime-tilt`)
        .flush({ detail: 'regime detection failed' }, { status: 500, statusText: 'Server Error' });
      expect(error?.status).toBe(500);
    });
  });

  describe('macroCalibration()', () => {
    it('GETs /views/macro-calibration and returns BL calibration payload', () => {
      svc.macroCalibration({ country: 'USA' }).subscribe();
      const req = http.expectOne(
        (r) => r.url === `${API}views/macro-calibration` && r.params.get('country') === 'USA',
      );
      expect(req.request.method).toBe('GET');
      req.flush({
        phase: 'EARLY_EXPANSION',
        delta: 3.0, tau: 0.05, confidence: 0.8,
        rationale: 'growth accelerating',
        macro_summary: 'pmi above 55',
        timestamp: '2026-04-28T00:00:00Z',
        bl_config: { views: [], tau: 0.05, prior_config: { mu_estimator: 'shrunk', risk_aversion: 3.0, cov_estimator: 'ledoit_wolf' } },
      });
    });

    it('when macro_text is supplied, sends macro_text query param', () => {
      svc.macroCalibration({ macroText: 'PMI above 55' }).subscribe();
      const req = http.expectOne((r) => r.url === `${API}views/macro-calibration`);
      expect(req.request.params.get('macro_text')).toBe('PMI above 55');
      req.flush({ phase: 'EXPANSION', delta: 2.5, tau: 0.05, confidence: 0.7, rationale: '', macro_summary: '', timestamp: '', bl_config: { views: [], tau: 0.05, prior_config: { mu_estimator: 'shrunk', risk_aversion: 2.5, cov_estimator: 'empirical' } } });
    });

    it('when refresh=true, sends refresh=true query param', () => {
      svc.macroCalibration({ refresh: true }).subscribe();
      const req = http.expectOne((r) => r.url === `${API}views/macro-calibration`);
      expect(req.request.params.get('refresh')).toBe('true');
      req.flush({ phase: 'SLOWDOWN', delta: 2.0, tau: 0.05, confidence: 0.6, rationale: '', macro_summary: '', timestamp: '', bl_config: { views: [], tau: 0.05, prior_config: { mu_estimator: 'empirical', risk_aversion: 2.0, cov_estimator: 'empirical' } } });
    });

    it('when no query is provided, omits all query params', () => {
      svc.macroCalibration({}).subscribe();
      const req = http.expectOne(`${API}views/macro-calibration`);
      expect(req.request.params.keys().length).toBe(0);
      req.flush({ phase: 'RECOVERY', delta: 3.5, tau: 0.05, confidence: 0.9, rationale: '', macro_summary: '', timestamp: '', bl_config: { views: [], tau: 0.05, prior_config: { mu_estimator: 'shrunk', risk_aversion: 3.5, cov_estimator: 'ledoit_wolf' } } });
    });

    it('when macro-calibration response is returned, maps phase and delta', () => {
      let result: import('../core/models/macro-intelligence.model').MacroCalibrationResponse | undefined;
      svc.macroCalibration({ country: 'EUR' }).subscribe((r) => (result = r));
      http.expectOne((r) => r.url === `${API}views/macro-calibration`).flush({
        phase: 'LATE_EXPANSION', delta: 4.0, tau: 0.03, confidence: 0.75,
        rationale: 'yield curve flattening',
        macro_summary: 'gdp slowing',
        timestamp: '2026-01-01T00:00:00Z',
        bl_config: { views: [], tau: 0.03, prior_config: { mu_estimator: 'ew', risk_aversion: 4.0, cov_estimator: 'ew' } },
      });
      expect(result?.phase).toBe('LATE_EXPANSION');
      expect(result?.delta).toBe(4.0);
    });

    it('propagates 503 from /views/macro-calibration', () => {
      let error: HttpErrorResponse | undefined;
      svc.macroCalibration({ country: 'GBR' }).subscribe({ error: (e) => (error = e) });
      http
        .expectOne((r) => r.url === `${API}views/macro-calibration`)
        .flush({ detail: 'service unavailable' }, { status: 503, statusText: 'Service Unavailable' });
      expect(error?.status).toBe(503);
    });
  });

  describe('teObservations()', () => {
    it('GETs /macro-data/te-observations with optional country + indicator filters', () => {
      svc.teObservations({
        country: 'USA',
        indicatorKeys: ['manufacturing_pmi', 'services_pmi'],
      }).subscribe();
      const req = http.expectOne(
        (r) => r.url === `${API}macro-data/te-observations`,
      );
      expect(req.request.params.get('country')).toBe('USA');
      expect(req.request.params.getAll('indicator_keys')).toEqual([
        'manufacturing_pmi',
        'services_pmi',
      ]);
      req.flush([]);
    });

    it('omits query params when no filters are provided', () => {
      svc.teObservations({}).subscribe();
      const req = http.expectOne(`${API}macro-data/te-observations`);
      expect(req.request.params.keys().length).toBe(0);
      req.flush([]);
    });

    it('when start_date, end_date and limit are supplied, sends them as query params', () => {
      svc.teObservations({ startDate: '2024-01-01', endDate: '2024-12-31', limit: 50 }).subscribe();
      const req = http.expectOne((r) => r.url === `${API}macro-data/te-observations`);
      expect(req.request.params.get('start_date')).toBe('2024-01-01');
      expect(req.request.params.get('end_date')).toBe('2024-12-31');
      expect(req.request.params.get('limit')).toBe('50');
      req.flush([]);
    });

    it('when observations are returned, maps the array response', () => {
      let result: import('../models/factor.model').TradingEconomicsObservation[] | undefined;
      svc.teObservations({ country: 'USA' }).subscribe((r) => (result = r));
      const payload: import('../models/factor.model').TradingEconomicsObservation[] = [
        { id: 'obs1', country: 'USA', indicator_key: 'manufacturing_pmi', date: '2024-01-01', value: 52.3, created_at: '', updated_at: '' },
      ];
      http.expectOne((r) => r.url === `${API}macro-data/te-observations`).flush(payload);
      expect(result?.length).toBe(1);
      expect(result?.[0].value).toBe(52.3);
    });

    it('propagates 400 from /macro-data/te-observations', () => {
      let error: HttpErrorResponse | undefined;
      svc.teObservations({ country: 'INVALID' }).subscribe({ error: (e) => (error = e) });
      http
        .expectOne((r) => r.url === `${API}macro-data/te-observations`)
        .flush({ detail: 'invalid country' }, { status: 400, statusText: 'Bad Request' });
      expect(error?.status).toBe(400);
    });
  });
});
