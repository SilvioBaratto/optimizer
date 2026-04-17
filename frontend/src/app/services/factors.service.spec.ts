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
      expect(req.request.body.method).toBe('fixed_count');
      req.flush({
        selected_tickers: ['AAPL', 'MSFT'],
        count: 2,
        turnover: null,
        buffer_zone: { entered: [], exited: [] },
      });
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
      req.flush({ left_inequality: [[0.1]], right_inequality: [0.5] });
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
      req.flush({
        quintile_cumulative_returns: { Q1: [0, 0.01], Q5: [0, 0.03] },
        spread_cumulative_return: [0, 0.02],
        annualized_spread: 0.05,
      });
      expect(result?.annualized_spread).toBe(0.05);
    });
  });

  describe('regimeTilt()', () => {
    it('POSTs /factors/regime-tilt with group_weights and returns tilted weights', () => {
      let result: FactorRegimeTiltApiResponse | undefined;
      svc.regimeTilt({
        group_weights: { value: 1, momentum: 1 },
      }).subscribe((r) => (result = r));
      const req = http.expectOne(`${API}factors/regime-tilt`);
      req.flush({
        regime: 'expansion',
        tilted_weights: { value: 0.9, momentum: 1.1 },
        tilt_multipliers: { value: 0.9, momentum: 1.1 },
      });
      expect(result?.regime).toBe('expansion');
      expect(result?.tilted_weights['momentum']).toBe(1.1);
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
        macroSummary: 'pmi above 55',
        blConfig: {},
      });
    });
  });

  describe('regimeHistory()', () => {
    it('GETs /market/regime/history with optional date range', () => {
      svc.regimeHistory({ startDate: '2024-01-01', endDate: '2024-12-31' }).subscribe();
      const req = http.expectOne(
        (r) => r.url === `${API}market/regime/history`,
      );
      expect(req.request.params.get('start_date')).toBe('2024-01-01');
      expect(req.request.params.get('end_date')).toBe('2024-12-31');
      req.flush({ points: [], total: 0 });
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
  });
});
