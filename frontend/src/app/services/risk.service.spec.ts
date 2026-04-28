import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { RiskService } from './risk.service';
import { environment } from '../../environments/environment';
import type {
  RiskLimitDto,
  RiskLimitListResponse,
  StressScenarioRequest,
} from '../models/risk.model';

const API = environment.apiUrl;

function limit(overrides: Partial<RiskLimitDto> = {}): RiskLimitDto {
  return {
    id: 'l1',
    portfolioId: 'p1',
    metric: 'max_drawdown',
    limitType: 'upper',
    threshold: 0.15,
    currentValue: null,
    isBreached: false,
    lastCheckedAt: null,
    createdAt: '',
    updatedAt: '',
    ...overrides,
  };
}

describe('RiskService', () => {
  let svc: RiskService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        RiskService,
      ],
    });
    svc = TestBed.inject(RiskService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  describe('getVar()', () => {
    it('GETs /portfolio/{name}/risk/var with lookback + method query', () => {
      svc.getVar('my-port', { lookback: 504, method: 'parametric' }).subscribe();
      const req = http.expectOne(
        (r) => r.url === `${API}portfolio/my-port/risk/var`,
      );
      expect(req.request.method).toBe('GET');
      expect(req.request.params.get('lookback')).toBe('504');
      expect(req.request.params.get('method')).toBe('parametric');
      req.flush({
        var: { '95': 0.03 },
        cvar: { '95': 0.05 },
        method: 'parametric',
        lookback: 504,
        nObservations: 500,
      });
    });

    it('propagates 404 when portfolio has no snapshot', () => {
      let error: HttpErrorResponse | undefined;
      svc.getVar('unknown').subscribe({ error: (e) => (error = e) });
      http
        .expectOne((r) => r.url === `${API}portfolio/unknown/risk/var`)
        .flush({ detail: 'no snapshot' }, { status: 404, statusText: 'Not Found' });
      expect(error?.status).toBe(404);
    });
  });

  describe('getCorrelation()', () => {
    it('GETs /portfolio/{name}/risk/correlation and returns clustered matrix', () => {
      svc.getCorrelation('my-port').subscribe();
      const req = http.expectOne((r) => r.url === `${API}portfolio/my-port/risk/correlation`);
      expect(req.request.method).toBe('GET');
      req.flush({
        assets: ['AAPL', 'MSFT'],
        matrix: [[1, 0.8], [0.8, 1]],
        clusterLabels: [0, 0],
      });
    });
  });

  describe('getFactorExposure()', () => {
    it('GETs /portfolio/{name}/risk/factor-exposure', () => {
      svc.getFactorExposure('my-port').subscribe();
      const req = http.expectOne(
        (r) => r.url === `${API}portfolio/my-port/risk/factor-exposure`,
      );
      expect(req.request.method).toBe('GET');
      req.flush({ exposures: { value: 0.2 }, assetExposures: { AAPL: { value: 0.5 } } });
    });
  });

  describe('getConcentration()', () => {
    it('GETs /portfolio/{name}/risk/concentration with top-N param', () => {
      svc.getConcentration('my-port', 10).subscribe();
      const req = http.expectOne((r) => r.url === `${API}portfolio/my-port/risk/concentration`);
      expect(req.request.params.get('n')).toBe('10');
      req.flush({
        assets: [{ ticker: 'AAPL', name: 'Apple', weight: 0.4 }],
        summary: { hhi: 0.2, effectiveN: 5, topNRatio: 0.8 },
      });
    });
  });

  describe('getLiquidity()', () => {
    it('GETs /portfolio/{name}/risk/liquidity with lookback and participation rate', () => {
      svc.getLiquidity('my-port', { lookbackDays: 30, participationRate: 0.2 }).subscribe();
      const req = http.expectOne(
        (r) => r.url === `${API}portfolio/my-port/risk/liquidity`,
      );
      expect(req.request.params.get('lookback_days')).toBe('30');
      expect(req.request.params.get('participation_rate')).toBe('0.2');
      req.flush({
        assets: [],
        summary: { weightedAvgDaysToLiquidate: 1.5 },
      });
    });
  });

  describe('generateStressScenarios()', () => {
    it('POSTs /risk/stress-scenarios with current_portfolio + macro_context', () => {
      const body: StressScenarioRequest = {
        current_portfolio: { AAPL: 0.5, MSFT: 0.5 },
        macro_context: 'Fed raised rates 25bp, VIX elevated',
        n_scenarios: 4,
      };
      svc.generateStressScenarios(body).subscribe();
      const req = http.expectOne(`${API}risk/stress-scenarios`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(body);
      req.flush({ nScenarios: 4, tickers: ['AAPL', 'MSFT'], scenarios: [] });
    });

    it('propagates 502 from LLM failure', () => {
      let error: HttpErrorResponse | undefined;
      svc
        .generateStressScenarios({
          current_portfolio: { AAPL: 1 },
          macro_context: 'a'.repeat(15),
        })
        .subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}risk/stress-scenarios`)
        .flush({ detail: 'llm down' }, { status: 502, statusText: 'Bad Gateway' });
      expect(error?.status).toBe(502);
    });
  });

  describe('Risk-limits CRUD', () => {
    it('GETs /risk/{portfolio_name}/limits and returns list with breachCount', () => {
      let result: RiskLimitListResponse | undefined;
      svc.listLimits('my-port').subscribe((r) => (result = r));
      http.expectOne(`${API}risk/my-port/limits`).flush({
        items: [limit({ isBreached: true })],
        breachCount: 1,
      });
      expect(result?.breachCount).toBe(1);
    });

    it('POSTs /risk/{portfolio_name}/limits to create a new limit', () => {
      svc
        .createLimit('my-port', {
          metric: 'max_drawdown',
          limit_type: 'upper',
          threshold: 0.2,
        })
        .subscribe();
      const req = http.expectOne(`${API}risk/my-port/limits`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body.threshold).toBe(0.2);
      req.flush(limit());
    });

    it('PUTs /risk/{portfolio_name}/limits/{id} to update current value + breach flag', () => {
      svc
        .updateLimit('my-port', 'l1', { current_value: 0.19, is_breached: true })
        .subscribe();
      const req = http.expectOne(`${API}risk/my-port/limits/l1`);
      expect(req.request.method).toBe('PUT');
      expect(req.request.body.is_breached).toBe(true);
      req.flush(limit({ isBreached: true }));
    });

    it('DELETEs /risk/{portfolio_name}/limits/{id}', () => {
      let done = false;
      svc.deleteLimit('my-port', 'l1').subscribe(() => (done = true));
      const req = http.expectOne(`${API}risk/my-port/limits/l1`);
      expect(req.request.method).toBe('DELETE');
      req.flush(null, { status: 204, statusText: 'No Content' });
      expect(done).toBe(true);
    });

    it('URI-encodes the portfolio name in limits endpoints', () => {
      svc.listLimits('my port').subscribe();
      http.expectOne(`${API}risk/my%20port/limits`).flush({ items: [], breachCount: 0 });
    });
  });
});
