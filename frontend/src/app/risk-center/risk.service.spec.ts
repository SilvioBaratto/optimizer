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
} from './risk.model';

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

    it('when lookback is provided, sends lookback query param', () => {
      svc.getCorrelation('my-port', 252).subscribe();
      const req = http.expectOne((r) => r.url === `${API}portfolio/my-port/risk/correlation`);
      expect(req.request.params.get('lookback')).toBe('252');
      req.flush({ assets: [], matrix: [], clusterLabels: [] });
    });

    it('propagates 404 when portfolio has no snapshot', () => {
      let error: HttpErrorResponse | undefined;
      svc.getCorrelation('unknown').subscribe({ error: (e) => (error = e) });
      http
        .expectOne((r) => r.url === `${API}portfolio/unknown/risk/correlation`)
        .flush({ detail: 'no snapshot' }, { status: 404, statusText: 'Not Found' });
      expect(error?.status).toBe(404);
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

    it('when factor data is missing, propagates 404', () => {
      let error: HttpErrorResponse | undefined;
      svc.getFactorExposure('unknown').subscribe({ error: (e) => (error = e) });
      http
        .expectOne((r) => r.url === `${API}portfolio/unknown/risk/factor-exposure`)
        .flush({ detail: 'not found' }, { status: 404, statusText: 'Not Found' });
      expect(error?.status).toBe(404);
    });
  });

  describe('getConcentration()', () => {
    it('GETs /portfolio/{name}/risk/concentration with top-N param', () => {
      svc.getConcentration('my-port', 10).subscribe();
      const req = http.expectOne((r) => r.url === `${API}portfolio/my-port/risk/concentration`);
      expect(req.request.method).toBe('GET');
      expect(req.request.params.get('n')).toBe('10');
      req.flush({
        assets: [{ ticker: 'AAPL', name: 'Apple', weight: 0.4 }],
        summary: { hhi: 0.2, effectiveN: 5, topNRatio: 0.8 },
      });
    });

    it('when n is omitted, sends no n query param', () => {
      svc.getConcentration('my-port').subscribe();
      const req = http.expectOne((r) => r.url === `${API}portfolio/my-port/risk/concentration`);
      expect(req.request.params.has('n')).toBe(false);
      req.flush({ assets: [], summary: { hhi: 0, effectiveN: 0, topNRatio: 0 } });
    });

    it('propagates 404 when portfolio has no snapshot', () => {
      let error: HttpErrorResponse | undefined;
      svc.getConcentration('unknown').subscribe({ error: (e) => (error = e) });
      http
        .expectOne((r) => r.url === `${API}portfolio/unknown/risk/concentration`)
        .flush({ detail: 'not found' }, { status: 404, statusText: 'Not Found' });
      expect(error?.status).toBe(404);
    });
  });

  describe('getLiquidity()', () => {
    it('GETs /portfolio/{name}/risk/liquidity with lookback and participation rate', () => {
      svc.getLiquidity('my-port', { lookbackDays: 30, participationRate: 0.2 }).subscribe();
      const req = http.expectOne(
        (r) => r.url === `${API}portfolio/my-port/risk/liquidity`,
      );
      expect(req.request.method).toBe('GET');
      expect(req.request.params.get('lookback_days')).toBe('30');
      expect(req.request.params.get('participation_rate')).toBe('0.2');
      req.flush({
        assets: [],
        summary: { weightedAvgDaysToLiquidate: 1.5 },
      });
    });

    it('when no query is provided, sends no query params', () => {
      svc.getLiquidity('my-port').subscribe();
      const req = http.expectOne((r) => r.url === `${API}portfolio/my-port/risk/liquidity`);
      expect(req.request.params.keys().length).toBe(0);
      req.flush({ assets: [], summary: { weightedAvgDaysToLiquidate: 0 } });
    });

    it('propagates 404 when portfolio has no snapshot', () => {
      let error: HttpErrorResponse | undefined;
      svc.getLiquidity('unknown').subscribe({ error: (e) => (error = e) });
      http
        .expectOne((r) => r.url === `${API}portfolio/unknown/risk/liquidity`)
        .flush({ detail: 'not found' }, { status: 404, statusText: 'Not Found' });
      expect(error?.status).toBe(404);
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

    it('when listLimits encounters 404, propagates the error', () => {
      let error: HttpErrorResponse | undefined;
      svc.listLimits('ghost').subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}risk/ghost/limits`)
        .flush({ detail: 'portfolio not found' }, { status: 404, statusText: 'Not Found' });
      expect(error?.status).toBe(404);
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

    it('when createLimit encounters 422, propagates the error', () => {
      let error: HttpErrorResponse | undefined;
      svc
        .createLimit('my-port', { metric: '', limit_type: 'upper', threshold: -1 })
        .subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}risk/my-port/limits`)
        .flush({ detail: 'invalid threshold' }, { status: 422, statusText: 'Unprocessable Entity' });
      expect(error?.status).toBe(422);
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

    it('when updateLimit encounters 404, propagates the error', () => {
      let error: HttpErrorResponse | undefined;
      svc
        .updateLimit('my-port', 'missing', { current_value: 0.1, is_breached: false })
        .subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}risk/my-port/limits/missing`)
        .flush({ detail: 'limit not found' }, { status: 404, statusText: 'Not Found' });
      expect(error?.status).toBe(404);
    });

    it('DELETEs /risk/{portfolio_name}/limits/{id}', () => {
      let done = false;
      svc.deleteLimit('my-port', 'l1').subscribe(() => (done = true));
      const req = http.expectOne(`${API}risk/my-port/limits/l1`);
      expect(req.request.method).toBe('DELETE');
      req.flush(null, { status: 204, statusText: 'No Content' });
      expect(done).toBe(true);
    });

    it('when deleteLimit encounters 404, propagates the error', () => {
      let error: HttpErrorResponse | undefined;
      svc.deleteLimit('my-port', 'ghost').subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}risk/my-port/limits/ghost`)
        .flush({ detail: 'limit not found' }, { status: 404, statusText: 'Not Found' });
      expect(error?.status).toBe(404);
    });

    it('URI-encodes the portfolio name in limits endpoints', () => {
      svc.listLimits('my port').subscribe();
      http.expectOne(`${API}risk/my%20port/limits`).flush({ items: [], breachCount: 0 });
    });
  });

  describe('error-path coverage (issue #912)', () => {
    // risk.service.ts has zero catchError — every error propagates as a raw
    // HttpErrorResponse. Assert error.status and error.error (the flushed
    // body), never error.message.
    it('when getVar times out (status 0), the raw HttpErrorResponse propagates', () => {
      let error: HttpErrorResponse | undefined;
      svc.getVar('my-port').subscribe({ error: (e) => (error = e) });

      http
        .expectOne((r) => r.url === `${API}portfolio/my-port/risk/var`)
        .error(new ProgressEvent('timeout'), { status: 0 });

      expect(error instanceof HttpErrorResponse).toBe(true);
      expect(error?.status).toBe(0);
    });

    it('when getCorrelation returns 503, error.status is 503', () => {
      let error: HttpErrorResponse | undefined;
      svc.getCorrelation('my-port').subscribe({ error: (e) => (error = e) });

      http
        .expectOne((r) => r.url === `${API}portfolio/my-port/risk/correlation`)
        .flush({ detail: 'down' }, { status: 503, statusText: 'Service Unavailable' });

      expect(error?.status).toBe(503);
    });

    it('when getLiquidity returns 503, error.status is 503', () => {
      let error: HttpErrorResponse | undefined;
      svc.getLiquidity('my-port').subscribe({ error: (e) => (error = e) });

      http
        .expectOne((r) => r.url === `${API}portfolio/my-port/risk/liquidity`)
        .flush({ detail: 'down' }, { status: 503, statusText: 'Service Unavailable' });

      expect(error?.status).toBe(503);
    });

    it('when createLimit returns 422 with a structured validation_errors body, the body is on error.error', () => {
      let error: HttpErrorResponse | undefined;
      svc
        .createLimit('my-port', { metric: 'max_drawdown', limit_type: 'upper', threshold: -1 })
        .subscribe({ error: (e) => (error = e) });

      const body = { detail: 'invalid', validation_errors: { threshold: ['must be > 0'] } };
      http
        .expectOne(`${API}risk/my-port/limits`)
        .flush(body, { status: 422, statusText: 'Unprocessable Entity' });

      expect(error?.status).toBe(422);
      expect(error?.error).toEqual(body);
    });

    it('when generateStressScenarios returns 422 with a nested validation_errors body, the body is reachable on error.error', () => {
      let error: HttpErrorResponse | undefined;
      svc
        .generateStressScenarios({ current_portfolio: { AAPL: 1 }, macro_context: 'a'.repeat(15) })
        .subscribe({ error: (e) => (error = e) });

      const nested = {
        error: { details: { validation_errors: { macro_context: ['too short'] } } },
      };
      http
        .expectOne(`${API}risk/stress-scenarios`)
        .flush(nested, { status: 422, statusText: 'Unprocessable Entity' });

      expect(error?.status).toBe(422);
      expect(error?.error).toEqual(nested);
    });

    it('when getVar returns a malformed body, the result is defined and var/cvar are undefined', () => {
      let result: import('./risk.model').VarApiResponse | undefined;
      svc.getVar('my-port').subscribe((r) => (result = r));

      http.expectOne((r) => r.url === `${API}portfolio/my-port/risk/var`).flush({});

      expect(result).toBeDefined();
      expect(result?.var).toBeUndefined();
      expect(result?.cvar).toBeUndefined();
    });

    it('when the caller unsubscribes from getVar before the flush, next never fires', () => {
      let emitted = false;
      const sub = svc
        .getVar('my-port')
        .subscribe({ next: () => (emitted = true), error: () => (emitted = true) });

      sub.unsubscribe();

      // A cancelled TestRequest cannot be flushed (Angular 21.1.1 throws), and
      // default verify() counts cancelled requests as open — so drain it with
      // expectOne (removes it from the open list) and assert its cancelled
      // state instead of flushing.
      const req = http.expectOne((r) => r.url === `${API}portfolio/my-port/risk/var`);
      expect(req.cancelled).toBe(true);
      expect(emitted).toBe(false);
    });
  });
});
