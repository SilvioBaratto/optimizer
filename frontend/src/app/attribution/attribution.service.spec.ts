import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { AttributionService } from './attribution.service';
import { environment } from '../../environments/environment';
import type {
  BrinsonApiRequest,
  BrinsonApiResponse,
  FactorAttributionApiRequest,
  FactorAttributionApiResponse,
} from './attribution.model';

const API = environment.apiUrl;

function brinsonRequest(): BrinsonApiRequest {
  return {
    portfolio_weights: { AAPL: 0.5, MSFT: 0.5 },
    benchmark_weights: { AAPL: 0.4, MSFT: 0.6 },
    start_date: '2024-01-01',
    end_date: '2024-12-31',
  };
}

function factorRequest(): FactorAttributionApiRequest {
  return {
    portfolio_weights: { AAPL: 0.5, MSFT: 0.5 },
    start_date: '2024-01-01',
    end_date: '2024-12-31',
  };
}

describe('AttributionService', () => {
  let svc: AttributionService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        AttributionService,
      ],
    });
    svc = TestBed.inject(AttributionService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  describe('brinson()', () => {
    it('POSTs /attribution/brinson with portfolio + benchmark weights and the date range', () => {
      const body = brinsonRequest();
      let result: BrinsonApiResponse | undefined;
      svc.brinson(body).subscribe((r) => (result = r));

      const req = http.expectOne(`${API}attribution/brinson`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(body);

      req.flush({
        sectors: [
          {
            sector: 'Technology',
            portfolioWeight: 1.0,
            benchmarkWeight: 1.0,
            portfolioReturn: 0.12,
            benchmarkReturn: 0.10,
            allocationEffect: 0.005,
            selectionEffect: 0.015,
            interactionEffect: 0.001,
            totalEffect: 0.021,
          },
        ],
        totalAllocation: 0.005,
        totalSelection: 0.015,
        totalInteraction: 0.001,
        totalActiveReturn: 0.021,
        portfolioReturn: 0.12,
        benchmarkReturn: 0.10,
      });

      expect(result?.totalActiveReturn).toBe(0.021);
      expect(result?.sectors[0].allocationEffect).toBe(0.005);
    });

    it('propagates 422 when weights do not sum to 1', () => {
      let error: HttpErrorResponse | undefined;
      svc.brinson(brinsonRequest()).subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}attribution/brinson`)
        .flush(
          { detail: 'weights do not sum to 1' },
          { status: 422, statusText: 'Unprocessable' },
        );
      expect(error?.status).toBe(422);
    });
  });

  describe('factor()', () => {
    it('POSTs /attribution/factor and returns factors + residual', () => {
      const body = factorRequest();
      let result: FactorAttributionApiResponse | undefined;
      svc.factor(body).subscribe((r) => (result = r));

      const req = http.expectOne(`${API}attribution/factor`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(body);

      req.flush({
        factors: [
          { factorName: 'value', exposure: 0.3, factorReturn: 0.05, contribution: 0.015 },
          { factorName: 'momentum', exposure: -0.1, factorReturn: 0.08, contribution: -0.008 },
        ],
        portfolioReturn: 0.10,
        explainedReturn: 0.007,
        residual: 0.093,
      });

      expect(result?.factors.length).toBe(2);
      expect(result?.residual).toBe(0.093);
    });

    it('propagates 422 when portfolio weights are invalid', () => {
      let error: HttpErrorResponse | undefined;
      svc.factor(factorRequest()).subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}attribution/factor`)
        .flush({ detail: 'bad weights' }, { status: 422, statusText: 'Unprocessable' });
      expect(error?.status).toBe(422);
    });

    it('propagates 500 on backend failure', () => {
      let error: HttpErrorResponse | undefined;
      svc.factor(factorRequest()).subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}attribution/factor`)
        .flush({ detail: 'factor scores missing' }, { status: 500, statusText: 'Server Error' });
      expect(error?.status).toBe(500);
    });
  });

  describe('error propagation via mapHttpError re-throw (issue #911)', () => {
    // attribution.service.ts wires `catchError(mapHttpError())` =
    // `(err) => throwError(() => err)` — a pure identity re-throw. The
    // module-private extractApiMessage / readBodyMessage /
    // formatValidationErrors helpers are never called in the chain (dead code
    // relative to the HTTP flow); their branches are unreachable without a
    // test-only production export, which review rejected. These tests assert
    // the real contract: the RAW HttpErrorResponse propagates (status + body
    // on error.error).
    it('when factor times out (status 0), the subscriber error fires', () => {
      let error: HttpErrorResponse | undefined;
      svc.factor(factorRequest()).subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}attribution/factor`)
        .error(new ProgressEvent('timeout'), { status: 0 });

      expect(error instanceof HttpErrorResponse).toBe(true);
      expect(error?.status).toBe(0);
    });

    it('when brinson returns 503, error.status is 503', () => {
      let error: HttpErrorResponse | undefined;
      svc.brinson(brinsonRequest()).subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}attribution/brinson`)
        .flush({ detail: 'down' }, { status: 503, statusText: 'Service Unavailable' });

      expect(error?.status).toBe(503);
    });

    it('when factor returns 422 with a validation_errors body, the body is on error.error', () => {
      let error: HttpErrorResponse | undefined;
      svc.factor(factorRequest()).subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}attribution/factor`)
        .flush(
          { validation_errors: { portfolio_weights: ['must sum to 1'] } },
          { status: 422, statusText: 'Unprocessable' },
        );

      expect(error?.status).toBe(422);
      expect(error?.error).toEqual({
        validation_errors: { portfolio_weights: ['must sum to 1'] },
      });
    });

    it('when brinson returns 422 with a nested error.details.validation_errors body, the body is reachable on error.error', () => {
      let error: HttpErrorResponse | undefined;
      svc.brinson(brinsonRequest()).subscribe({ error: (e) => (error = e) });

      const nested = {
        error: { details: { validation_errors: { benchmark_weights: ['required'] } } },
      };
      http
        .expectOne(`${API}attribution/brinson`)
        .flush(nested, { status: 422, statusText: 'Unprocessable' });

      expect(error?.status).toBe(422);
      expect(error?.error).toEqual(nested);
    });
  });

  describe('malformed success body — no schema validation (issue #911)', () => {
    it('when /attribution/factor returns a body missing factors, the result is defined and factors is undefined', () => {
      let result: FactorAttributionApiResponse | undefined;
      svc.factor(factorRequest()).subscribe((r) => (result = r));

      http
        .expectOne(`${API}attribution/factor`)
        .flush({ portfolioReturn: 0.1, residual: 0.1 });

      expect(result).toBeDefined();
      expect(result?.factors).toBeUndefined();
    });
  });

  describe('request cancellation (issue #911)', () => {
    it('when the caller unsubscribes before the flush, brinson never emits to next', () => {
      let emitted = false;
      const sub = svc
        .brinson(brinsonRequest())
        .subscribe({ next: () => (emitted = true), error: () => (emitted = true) });

      sub.unsubscribe();

      // A cancelled TestRequest cannot be flushed (Angular 21.1.1 throws), and
      // default verify() counts cancelled requests as open — so drain it with
      // expectOne (removes it from the open list) and assert its cancelled
      // state instead of flushing.
      const req = http.expectOne(`${API}attribution/brinson`);
      expect(req.cancelled).toBe(true);
      expect(emitted).toBe(false);
    });
  });
});
