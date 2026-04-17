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
} from '../models/attribution.model';

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
});
