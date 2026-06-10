import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { MarketService } from './market.service';
import { environment } from '../../../environments/environment';
import type { ReferenceIndicesResponse } from '../models/dashboard-api.model';

const API = environment.apiUrl;

describe('MarketService', () => {
  let svc: MarketService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        MarketService,
      ],
    });
    svc = TestBed.inject(MarketService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  describe('getIndices()', () => {
    it('GETs /market/indices and returns the ETF list', () => {
      let result: ReferenceIndicesResponse | undefined;
      svc.getIndices().subscribe((r) => (result = r));

      const req = http.expectOne(`${API}market/indices`);
      expect(req.request.method).toBe('GET');

      req.flush({
        indices: [
          { ticker: 'SPY', name: 'SPDR S&P 500', instrumentType: 'ETF' },
          { ticker: 'QQQ', name: 'Invesco QQQ', instrumentType: 'ETF' },
        ],
        total: 2,
      });

      expect(result?.total).toBe(2);
      expect(result?.indices[0].ticker).toBe('SPY');
    });

    it('propagates 500 when the market indices endpoint fails', () => {
      let error: HttpErrorResponse | undefined;
      svc.getIndices().subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}market/indices`)
        .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
      expect(error?.status).toBe(500);
    });
  });

  describe('error-path coverage (issue #914)', () => {
    // market.service.ts has no catchError → raw HttpErrorResponse propagates.
    it('when getIndices times out (status 0), the raw HttpErrorResponse propagates', () => {
      let error: HttpErrorResponse | undefined;
      svc.getIndices().subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}market/indices`)
        .error(new ProgressEvent('timeout'), { status: 0 });

      expect(error instanceof HttpErrorResponse).toBe(true);
      expect(error?.status).toBe(0);
    });

    it('when getIndices returns 503 with a circuit-breaker body, the body is on error.error', () => {
      let error: HttpErrorResponse | undefined;
      svc.getIndices().subscribe({ error: (e) => (error = e) });

      const body = { detail: 'circuit breaker open' };
      http
        .expectOne(`${API}market/indices`)
        .flush(body, { status: 503, statusText: 'Service Unavailable' });

      expect(error?.status).toBe(503);
      expect(error?.error).toEqual(body);
    });

    it('when getIndices returns 422, the raw body is on error.error', () => {
      let error: HttpErrorResponse | undefined;
      svc.getIndices().subscribe({ error: (e) => (error = e) });

      const body = { validation_errors: { instrument_type: ['invalid'] } };
      http
        .expectOne(`${API}market/indices`)
        .flush(body, { status: 422, statusText: 'Unprocessable' });

      expect(error?.status).toBe(422);
      expect(error?.error).toEqual(body);
    });

    it('when getIndices returns a malformed body, the result is defined and indices is undefined', () => {
      let result: ReferenceIndicesResponse | undefined;
      svc.getIndices().subscribe((r) => (result = r));

      http.expectOne(`${API}market/indices`).flush({ total: 0 });

      expect(result).toBeDefined();
      expect(result?.indices).toBeUndefined();
    });

    it('when the caller unsubscribes from getIndices before the flush, next never fires', () => {
      let emitted = false;
      const sub = svc
        .getIndices()
        .subscribe({ next: () => (emitted = true), error: () => (emitted = true) });
      sub.unsubscribe();

      // A cancelled TestRequest cannot be flushed (Angular 21.1.1 throws), and
      // default verify() counts it as open — drain it with expectOne instead.
      const req = http.expectOne(`${API}market/indices`);
      expect(req.cancelled).toBe(true);
      expect(emitted).toBe(false);
    });
  });
});
