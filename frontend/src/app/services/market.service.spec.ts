import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { MarketService } from './market.service';
import { environment } from '../../environments/environment';
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
});
