import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { YfinanceService } from './yfinance.service';
import { environment } from '../../environments/environment';
import {
  AnalystPriceTarget,
  Dividend,
  PriceHistory,
  TickerProfile,
} from '../models/yfinance.model';

const BASE = `${environment.apiUrl}yfinance-data/instruments`;
const ID = 'instrument-123';

describe('YfinanceService', () => {
  let svc: YfinanceService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        YfinanceService,
      ],
    });
    svc = TestBed.inject(YfinanceService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
  });

  // ─── Single-item endpoints ─────────────────────────────────────────────

  describe('getProfile()', () => {
    it('calls GET /{id}/profile and returns the payload', () => {
      const payload = { id: 'p1', instrument_id: ID, sector: 'Technology' } as TickerProfile;
      let result: TickerProfile | null | undefined;
      svc.getProfile(ID).subscribe((r) => (result = r));

      const req = http.expectOne(`${BASE}/${ID}/profile`);
      expect(req.request.method).toBe('GET');
      req.flush(payload);

      expect(result).toEqual(payload);
    });

    it('falls back to null on HTTP error', () => {
      let result: TickerProfile | null | undefined = undefined;
      svc.getProfile(ID).subscribe((r) => (result = r));

      http
        .expectOne(`${BASE}/${ID}/profile`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toBeNull();
    });
  });

  describe('getPriceTargets()', () => {
    it('returns null on HTTP error (single-item fallback)', () => {
      let result: AnalystPriceTarget | null | undefined = undefined;
      svc.getPriceTargets(ID).subscribe((r) => (result = r));

      http
        .expectOne(`${BASE}/${ID}/price-targets`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toBeNull();
    });
  });

  // ─── List endpoints ────────────────────────────────────────────────────

  describe('getPrices()', () => {
    it('passes start_date, end_date, limit as query params', () => {
      svc
        .getPrices(ID, { startDate: '2026-01-01', endDate: '2026-02-01', limit: 100 })
        .subscribe();

      const req = http.expectOne(
        (r) => r.url === `${BASE}/${ID}/prices`,
      );
      expect(req.request.params.get('start_date')).toBe('2026-01-01');
      expect(req.request.params.get('end_date')).toBe('2026-02-01');
      expect(req.request.params.get('limit')).toBe('100');
      req.flush([]);
    });

    it('calls /prices without params when none are provided', () => {
      svc.getPrices(ID).subscribe();

      const req = http.expectOne((r) => r.url === `${BASE}/${ID}/prices`);
      expect(req.request.params.keys().length).toBe(0);
      req.flush([]);
    });

    it('falls back to [] on HTTP error', () => {
      let result: PriceHistory[] | undefined;
      svc.getPrices(ID).subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url === `${BASE}/${ID}/prices`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toEqual([]);
    });
  });

  describe('getFinancials()', () => {
    it('passes statement_type and period_type as query params', () => {
      svc
        .getFinancials(ID, { statementType: 'income', periodType: 'quarterly' })
        .subscribe();

      const req = http.expectOne(
        (r) => r.url === `${BASE}/${ID}/financials`,
      );
      expect(req.request.params.get('statement_type')).toBe('income');
      expect(req.request.params.get('period_type')).toBe('quarterly');
      req.flush([]);
    });
  });

  describe('getDividends()', () => {
    it('calls /dividends and returns the list', () => {
      const payload: Dividend[] = [
        { id: 'd1', instrument_id: ID, date: '2026-01-01', amount: 0.24, created_at: '', updated_at: '' },
      ];
      let result: Dividend[] | undefined;
      svc.getDividends(ID).subscribe((r) => (result = r));

      http.expectOne(`${BASE}/${ID}/dividends`).flush(payload);

      expect(result).toEqual(payload);
    });
  });

  describe('other list endpoints just hit the right URL', () => {
    const cases: [string, keyof YfinanceService, string][] = [
      ['splits', 'getSplits', 'splits'],
      ['recommendations', 'getRecommendations', 'recommendations'],
      ['institutional-holders', 'getInstitutionalHolders', 'institutional-holders'],
      ['mutualfund-holders', 'getMutualfundHolders', 'mutualfund-holders'],
      ['insider-transactions', 'getInsiderTransactions', 'insider-transactions'],
      ['news', 'getNews', 'news'],
    ];

    for (const [name, method, path] of cases) {
      it(`${name} → GET /${path}`, () => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (svc[method] as (id: string) => { subscribe: (fn: () => void) => void })(ID)
          .subscribe(() => {});
        const req = http.expectOne(`${BASE}/${ID}/${path}`);
        expect(req.request.method).toBe('GET');
        req.flush([]);
      });
    }
  });
});
