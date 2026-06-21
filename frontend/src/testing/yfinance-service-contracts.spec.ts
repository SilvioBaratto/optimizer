/**
 * Request/response contract parity for YfinanceService (issue #1027).
 *
 * Covers: getProfile(), getPrices() (with query params), getDividends(), getNews()
 * — HTTP method, exact encoded URL, query-param placement, and response field-parity
 * against market_data.json snapshot. Also verifies one<T>() swallows errors (returns
 * null) and list<T>() swallows errors (returns []).
 */
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
  type TestRequest,
} from '@angular/common/http/testing';

import { assertMethod, assertUrl, assertQueryParams } from './contract-request.helper';
import { schemaOf } from './contract-parity';
import { assertFieldParity } from './contract-field-parity';
import { makePriceHistoryResponse, makeTickerProfileResponse } from './domain-fixtures';
import marketDataSnapshot from './contract-snapshots/market_data.json';
import { YfinanceService } from '../app/core/services/yfinance.service';
import { environment } from '../environments/environment';

const API = environment.apiUrl;
const BASE = `${API}yfinance-data/instruments`;

const ID = 'AAPL SP';
const ID_ENC = encodeURIComponent(ID);

function setupHttp(): { http: HttpTestingController } {
  TestBed.configureTestingModule({
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
    ],
  });
  return { http: TestBed.inject(HttpTestingController) };
}

function capture(http: HttpTestingController, url: string, flushBody: object | null = null): TestRequest {
  const req = http.expectOne((r) => r.url === url);
  req.flush(flushBody);
  return req;
}

describe('YfinanceService — request/response contract parity (issue #1027)', () => {
  let svc: YfinanceService;
  let http: HttpTestingController;

  beforeEach(() => {
    ({ http } = setupHttp());
    svc = TestBed.inject(YfinanceService);
  });

  afterEach(() => http.verify());

  // ── getProfile() ───────────────────────────────────────────────────────────

  describe('getProfile(id)', () => {
    it('sends GET to exact encoded instruments/{id}/profile URL', () => {
      svc.getProfile(ID).subscribe();
      const req = capture(http, `${BASE}/${ID_ENC}/profile`, makeTickerProfileResponse());
      assertMethod(req, 'GET');
      assertUrl(req, `${BASE}/${ID_ENC}/profile`);
    });

    it('id containing spaces is percent-encoded in the URL path', () => {
      svc.getProfile(ID).subscribe();
      const req = http.expectOne((r) => r.url === `${BASE}/${ID_ENC}/profile`);
      expect(req.request.url).toContain(ID_ENC);
      req.flush(makeTickerProfileResponse());
    });

    it('response matches TickerProfileResponse snapshot field-parity', () => {
      const FIXTURE = makeTickerProfileResponse();
      svc.getProfile(ID).subscribe();
      http.expectOne((r) => r.url === `${BASE}/${ID_ENC}/profile`).flush(FIXTURE);
      const schema = schemaOf(marketDataSnapshot, 'TickerProfileResponse');
      assertFieldParity(schema, FIXTURE, marketDataSnapshot);
    });

    it('on 4xx, emits null instead of throwing (one<T> swallows errors)', () => {
      let result: unknown = 'sentinel';
      svc.getProfile(ID).subscribe((r) => (result = r));
      http.expectOne((r) => r.url === `${BASE}/${ID_ENC}/profile`)
        .flush('Not found', { status: 404, statusText: 'Not Found' });
      expect(result).toBeNull();
    });
  });

  // ── getPrices() ────────────────────────────────────────────────────────────

  describe('getPrices(id, query?)', () => {
    it('without query, sends GET to exact instruments/{id}/prices URL', () => {
      svc.getPrices(ID).subscribe();
      const req = capture(http, `${BASE}/${ID_ENC}/prices`, [makePriceHistoryResponse()]);
      assertMethod(req, 'GET');
      assertUrl(req, `${BASE}/${ID_ENC}/prices`);
    });

    it('startDate puts start_date in query params (not the body)', () => {
      svc.getPrices(ID, { startDate: '2026-01-01' }).subscribe();
      const req = http.expectOne((r) => r.url === `${BASE}/${ID_ENC}/prices`);
      assertQueryParams(req, { start_date: '2026-01-01' });
      req.flush([]);
    });

    it('endDate puts end_date in query params', () => {
      svc.getPrices(ID, { endDate: '2026-06-01' }).subscribe();
      const req = http.expectOne((r) => r.url === `${BASE}/${ID_ENC}/prices`);
      assertQueryParams(req, { end_date: '2026-06-01' });
      req.flush([]);
    });

    it('limit puts limit in query params', () => {
      svc.getPrices(ID, { limit: 100 }).subscribe();
      const req = http.expectOne((r) => r.url === `${BASE}/${ID_ENC}/prices`);
      assertQueryParams(req, { limit: 100 });
      req.flush([]);
    });

    it('on 5xx, emits [] instead of throwing (list<T> swallows errors)', () => {
      let result: unknown = 'sentinel';
      svc.getPrices(ID).subscribe((r) => (result = r));
      http.expectOne((r) => r.url === `${BASE}/${ID_ENC}/prices`)
        .flush('Error', { status: 500, statusText: 'Server Error' });
      expect(result).toEqual([]);
    });

    it('response item matches PriceHistoryResponse snapshot field-parity', () => {
      const FIXTURE = makePriceHistoryResponse();
      svc.getPrices(ID).subscribe();
      http.expectOne((r) => r.url === `${BASE}/${ID_ENC}/prices`).flush([FIXTURE]);
      const schema = schemaOf(marketDataSnapshot, 'PriceHistoryResponse');
      assertFieldParity(schema, FIXTURE, marketDataSnapshot);
    });
  });

  // ── getDividends() ─────────────────────────────────────────────────────────

  describe('getDividends(id)', () => {
    it('sends GET to exact instruments/{id}/dividends URL', () => {
      svc.getDividends(ID).subscribe();
      const req = capture(http, `${BASE}/${ID_ENC}/dividends`, []);
      assertMethod(req, 'GET');
      assertUrl(req, `${BASE}/${ID_ENC}/dividends`);
    });

    it('on 5xx, emits [] instead of throwing', () => {
      let result: unknown = 'sentinel';
      svc.getDividends(ID).subscribe((r) => (result = r));
      http.expectOne((r) => r.url === `${BASE}/${ID_ENC}/dividends`)
        .flush('Error', { status: 500, statusText: 'Server Error' });
      expect(result).toEqual([]);
    });
  });

  // ── getNews() ──────────────────────────────────────────────────────────────

  describe('getNews(id)', () => {
    it('sends GET to exact instruments/{id}/news URL', () => {
      svc.getNews(ID).subscribe();
      const req = capture(http, `${BASE}/${ID_ENC}/news`, []);
      assertMethod(req, 'GET');
      assertUrl(req, `${BASE}/${ID_ENC}/news`);
    });
  });
});
