/**
 * Micro-spec for the request-side contract assertion helper (issue #998).
 *
 * Each assertion is proven to FAIL on a mismatch and PASS on a match, using real
 * `TestRequest` objects captured through Angular's `HttpTestingController` (the
 * exact type the per-domain contract specs feed the helper).
 *
 * Tests import directly from the module (not the barrel) so they fail to compile
 * (RED) if the module does not exist yet.
 */
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { HttpClient, HttpParams, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
  type TestRequest,
} from '@angular/common/http/testing';

import {
  assertBodyFieldTypes,
  assertBodyKeys,
  assertMethod,
  assertQueryParams,
  assertUrl,
} from './contract-request.helper';

const BASE = 'https://api.test/v1/';

describe('contract-request.helper', () => {
  let http: HttpClient;
  let mock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    http = TestBed.inject(HttpClient);
    mock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => mock.verify());

  function get(url: string, params?: HttpParams): TestRequest {
    http.get(url, params ? { params } : undefined).subscribe({ next: () => {}, error: () => {} });
    return mock.expectOne((r) => r.method === 'GET' && r.url === url);
  }

  function post(url: string, body: unknown): TestRequest {
    http.post(url, body).subscribe({ next: () => {}, error: () => {} });
    return mock.expectOne((r) => r.method === 'POST' && r.url === url);
  }

  // ── assertMethod ───────────────────────────────────────────────────────────

  it('when the method matches, no error is thrown', () => {
    const req = post(`${BASE}optimize`, {});
    expect(() => assertMethod(req, 'POST')).not.toThrow();
    req.flush({});
  });

  it('when the method differs, an error is thrown', () => {
    const req = post(`${BASE}optimize`, {});
    expect(() => assertMethod(req, 'GET')).toThrow();
    req.flush({});
  });

  // ── assertUrl ──────────────────────────────────────────────────────────────

  it('when the interpolated, encoded URL matches expected, no error is thrown', () => {
    const req = get(`${BASE}portfolio/${encodeURIComponent('a b')}/risk`);
    expect(() => assertUrl(req, `${BASE}portfolio/a%20b/risk`)).not.toThrow();
    req.flush({});
  });

  it('when a raw {token} remains in the URL, an error is thrown', () => {
    const req = get(`${BASE}portfolio/{name}/risk`);
    expect(() => assertUrl(req, `${BASE}portfolio/pf-1/risk`)).toThrow();
    req.flush({});
  });

  it('when the URL differs from the expected path, an error is thrown', () => {
    const req = get(`${BASE}portfolio/other/risk`);
    expect(() => assertUrl(req, `${BASE}portfolio/pf-1/risk`)).toThrow();
    req.flush({});
  });

  // ── assertQueryParams ────────────────────────────────────────────────────────

  it('when the expected query params are present in req.params, no error is thrown', () => {
    const req = get(`${BASE}portfolio/pf-1/risk/var`, new HttpParams().set('lookback', '252'));
    expect(() => assertQueryParams(req, { lookback: 252 })).not.toThrow();
    req.flush({});
  });

  it('when a query param is placed in the body instead of params, an error is thrown', () => {
    const req = post(`${BASE}risk/stress`, { lookback: 252 });
    expect(() => assertQueryParams(req, { lookback: 252 })).toThrow();
    req.flush({});
  });

  it('when a query param value differs, an error is thrown', () => {
    const req = get(`${BASE}portfolio/pf-1/risk/var`, new HttpParams().set('lookback', '63'));
    expect(() => assertQueryParams(req, { lookback: 252 })).toThrow();
    req.flush({});
  });

  // ── assertBodyKeys ───────────────────────────────────────────────────────────

  it('when the body key-set equals expected (order-insensitive), no error is thrown', () => {
    const req = post(`${BASE}optimize`, { tickers: ['A'], start_date: '2024-01-01' });
    expect(() => assertBodyKeys(req, ['start_date', 'tickers'])).not.toThrow();
    req.flush({});
  });

  it('when the body has an extra key, an error is thrown', () => {
    const req = post(`${BASE}optimize`, { tickers: ['A'], extra: 1 });
    expect(() => assertBodyKeys(req, ['tickers'])).toThrow();
    req.flush({});
  });

  it('when the body is missing a key, an error is thrown', () => {
    const req = post(`${BASE}optimize`, { tickers: ['A'] });
    expect(() => assertBodyKeys(req, ['tickers', 'start_date'])).toThrow();
    req.flush({});
  });

  it('when the body has the same number of keys but a different one, an error is thrown', () => {
    const req = post(`${BASE}optimize`, { tickers: ['A'], end_date: '2024-12-31' });
    expect(() => assertBodyKeys(req, ['tickers', 'start_date'])).toThrow();
    req.flush({});
  });

  // ── assertBodyFieldTypes ─────────────────────────────────────────────────────

  it('when each body field matches its expected JS type, no error is thrown', () => {
    const req = post(`${BASE}optimize`, { name: 'mr', count: 3, tickers: ['A'] });
    expect(() =>
      assertBodyFieldTypes(req, { name: 'string', count: 'number', tickers: 'array' }),
    ).not.toThrow();
    req.flush({});
  });

  it('when a body field has the wrong type, an error naming the field is thrown', () => {
    const req = post(`${BASE}optimize`, { count: 'three' });
    let message = '';
    try {
      assertBodyFieldTypes(req, { count: 'number' });
    } catch (err) {
      message = (err as Error).message;
    }
    expect(message).toContain('count');
    req.flush({});
  });
});
