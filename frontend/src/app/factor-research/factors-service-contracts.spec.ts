/**
 * Request-shape contract parity for FactorsService (issue #1012).
 *
 * Distinct from the component-level factor-research-contracts.spec.ts. This file
 * covers the SERVICE layer only — no component rendering, no fixture.
 *
 * Ten HTTP methods covered:
 *  1.  compute              → POST /factors/compute
 *  2.  pollCompute          → GET  /factors/compute/{id}
 *  3.  validate             → POST /factors/validate
 *  4.  score                → POST /factors/score
 *  5.  select               → POST /factors/select
 *  6.  exposureConstraints  → POST /factors/exposure-constraints
 *  7.  quintileSpread       → POST /factors/quintile-spread
 *  8.  regimeTilt           → POST /factors/regime-tilt
 *  9.  macroCalibration     → GET  /views/macro-calibration
 * 10.  teObservations       → GET  /macro-data/te-observations
 *
 * Assertions are request-side only (verb, exact interpolated URL, body forwarded,
 * query-param placement). No response field-parity assertions.
 *
 * URL-encoding contract: every path-parameter segment is URL-encoded so that job
 * IDs containing spaces, slashes, or ampersands cannot corrupt the path.
 *
 * Query-param placement contract (methods 9 and 10): parameters must be in the
 * URL query string, NOT in the request body. Callers must translate camelCase
 * option keys to snake_case before appending them to the URL:
 *   macroText  → macro_text
 *   startDate  → start_date
 *   endDate    → end_date
 */
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
  type TestRequest,
} from '@angular/common/http/testing';

import {
  assertBodyKeys,
  assertMethod,
  assertQueryParams,
  assertUrl,
} from '../../testing';
import { FactorsService } from './factors.service';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;

// Path-segment fixture that requires URL-encoding.
const JOB_ID  = 'job 1';
const JOB_ENC = encodeURIComponent(JOB_ID);

// A minimal payload that satisfies every POST method (tickers + dates).
const FWD = { tickers: ['AAPL', 'MSFT'], start_date: '2024-01-01', end_date: '2024-12-31' };

describe('FactorsService — request contract parity (issue #1012)', () => {
  let svc: FactorsService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    svc  = TestBed.inject(FactorsService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function capture(url: string): TestRequest {
    const req = http.expectOne((r) => r.url === url);
    req.flush({});
    return req;
  }

  // ── 1. compute — POST /factors/compute ──────────────────────────────────────

  describe('compute()', () => {
    it('when compute is called, a POST is issued to /factors/compute', () => {
      svc.compute(FWD as Parameters<typeof svc.compute>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/compute`);
      assertMethod(req, 'POST');
      assertUrl(req, `${API}factors/compute`);
    });

    it('when compute is called, the body is forwarded verbatim to /factors/compute', () => {
      svc.compute(FWD as Parameters<typeof svc.compute>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/compute`);
      expect(req.request.body).toEqual(FWD);
    });

    it('when compute is called, tickers is an array in the request body', () => {
      svc.compute(FWD as Parameters<typeof svc.compute>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/compute`);
      expect(Array.isArray((req.request.body as Record<string, unknown>)['tickers'])).toBe(true);
    });

    it('when compute is called, the body contains no camelCase key leakage', () => {
      svc.compute(FWD as Parameters<typeof svc.compute>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/compute`);
      const body = req.request.body as Record<string, unknown>;
      expect(body['startDate']).toBeUndefined();
      expect(body['endDate']).toBeUndefined();
    });

    it('when compute returns 5xx, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.compute(FWD as Parameters<typeof svc.compute>[0]).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${API}factors/compute`).flush(
        { detail: 'compute crashed' },
        { status: 500, statusText: 'Internal Server Error' },
      );
      expect(err?.status).toBe(500);
    });
  });

  // ── 2. pollCompute — GET /factors/compute/{id} ──────────────────────────────

  describe('pollCompute()', () => {
    it('when pollCompute is called, a GET is issued to the interpolated /factors/compute/{id} URL', () => {
      svc.pollCompute(JOB_ID).subscribe({ error: () => {} });
      const req = capture(`${API}factors/compute/${JOB_ENC}`);
      assertMethod(req, 'GET');
      assertUrl(req, `${API}factors/compute/${JOB_ENC}`);
    });

    it('when pollCompute is called, the URL contains no raw {token} placeholder', () => {
      svc.pollCompute('fc-1').subscribe({ error: () => {} });
      const req = http.expectOne(`${API}factors/compute/fc-1`);
      expect(req.request.url).not.toMatch(/\{[^}]*\}/);
      req.flush({});
    });

    it('when pollCompute is called with a job ID containing spaces, the ID is URL-encoded', () => {
      svc.pollCompute(JOB_ID).subscribe({ error: () => {} });
      const req = http.expectOne(`${API}factors/compute/${JOB_ENC}`);
      expect(req.request.url).toContain(JOB_ENC);
      req.flush({});
    });

    it('when pollCompute is called with a job ID containing a slash, the slash is encoded', () => {
      const id = 'fc/2';
      svc.pollCompute(id).subscribe({ error: () => {} });
      const req = http.expectOne(`${API}factors/compute/${encodeURIComponent(id)}`);
      expect(req.request.url).toContain(encodeURIComponent(id));
      req.flush({});
    });

    it('when pollCompute returns 404, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.pollCompute('gone').subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${API}factors/compute/gone`).flush(
        { detail: 'not found' },
        { status: 404, statusText: 'Not Found' },
      );
      expect(err?.status).toBe(404);
    });
  });

  // ── 3. validate — POST /factors/validate ────────────────────────────────────

  describe('validate()', () => {
    it('when validate is called, a POST is issued to /factors/validate', () => {
      svc.validate(FWD as Parameters<typeof svc.validate>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/validate`);
      assertMethod(req, 'POST');
      assertUrl(req, `${API}factors/validate`);
    });

    it('when validate is called, the body is forwarded verbatim to /factors/validate', () => {
      svc.validate(FWD as Parameters<typeof svc.validate>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/validate`);
      expect(req.request.body).toEqual(FWD);
    });

    it('when validate returns 5xx, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.validate(FWD as Parameters<typeof svc.validate>[0]).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${API}factors/validate`).flush(
        { detail: 'validate crashed' },
        { status: 500, statusText: 'Internal Server Error' },
      );
      expect(err?.status).toBe(500);
    });

    it('when validate returns 422, the error propagates with status 422', () => {
      let err: { status: number } | undefined;
      svc.validate(FWD as Parameters<typeof svc.validate>[0]).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${API}factors/validate`).flush(
        { detail: 'invalid tickers' },
        { status: 422, statusText: 'Unprocessable Entity' },
      );
      expect(err?.status).toBe(422);
    });
  });

  // ── 4. score — POST /factors/score ──────────────────────────────────────────

  describe('score()', () => {
    it('when score is called, a POST is issued to /factors/score', () => {
      svc.score(FWD as unknown as Parameters<typeof svc.score>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/score`);
      assertMethod(req, 'POST');
      assertUrl(req, `${API}factors/score`);
    });

    it('when score is called, the body is forwarded verbatim to /factors/score', () => {
      svc.score(FWD as unknown as Parameters<typeof svc.score>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/score`);
      expect(req.request.body).toEqual(FWD);
    });

    it('when score returns 5xx, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.score(FWD as unknown as Parameters<typeof svc.score>[0]).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${API}factors/score`).flush(
        { detail: 'score crashed' },
        { status: 500, statusText: 'Internal Server Error' },
      );
      expect(err?.status).toBe(500);
    });

    it('when score returns 422, the error propagates with status 422', () => {
      let err: { status: number } | undefined;
      svc.score(FWD as unknown as Parameters<typeof svc.score>[0]).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${API}factors/score`).flush(
        { detail: 'bad body' },
        { status: 422, statusText: 'Unprocessable Entity' },
      );
      expect(err?.status).toBe(422);
    });
  });

  // ── 5. select — POST /factors/select ────────────────────────────────────────

  describe('select()', () => {
    it('when select is called, a POST is issued to /factors/select', () => {
      svc.select(FWD as Parameters<typeof svc.select>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/select`);
      assertMethod(req, 'POST');
      assertUrl(req, `${API}factors/select`);
    });

    it('when select is called, the body is forwarded verbatim to /factors/select', () => {
      svc.select(FWD as Parameters<typeof svc.select>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/select`);
      expect(req.request.body).toEqual(FWD);
    });

    it('when select returns 5xx, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.select(FWD as Parameters<typeof svc.select>[0]).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${API}factors/select`).flush(
        { detail: 'server error' },
        { status: 500, statusText: 'Internal Server Error' },
      );
      expect(err?.status).toBe(500);
    });
  });

  // ── 6. exposureConstraints — POST /factors/exposure-constraints ─────────────

  describe('exposureConstraints()', () => {
    it('when exposureConstraints is called, a POST is issued to /factors/exposure-constraints', () => {
      svc.exposureConstraints(FWD as Parameters<typeof svc.exposureConstraints>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/exposure-constraints`);
      assertMethod(req, 'POST');
      assertUrl(req, `${API}factors/exposure-constraints`);
    });

    it('when exposureConstraints is called, the body is forwarded verbatim to /factors/exposure-constraints', () => {
      svc.exposureConstraints(FWD as Parameters<typeof svc.exposureConstraints>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/exposure-constraints`);
      expect(req.request.body).toEqual(FWD);
    });

    it('when exposureConstraints returns 5xx, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.exposureConstraints(FWD as Parameters<typeof svc.exposureConstraints>[0]).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${API}factors/exposure-constraints`).flush(
        { detail: 'crash' },
        { status: 500, statusText: 'Internal Server Error' },
      );
      expect(err?.status).toBe(500);
    });

    it('when exposureConstraints returns 422, the error propagates with status 422', () => {
      let err: { status: number } | undefined;
      svc.exposureConstraints(FWD as Parameters<typeof svc.exposureConstraints>[0]).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${API}factors/exposure-constraints`).flush(
        { detail: 'bad request' },
        { status: 422, statusText: 'Unprocessable Entity' },
      );
      expect(err?.status).toBe(422);
    });
  });

  // ── 7. quintileSpread — POST /factors/quintile-spread ───────────────────────

  describe('quintileSpread()', () => {
    it('when quintileSpread is called, a POST is issued to /factors/quintile-spread', () => {
      svc.quintileSpread(FWD as Parameters<typeof svc.quintileSpread>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/quintile-spread`);
      assertMethod(req, 'POST');
      assertUrl(req, `${API}factors/quintile-spread`);
    });

    it('when quintileSpread is called, the body is forwarded verbatim to /factors/quintile-spread', () => {
      svc.quintileSpread(FWD as Parameters<typeof svc.quintileSpread>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/quintile-spread`);
      expect(req.request.body).toEqual(FWD);
    });

    it('when quintileSpread returns 5xx, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.quintileSpread(FWD as Parameters<typeof svc.quintileSpread>[0]).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${API}factors/quintile-spread`).flush(
        { detail: 'crash' },
        { status: 500, statusText: 'Internal Server Error' },
      );
      expect(err?.status).toBe(500);
    });
  });

  // ── 8. regimeTilt — POST /factors/regime-tilt ───────────────────────────────

  describe('regimeTilt()', () => {
    it('when regimeTilt is called, a POST is issued to /factors/regime-tilt', () => {
      svc.regimeTilt(FWD as unknown as Parameters<typeof svc.regimeTilt>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/regime-tilt`);
      assertMethod(req, 'POST');
      assertUrl(req, `${API}factors/regime-tilt`);
    });

    it('when regimeTilt is called, the body is forwarded verbatim to /factors/regime-tilt', () => {
      svc.regimeTilt(FWD as unknown as Parameters<typeof svc.regimeTilt>[0]).subscribe({ error: () => {} });
      const req = capture(`${API}factors/regime-tilt`);
      expect(req.request.body).toEqual(FWD);
    });

    it('when regimeTilt returns 5xx, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.regimeTilt(FWD as unknown as Parameters<typeof svc.regimeTilt>[0]).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${API}factors/regime-tilt`).flush(
        { detail: 'crash' },
        { status: 500, statusText: 'Internal Server Error' },
      );
      expect(err?.status).toBe(500);
    });

    it('when regimeTilt returns 422, the error propagates with status 422', () => {
      let err: { status: number } | undefined;
      svc.regimeTilt(FWD as unknown as Parameters<typeof svc.regimeTilt>[0]).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne(`${API}factors/regime-tilt`).flush(
        { detail: 'bad request' },
        { status: 422, statusText: 'Unprocessable Entity' },
      );
      expect(err?.status).toBe(422);
    });
  });

  // ── 9. macroCalibration — GET /views/macro-calibration ──────────────────────
  //
  // Contract: the method issues a GET (not POST) and places all parameters as
  // URL query strings. The camelCase input key `macroText` maps to the snake_case
  // query param `macro_text`.

  describe('macroCalibration()', () => {
    it('when macroCalibration is called, a GET is issued to /views/macro-calibration', () => {
      svc.macroCalibration({ country: 'US', macroText: 'recession', refresh: true }).subscribe({ error: () => {} });
      const req = capture(`${API}views/macro-calibration`);
      assertMethod(req, 'GET');
      assertUrl(req, `${API}views/macro-calibration`);
    });

    it('when macroCalibration is called, all params are in the URL query string, not the body', () => {
      svc.macroCalibration({ country: 'US', macroText: 'recession', refresh: true }).subscribe({ error: () => {} });
      const req = capture(`${API}views/macro-calibration`);
      assertQueryParams(req, { country: 'US', macro_text: 'recession', refresh: true });
    });

    it('when macroCalibration is called, macroText is translated to the snake_case macro_text param', () => {
      svc.macroCalibration({ country: 'US', macroText: 'stagflation', refresh: false }).subscribe({ error: () => {} });
      const req = capture(`${API}views/macro-calibration`);
      expect(req.request.params.get('macro_text')).toBe('stagflation');
      expect(req.request.params.get('macroText')).toBeNull();
    });

    it('when macroCalibration is called, the request body is null (GET has no body)', () => {
      svc.macroCalibration({ country: 'US', macroText: 'recession', refresh: false }).subscribe({ error: () => {} });
      const req = capture(`${API}views/macro-calibration`);
      expect(req.request.body).toBeNull();
    });

    it('when macroCalibration is called, country param is not URL-encoded if it is a plain string', () => {
      svc.macroCalibration({ country: 'DE', macroText: '', refresh: false }).subscribe({ error: () => {} });
      const req = capture(`${API}views/macro-calibration`);
      expect(req.request.params.get('country')).toBe('DE');
    });

    it('when macroCalibration returns 5xx, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.macroCalibration({ country: 'US', macroText: '', refresh: false }).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne((r) => r.url === `${API}views/macro-calibration`).flush(
        { detail: 'crash' },
        { status: 500, statusText: 'Internal Server Error' },
      );
      expect(err?.status).toBe(500);
    });
  });

  // ── 10. teObservations — GET /macro-data/te-observations ────────────────────
  //
  // Contract: the method issues a GET and places all parameters as URL query
  // strings. The camelCase input keys `startDate`/`endDate` map to the snake_case
  // query params `start_date`/`end_date`.

  describe('teObservations()', () => {
    it('when teObservations is called, a GET is issued to /macro-data/te-observations', () => {
      svc.teObservations({ country: 'US', startDate: '2024-01-01', endDate: '2024-12-31', limit: 50 }).subscribe({ error: () => {} });
      const req = capture(`${API}macro-data/te-observations`);
      assertMethod(req, 'GET');
      assertUrl(req, `${API}macro-data/te-observations`);
    });

    it('when teObservations is called, all params are in the URL query string, not the body', () => {
      svc.teObservations({ country: 'US', startDate: '2024-01-01', endDate: '2024-12-31', limit: 50 }).subscribe({ error: () => {} });
      const req = capture(`${API}macro-data/te-observations`);
      assertQueryParams(req, { country: 'US', start_date: '2024-01-01', end_date: '2024-12-31', limit: 50 });
    });

    it('when teObservations is called, startDate is translated to the snake_case start_date param', () => {
      svc.teObservations({ country: 'US', startDate: '2024-06-01', endDate: '2024-12-31', limit: 10 }).subscribe({ error: () => {} });
      const req = capture(`${API}macro-data/te-observations`);
      expect(req.request.params.get('start_date')).toBe('2024-06-01');
      expect(req.request.params.get('startDate')).toBeNull();
    });

    it('when teObservations is called, endDate is translated to the snake_case end_date param', () => {
      svc.teObservations({ country: 'US', startDate: '2024-01-01', endDate: '2024-09-30', limit: 10 }).subscribe({ error: () => {} });
      const req = capture(`${API}macro-data/te-observations`);
      expect(req.request.params.get('end_date')).toBe('2024-09-30');
      expect(req.request.params.get('endDate')).toBeNull();
    });

    it('when teObservations is called, the request body is null (GET has no body)', () => {
      svc.teObservations({ country: 'DE', startDate: '2024-01-01', endDate: '2024-12-31', limit: 100 }).subscribe({ error: () => {} });
      const req = capture(`${API}macro-data/te-observations`);
      expect(req.request.body).toBeNull();
    });

    it('when teObservations is called with a different country, the country param is forwarded', () => {
      svc.teObservations({ country: 'JP', startDate: '2024-01-01', endDate: '2024-12-31', limit: 25 }).subscribe({ error: () => {} });
      const req = capture(`${API}macro-data/te-observations`);
      expect(req.request.params.get('country')).toBe('JP');
    });

    it('when teObservations returns 5xx, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.teObservations({ country: 'US', startDate: '2024-01-01', endDate: '2024-12-31', limit: 10 }).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne((r) => r.url === `${API}macro-data/te-observations`).flush(
        { detail: 'crash' },
        { status: 500, statusText: 'Internal Server Error' },
      );
      expect(err?.status).toBe(500);
    });

    it('when teObservations returns 404, the error propagates to the subscriber', () => {
      let err: { status: number } | undefined;
      svc.teObservations({ country: 'XX', startDate: '2024-01-01', endDate: '2024-12-31', limit: 10 }).subscribe({ error: (e) => (err = e as { status: number }) });
      http.expectOne((r) => r.url === `${API}macro-data/te-observations`).flush(
        { detail: 'country not found' },
        { status: 404, statusText: 'Not Found' },
      );
      expect(err?.status).toBe(404);
    });
  });
});
