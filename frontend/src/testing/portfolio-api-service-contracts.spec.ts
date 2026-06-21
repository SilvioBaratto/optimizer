/**
 * Request/response contract parity for PortfolioApiService (issue #1028).
 *
 * Covers: list, get (encoded name), getLatestSnapshot (encoded name),
 *         update (PUT), delete (DELETE) — asserting:
 *   – HTTP method and exact interpolated URL (no raw {token} placeholders)
 *   – Portfolio name with spaces is percent-encoded in the URL path segment
 *   – getLatestSnapshot name is encoded AND followed by /snapshots/latest
 *   – update PUT body key-set
 *   – delete DELETE carries no body
 *
 * Criteria (UNIT, issue #1028):
 *   PortfolioApiService GET/list/getLatestSnapshot(encoded name)/update/delete
 *   asserted field-by-field.
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
  assertMethod,
  assertUrl,
  assertBodyKeys,
} from './contract-request.helper';
import { makePortfolioDto, makeSnapshotDto } from './domain-fixtures';
import { PortfolioApiService } from '../app/core/services/portfolio-api.service';
import type { UpdatePortfolioDto } from '../app/core/models/portfolio-api.model';
import { environment } from '../environments/environment';

const API = environment.apiUrl;
const BASE = `${API}portfolio`;

// Portfolio name with space to verify percent-encoding
const NAME = 'my portfolio';
const NAME_ENC = encodeURIComponent(NAME);

const UPDATE_PAYLOAD: UpdatePortfolioDto = {
  name: 'Renamed Portfolio',
  description: 'Updated description',
  currency: 'USD',
  benchmark_ticker: 'QQQ',
};

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

function capture(http: HttpTestingController, url: string, flushBody: object | null = {}): TestRequest {
  const req = http.expectOne(r => r.url === url);
  req.flush(flushBody);
  return req;
}

describe('PortfolioApiService — request/response contract parity (issue #1028)', () => {
  let svc: PortfolioApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    ({ http } = setupHttp());
    svc = TestBed.inject(PortfolioApiService);
  });

  afterEach(() => http.verify());

  // ── list() — GET ─────────────────────────────────────────────────────────

  describe('list()', () => {
    const LIST_URL = `${BASE}/`;

    it('sends GET to exact portfolio/ URL', () => {
      svc.list().subscribe({ error: () => {} });
      const req = capture(http, LIST_URL, { items: [], total: 0 });
      assertMethod(req, 'GET');
      assertUrl(req, LIST_URL);
    });

    it('request body is null (GET carries no body)', () => {
      svc.list().subscribe({ error: () => {} });
      const req = capture(http, LIST_URL, { items: [], total: 0 });
      expect(req.request.body).toBeNull();
    });

    it('response PortfolioListResponseDto key-set matches fixture', () => {
      const FIXTURE = { items: [makePortfolioDto()], total: 1 };
      let result: Record<string, unknown> | undefined;
      svc.list().subscribe(r => { result = r as unknown as Record<string, unknown>; });
      http.expectOne(r => r.url === LIST_URL).flush(FIXTURE);
      expect(Object.keys(result ?? {})).toEqual(
        jasmine.arrayContaining(Object.keys(FIXTURE)),
      );
    });
  });

  // ── get(name) — GET with encoded name ────────────────────────────────────

  describe('get(name)', () => {
    const GET_URL = `${BASE}/${NAME_ENC}`;

    it('sends GET to exact percent-encoded portfolio/{name} URL', () => {
      svc.get(NAME).subscribe({ error: () => {} });
      const req = capture(http, GET_URL, makePortfolioDto());
      assertMethod(req, 'GET');
      assertUrl(req, GET_URL);
    });

    it('portfolio name with spaces is percent-encoded in the URL path', () => {
      svc.get(NAME).subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === GET_URL);
      expect(req.request.url).toContain(NAME_ENC);
      req.flush(makePortfolioDto());
    });

    it('no raw {token} placeholder survives in the URL', () => {
      svc.get(NAME).subscribe({ error: () => {} });
      const req = capture(http, GET_URL, makePortfolioDto());
      assertUrl(req, GET_URL);
    });

    it('response carries PortfolioDto key-set', () => {
      const FIXTURE = makePortfolioDto();
      let result: Record<string, unknown> | undefined;
      svc.get(NAME).subscribe(r => { result = r as unknown as Record<string, unknown>; });
      http.expectOne(r => r.url === GET_URL).flush(FIXTURE);
      expect(Object.keys(result ?? {})).toEqual(
        jasmine.arrayContaining(Object.keys(FIXTURE)),
      );
    });
  });

  // ── getLatestSnapshot(name) — GET with encoded name ───────────────────────

  describe('getLatestSnapshot(name)', () => {
    const SNAP_URL = `${BASE}/${NAME_ENC}/snapshots/latest`;

    it('sends GET to exact percent-encoded portfolio/{name}/snapshots/latest URL', () => {
      svc.getLatestSnapshot(NAME).subscribe({ error: () => {} });
      const req = capture(http, SNAP_URL, makeSnapshotDto());
      assertMethod(req, 'GET');
      assertUrl(req, SNAP_URL);
    });

    it('portfolio name with spaces is percent-encoded in the URL path before /snapshots/latest', () => {
      svc.getLatestSnapshot(NAME).subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === SNAP_URL);
      expect(req.request.url).toContain(NAME_ENC);
      expect(req.request.url).toContain('/snapshots/latest');
      req.flush(makeSnapshotDto());
    });

    it('no raw {token} placeholder survives in the URL', () => {
      svc.getLatestSnapshot(NAME).subscribe({ error: () => {} });
      const req = capture(http, SNAP_URL, makeSnapshotDto());
      assertUrl(req, SNAP_URL);
    });

    it('response carries SnapshotDto key-set', () => {
      const FIXTURE = makeSnapshotDto();
      let result: Record<string, unknown> | undefined;
      svc.getLatestSnapshot(NAME).subscribe(r => {
        result = r as unknown as Record<string, unknown>;
      });
      http.expectOne(r => r.url === SNAP_URL).flush(FIXTURE);
      expect(Object.keys(result ?? {})).toEqual(
        jasmine.arrayContaining(Object.keys(FIXTURE)),
      );
    });
  });

  // ── update(name, payload) — PUT with encoded name ────────────────────────

  describe('update(name, payload)', () => {
    const UPDATE_URL = `${BASE}/${NAME_ENC}`;

    it('sends PUT to exact percent-encoded portfolio/{name} URL', () => {
      svc.update(NAME, UPDATE_PAYLOAD).subscribe({ error: () => {} });
      const req = capture(http, UPDATE_URL, makePortfolioDto());
      assertMethod(req, 'PUT');
      assertUrl(req, UPDATE_URL);
    });

    it('portfolio name with spaces is percent-encoded in the URL path', () => {
      svc.update(NAME, UPDATE_PAYLOAD).subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === UPDATE_URL);
      expect(req.request.url).toContain(NAME_ENC);
      req.flush(makePortfolioDto());
    });

    it('request body contains exactly the UpdatePortfolioDto keys', () => {
      svc.update(NAME, UPDATE_PAYLOAD).subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === UPDATE_URL);
      assertBodyKeys(req, Object.keys(UPDATE_PAYLOAD));
      req.flush(makePortfolioDto());
    });

    it('response carries PortfolioDto key-set', () => {
      const FIXTURE = makePortfolioDto();
      let result: Record<string, unknown> | undefined;
      svc.update(NAME, UPDATE_PAYLOAD).subscribe(r => {
        result = r as unknown as Record<string, unknown>;
      });
      http.expectOne(r => r.url === UPDATE_URL).flush(FIXTURE);
      expect(Object.keys(result ?? {})).toEqual(
        jasmine.arrayContaining(Object.keys(FIXTURE)),
      );
    });
  });

  // ── delete(name) — DELETE with encoded name ──────────────────────────────

  describe('delete(name)', () => {
    const DELETE_URL = `${BASE}/${NAME_ENC}`;

    it('sends DELETE to exact percent-encoded portfolio/{name} URL', () => {
      svc.delete(NAME).subscribe({ error: () => {} });
      const req = capture(http, DELETE_URL, null);
      assertMethod(req, 'DELETE');
      assertUrl(req, DELETE_URL);
    });

    it('portfolio name with spaces is percent-encoded in the URL path', () => {
      svc.delete(NAME).subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === DELETE_URL);
      expect(req.request.url).toContain(NAME_ENC);
      req.flush(null, { status: 204, statusText: 'No Content' });
    });

    it('request body is null (DELETE carries no body)', () => {
      svc.delete(NAME).subscribe({ error: () => {} });
      const req = capture(http, DELETE_URL, null);
      expect(req.request.body).toBeNull();
    });
  });
});
