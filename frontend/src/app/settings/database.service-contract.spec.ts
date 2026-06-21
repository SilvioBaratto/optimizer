/**
 * Source-blind contract tests for issue #966.
 *
 * Derives every assertion from the acceptance criteria text only.
 * Tests DatabaseService method-by-method: HTTP method, URL segment,
 * required query params, and error propagation to the caller.
 *
 * Criteria pinned (T3, issue #966):
 *   - Every page/panel shows a visible, non-blank error state when its
 *     primary API call returns an error → service must propagate errors
 *   - Every wired page has specs covering:
 *       (a) loads data on init → service fires HTTP on subscription
 *       (b) handles API error  → observable errors on 4xx/5xx
 *
 * "field-by-field request/response" criterion (marked NOT VERIFIABLE by
 * oracle) is narrowed here to the runtime-checkable subset: HTTP method,
 * URL path segment, and required query parameters.
 */

import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';

import { DatabaseService } from './database.service';
import { assertMethod, assertUrl, assertQueryParams } from '../../testing/contract-request.helper';
import { makeHealthCheck, makeTableInfo, makeDatabaseStatus, makeTruncateResponse } from '../../testing/domain-fixtures';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;

describe('DatabaseService — contract-parity (issue #966)', () => {
  let service: DatabaseService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    http = TestBed.inject(HttpTestingController);
    service = TestBed.inject(DatabaseService);
  });

  afterEach(() => http.verify());

  // ── getHealth() ──────────────────────────────────────────────────────────

  describe('getHealth()', () => {
    it('when getHealth() is called, sends a GET request to /database/health', () => {
      service.getHealth().subscribe();

      const req = http.expectOne((r) => r.url.includes('/database/health'));
      expect(req.request.method).toBe('GET');
      req.flush({ healthy: true, latency_ms: 3, database_url: 'db://test' });
    });

    it('when API returns { healthy: true }, the observable emits a value with healthy === true', () => {
      let result: { healthy: boolean } | undefined;
      service.getHealth().subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url.includes('/database/health'))
        .flush({ healthy: true, latency_ms: 3, database_url: 'db://test' });

      expect(result?.healthy).toBe(true);
    });

    it('when getHealth() API returns 500, the observable errors (propagates to caller)', () => {
      let errored = false;
      service.getHealth().subscribe({ error: () => (errored = true) });

      http
        .expectOne((r) => r.url.includes('/database/health'))
        .flush('Internal Server Error', { status: 500, statusText: 'Server Error' });

      expect(errored).toBe(true);
    });

    it('when getHealth() API returns 404, the observable errors', () => {
      let errored = false;
      service.getHealth().subscribe({ error: () => (errored = true) });

      http
        .expectOne((r) => r.url.includes('/database/health'))
        .flush('Not Found', { status: 404, statusText: 'Not Found' });

      expect(errored).toBe(true);
    });
  });

  // ── getTables() ───────────────────────────────────────────────────────────

  describe('getTables()', () => {
    it('when getTables() is called, sends a GET request to /database/tables', () => {
      service.getTables().subscribe();

      const req = http.expectOne((r) => r.url.includes('/database/tables'));
      expect(req.request.method).toBe('GET');
      req.flush([]);
    });

    it('when getTables() API returns an array, the observable emits that array', () => {
      const tables = [
        { name: 'prices', schema: 'public', row_count: 100, size_bytes: 8192, size_pretty: '8 kB' },
      ];
      let result: unknown[] | undefined;
      service.getTables().subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url.includes('/database/tables'))
        .flush(tables);

      expect(Array.isArray(result)).toBe(true);
      expect((result as unknown[]).length).toBe(1);
    });

    it('when getTables() API returns 500, the observable errors', () => {
      let errored = false;
      service.getTables().subscribe({ error: () => (errored = true) });

      http
        .expectOne((r) => r.url.includes('/database/tables'))
        .flush('Server Error', { status: 500, statusText: 'Server Error' });

      expect(errored).toBe(true);
    });
  });

  // ── getStatus() ───────────────────────────────────────────────────────────

  describe('getStatus()', () => {
    it('when getStatus() is called, sends a GET request to /database/status', () => {
      service.getStatus().subscribe();

      const req = http.expectOne((r) => r.url.includes('/database/status'));
      expect(req.request.method).toBe('GET');
      req.flush({ total_size_pretty: '12 MB', tables: [], health: {} });
    });

    it('when getStatus() API returns 500, the observable errors', () => {
      let errored = false;
      service.getStatus().subscribe({ error: () => (errored = true) });

      http
        .expectOne((r) => r.url.includes('/database/status'))
        .flush('Server Error', { status: 500, statusText: 'Server Error' });

      expect(errored).toBe(true);
    });
  });

  // ── truncateTable(name) ───────────────────────────────────────────────────

  describe('truncateTable(name)', () => {
    it('when truncateTable("prices") is called, sends a DELETE to /database/tables/prices', () => {
      service.truncateTable('prices').subscribe();

      const req = http.expectOne(
        (r) => r.method === 'DELETE' && r.url.includes('/database/tables/prices'),
      );
      expect(req.request.method).toBe('DELETE');
      req.flush({ status: 'truncated', table: 'prices' });
    });

    it('when truncateTable(name) is called, the request includes confirm=true as a query param', () => {
      service.truncateTable('prices').subscribe();

      const req = http.expectOne(
        (r) => r.method === 'DELETE' && r.url.includes('/database/tables/'),
      );
      expect(req.request.params.get('confirm')).toBe('true');
      req.flush({ status: 'truncated', table: 'prices' });
    });

    it('when the table name contains special characters, the request URL percent-encodes them', () => {
      service.truncateTable('my table').subscribe();

      const req = http.expectOne(
        (r) => r.method === 'DELETE' && r.url.includes('my%20table'),
      );
      req.flush({ status: 'truncated', table: 'my table' });
    });

    it('when truncateTable(name) API returns 500, the observable errors', () => {
      let errored = false;
      service.truncateTable('prices').subscribe({ error: () => (errored = true) });

      http
        .expectOne((r) => r.method === 'DELETE')
        .flush('Server Error', { status: 500, statusText: 'Server Error' });

      expect(errored).toBe(true);
    });
  });

  // ── request-shape assertions (issue #1027) ──────────────────────────────────

  describe('request-shape assertions (issue #1027)', () => {
    it('getHealth() — GET to exact URL, response matches HealthCheck keys', () => {
      const FIXTURE = makeHealthCheck();
      let result: Record<string, unknown> | undefined;
      service.getHealth().subscribe((r) => (result = r as unknown as Record<string, unknown>));

      const req = http.expectOne((r) => r.url === `${API}database/health`);
      assertMethod(req, 'GET');
      assertUrl(req, `${API}database/health`);
      req.flush(FIXTURE);

      expect(Object.keys(result ?? {})).toEqual(jasmine.arrayContaining(Object.keys(FIXTURE)));
    });

    it('getTables() — GET to exact URL, response array items match TableInfo keys', () => {
      const ITEM = makeTableInfo();
      let result: unknown[] | undefined;
      service.getTables().subscribe((r) => (result = r));

      const req = http.expectOne((r) => r.url === `${API}database/tables`);
      assertMethod(req, 'GET');
      assertUrl(req, `${API}database/tables`);
      req.flush([ITEM]);

      expect(Object.keys((result as unknown[])[0] as object)).toEqual(
        jasmine.arrayContaining(Object.keys(ITEM)),
      );
    });

    it('getStatus() — GET to exact URL, response matches DatabaseStatus keys', () => {
      const FIXTURE = makeDatabaseStatus();
      let result: Record<string, unknown> | undefined;
      service.getStatus().subscribe((r) => (result = r as unknown as Record<string, unknown>));

      const req = http.expectOne((r) => r.url === `${API}database/status`);
      assertMethod(req, 'GET');
      assertUrl(req, `${API}database/status`);
      req.flush(FIXTURE);

      expect(Object.keys(result ?? {})).toEqual(jasmine.arrayContaining(Object.keys(FIXTURE)));
    });

    it('truncateTable(name) — DELETE to encoded URL with confirm=true param', () => {
      const TABLE = 'price history';
      service.truncateTable(TABLE).subscribe();

      const req = http.expectOne(
        (r) => r.url === `${API}database/tables/${encodeURIComponent(TABLE)}`,
      );
      assertMethod(req, 'DELETE');
      assertUrl(req, `${API}database/tables/${encodeURIComponent(TABLE)}`);
      assertQueryParams(req, { confirm: 'true' });
      req.flush(makeTruncateResponse());
    });
  });
});
