import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { DatabaseService } from './database.service';
import { environment } from '../../environments/environment';
import {
  DatabaseStatus,
  HealthCheck,
  TableInfo,
  TruncateResponse,
} from '../models/database.model';

const BASE = `${environment.apiUrl}database`;

describe('DatabaseService', () => {
  let svc: DatabaseService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        DatabaseService,
      ],
    });
    svc = TestBed.inject(DatabaseService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
  });

  describe('getHealth()', () => {
    it('calls GET /database/health and returns the payload', () => {
      const payload: HealthCheck = {
        status: 'healthy',
        latency_ms: 12,
        database: 'optimizer_db',
        version: 'PostgreSQL 16',
      };
      let result: HealthCheck | undefined;
      svc.getHealth().subscribe((r) => (result = r));

      const req = http.expectOne(`${BASE}/health`);
      expect(req.request.method).toBe('GET');
      req.flush(payload);

      expect(result).toEqual(payload);
    });

    it('propagates HTTP errors (no silent fallback)', () => {
      let error: unknown;
      svc.getHealth().subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${BASE}/health`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(error).toBeDefined();
    });
  });

  describe('getTables()', () => {
    it('calls GET /database/tables and returns the list', () => {
      const payload: TableInfo[] = [
        { name: 't1', schema: 'public', row_count: 10, size_bytes: 1024, size_pretty: '1 KB' },
      ];
      let result: TableInfo[] | undefined;
      svc.getTables().subscribe((r) => (result = r));

      http.expectOne(`${BASE}/tables`).flush(payload);

      expect(result).toEqual(payload);
    });
  });

  describe('getStatus()', () => {
    it('calls GET /database/status and returns the payload', () => {
      const payload: DatabaseStatus = {
        health: { status: 'healthy', latency_ms: 5, database: 'd', version: '16' },
        tables: [],
        total_size_pretty: '0 B',
      };
      let result: DatabaseStatus | undefined;
      svc.getStatus().subscribe((r) => (result = r));

      http.expectOne(`${BASE}/status`).flush(payload);

      expect(result).toEqual(payload);
    });
  });

  describe('truncateTable()', () => {
    it('calls DELETE /database/tables/{name}?confirm=true', () => {
      const payload: TruncateResponse = { table: 'users', status: 'truncated' };
      let result: TruncateResponse | undefined;
      svc.truncateTable('users').subscribe((r) => (result = r));

      const req = http.expectOne(
        (r) => r.url === `${BASE}/tables/users`,
      );
      expect(req.request.method).toBe('DELETE');
      expect(req.request.params.get('confirm')).toBe('true');
      req.flush(payload);

      expect(result).toEqual(payload);
    });

    it('URI-encodes the table name', () => {
      svc.truncateTable('schema.weird-name').subscribe();

      const req = http.expectOne(
        (r) => r.url === `${BASE}/tables/schema.weird-name`,
      );
      req.flush({ table: 'schema.weird-name', status: 'truncated' });
    });

    it('propagates 404 when the table is not managed', () => {
      let error: unknown;
      svc.truncateTable('ghost').subscribe({ error: (e) => (error = e) });

      http
        .expectOne((r) => r.url === `${BASE}/tables/ghost`)
        .flush({ detail: 'not a managed table' }, { status: 404, statusText: 'Not Found' });

      expect(error).toBeDefined();
    });
  });
});
