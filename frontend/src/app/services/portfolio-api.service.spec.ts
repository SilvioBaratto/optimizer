import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { PortfolioApiService } from './portfolio-api.service';
import { environment } from '../../environments/environment';
import {
  BrokerAccountDto,
  BrokerPositionDto,
  CreatePortfolioDto,
  CreateSnapshotDto,
  PortfolioDto,
  SnapshotDto,
  SnapshotListResponseDto,
  SyncJobResponseDto,
  SyncProgressResponseDto,
} from '../models/portfolio-api.model';

const BASE = `${environment.apiUrl}portfolio`;
const NAME = 'alpha';

describe('PortfolioApiService', () => {
  let svc: PortfolioApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        PortfolioApiService,
      ],
    });
    svc = TestBed.inject(PortfolioApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
  });

  describe('existing list/get (preserved)', () => {
    it('list() → GET /portfolio/', () => {
      svc.list().subscribe();
      const req = http.expectOne(`${BASE}/`);
      expect(req.request.method).toBe('GET');
      req.flush({ items: [], total: 0 });
    });

    it('get(name) → GET /portfolio/{name} with URI encoding', () => {
      svc.get('my portfolio').subscribe();
      const req = http.expectOne(`${BASE}/my%20portfolio`);
      expect(req.request.method).toBe('GET');
      req.flush({} as PortfolioDto);
    });
  });

  describe('create()', () => {
    it('POSTs the payload to /portfolio/', () => {
      const payload: CreatePortfolioDto = {
        name: 'beta',
        description: 'Test',
        currency: 'USD',
        benchmark_ticker: 'SPY',
      };
      svc.create(payload).subscribe();
      const req = http.expectOne(`${BASE}/`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(payload);
      req.flush({} as PortfolioDto);
    });

    it('propagates errors with a descriptive message', () => {
      let error: Error | undefined;
      svc.create({ name: 'beta' }).subscribe({
        error: (e: Error) => (error = e),
      });
      http.expectOne(`${BASE}/`).flush(
        { detail: 'duplicate name' },
        { status: 409, statusText: 'Conflict' },
      );
      expect(error?.message).toContain('duplicate name');
    });
  });

  describe('getSnapshots()', () => {
    it('GETs /portfolio/{name}/snapshots', () => {
      let result: SnapshotListResponseDto | undefined;
      svc.getSnapshots(NAME).subscribe((r) => (result = r));
      const payload: SnapshotListResponseDto = { items: [], total: 0 };
      http.expectOne(`${BASE}/${NAME}/snapshots`).flush(payload);
      expect(result).toEqual(payload);
    });

    it('propagates errors', () => {
      let error: Error | undefined;
      svc.getSnapshots(NAME).subscribe({ error: (e: Error) => (error = e) });
      http.expectOne(`${BASE}/${NAME}/snapshots`).flush(
        { detail: 'nope' },
        { status: 500, statusText: 'Server Error' },
      );
      expect(error).toBeDefined();
    });
  });

  describe('createSnapshot()', () => {
    it('POSTs the payload to /portfolio/{name}/snapshots', () => {
      const payload: CreateSnapshotDto = {
        snapshot_date: '2026-04-17',
        snapshot_type: 'manual',
        weights: { AAPL: 0.5, MSFT: 0.5 },
      };
      svc.createSnapshot(NAME, payload).subscribe();
      const req = http.expectOne(`${BASE}/${NAME}/snapshots`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(payload);
      req.flush({} as SnapshotDto);
    });
  });

  describe('getLatestSnapshot()', () => {
    it('GETs /portfolio/{name}/snapshots/latest', () => {
      svc.getLatestSnapshot(NAME).subscribe();
      const req = http.expectOne(`${BASE}/${NAME}/snapshots/latest`);
      expect(req.request.method).toBe('GET');
      req.flush({} as SnapshotDto);
    });
  });

  describe('syncPortfolio()', () => {
    it('POSTs to /portfolio/{name}/sync and returns the job id', () => {
      let result: SyncJobResponseDto | undefined;
      svc.syncPortfolio(NAME).subscribe((r) => (result = r));
      const req = http.expectOne(`${BASE}/${NAME}/sync`);
      expect(req.request.method).toBe('POST');
      req.flush({ job_id: 'abc', status: 'pending', message: '' });
      expect(result?.job_id).toBe('abc');
    });
  });

  describe('getSyncProgress()', () => {
    it('GETs /portfolio/{name}/sync/{job_id}', () => {
      let result: SyncProgressResponseDto | undefined;
      svc.getSyncProgress(NAME, 'abc').subscribe((r) => (result = r));
      const payload: SyncProgressResponseDto = {
        job_id: 'abc',
        status: 'running',
        current: 3,
        total: 10,
        result: null,
        error: null,
      };
      http.expectOne(`${BASE}/${NAME}/sync/abc`).flush(payload);
      expect(result).toEqual(payload);
    });
  });

  describe('getPositions()', () => {
    it('GETs /portfolio/{name}/positions', () => {
      let result: BrokerPositionDto[] | undefined;
      svc.getPositions(NAME).subscribe((r) => (result = r));
      http.expectOne(`${BASE}/${NAME}/positions`).flush([]);
      expect(result).toEqual([]);
    });
  });

  describe('getAccount()', () => {
    it('GETs /portfolio/{name}/account', () => {
      let result: BrokerAccountDto | undefined;
      svc.getAccount(NAME).subscribe((r) => (result = r));
      const payload: BrokerAccountDto = {
        id: 'a1',
        total: 1000,
        free: 100,
        invested: 900,
        blocked: null,
        result: null,
        currency: 'EUR',
        synced_at: '2026-04-17T00:00:00Z',
      };
      http.expectOne(`${BASE}/${NAME}/account`).flush(payload);
      expect(result).toEqual(payload);
    });

    it('propagates errors with a descriptive message', () => {
      let error: Error | undefined;
      svc.getAccount(NAME).subscribe({ error: (e: Error) => (error = e) });
      http.expectOne(`${BASE}/${NAME}/account`).flush(
        { detail: 'broker unavailable' },
        { status: 503, statusText: 'Service Unavailable' },
      );
      expect(error?.message).toContain('broker unavailable');
    });
  });
});
