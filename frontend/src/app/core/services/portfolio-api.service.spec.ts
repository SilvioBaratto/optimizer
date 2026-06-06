import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { PortfolioApiService } from './portfolio-api.service';
import { environment } from '../../../environments/environment';
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

  describe('get()', () => {
    it('when the backend returns 404, the error message is propagated as an Error', () => {
      let error: Error | undefined;
      svc.get('unknown').subscribe({ error: (e: Error) => (error = e) });
      http
        .expectOne(`${BASE}/unknown`)
        .flush({ detail: 'portfolio not found' }, { status: 404, statusText: 'Not Found' });
      expect(error?.message).toContain('portfolio not found');
    });
  });

  describe('list() deduplication', () => {
    it('issues exactly one HTTP call when two subscribers attach in the same tick', () => {
      svc.list().subscribe();
      svc.list().subscribe();

      // expectOne throws if more than one request matches the URL
      const req = http.expectOne(`${BASE}/`);
      expect(req.request.method).toBe('GET');
      req.flush({ items: [], total: 0 });
    });

    it('both parallel subscribers receive the same emitted value', () => {
      const payload = { items: [{ name: 'alpha' }], total: 1 };
      let first: unknown;
      let second: unknown;

      svc.list().subscribe((v) => (first = v));
      svc.list().subscribe((v) => (second = v));

      http.expectOne(`${BASE}/`).flush(payload);

      expect(first).toEqual(payload);
      expect(second).toEqual(payload);
      expect(first).toBe(second);
    });

    it('issues a new HTTP call for a later list() call after the previous subscribers complete', () => {
      svc.list().subscribe();
      http.expectOne(`${BASE}/`).flush({ items: [], total: 0 });

      svc.list().subscribe();
      const req = http.expectOne(`${BASE}/`);
      expect(req.request.method).toBe('GET');
      req.flush({ items: [], total: 0 });
    });

    it('busts the cache after a successful create() so the next list() refetches', () => {
      svc.list().subscribe();
      http.expectOne(`${BASE}/`).flush({ items: [], total: 0 });

      svc
        .create({
          name: 'beta',
          description: null,
          currency: 'USD',
          benchmark_ticker: 'SPY',
        })
        .subscribe();
      http.expectOne((r) => r.url === `${BASE}/` && r.method === 'POST').flush(
        {} as PortfolioDto,
      );

      svc.list().subscribe();
      const listReq = http.expectOne(
        (r) => r.url === `${BASE}/` && r.method === 'GET',
      );
      expect(listReq.request.method).toBe('GET');
      listReq.flush({ items: [], total: 0 });
    });

    it('does not poison the cache on error: the next list() issues a new HTTP call and succeeds', () => {
      let firstError: Error | undefined;
      svc.list().subscribe({ error: (e: Error) => (firstError = e) });
      http
        .expectOne(`${BASE}/`)
        .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
      expect(firstError?.message).toContain('boom');

      let secondResult: unknown;
      svc.list().subscribe((v) => (secondResult = v));
      http.expectOne(`${BASE}/`).flush({ items: [], total: 0 });
      expect(secondResult).toEqual({ items: [], total: 0 });
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

    it('when snapshot creation returns a SnapshotDto, the response value is delivered to the subscriber', () => {
      const created: SnapshotDto = {
        id: 'sn-1',
        portfolio_id: 'p-1',
        snapshot_date: '2026-04-17',
        snapshot_type: 'manual',
        weights: { AAPL: 0.5, MSFT: 0.5 },
        sector_mapping: null,
        summary: null,
        optimizer_config: null,
        turnover: null,
        holding_count: 2,
        created_at: '2026-04-17T00:00:00Z',
      };
      let result: SnapshotDto | undefined;
      svc.createSnapshot(NAME, { snapshot_date: '2026-04-17', weights: { AAPL: 0.5, MSFT: 0.5 } }).subscribe((r) => (result = r));
      http.expectOne(`${BASE}/${NAME}/snapshots`).flush(created);
      expect(result?.id).toBe('sn-1');
    });

    it('when the backend returns 422, the error message is propagated as an Error', () => {
      let error: Error | undefined;
      svc.createSnapshot(NAME, { snapshot_date: 'bad', weights: {} }).subscribe({ error: (e: Error) => (error = e) });
      http
        .expectOne(`${BASE}/${NAME}/snapshots`)
        .flush({ detail: 'invalid date' }, { status: 422, statusText: 'Unprocessable Entity' });
      expect(error?.message).toContain('invalid date');
    });
  });

  describe('getLatestSnapshot()', () => {
    it('GETs /portfolio/{name}/snapshots/latest', () => {
      svc.getLatestSnapshot(NAME).subscribe();
      const req = http.expectOne(`${BASE}/${NAME}/snapshots/latest`);
      expect(req.request.method).toBe('GET');
      req.flush({} as SnapshotDto);
    });

    it('when the snapshot is returned, the response value is delivered to the subscriber', () => {
      const snap: SnapshotDto = {
        id: 'sn-2',
        portfolio_id: 'p-1',
        snapshot_date: '2026-04-18',
        snapshot_type: 'optimization',
        weights: { AAPL: 0.6 },
        sector_mapping: null,
        summary: null,
        optimizer_config: null,
        turnover: 0.05,
        holding_count: 1,
        created_at: '2026-04-18T00:00:00Z',
      };
      let result: SnapshotDto | undefined;
      svc.getLatestSnapshot(NAME).subscribe((r) => (result = r));
      http.expectOne(`${BASE}/${NAME}/snapshots/latest`).flush(snap);
      expect(result?.id).toBe('sn-2');
      expect(result?.snapshot_type).toBe('optimization');
    });

    it('when the backend returns 404, the error message is propagated as an Error', () => {
      let error: Error | undefined;
      svc.getLatestSnapshot(NAME).subscribe({ error: (e: Error) => (error = e) });
      http
        .expectOne(`${BASE}/${NAME}/snapshots/latest`)
        .flush({ detail: 'no snapshot' }, { status: 404, statusText: 'Not Found' });
      expect(error?.message).toContain('no snapshot');
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

    it('when the broker is unavailable, the error message is propagated as an Error', () => {
      let error: Error | undefined;
      svc.syncPortfolio(NAME).subscribe({ error: (e: Error) => (error = e) });
      http
        .expectOne(`${BASE}/${NAME}/sync`)
        .flush({ detail: 'broker offline' }, { status: 503, statusText: 'Service Unavailable' });
      expect(error?.message).toContain('broker offline');
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

    it('URI-encodes the job_id in the URL', () => {
      svc.getSyncProgress(NAME, 'a b/c').subscribe();
      const req = http.expectOne(`${BASE}/${NAME}/sync/a%20b%2Fc`);
      expect(req.request.method).toBe('GET');
      req.flush({ job_id: 'a b/c', status: 'running', current: 1, total: 5, result: null, error: null });
    });

    it('when the backend returns 404, the error message is propagated as an Error', () => {
      let error: Error | undefined;
      svc.getSyncProgress(NAME, 'gone').subscribe({ error: (e: Error) => (error = e) });
      http
        .expectOne(`${BASE}/${NAME}/sync/gone`)
        .flush({ detail: 'job not found' }, { status: 404, statusText: 'Not Found' });
      expect(error?.message).toContain('job not found');
    });
  });

  describe('getPositions()', () => {
    it('GETs /portfolio/{name}/positions', () => {
      let result: BrokerPositionDto[] | undefined;
      svc.getPositions(NAME).subscribe((r) => (result = r));
      http.expectOne(`${BASE}/${NAME}/positions`).flush([]);
      expect(result).toEqual([]);
    });

    it('when positions are returned, the full list is delivered to the subscriber', () => {
      const pos: BrokerPositionDto = {
        id: 'pos-1',
        ticker: 'AAPL',
        yfinance_ticker: 'AAPL',
        name: 'Apple Inc.',
        quantity: 10,
        average_price: 150,
        current_price: 180,
        ppl: 300,
        fx_ppl: null,
        initial_fill_date: '2025-01-01',
        synced_at: '2026-04-17T00:00:00Z',
      };
      let result: BrokerPositionDto[] | undefined;
      svc.getPositions(NAME).subscribe((r) => (result = r));
      http.expectOne(`${BASE}/${NAME}/positions`).flush([pos]);
      expect(result?.length).toBe(1);
      expect(result?.[0].ticker).toBe('AAPL');
    });

    it('when the backend returns 503, the error message is propagated as an Error', () => {
      let error: Error | undefined;
      svc.getPositions(NAME).subscribe({ error: (e: Error) => (error = e) });
      http
        .expectOne(`${BASE}/${NAME}/positions`)
        .flush({ detail: 'broker unavailable' }, { status: 503, statusText: 'Service Unavailable' });
      expect(error?.message).toContain('broker unavailable');
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
