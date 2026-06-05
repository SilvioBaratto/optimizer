import { TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp } from '../../testing';
import {
  ReferenceIndexService,
  type ReferenceIndexSeedProgress,
} from './reference-index.service';
import { environment } from '../../environments/environment';

const BASE = `${environment.apiUrl}reference-indices`;

function progress(status: ReferenceIndexSeedProgress['status']): ReferenceIndexSeedProgress {
  return { job_id: 'j1', status, current: 0, total: 1, errors: [], result: null, error: null };
}

describe('ReferenceIndexService', () => {
  let svc: ReferenceIndexService;
  let http: HttpTestingController;

  beforeEach(async () => {
    await configureTestBed({ withHttp: true });
    svc = TestBed.inject(ReferenceIndexService);
    http = injectHttp();
  });

  afterEach(() => http.verify());

  describe('startSeed()', () => {
    it('when called, it POSTs seed with the tickers body', () => {
      svc.startSeed(['AAPL', 'MSFT']).subscribe();
      const req = http.expectOne(`${BASE}/seed`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({ tickers: ['AAPL', 'MSFT'] });
      req.flush({ job_id: 'j1', status: 'pending' });
    });

    it('when the server returns an error detail, it is surfaced as the Error message', () => {
      let message: string | undefined;
      svc.startSeed(['AAPL']).subscribe({ error: (e: Error) => (message = e.message) });
      http.expectOne(`${BASE}/seed`).flush({ detail: 'bad tickers' }, { status: 422, statusText: 'Unprocessable' });
      expect(message).toBe('bad tickers');
    });

    it('when a different status carries a detail, the same Error mapping applies (status-agnostic)', () => {
      let message: string | undefined;
      svc.startSeed(['AAPL']).subscribe({ error: (e: Error) => (message = e.message) });
      http.expectOne(`${BASE}/seed`).flush({ detail: 'server exploded' }, { status: 500, statusText: 'Server Error' });
      expect(message).toBe('server exploded');
    });

    it('when no detail is present, a default message is used', () => {
      let message: string | undefined;
      svc.startSeed(['AAPL']).subscribe({ error: (e: Error) => (message = e.message) });
      http.expectOne(`${BASE}/seed`).flush(null, { status: 500, statusText: 'Server Error' });
      expect(message).toBe('Failed to start seed');
    });
  });

  describe('getJob()', () => {
    it('when called, it GETs seed/{jobId}', () => {
      svc.getJob('j1').subscribe();
      const req = http.expectOne(`${BASE}/seed/j1`);
      expect(req.request.method).toBe('GET');
      req.flush(progress('running'));
    });

    it('when it errors, the detail is mapped to an Error', () => {
      let message: string | undefined;
      svc.getJob('j1').subscribe({ error: (e: Error) => (message = e.message) });
      http.expectOne(`${BASE}/seed/j1`).flush({ detail: 'gone' }, { status: 404, statusText: 'Not Found' });
      expect(message).toBe('gone');
    });

    it('when it errors without a detail, the default message is used', () => {
      let message: string | undefined;
      svc.getJob('j1').subscribe({ error: (e: Error) => (message = e.message) });
      http.expectOne(`${BASE}/seed/j1`).flush(null, { status: 500, statusText: 'x' });
      expect(message).toBe('Failed to fetch job');
    });
  });

  describe('getStatus()', () => {
    it('when called, it GETs status', () => {
      svc.getStatus().subscribe();
      const req = http.expectOne(`${BASE}/status`);
      expect(req.request.method).toBe('GET');
      req.flush({ items: [], total: 0, healthyCount: 0, missingCount: 0, staleCount: 0, staleDays: 30 });
    });

    it('when it errors, the detail is mapped to an Error', () => {
      let message: string | undefined;
      svc.getStatus().subscribe({ error: (e: Error) => (message = e.message) });
      http.expectOne(`${BASE}/status`).flush({ detail: 'no status' }, { status: 500, statusText: 'x' });
      expect(message).toBe('no status');
    });

    it('when it errors without a detail, the default message is used', () => {
      let message: string | undefined;
      svc.getStatus().subscribe({ error: (e: Error) => (message = e.message) });
      http.expectOne(`${BASE}/status`).flush(null, { status: 500, statusText: 'x' });
      expect(message).toBe('Failed to load status');
    });
  });

  describe('getConfigured()', () => {
    it('when called, it GETs configured', () => {
      svc.getConfigured().subscribe();
      const req = http.expectOne(`${BASE}/configured`);
      expect(req.request.method).toBe('GET');
      req.flush({ tickers: [], settingsTickers: [], portfolioTickers: [] });
    });

    it('when it errors, the detail is mapped to an Error', () => {
      let message: string | undefined;
      svc.getConfigured().subscribe({ error: (e: Error) => (message = e.message) });
      http.expectOne(`${BASE}/configured`).flush({ detail: 'no config' }, { status: 500, statusText: 'x' });
      expect(message).toBe('no config');
    });

    it('when it errors without a detail, the default message is used', () => {
      let message: string | undefined;
      svc.getConfigured().subscribe({ error: (e: Error) => (message = e.message) });
      http.expectOne(`${BASE}/configured`).flush(null, { status: 500, statusText: 'x' });
      expect(message).toBe('Failed to load configured tickers');
    });
  });

  describe('pollUntilDone()', () => {
    // Zoneless app: drive the RxJS timer with jasmine.clock (setTimeout-backed).
    it('when polling, it re-GETs the job until a terminal status, inclusive', () => {
      jasmine.clock().install();
      try {
        const seen: string[] = [];
        svc.pollUntilDone('j1', 2000).subscribe((p) => seen.push(p.status));

        jasmine.clock().tick(0); // release the timer(0, …) first emission
        http.expectOne(`${BASE}/seed/j1`).flush(progress('running'));

        jasmine.clock().tick(2000);
        http.expectOne(`${BASE}/seed/j1`).flush(progress('completed'));

        expect(seen).toEqual(['running', 'completed']);
      } finally {
        jasmine.clock().uninstall();
      }
    });

    it('when the job fails, the failed status is terminal and stops polling', () => {
      jasmine.clock().install();
      try {
        const seen: string[] = [];
        svc.pollUntilDone('j1', 2000).subscribe((p) => seen.push(p.status));

        jasmine.clock().tick(0);
        http.expectOne(`${BASE}/seed/j1`).flush(progress('failed'));

        // No further poll should be scheduled after a terminal status.
        jasmine.clock().tick(2000);
        http.expectNone(`${BASE}/seed/j1`);
        expect(seen).toEqual(['failed']);
      } finally {
        jasmine.clock().uninstall();
      }
    });
  });
});
