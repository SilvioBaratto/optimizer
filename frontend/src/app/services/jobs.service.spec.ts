import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { JobsService } from './jobs.service';
import { environment } from '../../environments/environment';
import {
  DOMAIN_META,
  DomainStatus,
  JobDomain,
  JobSummary,
} from '../models/jobs.model';

const JOBS_URL = `${environment.apiUrl}jobs`;

function job(overrides: Partial<JobSummary> = {}): JobSummary {
  return {
    id: 'j1',
    domain: 'yfinance_fetch',
    status: 'completed',
    current: 100,
    total: 100,
    error: null,
    errors_count: 0,
    started_at: '2026-04-17T00:00:00Z',
    finished_at: new Date().toISOString(),
    duration_seconds: 60,
    ...overrides,
  };
}

describe('JobsService', () => {
  let svc: JobsService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        JobsService,
      ],
    });
    svc = TestBed.inject(JobsService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
  });

  describe('DOMAIN_META', () => {
    it('declares all 15 backend job types with underscore format', () => {
      const expected: JobDomain[] = [
        'yfinance_fetch',
        'macro_fetch',
        'fred_fetch',
        'macro_news_fetch',
        'news_summarize',
        'macro_calibrate',
        'portfolio_sync',
        'universe_build',
        'reference_index_seed',
        'optimize',
        'backtest',
        'validate',
        'tune',
        'factors_compute',
        'report_generate',
      ];
      const actual = DOMAIN_META.map((m) => m.domain);
      expect(actual.sort()).toEqual(expected.sort());
    });

    it('gives fred_fetch a monthly-scoped threshold (>= 30 days)', () => {
      const fred = DOMAIN_META.find((m) => m.domain === 'fred_fetch');
      expect(fred).toBeDefined();
      expect(fred!.staleThresholdHours).toBeGreaterThanOrEqual(30 * 24);
    });
  });

  describe('getDomainStatuses()', () => {
    it('returns an empty array when the API reports no jobs (issue #440)', () => {
      // Regression: previously the service would fan out the empty result
      // over every DOMAIN_META entry, producing 15 placeholder cards in the
      // Pipeline Health grid. The contract is now: no jobs → no statuses,
      // letting the page render its single "no pipeline activity yet" tile.
      let result: DomainStatus[] | undefined;
      svc.getDomainStatuses().subscribe((r) => (result = r));

      http.expectOne((r) => r.url === JOBS_URL).flush({
        jobs: [],
        total: 0,
        limit: 100,
        offset: 0,
      });

      expect(result).toEqual([]);
    });

    it('builds one status per domain present in the API response', () => {
      let result: DomainStatus[] | undefined;
      svc.getDomainStatuses().subscribe((r) => (result = r));

      http.expectOne((r) => r.url === JOBS_URL).flush({
        jobs: [
          job({ id: 'a', domain: 'optimize', status: 'completed' }),
          job({ id: 'b', domain: 'yfinance_fetch', status: 'completed' }),
        ],
        total: 2,
        limit: 100,
        offset: 0,
      });

      expect(result!.length).toBe(2);
      expect(result!.map((s) => s.meta.domain).sort()).toEqual(
        ['optimize', 'yfinance_fetch'].sort(),
      );
    });

    it('groups running/completed/failed jobs by underscore domain name', () => {
      let result: DomainStatus[] | undefined;
      svc.getDomainStatuses().subscribe((r) => (result = r));

      http.expectOne((r) => r.url === JOBS_URL).flush({
        jobs: [
          job({ id: 'a', domain: 'optimize', status: 'running' }),
          job({ id: 'b', domain: 'optimize', status: 'completed' }),
          job({ id: 'c', domain: 'optimize', status: 'failed' }),
        ],
        total: 3,
        limit: 100,
        offset: 0,
      });

      const optimize = result!.find((s) => s.meta.domain === 'optimize');
      expect(optimize).toBeDefined();
      expect(optimize!.running?.id).toBe('a');
      expect(optimize!.lastSuccess?.id).toBe('b');
      expect(optimize!.recentFailures.length).toBe(1);
    });

    it('surfaces unknown domains from the API as synthesized statuses', () => {
      let result: DomainStatus[] | undefined;
      svc.getDomainStatuses().subscribe((r) => (result = r));

      http.expectOne((r) => r.url === JOBS_URL).flush({
        jobs: [job({ id: 'x', domain: 'brand_new_job', status: 'running' })],
        total: 1,
        limit: 100,
        offset: 0,
      });

      const unknown = result!.find((s) => s.meta.domain === 'brand_new_job');
      expect(unknown).toBeDefined();
      expect(unknown!.running?.id).toBe('x');
    });

    it('propagates the error so the caller can render a banner (issue #440)', () => {
      // Regression: the previous silent catchError fallback masked HTTP errors
      // as "15 empty cards" — indistinguishable from a successful empty result.
      let error: unknown;
      let next: DomainStatus[] | undefined;
      svc.getDomainStatuses().subscribe({
        next: (r) => (next = r),
        error: (e) => (error = e),
      });

      http
        .expectOne((r) => r.url === JOBS_URL)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(next).toBeUndefined();
      expect(error).toBeDefined();
    });
  });

  describe('getJob()', () => {
    it('GETs /jobs/{id} (URI-encoded) and returns the payload', () => {
      const payload = job({ id: 'abc 123', domain: 'optimize', status: 'running' });
      let result: JobSummary | undefined;
      svc.getJob('abc 123').subscribe((r) => (result = r));

      const req = http.expectOne(`${JOBS_URL}/abc%20123`);
      expect(req.request.method).toBe('GET');
      req.flush(payload);

      expect(result).toEqual(payload);
    });
  });
});

describe('JobsService.listJobs()', () => {
  let svc: JobsService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        JobsService,
      ],
    });
    svc = TestBed.inject(JobsService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  const emptyResponse = { jobs: [], total: 0, limit: 25, offset: 0 };

  it('sends limit and offset as query params and no domain/status when empty', () => {
    svc.listJobs({ limit: 25, offset: 0 }).subscribe();

    const req = http.expectOne((r) => r.url === JOBS_URL);
    expect(req.request.method).toBe('GET');
    expect(req.request.params.get('limit')).toBe('25');
    expect(req.request.params.get('offset')).toBe('0');
    expect(req.request.params.has('domain')).toBe(false);
    expect(req.request.params.has('status')).toBe(false);
    req.flush(emptyResponse);
  });

  it('emits one domain= param per selected domain (multi-value)', () => {
    svc
      .listJobs({
        domain: ['yfinance_fetch', 'macro_fetch'],
        limit: 10,
        offset: 0,
      })
      .subscribe();

    const req = http.expectOne((r) => r.url === JOBS_URL);
    expect(req.request.params.getAll('domain')).toEqual([
      'yfinance_fetch',
      'macro_fetch',
    ]);
    req.flush(emptyResponse);
  });

  it('emits one status= param per selected status (multi-value)', () => {
    svc
      .listJobs({ status: ['running', 'failed'], limit: 10, offset: 0 })
      .subscribe();

    const req = http.expectOne((r) => r.url === JOBS_URL);
    expect(req.request.params.getAll('status')).toEqual(['running', 'failed']);
    req.flush(emptyResponse);
  });

  it('forwards explicit limit/offset', () => {
    svc.listJobs({ limit: 50, offset: 100 }).subscribe();

    const req = http.expectOne((r) => r.url === JOBS_URL);
    expect(req.request.params.get('limit')).toBe('50');
    expect(req.request.params.get('offset')).toBe('100');
    req.flush(emptyResponse);
  });
});
