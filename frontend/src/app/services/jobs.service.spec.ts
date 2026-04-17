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
    it('builds a status entry for each known DOMAIN_META domain', () => {
      let result: DomainStatus[] | undefined;
      svc.getDomainStatuses().subscribe((r) => (result = r));

      http.expectOne((r) => r.url === JOBS_URL).flush({
        jobs: [],
        total: 0,
        limit: 100,
        offset: 0,
      });

      expect(result!.length).toBeGreaterThanOrEqual(DOMAIN_META.length);
      for (const meta of DOMAIN_META) {
        expect(result!.some((s) => s.meta.domain === meta.domain)).toBe(true);
      }
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

    it('falls back to empty statuses for all known domains on error', () => {
      let result: DomainStatus[] | undefined;
      svc.getDomainStatuses().subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url === JOBS_URL)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result!.length).toBe(DOMAIN_META.length);
      expect(result!.every((s) => s.lastSuccess === null && s.running === null)).toBe(
        true,
      );
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
