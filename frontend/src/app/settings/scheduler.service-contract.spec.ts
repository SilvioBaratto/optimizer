/**
 * Source-blind contract tests for issue #966.
 *
 * Derives every assertion from the acceptance criteria text only.
 * Tests SchedulerService.getStatus(): HTTP method, URL segment, response
 * shape, and error propagation to the caller.
 *
 * Criteria pinned (T3, issue #966):
 *   - Every page/panel shows a visible, non-blank error state when its
 *     primary API call returns an error → service must propagate errors
 *   - Every wired page has specs covering:
 *       (a) loads data on init → service fires HTTP on subscription
 *       (b) handles API error  → observable errors on 4xx/5xx
 */

import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';

import { SchedulerService } from './scheduler.service';
import { assertMethod, assertUrl } from '../../testing/contract-request.helper';
import { makeSchedulerStatusResponse, makeSchedulerJobStatusDto } from '../../testing/domain-fixtures';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;

describe('SchedulerService — contract-parity (issue #966)', () => {
  let service: SchedulerService;
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
    service = TestBed.inject(SchedulerService);
  });

  afterEach(() => http.verify());

  // ── getStatus() ───────────────────────────────────────────────────────────

  describe('getStatus()', () => {
    it('when getStatus() is called, sends a GET request to /scheduler/status', () => {
      service.getStatus().subscribe();

      const req = http.expectOne((r) => r.url.includes('/scheduler/status'));
      expect(req.request.method).toBe('GET');
      req.flush({ schedulerRunning: true, jobs: [] });
    });

    it('when API returns schedulerRunning: true, the observable emits that value', () => {
      let result: { schedulerRunning: boolean; jobs: unknown[] } | undefined;
      service.getStatus().subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url.includes('/scheduler/status'))
        .flush({ schedulerRunning: true, jobs: [] });

      expect(result?.schedulerRunning).toBe(true);
    });

    it('when API returns schedulerRunning: false, the observable emits that value', () => {
      let result: { schedulerRunning: boolean; jobs: unknown[] } | undefined;
      service.getStatus().subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url.includes('/scheduler/status'))
        .flush({ schedulerRunning: false, jobs: [] });

      expect(result?.schedulerRunning).toBe(false);
    });

    it('when API returns a jobs array, the observable emits that jobs array', () => {
      const jobs = [
        {
          jobId: 'daily_pipeline',
          name: 'Daily pipeline',
          nextRunTime: '2026-04-18T07:00:00Z',
          lastRunTime: '2026-04-17T07:00:00Z',
          lastStatus: 'completed',
          trigger: 'cron[hour=7]',
        },
      ];
      let result: { schedulerRunning: boolean; jobs: unknown[] } | undefined;
      service.getStatus().subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url.includes('/scheduler/status'))
        .flush({ schedulerRunning: true, jobs });

      expect(Array.isArray(result?.jobs)).toBe(true);
      expect((result?.jobs as unknown[]).length).toBe(1);
    });

    it('when getStatus() API returns 500, the observable errors (propagates to caller)', () => {
      let errored = false;
      service.getStatus().subscribe({ error: () => (errored = true) });

      http
        .expectOne((r) => r.url.includes('/scheduler/status'))
        .flush('Internal Server Error', { status: 500, statusText: 'Server Error' });

      expect(errored).toBe(true);
    });

    it('when getStatus() API returns 503, the observable errors', () => {
      let errored = false;
      service.getStatus().subscribe({ error: () => (errored = true) });

      http
        .expectOne((r) => r.url.includes('/scheduler/status'))
        .flush('Service Unavailable', { status: 503, statusText: 'Service Unavailable' });

      expect(errored).toBe(true);
    });
  });

  // ── request-shape assertions (issue #1027) ──────────────────────────────────

  describe('request-shape assertions (issue #1027)', () => {
    it('getStatus() — GET to exact URL', () => {
      service.getStatus().subscribe();

      const req = http.expectOne((r) => r.url === `${API}scheduler/status`);
      assertMethod(req, 'GET');
      assertUrl(req, `${API}scheduler/status`);
      req.flush(makeSchedulerStatusResponse());
    });

    it('getStatus() — response matches SchedulerStatusResponse fixture keys', () => {
      const FIXTURE = makeSchedulerStatusResponse();
      let result: Record<string, unknown> | undefined;
      service.getStatus().subscribe((r) => (result = r as unknown as Record<string, unknown>));

      http.expectOne((r) => r.url === `${API}scheduler/status`).flush(FIXTURE);

      expect(Object.keys(result ?? {})).toEqual(jasmine.arrayContaining(Object.keys(FIXTURE)));
    });

    it('getStatus() — jobs array items match SchedulerJobStatusDto fixture keys', () => {
      const FIXTURE = makeSchedulerStatusResponse();
      let result: { schedulerRunning: boolean; jobs: Record<string, unknown>[] } | undefined;
      service.getStatus().subscribe(
        (r) => (result = r as unknown as typeof result),
      );

      http.expectOne((r) => r.url === `${API}scheduler/status`).flush(FIXTURE);

      const JOB_KEYS = Object.keys(makeSchedulerJobStatusDto());
      expect(Object.keys(result?.jobs?.[0] ?? {})).toEqual(jasmine.arrayContaining(JOB_KEYS));
    });
  });
});
