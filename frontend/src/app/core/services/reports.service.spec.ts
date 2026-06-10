import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { ReportsService } from './reports.service';
import { environment } from '../../../environments/environment';
import type {
  ReportGenerateRequest,
  ReportJobCreateResponse,
  ReportJobProgress,
} from '../models/report.model';

const API = environment.apiUrl;

function generateRequest(): ReportGenerateRequest {
  return {
    template: 'standard',
    sections: ['portfolio_summary', 'weights'],
    format: 'pdf',
    orientation: 'portrait',
  };
}

describe('ReportsService', () => {
  let svc: ReportsService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ReportsService,
      ],
    });
    svc = TestBed.inject(ReportsService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  describe('generate()', () => {
    it('POSTs /reports/generate and returns the 202 + job_id payload', () => {
      let result: ReportJobCreateResponse | undefined;
      svc.generate(generateRequest()).subscribe((r) => (result = r));

      const req = http.expectOne(`${API}reports/generate`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body.sections).toEqual(['portfolio_summary', 'weights']);

      req.flush(
        { job_id: 'rpt-1', status: 'pending', message: 'started' },
        { status: 202, statusText: 'Accepted' },
      );

      expect(result).toEqual({ job_id: 'rpt-1', status: 'pending', message: 'started' });
    });

    it('propagates 409 when a report job is already running', () => {
      let error: HttpErrorResponse | undefined;
      svc.generate(generateRequest()).subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}reports/generate`)
        .flush({ detail: 'busy' }, { status: 409, statusText: 'Conflict' });
      expect(error?.status).toBe(409);
    });
  });

  describe('pollJob()', () => {
    it('GETs /reports/jobs/{job_id} (NOT the global /jobs path)', () => {
      let result: ReportJobProgress | undefined;
      svc.pollJob('rpt-1').subscribe((r) => (result = r));

      const req = http.expectOne(`${API}reports/jobs/rpt-1`);
      expect(req.request.method).toBe('GET');
      req.flush({
        job_id: 'rpt-1',
        status: 'completed',
        current: 100,
        total: 100,
        errors: [],
        result: { report_id: 'r-42', download_url: '/api/v1/reports/r-42/download' },
        error: null,
      });

      expect(result?.status).toBe('completed');
      expect(result?.result?.report_id).toBe('r-42');
    });

    it('URI-encodes the job_id in the polling URL', () => {
      let result: ReportJobProgress | undefined;
      svc.pollJob('a b/c').subscribe((r) => (result = r));
      http.expectOne(`${API}reports/jobs/a%20b%2Fc`).flush({
        job_id: 'a b/c',
        status: 'running',
        current: 1,
        total: 10,
        errors: [],
        result: null,
        error: null,
      });
      expect(result?.job_id).toBe('a b/c');
    });

    it('propagates 404 for an unknown job_id', () => {
      let error: HttpErrorResponse | undefined;
      svc.pollJob('gone').subscribe({ error: (e) => (error = e) });
      http
        .expectOne(`${API}reports/jobs/gone`)
        .flush({ detail: 'not found' }, { status: 404, statusText: 'Not Found' });
      expect(error?.status).toBe(404);
    });
  });

  describe('downloadUrl()', () => {
    it('builds the download URL from a report_id against the API base', () => {
      expect(svc.downloadUrl('r-42')).toBe(`${API}reports/r-42/download`);
    });

    it('URI-encodes the report id', () => {
      expect(svc.downloadUrl('r 42/x')).toBe(`${API}reports/r%2042%2Fx/download`);
    });
  });

  describe('error-path coverage (issue #915)', () => {
    // reports.service.ts has no catchError → raw HttpErrorResponse propagates.
    it('when pollJob times out (status 0), the raw HttpErrorResponse propagates', () => {
      let error: HttpErrorResponse | undefined;
      svc.pollJob('rpt-1').subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}reports/jobs/rpt-1`)
        .error(new ProgressEvent('timeout'), { status: 0 });

      expect(error instanceof HttpErrorResponse).toBe(true);
      expect(error?.status).toBe(0);
    });

    it('when generate returns 503, error.status is 503 and the body is on error.error', () => {
      let error: HttpErrorResponse | undefined;
      svc.generate(generateRequest()).subscribe({ error: (e) => (error = e) });

      const body = { detail: 'worker pool exhausted' };
      http
        .expectOne(`${API}reports/generate`)
        .flush(body, { status: 503, statusText: 'Service Unavailable' });

      expect(error?.status).toBe(503);
      expect(error?.error).toEqual(body);
    });

    it('when generate returns 422 with a top-level validation_errors body, the body is on error.error', () => {
      let error: HttpErrorResponse | undefined;
      svc.generate(generateRequest()).subscribe({ error: (e) => (error = e) });

      const body = { validation_errors: { template: ['required'] } };
      http
        .expectOne(`${API}reports/generate`)
        .flush(body, { status: 422, statusText: 'Unprocessable' });

      expect(error?.status).toBe(422);
      expect(error?.error).toEqual(body);
    });

    it('when generate returns 422 with a nested error.details.validation_errors body, the body is reachable on error.error', () => {
      let error: HttpErrorResponse | undefined;
      svc.generate(generateRequest()).subscribe({ error: (e) => (error = e) });

      const nested = {
        error: { details: { validation_errors: { sections: ['required'] } } },
      };
      http
        .expectOne(`${API}reports/generate`)
        .flush(nested, { status: 422, statusText: 'Unprocessable' });

      expect(error?.status).toBe(422);
      expect(error?.error).toEqual(nested);
    });

    it('when pollJob returns a malformed body, the result is defined and status is undefined', () => {
      let result: ReportJobProgress | undefined;
      svc.pollJob('rpt-1').subscribe((r) => (result = r));

      http.expectOne(`${API}reports/jobs/rpt-1`).flush({ job_id: 'rpt-1' });

      expect(result).toBeDefined();
      expect(result?.status).toBeUndefined();
    });

    it('when the caller unsubscribes from pollJob before the flush, next never fires', () => {
      let emitted = false;
      const sub = svc
        .pollJob('rpt-1')
        .subscribe({ next: () => (emitted = true), error: () => (emitted = true) });
      sub.unsubscribe();

      // A cancelled TestRequest cannot be flushed (Angular 21.1.1 throws), and
      // default verify() counts it as open — drain it with expectOne instead.
      const req = http.expectOne(`${API}reports/jobs/rpt-1`);
      expect(req.cancelled).toBe(true);
      expect(emitted).toBe(false);
    });
  });
});
