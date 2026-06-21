/**
 * Request/response contract parity for ReportsService (issue #1027).
 *
 * Covers: generate() POST, pollJob() GET — HTTP method, exact URL, body keys,
 * and response field-parity against the committed reports.json snapshot.
 */
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
  type TestRequest,
} from '@angular/common/http/testing';

import { assertMethod, assertUrl, assertBodyKeys } from './contract-request.helper';
import { schemaOf } from './contract-parity';
import { assertFieldParity } from './contract-field-parity';
import { makeReportJobCreateResponse } from './domain-fixtures';
import reportsSnapshot from './contract-snapshots/reports.json';
import { ReportsService } from '../app/core/services/reports.service';
import type { ReportGenerateRequest } from '../app/core/models/report.model';
import { environment } from '../environments/environment';

const API = environment.apiUrl;

const GENERATE_REQUEST: ReportGenerateRequest = {
  sections: ['portfolio_summary', 'weights'],
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

function capture(http: HttpTestingController, url: string, flushBody: object = {}): TestRequest {
  const req = http.expectOne((r) => r.url === url);
  req.flush(flushBody);
  return req;
}

describe('ReportsService — request/response contract parity (issue #1027)', () => {
  let svc: ReportsService;
  let http: HttpTestingController;

  beforeEach(() => {
    ({ http } = setupHttp());
    svc = TestBed.inject(ReportsService);
  });

  afterEach(() => http.verify());

  // ── generate() ────────────────────────────────────────────────────────────

  describe('generate()', () => {
    it('sends POST to exact reports/generate URL', () => {
      svc.generate(GENERATE_REQUEST).subscribe({ error: () => {} });
      const req = capture(http, `${API}reports/generate`, makeReportJobCreateResponse());
      assertMethod(req, 'POST');
      assertUrl(req, `${API}reports/generate`);
    });

    it('request body includes exactly the keys present in the request object', () => {
      svc.generate(GENERATE_REQUEST).subscribe({ error: () => {} });
      const req = capture(http, `${API}reports/generate`, makeReportJobCreateResponse());
      assertBodyKeys(req, Object.keys(GENERATE_REQUEST));
    });

    it('request body sections field is an array', () => {
      svc.generate(GENERATE_REQUEST).subscribe({ error: () => {} });
      const req = http.expectOne((r) => r.url === `${API}reports/generate`);
      expect(Array.isArray(req.request.body?.['sections'])).toBe(true);
      req.flush(makeReportJobCreateResponse());
    });

    it('response matches ReportJobCreateResponse snapshot field-parity', () => {
      const FIXTURE = makeReportJobCreateResponse();
      let result: Record<string, unknown> | undefined;
      svc.generate(GENERATE_REQUEST)
        .subscribe((r) => (result = r as unknown as Record<string, unknown>));

      http.expectOne((r) => r.url === `${API}reports/generate`).flush(FIXTURE);

      const schema = schemaOf(reportsSnapshot, 'ReportJobCreateResponse');
      assertFieldParity(schema, FIXTURE);
      expect(Object.keys(result ?? {})).toEqual(jasmine.arrayContaining(Object.keys(FIXTURE)));
    });

    it('when API returns 500, the observable errors', () => {
      let errored = false;
      svc.generate(GENERATE_REQUEST).subscribe({ error: () => (errored = true) });
      http.expectOne((r) => r.url === `${API}reports/generate`)
        .flush('Server Error', { status: 500, statusText: 'Server Error' });
      expect(errored).toBe(true);
    });
  });

  // ── pollJob() ──────────────────────────────────────────────────────────────

  describe('pollJob(jobId)', () => {
    const JOB_ID = 'job abc';
    const ENC_ID = encodeURIComponent(JOB_ID);

    it('sends GET to exact encoded reports/jobs/{jobId} URL', () => {
      svc.pollJob(JOB_ID).subscribe({ error: () => {} });
      const req = capture(http, `${API}reports/jobs/${ENC_ID}`);
      assertMethod(req, 'GET');
      assertUrl(req, `${API}reports/jobs/${ENC_ID}`);
    });

    it('path parameter is percent-encoded (no raw {token} in URL)', () => {
      svc.pollJob(JOB_ID).subscribe({ error: () => {} });
      const req = http.expectOne((r) => r.url === `${API}reports/jobs/${ENC_ID}`);
      expect(req.request.url).toContain(ENC_ID);
      expect(req.request.url).not.toContain(JOB_ID.replace(/%/g, ''));
      req.flush({});
    });

    it('when API returns 404, the observable errors', () => {
      let errored = false;
      svc.pollJob(JOB_ID).subscribe({ error: () => (errored = true) });
      http.expectOne((r) => r.url === `${API}reports/jobs/${ENC_ID}`)
        .flush('Not found', { status: 404, statusText: 'Not Found' });
      expect(errored).toBe(true);
    });
  });
});
