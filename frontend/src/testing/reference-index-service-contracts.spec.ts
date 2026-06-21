/**
 * Request/response contract parity for ReferenceIndexService (issue #1027).
 *
 * Covers: startSeed() POST, getJob() GET, getStatus() GET, getConfigured() GET
 * — HTTP method, exact URL, body keys, field types, and error propagation.
 */
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
  type TestRequest,
} from '@angular/common/http/testing';

import { assertMethod, assertUrl, assertBodyKeys, assertBodyFieldTypes } from './contract-request.helper';
import {
  ReferenceIndexService,
  type ReferenceIndexSeedJobResponse,
  type ReferenceIndexSeedProgress,
  type ReferenceIndexStatusResponse,
  type ReferenceIndexConfiguredResponse,
} from '../app/core/services/reference-index.service';
import { environment } from '../environments/environment';

const API = environment.apiUrl;
const BASE = `${API}reference-indices`;

const SEED_RESPONSE: ReferenceIndexSeedJobResponse = { job_id: 'job-1', status: 'pending', message: 'Seeding' };
const JOB_PROGRESS: ReferenceIndexSeedProgress = {
  job_id: 'job-1', status: 'running', current: 3, total: 10, errors: [], result: null, error: null,
};
const STATUS_RESPONSE: ReferenceIndexStatusResponse = {
  items: [], total: 0, healthyCount: 0, missingCount: 0, staleCount: 0, staleDays: 7,
};
const CONFIGURED_RESPONSE: ReferenceIndexConfiguredResponse = {
  tickers: ['SPY'], settingsTickers: [], portfolioTickers: [],
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

function capture(http: HttpTestingController, url: string, flushBody: object | null = {}): TestRequest {
  const req = http.expectOne((r) => r.url === url);
  req.flush(flushBody);
  return req;
}

describe('ReferenceIndexService — request/response contract parity (issue #1027)', () => {
  let svc: ReferenceIndexService;
  let http: HttpTestingController;

  beforeEach(() => {
    ({ http } = setupHttp());
    svc = TestBed.inject(ReferenceIndexService);
  });

  afterEach(() => http.verify());

  // ── startSeed() ────────────────────────────────────────────────────────────

  describe('startSeed(tickers)', () => {
    const TICKERS = ['AAPL', 'MSFT'];

    it('sends POST to exact reference-indices/seed URL', () => {
      svc.startSeed(TICKERS).subscribe({ error: () => {} });
      const req = capture(http, `${BASE}/seed`, SEED_RESPONSE);
      assertMethod(req, 'POST');
      assertUrl(req, `${BASE}/seed`);
    });

    it('request body has exactly one key: tickers', () => {
      svc.startSeed(TICKERS).subscribe({ error: () => {} });
      const req = capture(http, `${BASE}/seed`, SEED_RESPONSE);
      assertBodyKeys(req, ['tickers']);
    });

    it('request body tickers field is an array', () => {
      svc.startSeed(TICKERS).subscribe({ error: () => {} });
      const req = http.expectOne((r) => r.url === `${BASE}/seed`);
      assertBodyFieldTypes(req, { tickers: 'array' });
      req.flush(SEED_RESPONSE);
    });

    it('response has expected ReferenceIndexSeedJobResponse keys', () => {
      let result: Record<string, unknown> | undefined;
      svc.startSeed(TICKERS).subscribe((r) => (result = r as unknown as Record<string, unknown>));
      http.expectOne((r) => r.url === `${BASE}/seed`).flush(SEED_RESPONSE);
      expect(Object.keys(result ?? {})).toEqual(jasmine.arrayContaining(Object.keys(SEED_RESPONSE)));
    });

    it('when API errors, the observable errors (propagates to caller)', () => {
      let errored = false;
      svc.startSeed(TICKERS).subscribe({ error: () => (errored = true) });
      http.expectOne((r) => r.url === `${BASE}/seed`)
        .flush('Bad Request', { status: 400, statusText: 'Bad Request' });
      expect(errored).toBe(true);
    });
  });

  // ── getJob() ────────────────────────────────────────────────────────────────

  describe('getJob(jobId)', () => {
    const JOB_ID = 'job-1';

    it('sends GET to exact reference-indices/seed/{jobId} URL', () => {
      svc.getJob(JOB_ID).subscribe({ error: () => {} });
      const req = capture(http, `${BASE}/seed/${JOB_ID}`, JOB_PROGRESS);
      assertMethod(req, 'GET');
      assertUrl(req, `${BASE}/seed/${JOB_ID}`);
    });

    it('response has expected ReferenceIndexSeedProgress keys', () => {
      let result: Record<string, unknown> | undefined;
      svc.getJob(JOB_ID).subscribe((r) => (result = r as unknown as Record<string, unknown>));
      http.expectOne((r) => r.url === `${BASE}/seed/${JOB_ID}`).flush(JOB_PROGRESS);
      expect(Object.keys(result ?? {})).toEqual(jasmine.arrayContaining(Object.keys(JOB_PROGRESS)));
    });

    it('when API errors, the observable errors', () => {
      let errored = false;
      svc.getJob(JOB_ID).subscribe({ error: () => (errored = true) });
      http.expectOne((r) => r.url === `${BASE}/seed/${JOB_ID}`)
        .flush('Not found', { status: 404, statusText: 'Not Found' });
      expect(errored).toBe(true);
    });
  });

  // ── getStatus() ────────────────────────────────────────────────────────────

  describe('getStatus()', () => {
    it('sends GET to exact reference-indices/status URL', () => {
      svc.getStatus().subscribe({ error: () => {} });
      const req = capture(http, `${BASE}/status`, STATUS_RESPONSE);
      assertMethod(req, 'GET');
      assertUrl(req, `${BASE}/status`);
    });

    it('response has expected ReferenceIndexStatusResponse keys', () => {
      let result: Record<string, unknown> | undefined;
      svc.getStatus().subscribe((r) => (result = r as unknown as Record<string, unknown>));
      http.expectOne((r) => r.url === `${BASE}/status`).flush(STATUS_RESPONSE);
      expect(Object.keys(result ?? {})).toEqual(jasmine.arrayContaining(Object.keys(STATUS_RESPONSE)));
    });

    it('when API errors, the observable errors', () => {
      let errored = false;
      svc.getStatus().subscribe({ error: () => (errored = true) });
      http.expectOne((r) => r.url === `${BASE}/status`)
        .flush('Server Error', { status: 500, statusText: 'Server Error' });
      expect(errored).toBe(true);
    });
  });

  // ── getConfigured() ────────────────────────────────────────────────────────

  describe('getConfigured()', () => {
    it('sends GET to exact reference-indices/configured URL', () => {
      svc.getConfigured().subscribe({ error: () => {} });
      const req = capture(http, `${BASE}/configured`, CONFIGURED_RESPONSE);
      assertMethod(req, 'GET');
      assertUrl(req, `${BASE}/configured`);
    });

    it('response has expected ReferenceIndexConfiguredResponse keys', () => {
      let result: Record<string, unknown> | undefined;
      svc.getConfigured().subscribe((r) => (result = r as unknown as Record<string, unknown>));
      http.expectOne((r) => r.url === `${BASE}/configured`).flush(CONFIGURED_RESPONSE);
      expect(Object.keys(result ?? {})).toEqual(jasmine.arrayContaining(Object.keys(CONFIGURED_RESPONSE)));
    });

    it('when API errors, the observable errors', () => {
      let errored = false;
      svc.getConfigured().subscribe({ error: () => (errored = true) });
      http.expectOne((r) => r.url === `${BASE}/configured`)
        .flush('Server Error', { status: 500, statusText: 'Server Error' });
      expect(errored).toBe(true);
    });
  });
});
