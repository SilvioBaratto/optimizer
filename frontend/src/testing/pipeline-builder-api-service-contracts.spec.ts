/**
 * Request/response contract parity for PipelineBuilderApiService (issue #1028).
 *
 * Covers: createSession POST, pollStep GET, deleteSession DELETE — asserting:
 *   – HTTP method and exact interpolated URL (no raw {token} placeholders)
 *   – RunLevelConfig body key-set (snake_case) and field types
 *   – Path params (sessionId, stepId) percent-encoded in the URL segment
 *   – DELETE responds with 204 No Content (no body keys to assert)
 *
 * Criteria (UNIT, issue #1028):
 *   PipelineBuilderApiService (createSession POST, pollStep GET, delete DELETE)
 *   asserted field-by-field.
 */
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
  type TestRequest,
} from '@angular/common/http/testing';

import {
  assertMethod,
  assertUrl,
  assertBodyKeys,
  assertBodyFieldTypes,
} from './contract-request.helper';
import { makeCreateSessionResponse, makeStepPollResponse } from './domain-fixtures';
import { PipelineBuilderApiService } from '../app/core/services/pipeline-builder-api.service';
import { environment } from '../environments/environment';

const API = environment.apiUrl;
const BASE = `${API}pipeline-builder`;

// Session and step IDs with spaces to verify encoding
const SESSION_ID = 'sess abc 1';
const SESSION_ENC = encodeURIComponent(SESSION_ID);
const STEP_ID = 'step xyz 2';
const STEP_ENC = encodeURIComponent(STEP_ID);

const RUN_CONFIG = {
  rebalance_freq: 63,
  n_selected: 20,
  cost_bps: 30,
  tax_rate: 0.26,
  base_currency: 'EUR',
  robust: false,
  persist: false,
  start_date: null,
  end_date: null,
  seed: 42,
} as const;

const RUN_CONFIG_KEYS = [
  'rebalance_freq', 'n_selected', 'cost_bps', 'tax_rate',
  'base_currency', 'robust', 'persist', 'start_date', 'end_date', 'seed',
];

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
  const req = http.expectOne(r => r.url === url);
  req.flush(flushBody);
  return req;
}

describe('PipelineBuilderApiService — request/response contract parity (issue #1028)', () => {
  let svc: PipelineBuilderApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    ({ http } = setupHttp());
    svc = TestBed.inject(PipelineBuilderApiService);
  });

  afterEach(() => http.verify());

  // ── createSession(config) — POST ─────────────────────────────────────────

  describe('createSession(config)', () => {
    const SESSIONS_URL = `${BASE}/sessions`;

    it('sends POST to exact pipeline-builder/sessions URL', () => {
      svc.createSession(RUN_CONFIG as never).subscribe({ error: () => {} });
      const req = capture(http, SESSIONS_URL, makeCreateSessionResponse());
      assertMethod(req, 'POST');
      assertUrl(req, SESSIONS_URL);
    });

    it('request body contains exactly the RunLevelConfig snake_case keys', () => {
      svc.createSession(RUN_CONFIG as never).subscribe({ error: () => {} });
      const req = capture(http, SESSIONS_URL, makeCreateSessionResponse());
      assertBodyKeys(req, RUN_CONFIG_KEYS);
    });

    it('request body numeric fields are sent as numbers (not strings)', () => {
      svc.createSession(RUN_CONFIG as never).subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === SESSIONS_URL);
      assertBodyFieldTypes(req, {
        rebalance_freq: 'number',
        n_selected: 'number',
        cost_bps: 'number',
        tax_rate: 'number',
        seed: 'number',
      });
      req.flush(makeCreateSessionResponse());
    });

    it('request body string field base_currency is sent as string', () => {
      svc.createSession(RUN_CONFIG as never).subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === SESSIONS_URL);
      assertBodyFieldTypes(req, { base_currency: 'string' });
      req.flush(makeCreateSessionResponse());
    });

    it('request body boolean fields robust and persist are sent as booleans', () => {
      svc.createSession(RUN_CONFIG as never).subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === SESSIONS_URL);
      assertBodyFieldTypes(req, { robust: 'boolean', persist: 'boolean' });
      req.flush(makeCreateSessionResponse());
    });

    it('response carries sessionId field', () => {
      let result: Record<string, unknown> | undefined;
      svc.createSession(RUN_CONFIG as never).subscribe(r => {
        result = r as unknown as Record<string, unknown>;
      });
      const FIXTURE = makeCreateSessionResponse();
      http.expectOne(r => r.url === SESSIONS_URL).flush(FIXTURE);
      expect(Object.keys(result ?? {})).toEqual(
        jasmine.arrayContaining(Object.keys(FIXTURE)),
      );
    });
  });

  // ── pollStep(sessionId, stepId) — GET ────────────────────────────────────

  describe('pollStep(sessionId, stepId)', () => {
    const POLL_URL = `${BASE}/sessions/${SESSION_ENC}/steps/${STEP_ENC}`;

    it('sends GET to exact percent-encoded sessions/{sessionId}/steps/{stepId} URL', () => {
      svc.pollStep(SESSION_ID, STEP_ID).subscribe({ error: () => {} });
      const req = capture(http, POLL_URL, makeStepPollResponse());
      assertMethod(req, 'GET');
      assertUrl(req, POLL_URL);
    });

    it('sessionId containing spaces is percent-encoded in the URL path', () => {
      svc.pollStep(SESSION_ID, STEP_ID).subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === POLL_URL);
      expect(req.request.url).toContain(SESSION_ENC);
      req.flush(makeStepPollResponse());
    });

    it('stepId containing spaces is percent-encoded in the URL path', () => {
      svc.pollStep(SESSION_ID, STEP_ID).subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === POLL_URL);
      expect(req.request.url).toContain(STEP_ENC);
      req.flush(makeStepPollResponse());
    });

    it('no raw {token} placeholder survives in the URL', () => {
      svc.pollStep(SESSION_ID, STEP_ID).subscribe({ error: () => {} });
      // assertUrl internally checks /\{[^}]*\}/ — this doubles as the coverage
      const req = capture(http, POLL_URL, makeStepPollResponse());
      assertUrl(req, POLL_URL);
    });

    it('request body is null (GET carries no body)', () => {
      svc.pollStep(SESSION_ID, STEP_ID).subscribe({ error: () => {} });
      const req = capture(http, POLL_URL, makeStepPollResponse());
      expect(req.request.body).toBeNull();
    });

    it('response carries StepPollResponse key-set', () => {
      let result: Record<string, unknown> | undefined;
      svc.pollStep(SESSION_ID, STEP_ID).subscribe(r => {
        result = r as unknown as Record<string, unknown>;
      });
      const FIXTURE = makeStepPollResponse();
      http.expectOne(r => r.url === POLL_URL).flush(FIXTURE);
      expect(Object.keys(result ?? {})).toEqual(
        jasmine.arrayContaining(Object.keys(FIXTURE)),
      );
    });
  });

  // ── deleteSession(sessionId) — DELETE ────────────────────────────────────

  describe('deleteSession(sessionId)', () => {
    const DELETE_URL = `${BASE}/sessions/${SESSION_ENC}`;

    it('sends DELETE to exact percent-encoded sessions/{sessionId} URL', () => {
      svc.deleteSession(SESSION_ID).subscribe({ error: () => {} });
      const req = capture(http, DELETE_URL, null);
      assertMethod(req, 'DELETE');
      assertUrl(req, DELETE_URL);
    });

    it('sessionId containing spaces is percent-encoded in the URL path', () => {
      svc.deleteSession(SESSION_ID).subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === DELETE_URL);
      expect(req.request.url).toContain(SESSION_ENC);
      req.flush(null, { status: 204, statusText: 'No Content' });
    });

    it('request body is null (DELETE carries no body)', () => {
      svc.deleteSession(SESSION_ID).subscribe({ error: () => {} });
      const req = capture(http, DELETE_URL, null);
      expect(req.request.body).toBeNull();
    });
  });
});
