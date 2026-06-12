/**
 * portfolio-builder-field-parity.spec.ts — issue #967
 *
 * Field-level + HTTP-contract parity for BuilderDriftService and
 * PipelineBuilderApiService.
 *
 * Covers:
 *   – DriftResponse field-parity vs portfolio.json snapshot
 *   – Drift URL: GET portfolio/{name%20encoded}/drift?base=…
 *   – createSession() POST + RunLevelConfig snake_case body
 *   – runStep() POST with percent-encoded path segments, empty body
 *   – pollStep() GET URL shape
 *   – Pipeline-builder session response shapes (CreateSession, StepRun, StepPoll)
 */

import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';

import { schemaOf } from './contract-parity';
import { assertFieldParity, requiredKeys } from './contract-field-parity';
import portfolioSnapshot from './contract-snapshots/portfolio.json';
import pipelineBuilderSnapshot from './contract-snapshots/pipeline_builder.json';
import {
  makeCreateSessionResponse,
  makeDriftResponseRich,
  makeStepPollResponse,
  makeStepRunWireResponse,
} from './domain-fixtures';
import { BuilderDriftService } from '../app/portfolio-builder/builder-drift.service';
import {
  BuilderStore,
  BUILDER_DRIFT_SERVICE,
} from '../app/portfolio-builder/state/builder.store';
import { PipelineBuilderApiService } from '../app/core/services/pipeline-builder-api.service';
import type { RunLevelConfig } from '../app/core/models/pipeline-builder.model';

// ── Helpers ───────────────────────────────────────────────────────────────────

const DRIFT_FLUSH = makeDriftResponseRich();

const RUN_CONFIG: RunLevelConfig = {
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
};

// ── DriftResponse — field-parity against portfolio.json ───────────────────────

describe('DriftResponse — assertFieldParity (portfolio.json, issue #967)', () => {
  const schema = schemaOf(portfolioSnapshot, 'DriftResponse');

  it('required fields: holdings, target, drift, trades, totals, diagnostics, request_id', () => {
    const req = requiredKeys(schema);
    ['holdings', 'target', 'drift', 'trades', 'totals', 'diagnostics', 'request_id'].forEach(
      (k) => expect(req.has(k)).withContext(`expected '${k}' in required`).toBe(true),
    );
  });

  it('assertFieldParity: DriftResponseRich fixture types match snapshot (schema as root for $ref)', () => {
    expect(() => assertFieldParity(schema, makeDriftResponseRich(), schema)).not.toThrow();
  });

  it('wire shape is snake_case: request_id (not requestId)', () => {
    const props = schema.properties ?? {};
    expect('request_id' in props).toBe(true);
    expect('requestId' in props).toBe(false);
  });
});

// ── BuilderDriftService — HTTP URL contract ───────────────────────────────────

describe('BuilderDriftService — HTTP URL contract (issue #967)', () => {
  let service: BuilderDriftService;
  let store: BuilderStore;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        BuilderStore,
        BuilderDriftService,
        { provide: BUILDER_DRIFT_SERVICE, useValue: { runExplicit: () => {} } },
      ],
    });
    http = TestBed.inject(HttpTestingController);
    store = TestBed.inject(BuilderStore);
    service = TestBed.inject(BuilderDriftService);
  });

  afterEach(() => http.verify());

  function triggerFetch(name: string): void {
    store.setPortfolioName(name);
    store.setResultStatus('ok');
    service.init();
    service.runExplicit();
  }

  it('URL percent-encodes name: "Tech Portfolio" → …/Tech%20Portfolio/drift', () => {
    triggerFetch('Tech Portfolio');

    const req = http.expectOne((r) =>
      r.url.includes('portfolio/Tech%20Portfolio/drift'),
    );
    expect(req.request.method).toBe('GET');
    req.flush(DRIFT_FLUSH);
  });

  it('URL path segment is the NAME, not a raw {name} token', () => {
    triggerFetch('Growth Fund');

    const req = http.expectOne((r) => r.url.includes('/drift'));
    expect(req.request.url).not.toContain('{name}');
    req.flush(DRIFT_FLUSH);
  });

  it('appends base as a query param (not embedded in the path)', () => {
    triggerFetch('Growth Fund');

    const req = http.expectOne((r) => r.url.includes('/drift'));
    expect(req.request.params.has('base')).toBe(true);
    req.flush(DRIFT_FLUSH);
  });

  it('when portfolioName is null, no HTTP request is sent', () => {
    store.setPortfolioName(null);
    store.setResultStatus('ok');
    service.init();
    service.runExplicit();

    expect(http.match(() => true).length).toBe(0);
  });
});

// ── PipelineBuilderApiService — HTTP contract ─────────────────────────────────

describe('PipelineBuilderApiService — HTTP contract (issue #967)', () => {
  let service: PipelineBuilderApiService;
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
    service = TestBed.inject(PipelineBuilderApiService);
  });

  afterEach(() => http.verify());

  describe('createSession()', () => {
    it('sends POST to .../pipeline-builder/sessions', () => {
      service.createSession(RUN_CONFIG).subscribe();

      const req = http.expectOne(
        (r) => r.method === 'POST' && r.url.includes('/pipeline-builder/sessions'),
      );
      req.flush({ sessionId: 'sess-1' });
    });

    it('request body equals the RunLevelConfig passed in', () => {
      service.createSession(RUN_CONFIG).subscribe();

      const req = http.expectOne((r) => r.method === 'POST' && r.url.includes('/sessions'));
      expect(req.request.body).toEqual(RUN_CONFIG);
      req.flush({ sessionId: 'sess-1' });
    });

    it('RunLevelConfig body uses snake_case (rebalance_freq, n_selected, base_currency)', () => {
      expect('rebalance_freq' in RUN_CONFIG).toBe(true);
      expect('n_selected' in RUN_CONFIG).toBe(true);
      expect('base_currency' in RUN_CONFIG).toBe(true);
      expect('rebalanceFreq' in RUN_CONFIG).toBe(false);
      expect('nSelected' in RUN_CONFIG).toBe(false);
      expect('baseCurrency' in RUN_CONFIG).toBe(false);
    });
  });

  describe('runStep()', () => {
    it('sends POST to .../sessions/{sessionId}/steps/{stepId} with encoded segments', () => {
      service.runStep('sess abc', 'load', {}).subscribe();

      const req = http.expectOne(
        (r) => r.method === 'POST' && r.url.includes('/sessions/sess%20abc/steps/load'),
      );
      req.flush({ job_id: 'job-1', status: 'pending', message: '' });
    });

    it('request body is the params object passed in (empty object when {})', () => {
      service.runStep('sess-1', 'load', {}).subscribe();

      const req = http.expectOne((r) => r.method === 'POST' && r.url.includes('/steps/'));
      expect(req.request.body).toEqual({});
      req.flush({ job_id: 'job-1', status: 'pending', message: '' });
    });

    it('URL has no raw {stepId} token', () => {
      service.runStep('sess-1', 'optimize', {}).subscribe();

      const req = http.expectOne((r) => r.method === 'POST' && r.url.includes('/steps/'));
      expect(req.request.url).not.toContain('{stepId}');
      req.flush({ job_id: 'job-1', status: 'pending', message: '' });
    });

    it('maps wire job_id to camelCase jobId in emitted StepRunResponse', () => {
      let result: { jobId: string } | undefined;
      service.runStep('sess-1', 'load', {}).subscribe((r) => (result = r));

      http
        .expectOne((r) => r.method === 'POST')
        .flush({ job_id: 'j-42', status: 'pending', message: 'queued' });

      expect(result?.jobId).toBe('j-42');
    });
  });

  describe('pollStep()', () => {
    it('sends GET to .../sessions/{sessionId}/steps/{stepId}', () => {
      service.pollStep('sess-1', 'load').subscribe();

      const req = http.expectOne(
        (r) => r.method === 'GET' && r.url.includes('/sessions/sess-1/steps/load'),
      );
      req.flush({ status: 'running', progress: { current: 1, total: 5 }, result: null, error: null, gateReason: null });
    });
  });
});

// ── Pipeline-builder session response shapes ──────────────────────────────────

describe('PipelineBuilderApiService — session response shapes (issue #967)', () => {

  describe('CreateSessionResponse — camelCase session init', () => {
    const schema = schemaOf(pipelineBuilderSnapshot, 'CreateSessionResponse');

    it('required field: sessionId', () => {
      expect(requiredKeys(schema).has('sessionId')).toBe(true);
    });

    it('camelCase sessionId (not session_id)', () => {
      const f = makeCreateSessionResponse();
      expect('sessionId' in f).toBe(true);
      expect('session_id' in f).toBe(false);
    });

    it('assertFieldParity: fixture matches snapshot', () => {
      expect(() =>
        assertFieldParity(schema, makeCreateSessionResponse(), pipelineBuilderSnapshot as never),
      ).not.toThrow();
    });
  });

  describe('StepRunResponse — snake_case wire; camelCase after mapping', () => {
    const schema = schemaOf(pipelineBuilderSnapshot, 'StepRunResponse');

    it('wire required fields: job_id, status', () => {
      const req = requiredKeys(schema);
      expect(req.has('job_id')).toBe(true);
      expect(req.has('status')).toBe(true);
    });

    it('assertFieldParity: wire fixture matches snapshot', () => {
      expect(() =>
        assertFieldParity(schema, makeStepRunWireResponse(), pipelineBuilderSnapshot as never),
      ).not.toThrow();
    });
  });

  describe('StepPollResponse — camelCase gateReason', () => {
    const schema = schemaOf(pipelineBuilderSnapshot, 'StepPollResponse');

    it('required field: status', () => {
      expect(requiredKeys(schema).has('status')).toBe(true);
    });

    it('gateReason uses camelCase (not gate_reason)', () => {
      const props = schema.properties ?? {};
      expect('gateReason' in props).toBe(true);
      expect('gate_reason' in props).toBe(false);
    });

    it('assertFieldParity: fixture matches snapshot', () => {
      expect(() =>
        assertFieldParity(schema, makeStepPollResponse(), pipelineBuilderSnapshot as never),
      ).not.toThrow();
    });
  });

});
