import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { PipelineBuilderApiService } from './pipeline-builder-api.service';
import { environment } from '../../environments/environment';
import {
  PIPELINE_STEPS,
  type CreateSessionResponse,
  type RunLevelConfig,
  type StepPollResponse,
  type StepRunResponse,
} from '../models/pipeline-builder.model';

const BASE = `${environment.apiUrl}pipeline-builder`;

function defaultConfig(): RunLevelConfig {
  return {
    rebalance_freq: 63,
    n_selected: 20,
    cost_bps: 10.0,
    tax_rate: 0.26,
    base_currency: 'EUR',
    robust: false,
    persist: false,
    start_date: null,
    end_date: null,
    seed: 42,
  };
}

describe('PipelineBuilderApiService', () => {
  let svc: PipelineBuilderApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        PipelineBuilderApiService,
      ],
    });
    svc = TestBed.inject(PipelineBuilderApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  describe('createSession()', () => {
    it('when called with a RunLevelConfig, POSTs to /sessions and returns the CreateSessionResponse', () => {
      const body = defaultConfig();
      let result: CreateSessionResponse | undefined;
      svc.createSession(body).subscribe((r) => (result = r));

      const req = http.expectOne(`${BASE}/sessions`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(body);

      const payload: CreateSessionResponse = { sessionId: 'sid-1' };
      req.flush(payload, { status: 201, statusText: 'Created' });
      expect(result).toEqual(payload);
    });

    it('when the backend returns 429, the capacity error is propagated', () => {
      let error: HttpErrorResponse | undefined;
      svc.createSession(defaultConfig()).subscribe({ error: (e) => (error = e) });

      http.expectOne(`${BASE}/sessions`).flush(
        { detail: 'max concurrent sessions reached' },
        { status: 429, statusText: 'Too Many Requests' },
      );
      expect(error?.status).toBe(429);
    });
  });

  describe('runStep()', () => {
    it('when called with a step body, POSTs to /sessions/{id}/steps/{stepId} and returns the StepRunResponse', () => {
      let result: StepRunResponse | undefined;
      svc
        .runStep('sid-1', 'load', { include_delisted: true })
        .subscribe((r) => (result = r));

      const req = http.expectOne(`${BASE}/sessions/sid-1/steps/load`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({ include_delisted: true });

      req.flush(
        { job_id: 'j-1', status: 'pending', message: 'dispatched' },
        { status: 202, statusText: 'Accepted' },
      );
      expect(result).toEqual({
        jobId: 'j-1',
        status: 'pending',
        message: 'dispatched',
      });
    });

    it('when the sessionId or stepId contains unsafe chars, the URI segments are encoded', () => {
      svc.runStep('a b/c', 'load step', {}).subscribe();
      const req = http.expectOne(
        `${BASE}/sessions/a%20b%2Fc/steps/load%20step`,
      );
      req.flush({ job_id: 'j', status: 'pending', message: '' });
      expect(req.request.method).toBe('POST');
    });

    it('when the backend returns 409, the conflict error is propagated', () => {
      let error: HttpErrorResponse | undefined;
      svc
        .runStep('sid-1', 'load', {})
        .subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${BASE}/sessions/sid-1/steps/load`)
        .flush({ detail: 'job already running' }, { status: 409, statusText: 'Conflict' });
      expect(error?.status).toBe(409);
    });
  });

  describe('pollStep()', () => {
    it('when called, GETs /sessions/{id}/steps/{stepId} and returns the StepPollResponse', () => {
      let result: StepPollResponse | undefined;
      svc.pollStep('sid-1', 'coverage_gate').subscribe((r) => (result = r));

      const req = http.expectOne(`${BASE}/sessions/sid-1/steps/coverage_gate`);
      expect(req.request.method).toBe('GET');

      const payload: StepPollResponse = {
        status: 'failed',
        progress: { current: 1, total: 1 },
        result: null,
        error: null,
        gateReason: 'no factors passed IS ∩ OOS',
      };
      req.flush(payload);
      expect(result).toEqual(payload);
    });

    it('URI-encodes both path segments', () => {
      svc.pollStep('s/1', 'load step').subscribe();
      const req = http.expectOne(`${BASE}/sessions/s%2F1/steps/load%20step`);
      req.flush({
        status: 'running',
        progress: {},
        result: null,
        error: null,
        gateReason: null,
      });
      expect(req.request.method).toBe('GET');
    });
  });

  describe('deleteSession()', () => {
    it('when called, DELETEs /sessions/{id} and completes with the 204 no-content response', () => {
      let completed = false;
      svc.deleteSession('sid-1').subscribe({ complete: () => (completed = true) });

      const req = http.expectOne(`${BASE}/sessions/sid-1`);
      expect(req.request.method).toBe('DELETE');
      req.flush(null, { status: 204, statusText: 'No Content' });
      expect(completed).toBe(true);
    });

    it('URI-encodes the session id', () => {
      svc.deleteSession('a/b').subscribe();
      const req = http.expectOne(`${BASE}/sessions/a%2Fb`);
      req.flush(null, { status: 204, statusText: 'No Content' });
      expect(req.request.method).toBe('DELETE');
    });
  });

  describe('getArtifactUrl()', () => {
    it('builds an encoded URL for the named artifact', () => {
      const url = svc.getArtifactUrl('sid-1', 'report.md');
      expect(url).toBe(`${BASE}/sessions/sid-1/artifacts/report.md`);
    });

    it('URI-encodes the session id', () => {
      const url = svc.getArtifactUrl('a/b', 'weights.csv');
      expect(url).toBe(`${BASE}/sessions/a%2Fb/artifacts/weights.csv`);
    });
  });

  describe('PIPELINE_STEPS', () => {
    it('declares 13 steps in CLI dependency order', () => {
      expect(PIPELINE_STEPS.length).toBe(13);
      expect(PIPELINE_STEPS.map((s) => s.id)).toEqual([
        'load',
        'screen',
        'clean_returns',
        'build_history',
        'validate_is',
        'validate_oos',
        'coverage_gate',
        'regime',
        'optimize',
        'rebalance_decision',
        'cost',
        'report',
        'persist',
      ]);
    });

    it('assigns each step a unique zero-based index matching position', () => {
      PIPELINE_STEPS.forEach((s, i) => expect(s.index).toBe(i));
    });
  });
});
