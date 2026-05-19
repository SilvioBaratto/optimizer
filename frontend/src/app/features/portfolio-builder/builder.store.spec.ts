import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { BuilderStore } from './builder.store';
import type { BuilderStage } from './builder-stage';
import {
  type RunLevelConfig,
  StepStatus,
} from '../../models/pipeline-builder.model';
import { STEP_PARAM_DEFAULTS } from '../../pages/pipeline-stepper/step-params.model';
import { environment } from '../../../environments/environment';

const SESSIONS_URL = `${environment.apiUrl}pipeline-builder/sessions`;
const STEP_URL = (sid: string, step: string) =>
  `${SESSIONS_URL}/${sid}/steps/${step}`;
const ACCEPTED = { status: 202, statusText: 'Accepted' };

function acceptResponse() {
  return { job_id: 'j-1', status: 'pending', message: '' };
}

function defaultConfig(): RunLevelConfig {
  return {
    rebalance_freq: 63,
    n_selected: 20,
    cost_bps: 10,
    tax_rate: 0.26,
    base_currency: 'EUR',
    robust: false,
    persist: false,
    start_date: null,
    end_date: null,
    seed: 42,
  };
}

describe('BuilderStore', () => {
  let store: BuilderStore;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        BuilderStore,
      ],
    });
    store = TestBed.inject(BuilderStore);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
  });

  it('when freshly constructed, initial summary has no config and status Pending', () => {
    expect(store.config()).toBeNull();
    expect(store.stepParams()).toBeNull();
    expect(store.currentStage()).toBeNull();
    expect(store.status()).toBe(StepStatus.Pending);
    expect(store.summary()).toEqual({
      stage: null,
      status: StepStatus.Pending,
      hasConfig: false,
    });
  });

  it('when freshly constructed, sessionId is null', () => {
    expect(store.sessionId()).toBeNull();
  });

  it('when setConfig is called, hasConfig flips to true', () => {
    const cfg = defaultConfig();
    store.setConfig(cfg);
    expect(store.config()).toBe(cfg);
    expect(store.summary().hasConfig).toBe(true);
  });

  it('when setStepParams is called, stepParams signal exposes the value', () => {
    const params = structuredClone(STEP_PARAM_DEFAULTS);
    store.setStepParams(params);
    expect(store.stepParams()).toBe(params);
  });

  it('when setStage is called with "universe", currentStage and summary.stage update', () => {
    store.setStage('universe');
    expect(store.currentStage()).toBe('universe');
    expect(store.summary().stage).toBe('universe');
  });

  it('when setStage cycles through all 6 BuilderStage ids, each is accepted and exposed', () => {
    const stages: readonly BuilderStage[] = [
      'universe',
      'objective',
      'constraints',
      'optimize',
      'review',
      'rebalance',
    ];
    for (const stage of stages) {
      store.setStage(stage);
      expect(store.currentStage()).toBe(stage);
      expect(store.summary().stage).toBe(stage);
    }
  });

  it('when setStage is called with null, currentStage clears back to null', () => {
    store.setStage('optimize');
    store.setStage(null);
    expect(store.currentStage()).toBeNull();
    expect(store.summary().stage).toBeNull();
  });

  it('when setStatus is called with Running, summary.status propagates', () => {
    store.setStatus(StepStatus.Running);
    expect(store.status()).toBe(StepStatus.Running);
    expect(store.summary().status).toBe(StepStatus.Running);
  });

  it('when setSessionId is called with a string, sessionId exposes that value', () => {
    store.setSessionId('sid-xyz');
    expect(store.sessionId()).toBe('sid-xyz');
  });

  it('when setSessionId is called with null after being set, sessionId returns to null', () => {
    store.setSessionId('sid-xyz');
    store.setSessionId(null);
    expect(store.sessionId()).toBeNull();
  });

  it('when freshly constructed, stepResults and stepStatuses are empty maps', () => {
    expect(store.stepResults().size).toBe(0);
    expect(store.stepStatuses().size).toBe(0);
  });

  it('when setStepResult is called, stepResults exposes the entry under the step id', () => {
    const result = { n_tickers: 42 } as Record<string, unknown>;
    store.setStepResult('load', result);
    expect(store.stepResults().get('load')).toBe(result);
  });

  it('when setStepResult is called twice for different steps, both entries are preserved', () => {
    const a = { n_tickers: 1 } as Record<string, unknown>;
    const b = { n_investable: 2 } as Record<string, unknown>;
    store.setStepResult('load', a);
    store.setStepResult('screen', b);
    expect(store.stepResults().get('load')).toBe(a);
    expect(store.stepResults().get('screen')).toBe(b);
    expect(store.stepResults().size).toBe(2);
  });

  it('when setStepResult is called with null, the entry is stored as null', () => {
    store.setStepResult('load', null);
    expect(store.stepResults().has('load')).toBe(true);
    expect(store.stepResults().get('load')).toBeNull();
  });

  it('when setStepStatus is called, stepStatuses exposes the entry under the step id', () => {
    store.setStepStatus('optimize', StepStatus.Running);
    expect(store.stepStatuses().get('optimize')).toBe(StepStatus.Running);
  });

  it('when setStepStatus mutates the same step twice, the latest value wins', () => {
    store.setStepStatus('load', StepStatus.Running);
    store.setStepStatus('load', StepStatus.Completed);
    expect(store.stepStatuses().get('load')).toBe(StepStatus.Completed);
  });

  it('when reset is called after mutations, all signals return to initial', () => {
    store.setConfig(defaultConfig());
    store.setStepParams(structuredClone(STEP_PARAM_DEFAULTS));
    store.setStage('optimize');
    store.setStatus(StepStatus.Completed);
    store.setStepResult('load', { n_tickers: 1 } as Record<string, unknown>);
    store.setStepStatus('load', StepStatus.Completed);
    store.setSessionId('sid-old');

    store.reset();

    expect(store.config()).toBeNull();
    expect(store.stepParams()).toBeNull();
    expect(store.currentStage()).toBeNull();
    expect(store.status()).toBe(StepStatus.Pending);
    expect(store.stepResults().size).toBe(0);
    expect(store.stepStatuses().size).toBe(0);
    expect(store.sessionId()).toBeNull();
    expect(store.summary()).toEqual({
      stage: null,
      status: StepStatus.Pending,
      hasConfig: false,
    });
  });

  describe('optimize()', () => {
    it('when sessionId is null, optimize() is a no-op and no HTTP request is made', () => {
      store.optimize();
      http.expectNone(() => true);
      expect(store.status()).toBe(StepStatus.Pending);
    });

    it('when status is already Running, optimize() is a no-op and no HTTP request is made', () => {
      store.setSessionId('sid-1');
      store.setStatus(StepStatus.Running);
      store.optimize();
      http.expectNone(() => true);
      expect(store.status()).toBe(StepStatus.Running);
    });

    it('when sessionId is set and status is not Running, optimize() POSTs to /sessions/{sid}/steps/optimize with empty body', () => {
      store.setSessionId('sid-1');
      store.optimize();
      const req = http.expectOne(STEP_URL('sid-1', 'optimize'));
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({});
      req.flush(acceptResponse(), ACCEPTED);
    });

    it('when optimize() dispatches, status is set to Running optimistically', () => {
      store.setSessionId('sid-1');
      store.optimize();
      expect(store.status()).toBe(StepStatus.Running);
      const req = http.expectOne(STEP_URL('sid-1', 'optimize'));
      req.flush(acceptResponse(), ACCEPTED);
    });

    it('when optimize() POST fails, status transitions to Error', () => {
      store.setSessionId('sid-1');
      store.optimize();
      const req = http.expectOne(STEP_URL('sid-1', 'optimize'));
      req.flush('boom', { status: 500, statusText: 'Server Error' });
      expect(store.status()).toBe(StepStatus.Error);
    });
  });

  describe('rebalance()', () => {
    it('when sessionId is null, rebalance() is a no-op and no HTTP request is made', () => {
      store.rebalance();
      http.expectNone(() => true);
      expect(store.status()).toBe(StepStatus.Pending);
    });

    it('when status is already Running, rebalance() is a no-op and no HTTP request is made', () => {
      store.setSessionId('sid-1');
      store.setStatus(StepStatus.Running);
      store.rebalance();
      http.expectNone(() => true);
      expect(store.status()).toBe(StepStatus.Running);
    });

    it('when sessionId is set and status is not Running, rebalance() POSTs to /sessions/{sid}/steps/rebalance_decision with empty body', () => {
      store.setSessionId('sid-2');
      store.rebalance();
      const req = http.expectOne(STEP_URL('sid-2', 'rebalance_decision'));
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({});
      req.flush(acceptResponse(), ACCEPTED);
    });

    it('when rebalance() dispatches, status is set to Running optimistically', () => {
      store.setSessionId('sid-2');
      store.rebalance();
      expect(store.status()).toBe(StepStatus.Running);
      const req = http.expectOne(STEP_URL('sid-2', 'rebalance_decision'));
      req.flush(acceptResponse(), ACCEPTED);
    });

    it('when rebalance() POST fails, status transitions to Error', () => {
      store.setSessionId('sid-2');
      store.rebalance();
      const req = http.expectOne(STEP_URL('sid-2', 'rebalance_decision'));
      req.flush('boom', { status: 500, statusText: 'Server Error' });
      expect(store.status()).toBe(StepStatus.Error);
    });
  });

  describe('export()', () => {
    it('when export() is invoked with no sessionId, it does not throw and no HTTP request is made', () => {
      expect(() => store.export()).not.toThrow();
      http.expectNone(() => true);
    });

    it('when export() is invoked with a sessionId, it does not throw and no HTTP request is made (Phase 2 stub)', () => {
      store.setSessionId('sid-3');
      expect(() => store.export()).not.toThrow();
      http.expectNone(() => true);
    });

    it('when export() is invoked, status is unchanged', () => {
      store.setSessionId('sid-3');
      store.setStatus(StepStatus.Completed);
      store.export();
      expect(store.status()).toBe(StepStatus.Completed);
    });
  });
});
