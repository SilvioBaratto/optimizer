/**
 * inputs-pane-render-coverage.spec.ts — issue #942
 *
 * Branch coverage for InputsPaneComponent's helper methods, continue/retry
 * dispatch, the poll lifecycle (completed / gate-failure / error-failure /
 * swallowed poll error) and the resubmit-reset path. Section mount/collapse is
 * covered in inputs-pane.spec.ts and not duplicated here.
 */

import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { Subject } from 'rxjs';

import { InputsPaneComponent } from './inputs-pane';
import { BuilderStore } from '../state/builder.store';
import { JOB_POLL_TICK } from '../../shared/job-progress-tracker/job-progress-tracker';
import { environment } from '../../../environments/environment';
import {
  STEP_PARAM_DEFAULTS,
  type PipelineConfigSubmit,
} from '../../pipeline-stepper/models/step-params.model';
import { StepStatus, type PipelineStepId } from '../../core/models/pipeline-builder.model';

const SESSIONS_URL = `${environment.apiUrl}pipeline-builder/sessions`;
const STEP_URL = (sid: string, step: string) => `${SESSIONS_URL}/${sid}/steps/${step}`;
const ACCEPTED = { status: 202, statusText: 'Accepted' };

function acceptResponse() {
  return { job_id: 'j-1', status: 'pending', message: '' };
}

function defaultSubmit(): PipelineConfigSubmit {
  return {
    config: {
      rebalance_freq: 21, n_selected: 25, cost_bps: 10, tax_rate: 0.26,
      base_currency: 'EUR', robust: false, persist: false,
      start_date: null, end_date: null, seed: 42,
    },
    steps: structuredClone(STEP_PARAM_DEFAULTS),
  };
}

function poll(status: string, over: Record<string, unknown> = {}) {
  return { status, progress: { current: 1, total: 1 }, result: null, error: null, gateReason: null, ...over };
}

function setup(): {
  fixture: ComponentFixture<InputsPaneComponent>;
  comp: InputsPaneComponent;
  store: BuilderStore;
  http: HttpTestingController;
  tick: Subject<number>;
} {
  const tick = new Subject<number>();
  TestBed.configureTestingModule({
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
      { provide: JOB_POLL_TICK, useValue: tick.asObservable() },
      BuilderStore,
    ],
  });
  const http = TestBed.inject(HttpTestingController);
  const store = TestBed.inject(BuilderStore);
  const fixture = TestBed.createComponent(InputsPaneComponent);
  fixture.detectChanges();
  return { fixture, comp: fixture.componentInstance, store, http, tick };
}

describe('InputsPaneComponent — render-coverage (#942)', () => {
  it('sectionAggStatus resolves every aggregate arm', () => {
    const { comp, store, http } = setup();
    store.setStepStatus('load', StepStatus.Error);
    expect(comp.sectionAggStatus('universe')).toBe(StepStatus.Error);
    store.setStepStatus('load', StepStatus.Running);
    expect(comp.sectionAggStatus('universe')).toBe(StepStatus.Running);
    store.setStepStatus('load', StepStatus.Completed);
    store.setStepStatus('screen', StepStatus.Completed);
    expect(comp.sectionAggStatus('universe')).toBe(StepStatus.Completed);
    store.setStepStatus('load', StepStatus.Ready);
    expect(comp.sectionAggStatus('universe')).toBe(StepStatus.Ready);
    store.setStepStatus('load', StepStatus.Pending);
    store.setStepStatus('screen', StepStatus.Pending);
    expect(comp.sectionAggStatus('universe')).toBe(StepStatus.Pending);
    store.setStepStatus('load', StepStatus.Locked);
    store.setStepStatus('screen', StepStatus.Locked);
    expect(comp.sectionAggStatus('universe')).toBe(StepStatus.Locked);
    http.verify();
  });

  it('sectionAggSummary returns a string', () => {
    const { comp, http } = setup();
    expect(typeof comp.sectionAggSummary('universe')).toBe('string');
    http.verify();
  });

  it('errorFor returns lastError only for the active step', () => {
    const { comp, http } = setup();
    comp.activeStepId.set('load');
    comp.lastError.set('boom');
    expect(comp.errorFor('load')).toBe('boom');
    expect(comp.errorFor('screen')).toBeNull();
    http.verify();
  });

  it('showContinueForSection is true only when the active step completed with a successor', () => {
    const { comp, store, http } = setup();
    comp.activeStepId.set('load');
    store.setStepStatus('load', StepStatus.Completed);
    expect(comp.showContinueForSection('universe')).toBe(true);
    store.setStepStatus('load', StepStatus.Running);
    expect(comp.showContinueForSection('universe')).toBe(false);
    comp.activeStepId.set(null);
    expect(comp.showContinueForSection('universe')).toBe(false);
    http.verify();
  });

  it('showRetryForSection is true only when the active step errored', () => {
    const { comp, store, http } = setup();
    comp.activeStepId.set('load');
    store.setStepStatus('load', StepStatus.Error);
    expect(comp.showRetryForSection('universe')).toBe(true);
    store.setStepStatus('load', StepStatus.Ready);
    expect(comp.showRetryForSection('universe')).toBe(false);
    comp.activeStepId.set(null);
    expect(comp.showRetryForSection('universe')).toBe(false);
    http.verify();
  });

  it('onContinue dispatches the next step when current is completed and a session exists', () => {
    const { comp, store, http } = setup();
    store.setSessionId('sid-1');
    store.setStepStatus('load', StepStatus.Completed);
    comp.onContinue('load');
    http.expectOne(STEP_URL('sid-1', 'screen')).flush(acceptResponse(), ACCEPTED);
    http.verify();
  });

  it('onContinue is a no-op when the current step is not completed', () => {
    const { comp, store, http } = setup();
    store.setSessionId('sid-1');
    comp.onContinue('load');
    http.expectNone(() => true);
    http.verify();
  });

  it('onRetry resets the step to Ready then re-dispatches it', () => {
    const { comp, store, http } = setup();
    store.setSessionId('sid-1');
    comp.onRetry('load');
    http.expectOne(STEP_URL('sid-1', 'load')).flush(acceptResponse(), ACCEPTED);
    expect(store.stepStatuses().get('load')).toBe(StepStatus.Running);
    http.verify();
  });

  it('onPanelRunStep skips when the step is already running', () => {
    const { comp, store, http } = setup();
    store.setSessionId('sid-1');
    store.setStepStatus('load', StepStatus.Running);
    comp.onPanelRunStep('load', {});
    http.expectNone(() => true);
    http.verify();
  });

  it('dispatchStep is a no-op without a session', () => {
    const { comp, http } = setup();
    comp.onPanelRunStep('load', {});
    http.expectNone(() => true);
    http.verify();
  });

  it('a dispatch failure records the error', () => {
    const { comp, store, http } = setup();
    store.setSessionId('sid-1');
    comp.onPanelRunStep('load', {});
    http.expectOne(STEP_URL('sid-1', 'load')).flush({ detail: 'x' }, { status: 500, statusText: 'Server Error' });
    expect(comp.lastError()).toContain('HTTP 500');
    http.verify();
  });

  it('a completed poll stores the result and marks the step completed', () => {
    const { comp, store, http, tick } = setup();
    store.setSessionId('sid-1');
    comp.onPanelRunStep('load', {});
    http.expectOne(STEP_URL('sid-1', 'load')).flush(acceptResponse(), ACCEPTED);
    tick.next(0);
    http.expectOne(STEP_URL('sid-1', 'load')).flush(poll('completed', { result: { ok: true } }));
    expect(store.stepStatuses().get('load')).toBe(StepStatus.Completed);
    expect(store.stepResults().get('load')).toEqual({ ok: true });
    http.verify();
  });

  it('a failed poll with a gate reason sets abortGateError', () => {
    const { comp, store, http, tick } = setup();
    store.setSessionId('sid-1');
    comp.onPanelRunStep('load', {});
    http.expectOne(STEP_URL('sid-1', 'load')).flush(acceptResponse(), ACCEPTED);
    tick.next(0);
    http.expectOne(STEP_URL('sid-1', 'load')).flush(poll('failed', { gateReason: 'too few factors' }));
    expect(comp.abortGateError()).toBe('too few factors');
    http.verify();
  });

  it('a failed poll without a gate reason sets lastError', () => {
    const { comp, store, http, tick } = setup();
    store.setSessionId('sid-1');
    comp.onPanelRunStep('load', {});
    http.expectOne(STEP_URL('sid-1', 'load')).flush(acceptResponse(), ACCEPTED);
    tick.next(0);
    http.expectOne(STEP_URL('sid-1', 'load')).flush(poll('failed', { error: 'step boom' }));
    expect(comp.lastError()).toBe('step boom');
    http.verify();
  });

  it('a poll error is swallowed and polling does not crash', () => {
    const { comp, store, http, tick } = setup();
    store.setSessionId('sid-1');
    comp.onPanelRunStep('load', {});
    http.expectOne(STEP_URL('sid-1', 'load')).flush(acceptResponse(), ACCEPTED);
    tick.next(0);
    http.expectOne(STEP_URL('sid-1', 'load')).flush({ detail: 'x' }, { status: 500, statusText: 'Server Error' });
    expect(comp.lastError()).toBeNull(); // catchError → of(null), no failure recorded
    http.verify();
  });

  it('onConfigSubmit on an existing session resets the store before creating a new one', () => {
    const { comp, store, http } = setup();
    store.setSessionId('old');
    store.setStepStatus('load', StepStatus.Completed);
    comp.onConfigSubmit(defaultSubmit());
    expect(store.stepStatuses().size).toBe(0); // reset() wiped per-step state
    http.expectOne(SESSIONS_URL).flush({ sessionId: 'new' });
    expect(store.sessionId()).toBe('new');
    http.verify();
  });

  it('onConfigSubmit records an error when session creation fails', () => {
    const { comp, http } = setup();
    comp.onConfigSubmit(defaultSubmit());
    http.expectOne(SESSIONS_URL).flush({ detail: 'x' }, { status: 500, statusText: 'Server Error' });
    expect(comp.lastError()).toContain('HTTP 500');
    http.verify();
  });

  it('isExpanded reflects the expanded section id', () => {
    const { comp, http } = setup();
    expect(comp.isExpanded('universe')).toBe(false);
    comp.toggleSection('universe');
    expect(comp.isExpanded('universe')).toBe(true);
    comp.toggleSection('universe');
    expect(comp.isExpanded('universe')).toBe(false);
    http.verify();
  });
});
