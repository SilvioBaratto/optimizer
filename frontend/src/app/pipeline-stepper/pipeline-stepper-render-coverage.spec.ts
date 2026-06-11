/**
 * pipeline-stepper-render-coverage.spec.ts — issue #943
 *
 * First spec for the orchestrator shell. The shell describe drives the
 * phase/abort/expand/switch-case template arms with JOB_POLL_TICK = NEVER; the
 * workflow describe drives session creation, dispatch and the poll lifecycle
 * with a controllable tick Subject.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';
import { NEVER, Subject } from 'rxjs';

import { configureTestBed, injectHttp, makeStepPollResponse } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { PipelineStepperComponent } from './pipeline-stepper';
import { JOB_POLL_TICK } from '../shared/job-progress-tracker/job-progress-tracker';
import { StepStatus } from '../core/models/pipeline-builder.model';
import { environment } from '../../environments/environment';
import { STEP_PARAM_DEFAULTS, type PipelineConfigSubmit } from './models/step-params.model';

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

describe('PipelineStepperComponent — shell render-coverage (#943)', () => {
  let fixture: ComponentFixture<PipelineStepperComponent>;
  let comp: PipelineStepperComponent;
  let el: HTMLElement;

  beforeEach(async () => {
    await configureTestBed({
      imports: [PipelineStepperComponent],
      withHttp: true,
      withRouter: true,
      providers: [ICON_PROVIDER, { provide: JOB_POLL_TICK, useValue: NEVER }],
    });
    fixture = TestBed.createComponent(PipelineStepperComponent);
    comp = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
  });

  function button(text: string): HTMLButtonElement | undefined {
    return Array.from(el.querySelectorAll('button')).find((b) => b.textContent?.includes(text)) as
      | HTMLButtonElement
      | undefined;
  }

  it('in the config phase, the run-config panel is shown and the stepper is absent', () => {
    expect(el.querySelector('app-run-config-panel')).not.toBeNull();
    expect(el.querySelector('app-step-section')).toBeNull();
    expect(button('Abort')).toBeUndefined();
  });

  it('in the running phase, the stepper, completed count and abort button render', () => {
    comp.phase.set('running');
    comp.stepStatuses.set(new Map([['load', StepStatus.Completed]]));
    fixture.detectChanges();
    expect(el.querySelectorAll('app-step-section').length).toBe(13);
    expect(el.textContent).toContain('/ 13 completed');
    expect(button('Abort')).not.toBeUndefined();
    expect(comp.completedCount()).toBe(1);
  });

  it('a session error renders an alert', () => {
    comp.sessionError.set('session boom');
    fixture.detectChanges();
    expect(el.querySelector('[role="alert"]')?.textContent).toContain('session boom');
  });

  it('a gate error renders the halted-at-coverage-gate alert', () => {
    comp.abortGateError.set('not enough factors');
    fixture.detectChanges();
    expect(el.textContent).toContain('Pipeline halted at Coverage Gate');
    expect(el.textContent).toContain('not enough factors');
  });

  it('each expanded step renders its matching @switch panel', () => {
    comp.phase.set('running');
    const cases: Array<{ id: string; selector: string }> = [
      { id: 'load', selector: 'app-step-load-panel' },
      { id: 'screen', selector: 'app-step-screen-panel' },
      { id: 'clean_returns', selector: 'app-step-clean-returns-panel' },
      { id: 'build_history', selector: 'app-step-build-history-panel' },
      { id: 'validate_is', selector: 'app-step-validate-is-panel' },
      { id: 'validate_oos', selector: 'app-step-validate-oos-panel' },
      { id: 'coverage_gate', selector: 'app-step-coverage-gate-panel' },
      { id: 'regime', selector: 'app-step-regime-panel' },
      { id: 'optimize', selector: 'app-step-optimize-panel' },
      { id: 'rebalance_decision', selector: 'app-step-rebalance-decision-panel' },
      { id: 'cost', selector: 'app-step-cost-panel' },
      { id: 'report', selector: 'app-step-report-panel' },
      { id: 'persist', selector: 'app-step-persist-panel' },
    ];
    for (const c of cases) {
      comp.expandedStepId.set(c.id as never);
      fixture.detectChanges();
      expect(el.querySelector(c.selector)).withContext(c.id).not.toBeNull();
    }
  });

  it('toggleExpand ignores locked/pending steps but toggles ready ones', () => {
    comp.stepStatuses.set(new Map([['load', StepStatus.Locked], ['screen', StepStatus.Ready]]));
    comp.toggleExpand('load');
    expect(comp.isExpanded('load')).toBe(false);
    comp.toggleExpand('screen');
    expect(comp.isExpanded('screen')).toBe(true);
    comp.toggleExpand('screen');
    expect(comp.isExpanded('screen')).toBe(false);
  });

  it('showContinueFor and showRetryFor reflect the active step status', () => {
    comp.activeStepId.set('load');
    comp.stepStatuses.set(new Map([['load', StepStatus.Completed]]));
    expect(comp.showContinueFor('load')).toBe(true);
    expect(comp.showContinueFor('screen')).toBe(false);
    comp.stepStatuses.set(new Map([['load', StepStatus.Error]]));
    expect(comp.showRetryFor('load')).toBe(true);
    expect(comp.showRetryFor('screen')).toBe(false);
  });

  it('onAbort reverts to the config phase', () => {
    comp.phase.set('running');
    comp.onAbort();
    expect(comp.phase()).toBe('config');
  });

  it('errorFor returns the active-step error and the last step cannot continue', () => {
    comp.phase.set('running');
    comp.activeStepId.set('load');
    comp.expandedStepId.set('load');
    comp.stepStatuses.set(new Map([['load', StepStatus.Ready]]));
    comp.lastError.set('oops');
    fixture.detectChanges();
    expect(comp.errorFor('load')).toBe('oops'); // id === activeStepId → lastError()

    comp.activeStepId.set('persist');
    comp.stepStatuses.set(new Map([['persist', StepStatus.Completed]]));
    expect(comp.showContinueFor('persist')).toBe(false); // nextStepId('persist') === null
  });
});

describe('PipelineStepperComponent — workflow coverage (#943)', () => {
  let fixture: ComponentFixture<PipelineStepperComponent>;
  let comp: PipelineStepperComponent;
  let http: HttpTestingController;
  let tick: Subject<number>;

  async function setup(): Promise<void> {
    tick = new Subject<number>();
    await configureTestBed({
      imports: [PipelineStepperComponent],
      withHttp: true,
      withRouter: true,
      providers: [ICON_PROVIDER, { provide: JOB_POLL_TICK, useValue: tick.asObservable() }],
    });
    fixture = TestBed.createComponent(PipelineStepperComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
    fixture.detectChanges();
  }

  function createSessionAndDispatchLoad(): void {
    comp.onConfigSubmit(defaultSubmit());
    http.expectOne(SESSIONS_URL).flush({ sessionId: 'sid-1' });
    http.expectOne(STEP_URL('sid-1', 'load')).flush(acceptResponse(), ACCEPTED);
  }

  it('onConfigSubmit creates a session, enters running and dispatches the first step', async () => {
    await setup();
    createSessionAndDispatchLoad();
    expect(comp.phase()).toBe('running');
    expect(comp.stepStatuses().get('load')).toBe(StepStatus.Running);
    http.verify();
  });

  it('a session-creation failure surfaces a session error', async () => {
    await setup();
    comp.onConfigSubmit(defaultSubmit());
    http.expectOne(SESSIONS_URL).flush({}, { status: 500, statusText: 'Server Error' });
    expect(comp.sessionError()).toContain('HTTP 500');
    http.verify();
  });

  it('a 429 session failure surfaces the capacity detail message', async () => {
    await setup();
    comp.onConfigSubmit(defaultSubmit());
    http.expectOne(SESSIONS_URL).flush({ detail: 'capacity reached' }, { status: 429, statusText: 'Too Many Requests' });
    expect(comp.sessionError()).toBe('capacity reached');
    http.verify();
  });

  it('a completed poll advances the next step to ready', async () => {
    await setup();
    createSessionAndDispatchLoad();
    tick.next(0);
    http.expectOne(STEP_URL('sid-1', 'load')).flush(makeStepPollResponse({ status: 'completed', result: { ok: true } }));
    expect(comp.stepStatuses().get('load')).toBe(StepStatus.Completed);
    expect(comp.stepStatuses().get('screen')).toBe(StepStatus.Ready);
    http.verify();
  });

  it('a failed poll with a gate reason sets the gate error', async () => {
    await setup();
    createSessionAndDispatchLoad();
    tick.next(0);
    http.expectOne(STEP_URL('sid-1', 'load')).flush(makeStepPollResponse({ status: 'failed', gateReason: 'too few factors' }));
    expect(comp.abortGateError()).toBe('too few factors');
    http.verify();
  });

  it('a failed poll without a gate reason sets the last error', async () => {
    await setup();
    createSessionAndDispatchLoad();
    tick.next(0);
    http.expectOne(STEP_URL('sid-1', 'load')).flush(makeStepPollResponse({ status: 'failed', gateReason: null, error: 'step boom' }));
    expect(comp.lastError()).toBe('step boom');
    http.verify();
  });

  it('a poll error is swallowed and does not record a failure', async () => {
    await setup();
    createSessionAndDispatchLoad();
    tick.next(0);
    http.expectOne(STEP_URL('sid-1', 'load')).flush({ detail: 'x' }, { status: 500, statusText: 'Server Error' });
    expect(comp.lastError()).toBeNull();
    http.verify();
  });

  it('onContinue dispatches the next step once the current is completed', async () => {
    await setup();
    createSessionAndDispatchLoad();
    comp.stepStatuses.update((m) => new Map(m).set('load', StepStatus.Completed));
    comp.onContinue();
    http.expectOne(STEP_URL('sid-1', 'screen')).flush(acceptResponse(), ACCEPTED);
    expect(comp.activeStepId()).toBe('screen');
    http.verify();
  });

  it('onRetry re-dispatches the active step', async () => {
    await setup();
    createSessionAndDispatchLoad();
    comp.onRetry();
    http.expectOne(STEP_URL('sid-1', 'load')).flush(acceptResponse(), ACCEPTED);
    expect(comp.stepStatuses().get('load')).toBe(StepStatus.Running);
    http.verify();
  });

  it('onPanelRunStep is a no-op while the active step is already running', async () => {
    await setup();
    createSessionAndDispatchLoad();
    comp.onPanelRunStep({}); // load is Running
    http.expectNone(STEP_URL('sid-1', 'load'));
    http.verify();
  });

  it('a 409 dispatch failure reports the already-running message', async () => {
    await setup();
    comp.onConfigSubmit(defaultSubmit());
    http.expectOne(SESSIONS_URL).flush({ sessionId: 'sid-1' });
    http.expectOne(STEP_URL('sid-1', 'load')).flush({ detail: 'x' }, { status: 409, statusText: 'Conflict' });
    expect(comp.sessionError()).toContain('already running');
    http.verify();
  });

  it('onAbort deletes the session and resets to config', async () => {
    await setup();
    createSessionAndDispatchLoad();
    comp.onAbort();
    http.expectOne(`${SESSIONS_URL}/sid-1`).flush({});
    expect(comp.phase()).toBe('config');
    expect(comp.sessionId()).toBeNull();
    http.verify();
  });

  it('a running (non-terminal) poll keeps polling until a terminal poll arrives', async () => {
    await setup();
    createSessionAndDispatchLoad();
    tick.next(0);
    http.expectOne(STEP_URL('sid-1', 'load')).flush(makeStepPollResponse({ status: 'running' }));
    expect(comp.stepStatuses().get('load')).toBe(StepStatus.Running);
    tick.next(1);
    http.expectOne(STEP_URL('sid-1', 'load')).flush(makeStepPollResponse({ status: 'completed' }));
    expect(comp.stepStatuses().get('load')).toBe(StepStatus.Completed);
    http.verify();
  });

  it('a failed poll with no error message uses the default text', async () => {
    await setup();
    createSessionAndDispatchLoad();
    tick.next(0);
    http.expectOne(STEP_URL('sid-1', 'load')).flush(makeStepPollResponse({ status: 'failed', gateReason: null, error: null }));
    expect(comp.lastError()).toBe('Step failed');
    http.verify();
  });

  it('a 429 with no detail body uses the default capacity message', async () => {
    await setup();
    comp.onConfigSubmit(defaultSubmit());
    http.expectOne(SESSIONS_URL).flush(null, { status: 429, statusText: 'Too Many Requests' });
    expect(comp.sessionError()).toContain('capacity reached');
    http.verify();
  });
});
