import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { Subject } from 'rxjs';

import { PipelineStepperComponent } from './pipeline-stepper';
import { environment } from '../../../environments/environment';
import { JOB_POLL_TICK } from '../../shared/job-progress-tracker/job-progress-tracker';
import {
  PIPELINE_STEPS,
  StepStatus,
} from '../../models/pipeline-builder.model';
import {
  STEP_PARAM_DEFAULTS,
  type PipelineConfigSubmit,
} from './step-params.model';

const SESSIONS_URL = `${environment.apiUrl}pipeline-builder/sessions`;
const STEPS_URL = (sid: string, step: string) =>
  `${SESSIONS_URL}/${sid}/steps/${step}`;

function defaultSubmit(): PipelineConfigSubmit {
  return {
    config: {
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
    },
    steps: structuredClone(STEP_PARAM_DEFAULTS),
  };
}

function accept() {
  return { job_id: 'j-1', status: 'pending', message: '' };
}
const ACCEPTED = { status: 202, statusText: 'Accepted' };

describe('PipelineStepperComponent', () => {
  let http: HttpTestingController;
  let tick: Subject<number>;

  beforeEach(() => {
    tick = new Subject<number>();
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        { provide: JOB_POLL_TICK, useValue: tick.asObservable() },
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function createComponent(): PipelineStepperComponent {
    return TestBed.createComponent(PipelineStepperComponent).componentInstance;
  }

  // Session created + step 1 auto-dispatched and Running (poller armed).
  function startedSession(): PipelineStepperComponent {
    const c = createComponent();
    c.onConfigSubmit(defaultSubmit());
    http.expectOne(SESSIONS_URL).flush({ sessionId: 'sid-1' });
    http.expectOne(STEPS_URL('sid-1', 'load')).flush(accept(), ACCEPTED);
    return c;
  }

  function completeActive(c: PipelineStepperComponent, step: string): void {
    tick.next(0);
    http.expectOne(STEPS_URL('sid-1', step)).flush({
      status: 'completed',
      progress: {},
      result: { ok: true },
      error: null,
      gateReason: null,
    });
  }

  describe('initial render', () => {
    it('phase is "config" and sessionId is null', () => {
      const c = createComponent();
      expect(c.phase()).toBe('config');
      expect(c.sessionId()).toBeNull();
    });
  });

  describe('onConfigSubmit()', () => {
    it('stores stepParams, POSTs /sessions, then auto-dispatches load', () => {
      const c = createComponent();
      c.onConfigSubmit(defaultSubmit());

      const s = http.expectOne(SESSIONS_URL);
      expect(s.request.method).toBe('POST');
      s.flush({ sessionId: 'sid-1' }, { status: 201, statusText: 'Created' });

      expect(c.stepParams()).toEqual(STEP_PARAM_DEFAULTS);
      expect(c.phase()).toBe('running');
      expect(c.activeStepId()).toBe('load');

      const run = http.expectOne(STEPS_URL('sid-1', 'load'));
      expect(run.request.method).toBe('POST');
      expect(run.request.body).toEqual({
        include_delisted: true,
        macro_country: 'USA',
      });
      run.flush(accept(), ACCEPTED);

      expect(c.stepStatuses().get('load')).toBe(StepStatus.Running);
      expect(c.expandedStepId()).toBe('load');
    });

    it('when session creation 429s, no load dispatch and phase stays config', () => {
      const c = createComponent();
      c.onConfigSubmit(defaultSubmit());

      http
        .expectOne(SESSIONS_URL)
        .flush(
          { detail: 'max concurrent sessions reached' },
          { status: 429, statusText: 'Too Many Requests' },
        );

      expect(c.sessionError()).not.toBeNull();
      expect(c.phase()).toBe('config');
      http.expectNone(STEPS_URL('sid-1', 'load'));
    });

    it('on session create, load is Ready and the rest Locked (before dispatch)', () => {
      const c = createComponent();
      c.onConfigSubmit(defaultSubmit());
      http.expectOne(SESSIONS_URL).flush({ sessionId: 'sid-1' });
      // load POST in flight — status set Running on its response
      const run = http.expectOne(STEPS_URL('sid-1', 'load'));
      for (const step of PIPELINE_STEPS) {
        if (step.id === 'load') continue;
        expect(c.stepStatuses().get(step.id)).toBe(StepStatus.Locked);
      }
      run.flush(accept(), ACCEPTED);
    });
  });

  describe('onAbort()', () => {
    it('DELETEs the session, resets state, tears down polling', () => {
      const c = startedSession();
      c.onAbort();
      const del = http.expectOne(`${SESSIONS_URL}/sid-1`);
      expect(del.request.method).toBe('DELETE');
      del.flush(null, { status: 204, statusText: 'No Content' });

      expect(c.sessionId()).toBeNull();
      expect(c.phase()).toBe('config');
      expect(c.activeStepId()).toBeNull();
      expect(c.expandedStepId()).toBeNull();
      expect(c.stepParams()).toBeNull();
      expect(c.stepStatuses().size).toBe(0);

      tick.next(1);
      expect(http.match(STEPS_URL('sid-1', 'load')).length).toBe(0);
    });

    it('with no session, no DELETE issued', () => {
      const c = createComponent();
      c.onAbort();
      http.expectNone(`${SESSIONS_URL}/`);
      expect(c.phase()).toBe('config');
    });
  });

  describe('polling + auto-advance', () => {
    it('on completed, step Completed, N+1 Ready, canContinue true, NO auto-dispatch', () => {
      const c = startedSession();
      completeActive(c, 'load');

      expect(c.stepStatuses().get('load')).toBe(StepStatus.Completed);
      expect(c.stepStatuses().get('screen')).toBe(StepStatus.Ready);
      expect(c.canContinue()).toBe(true);
      http.expectNone(STEPS_URL('sid-1', 'screen'));
    });

    it('after terminal response the next tick issues NO GET', () => {
      const c = startedSession();
      completeActive(c, 'load');
      tick.next(1);
      expect(http.match(STEPS_URL('sid-1', 'load')).length).toBe(0);
    });

    it('on plain failure, step Error, lastError set, abortGateError null', () => {
      const c = startedSession();
      tick.next(0);
      http.expectOne(STEPS_URL('sid-1', 'load')).flush({
        status: 'failed',
        progress: {},
        result: null,
        error: 'load failed',
        gateReason: null,
      });

      expect(c.stepStatuses().get('load')).toBe(StepStatus.Error);
      expect(c.lastError()).toBe('load failed');
      expect(c.abortGateError()).toBeNull();
      expect(c.showRetryFor('load')).toBe(true);
    });
  });

  describe('onContinue()', () => {
    it('dispatches the next step with its upfront params', () => {
      const c = startedSession();
      completeActive(c, 'load');

      c.onContinue();

      expect(c.activeStepId()).toBe('screen');
      expect(c.expandedStepId()).toBe('screen');
      const run = http.expectOne(STEPS_URL('sid-1', 'screen'));
      expect(run.request.body).toEqual({ preset: 'developed_markets' });
      run.flush(accept(), ACCEPTED);
      expect(c.stepStatuses().get('screen')).toBe(StepStatus.Running);
    });

    it('is a no-op when the active step is not Completed', () => {
      const c = startedSession(); // load Running
      c.onContinue();
      http.expectNone(STEPS_URL('sid-1', 'screen'));
      expect(c.activeStepId()).toBe('load');
    });
  });

  describe('onRetry()', () => {
    it('re-dispatches the failed active step', () => {
      const c = startedSession();
      tick.next(0);
      http.expectOne(STEPS_URL('sid-1', 'load')).flush({
        status: 'failed',
        progress: {},
        result: null,
        error: 'boom',
        gateReason: null,
      });
      expect(c.stepStatuses().get('load')).toBe(StepStatus.Error);

      c.onRetry();
      expect(c.lastError()).toBeNull();
      const run = http.expectOne(STEPS_URL('sid-1', 'load'));
      run.flush(accept(), ACCEPTED);
      expect(c.stepStatuses().get('load')).toBe(StepStatus.Running);
    });
  });

  describe('toggleExpand()', () => {
    it('issues no HTTP and leaves the pipeline cursor unchanged', () => {
      const c = startedSession();
      completeActive(c, 'load');
      c.onContinue();
      http.expectOne(STEPS_URL('sid-1', 'screen')).flush(accept(), ACCEPTED);
      expect(c.activeStepId()).toBe('screen');

      c.toggleExpand('load'); // view the done step
      expect(c.expandedStepId()).toBe('load');
      expect(c.activeStepId()).toBe('screen');
      http.expectNone(STEPS_URL('sid-1', 'load'));

      c.toggleExpand('load'); // collapse again
      expect(c.expandedStepId()).toBeNull();
    });

    it('ignores Locked / Pending steps', () => {
      const c = startedSession();
      c.toggleExpand('optimize'); // still Locked
      expect(c.expandedStepId()).toBe('load');
    });
  });

  describe('coverage_gate abort', () => {
    it('gate failure sets abortGateError and keeps downstream Locked', () => {
      const c = startedSession();
      // Jump the cursor to a Ready coverage_gate and dispatch it.
      const map = new Map(c.stepStatuses());
      map.set('coverage_gate', StepStatus.Ready);
      c.stepStatuses.set(map);
      c.activeStepId.set('coverage_gate');
      c.onPanelRunStep({});
      http
        .expectOne(STEPS_URL('sid-1', 'coverage_gate'))
        .flush(accept(), ACCEPTED);

      tick.next(0);
      http.expectOne(STEPS_URL('sid-1', 'coverage_gate')).flush({
        status: 'failed',
        progress: {},
        result: null,
        error: null,
        gateReason: 'Only 1 factor passed IS ∩ OOS (min_factors=2)',
      });

      expect(c.stepStatuses().get('coverage_gate')).toBe(StepStatus.Error);
      expect(c.abortGateError()).toBe(
        'Only 1 factor passed IS ∩ OOS (min_factors=2)',
      );
      expect(c.lastError()).toBeNull();
      for (const downstream of [
        'regime',
        'optimize',
        'rebalance_decision',
        'cost',
        'report',
        'persist',
      ] as const) {
        expect(c.stepStatuses().get(downstream)).toBe(StepStatus.Locked);
      }
    });
  });

  describe('completedCount', () => {
    it('counts Completed steps', () => {
      const c = startedSession();
      expect(c.completedCount()).toBe(0);
      completeActive(c, 'load');
      expect(c.completedCount()).toBe(1);
    });
  });
});
