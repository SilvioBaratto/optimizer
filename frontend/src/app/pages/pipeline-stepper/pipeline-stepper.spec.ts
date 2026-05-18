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
  type RunLevelConfig,
} from '../../models/pipeline-builder.model';

const SESSIONS_URL = `${environment.apiUrl}pipeline-builder/sessions`;
const STEPS_URL = (sid: string, step: string) =>
  `${SESSIONS_URL}/${sid}/steps/${step}`;

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

  function startedSession(): PipelineStepperComponent {
    const c = createComponent();
    c.onConfigSubmit(defaultConfig());
    http.expectOne(SESSIONS_URL).flush({ sessionId: 'sid-1' });
    return c;
  }

  describe('initial render', () => {
    it('when constructed, phase is "config" and sessionId is null', () => {
      const c = createComponent();
      expect(c.phase()).toBe('config');
      expect(c.sessionId()).toBeNull();
      expect(c.sessionError()).toBeNull();
    });
  });

  describe('onConfigSubmit()', () => {
    it('when called with a valid config, POSTs /sessions and transitions to running phase', () => {
      const c = createComponent();
      c.onConfigSubmit(defaultConfig());

      const req = http.expectOne(SESSIONS_URL);
      expect(req.request.method).toBe('POST');
      req.flush({ sessionId: 'sid-1' }, { status: 201, statusText: 'Created' });

      expect(c.sessionId()).toBe('sid-1');
      expect(c.phase()).toBe('running');
      expect(c.activeStepId()).toBe('load');
    });

    it('when session is created, "load" status is Ready and others are Locked', () => {
      const c = startedSession();
      const map = c.stepStatuses();
      expect(map.get('load')).toBe(StepStatus.Ready);
      for (const step of PIPELINE_STEPS) {
        if (step.id === 'load') continue;
        expect(map.get(step.id)).toBe(StepStatus.Locked);
      }
    });

    it('when the backend returns 429, sessionError is set and phase stays "config"', () => {
      const c = createComponent();
      c.onConfigSubmit(defaultConfig());

      http
        .expectOne(SESSIONS_URL)
        .flush(
          { detail: 'max concurrent sessions reached' },
          { status: 429, statusText: 'Too Many Requests' },
        );

      expect(c.sessionError()).not.toBeNull();
      expect(c.phase()).toBe('config');
      expect(c.sessionId()).toBeNull();
    });
  });

  describe('onAbort()', () => {
    it('when called with an active session, DELETEs /sessions/{id} and resets state', () => {
      const c = startedSession();

      c.onAbort();
      const del = http.expectOne(`${SESSIONS_URL}/sid-1`);
      expect(del.request.method).toBe('DELETE');
      del.flush(null, { status: 204, statusText: 'No Content' });

      expect(c.sessionId()).toBeNull();
      expect(c.phase()).toBe('config');
      expect(c.activeStepId()).toBeNull();
      expect(c.stepStatuses().size).toBe(0);
    });

    it('when called with no active session, no DELETE is issued and phase remains "config"', () => {
      const c = createComponent();
      c.onAbort();
      http.expectNone(`${SESSIONS_URL}/`);
      expect(c.phase()).toBe('config');
    });
  });

  describe('sidebar interactions', () => {
    it('when a Ready step is clicked, activeStepId updates', () => {
      const c = startedSession();
      c.selectStep('load');
      expect(c.activeStepId()).toBe('load');
    });

    it('when a Locked step is clicked, activeStepId is NOT changed', () => {
      const c = startedSession();
      c.selectStep('optimize');
      expect(c.activeStepId()).toBe('load');
    });
  });

  describe('runActiveStep()', () => {
    it('when active step is Ready, POSTs /sessions/{id}/steps/{stepId} and sets Running', () => {
      const c = startedSession();
      c.runActiveStep();

      const req = http.expectOne(STEPS_URL('sid-1', 'load'));
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({});
      req.flush(
        { job_id: 'j-1', status: 'pending', message: '' },
        { status: 202, statusText: 'Accepted' },
      );

      expect(c.stepStatuses().get('load')).toBe(StepStatus.Running);
    });

    it('when active step is Running, runActiveStep does NOT issue another POST', () => {
      const c = startedSession();
      c.runActiveStep();
      http
        .expectOne(STEPS_URL('sid-1', 'load'))
        .flush(
          { job_id: 'j-1', status: 'pending', message: '' },
          { status: 202, statusText: 'Accepted' },
        );

      // step is now Running — second call must be a no-op
      c.runActiveStep();
      expect(http.match(STEPS_URL('sid-1', 'load')).length).toBe(0);
    });

    it('when runStep returns 409, sessionError is set and status stays Ready', () => {
      const c = startedSession();
      c.runActiveStep();

      http
        .expectOne(STEPS_URL('sid-1', 'load'))
        .flush(
          { detail: 'job already running' },
          { status: 409, statusText: 'Conflict' },
        );

      expect(c.sessionError()).toContain('409');
      expect(c.stepStatuses().get('load')).toBe(StepStatus.Ready);
    });
  });

  describe('polling lifecycle', () => {
    function dispatchLoad(c: PipelineStepperComponent): void {
      c.runActiveStep();
      http
        .expectOne(STEPS_URL('sid-1', 'load'))
        .flush(
          { job_id: 'j-1', status: 'pending', message: '' },
          { status: 202, statusText: 'Accepted' },
        );
    }

    it('when a tick fires, GETs the active step and updates status to Running', () => {
      const c = startedSession();
      dispatchLoad(c);

      tick.next(0);
      http.expectOne(STEPS_URL('sid-1', 'load')).flush({
        status: 'running',
        progress: { current: 1, total: 10 },
        result: null,
        error: null,
        gateReason: null,
      });

      expect(c.stepStatuses().get('load')).toBe(StepStatus.Running);
    });

    it('when poll reports completed, step is Completed and N+1 unlocks to Ready', () => {
      const c = startedSession();
      dispatchLoad(c);

      tick.next(0);
      http.expectOne(STEPS_URL('sid-1', 'load')).flush({
        status: 'completed',
        progress: { current: 10, total: 10 },
        result: { ok: true },
        error: null,
        gateReason: null,
      });

      expect(c.stepStatuses().get('load')).toBe(StepStatus.Completed);
      expect(c.stepStatuses().get('screen')).toBe(StepStatus.Ready);
      expect(c.nextEnabled()).toBe(true);
    });

    it('when poll reports failed, step is Error and remaining steps stay Locked', () => {
      const c = startedSession();
      dispatchLoad(c);

      tick.next(0);
      http.expectOne(STEPS_URL('sid-1', 'load')).flush({
        status: 'failed',
        progress: {},
        result: null,
        error: 'load failed',
        gateReason: null,
      });

      expect(c.stepStatuses().get('load')).toBe(StepStatus.Error);
      expect(c.stepStatuses().get('screen')).toBe(StepStatus.Locked);
      expect(c.lastError()).toBe('load failed');
    });

    it('after terminal response, the next tick issues NO additional GET', () => {
      const c = startedSession();
      dispatchLoad(c);

      tick.next(0);
      http.expectOne(STEPS_URL('sid-1', 'load')).flush({
        status: 'completed',
        progress: {},
        result: null,
        error: null,
        gateReason: null,
      });

      tick.next(1);
      expect(http.match(STEPS_URL('sid-1', 'load')).length).toBe(0);
    });

    it('when onAbort is called mid-poll, the polling subscription is torn down', () => {
      const c = startedSession();
      dispatchLoad(c);
      tick.next(0);
      http.expectOne(STEPS_URL('sid-1', 'load')).flush({
        status: 'running',
        progress: {},
        result: null,
        error: null,
        gateReason: null,
      });

      c.onAbort();
      http.expectOne(`${SESSIONS_URL}/sid-1`).flush(null, { status: 204, statusText: 'No Content' });

      tick.next(1);
      expect(http.match(STEPS_URL('sid-1', 'load')).length).toBe(0);
    });
  });

  describe('coverage_gate abort', () => {
    function reachCoverageGate(c: PipelineStepperComponent): void {
      // Walk the stepper to coverage_gate as the active step. We don't run
      // intervening steps — we just set the gate step Ready manually via the
      // public Map signal, then make it active.
      const map = new Map(c.stepStatuses());
      map.set('coverage_gate', StepStatus.Ready);
      c.stepStatuses.set(map);
      c.selectStep('coverage_gate');
      c.runActiveStep();
      http
        .expectOne(STEPS_URL('sid-1', 'coverage_gate'))
        .flush(
          { job_id: 'j-gate', status: 'pending', message: '' },
          { status: 202, statusText: 'Accepted' },
        );
    }

    it('when coverage_gate fails with gateReason, abortGateError is set and downstream stays Locked', () => {
      const c = startedSession();
      reachCoverageGate(c);

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
      expect(c.lastError()).toBeNull();
    });

    it('when a non-gate step fails with plain error, abortGateError stays null', () => {
      const c = startedSession();
      c.runActiveStep();
      http
        .expectOne(STEPS_URL('sid-1', 'load'))
        .flush(
          { job_id: 'j-1', status: 'pending', message: '' },
          { status: 202, statusText: 'Accepted' },
        );

      tick.next(0);
      http.expectOne(STEPS_URL('sid-1', 'load')).flush({
        status: 'failed',
        progress: {},
        result: null,
        error: 'load failed',
        gateReason: null,
      });

      expect(c.stepStatuses().get('load')).toBe(StepStatus.Error);
      expect(c.abortGateError()).toBeNull();
      expect(c.lastError()).toBe('load failed');
    });

    it('onAbort clears abortGateError', () => {
      const c = startedSession();
      reachCoverageGate(c);

      tick.next(0);
      http.expectOne(STEPS_URL('sid-1', 'coverage_gate')).flush({
        status: 'failed',
        progress: {},
        result: null,
        error: null,
        gateReason: 'gate reason text',
      });
      expect(c.abortGateError()).not.toBeNull();

      c.onAbort();
      http
        .expectOne(`${SESSIONS_URL}/sid-1`)
        .flush(null, { status: 204, statusText: 'No Content' });
      expect(c.abortGateError()).toBeNull();
    });
  });

  describe('computed signals', () => {
    it('when no steps are completed, completedCount is 0', () => {
      const c = startedSession();
      expect(c.completedCount()).toBe(0);
    });

    it('nextEnabled is false when the active step has not completed', () => {
      const c = startedSession();
      expect(c.nextEnabled()).toBe(false);
    });
  });
});
