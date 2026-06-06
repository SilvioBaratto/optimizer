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
import { StepStatus, type PipelineStepId } from '../../models/pipeline-builder.model';

const SESSIONS_URL = `${environment.apiUrl}pipeline-builder/sessions`;
const STEP_URL = (sid: string, step: string) =>
  `${SESSIONS_URL}/${sid}/steps/${step}`;
const ACCEPTED = { status: 202, statusText: 'Accepted' };

function acceptResponse() {
  return { job_id: 'j-1', status: 'pending', message: '' };
}

function defaultSubmit(): PipelineConfigSubmit {
  return {
    config: {
      rebalance_freq: 21,
      n_selected: 25,
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

function setup(): {
  fixture: ComponentFixture<InputsPaneComponent>;
  host: HTMLElement;
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
  const fixture = TestBed.createComponent(InputsPaneComponent);
  fixture.detectChanges();
  return { fixture, host: fixture.nativeElement as HTMLElement, http, tick };
}

function sectionButtons(host: HTMLElement): HTMLButtonElement[] {
  return Array.from(
    host.querySelectorAll<HTMLButtonElement>(
      'app-step-section > section > button',
    ),
  );
}

// ACCORDION_SECTIONS index → button.
const SECTION_INDEX: Record<string, number> = {
  universe: 0,
  objective: 1,
  constraints: 2,
  moments: 3,
  horizon: 4,
};

function expand(
  fixture: ComponentFixture<InputsPaneComponent>,
  host: HTMLElement,
  section: keyof typeof SECTION_INDEX,
): void {
  sectionButtons(host)[SECTION_INDEX[section]].click();
  fixture.detectChanges();
}

describe('InputsPaneComponent', () => {
  it('when rendered, 5 app-step-section rows are present in ACCORDION_SECTIONS order', () => {
    const { host, http } = setup();
    const sections = host.querySelectorAll('app-step-section');
    expect(sections.length).toBe(5);
    const titles = Array.from(sections).map((el) =>
      el.querySelector('.text-data-sm')?.textContent?.trim(),
    );
    expect(titles).toEqual([
      'Universe',
      'Objective',
      'Constraints',
      'Moments',
      'Horizon',
    ]);
    http.verify();
  });

  it('when no section is expanded, no panel body is mounted', () => {
    const { host, http } = setup();
    expect(host.querySelector('app-step-load-panel')).toBeNull();
    expect(host.querySelector('app-step-screen-panel')).toBeNull();
    expect(host.querySelector('app-run-config-panel')).toBeNull();
    expect(host.querySelector('app-step-validate-is-panel')).toBeNull();
    expect(host.querySelector('app-step-regime-panel')).toBeNull();
    expect(host.querySelector('app-step-clean-returns-panel')).toBeNull();
    expect(host.querySelector('app-step-optimize-panel')).toBeNull();
    http.verify();
  });

  it('when the Universe section header is clicked, Universe expands and mounts run-config + load + screen panels', () => {
    const { fixture, host, http } = setup();
    expand(fixture, host, 'universe');

    expect(host.querySelector('app-run-config-panel')).not.toBeNull();
    expect(host.querySelector('app-step-load-panel')).not.toBeNull();
    expect(host.querySelector('app-step-screen-panel')).not.toBeNull();
    http.verify();
  });

  it('when the Universe section is collapsed again, the panel bodies unmount', () => {
    const { fixture, host, http } = setup();
    expand(fixture, host, 'universe');
    expand(fixture, host, 'universe');

    expect(host.querySelector('app-run-config-panel')).toBeNull();
    expect(host.querySelector('app-step-load-panel')).toBeNull();
    expect(host.querySelector('app-step-screen-panel')).toBeNull();
    http.verify();
  });

  it('when the Objective section is expanded, mounts validate-is, validate-oos, coverage-gate panels in DOM order', () => {
    const { fixture, host, http } = setup();
    expand(fixture, host, 'objective');

    expect(host.querySelector('app-step-validate-is-panel')).not.toBeNull();
    expect(host.querySelector('app-step-validate-oos-panel')).not.toBeNull();
    expect(host.querySelector('app-step-coverage-gate-panel')).not.toBeNull();

    const objSection = host.querySelectorAll('app-step-section')[1];
    const panels = Array.from(
      objSection.querySelectorAll(
        'app-step-validate-is-panel, app-step-validate-oos-panel, app-step-coverage-gate-panel',
      ),
    ).map((el) => el.tagName.toLowerCase());
    expect(panels).toEqual([
      'app-step-validate-is-panel',
      'app-step-validate-oos-panel',
      'app-step-coverage-gate-panel',
    ]);
    http.verify();
  });

  it('when the Objective section is collapsed, its panels unmount', () => {
    const { fixture, host, http } = setup();
    expand(fixture, host, 'objective');
    expand(fixture, host, 'objective');

    expect(host.querySelector('app-step-validate-is-panel')).toBeNull();
    expect(host.querySelector('app-step-validate-oos-panel')).toBeNull();
    expect(host.querySelector('app-step-coverage-gate-panel')).toBeNull();
    http.verify();
  });

  it('when the Constraints section is expanded, mounts regime + rebalance-decision panels in DOM order', () => {
    const { fixture, host, http } = setup();
    expand(fixture, host, 'constraints');

    const cSection = host.querySelectorAll('app-step-section')[2];
    const panels = Array.from(
      cSection.querySelectorAll(
        'app-step-regime-panel, app-step-rebalance-decision-panel',
      ),
    ).map((el) => el.tagName.toLowerCase());
    expect(panels).toEqual([
      'app-step-regime-panel',
      'app-step-rebalance-decision-panel',
    ]);
    http.verify();
  });

  it('when the Constraints section is collapsed, its panels unmount', () => {
    const { fixture, host, http } = setup();
    expand(fixture, host, 'constraints');
    expand(fixture, host, 'constraints');

    expect(host.querySelector('app-step-regime-panel')).toBeNull();
    expect(host.querySelector('app-step-rebalance-decision-panel')).toBeNull();
    http.verify();
  });

  it('when the Moments section is expanded, mounts clean-returns + build-history panels in DOM order', () => {
    const { fixture, host, http } = setup();
    expand(fixture, host, 'moments');

    const mSection = host.querySelectorAll('app-step-section')[3];
    const panels = Array.from(
      mSection.querySelectorAll(
        'app-step-clean-returns-panel, app-step-build-history-panel',
      ),
    ).map((el) => el.tagName.toLowerCase());
    expect(panels).toEqual([
      'app-step-clean-returns-panel',
      'app-step-build-history-panel',
    ]);
    http.verify();
  });

  it('when the Moments section is collapsed, its panels unmount', () => {
    const { fixture, host, http } = setup();
    expand(fixture, host, 'moments');
    expand(fixture, host, 'moments');

    expect(host.querySelector('app-step-clean-returns-panel')).toBeNull();
    expect(host.querySelector('app-step-build-history-panel')).toBeNull();
    http.verify();
  });

  it('when the Horizon section is expanded, mounts optimize + cost + report + persist panels in DOM order', () => {
    const { fixture, host, http } = setup();
    expand(fixture, host, 'horizon');

    const hSection = host.querySelectorAll('app-step-section')[4];
    const panels = Array.from(
      hSection.querySelectorAll(
        'app-step-optimize-panel, app-step-cost-panel, app-step-report-panel, app-step-persist-panel',
      ),
    ).map((el) => el.tagName.toLowerCase());
    expect(panels).toEqual([
      'app-step-optimize-panel',
      'app-step-cost-panel',
      'app-step-report-panel',
      'app-step-persist-panel',
    ]);
    http.verify();
  });

  it('when the Horizon section is collapsed, its panels unmount', () => {
    const { fixture, host, http } = setup();
    expand(fixture, host, 'horizon');
    expand(fixture, host, 'horizon');

    expect(host.querySelector('app-step-optimize-panel')).toBeNull();
    expect(host.querySelector('app-step-cost-panel')).toBeNull();
    expect(host.querySelector('app-step-report-panel')).toBeNull();
    expect(host.querySelector('app-step-persist-panel')).toBeNull();
    http.verify();
  });

  it('when abortGateError is set and Objective is expanded, the coverage-gate panel receives the value via [abortGateError]', () => {
    const { fixture, host, http } = setup();
    fixture.componentInstance.abortGateError.set('not enough factors');
    expand(fixture, host, 'objective');

    const cov = fixture.debugElement.query(
      (n) => n.name === 'app-step-coverage-gate-panel',
    );
    expect(cov).toBeTruthy();
    expect(cov.componentInstance.abortGateError()).toBe('not enough factors');
    http.verify();
  });

  it('when a sessionId is captured and Horizon is expanded, the persist panel receives the value via [sessionId]', () => {
    const { fixture, host, http } = setup();
    TestBed.inject(BuilderStore).setSessionId('sid-99');
    expand(fixture, host, 'horizon');

    const persist = fixture.debugElement.query(
      (n) => n.name === 'app-step-persist-panel',
    );
    expect(persist).toBeTruthy();
    expect(persist.componentInstance.sessionId()).toBe('sid-99');
    http.verify();
  });

  it('when a panel emits runStep and a session exists, the dispatch wrapper POSTs to the session step URL', () => {
    const { fixture, http } = setup();
    TestBed.inject(BuilderStore).setSessionId('sid-7');
    const stepId: PipelineStepId = 'optimize';

    fixture.componentInstance.onPanelRunStep(stepId, {});

    const req = http.expectOne(STEP_URL('sid-7', stepId));
    expect(req.request.method).toBe('POST');
    req.flush(acceptResponse(), ACCEPTED);
    http.verify();
  });

  it('when onConfigSubmit fires, BuilderStore receives setConfig and setStepParams and a session is created via the API', () => {
    const { fixture, http } = setup();
    const store = TestBed.inject(BuilderStore);
    const payload = defaultSubmit();

    fixture.componentInstance.onConfigSubmit(payload);

    expect(store.config()).toEqual(payload.config);
    expect(store.stepParams()).toEqual(payload.steps);

    const req = http.expectOne(SESSIONS_URL);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(payload.config);
    req.flush({ sessionId: 'sid-1' });

    expect(store.sessionId()).toBe('sid-1');
    http.verify();
  });

  it('when a session is created, the first step is unlocked to Ready so its run-step button can appear', () => {
    const { fixture, http } = setup();
    const store = TestBed.inject(BuilderStore);

    fixture.componentInstance.onConfigSubmit(defaultSubmit());
    http.expectOne(SESSIONS_URL).flush({ sessionId: 'sid-ready' });

    expect(store.stepStatuses().get('load')).toBe(StepStatus.Ready);
    http.verify();
  });
});
