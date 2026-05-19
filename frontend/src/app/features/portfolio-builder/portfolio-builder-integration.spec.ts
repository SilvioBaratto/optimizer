import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { Subject } from 'rxjs';

import { PortfolioBuilderComponent } from './portfolio-builder';
import { BuilderStore } from './builder.store';
import { JOB_POLL_TICK } from '../../shared/job-progress-tracker/job-progress-tracker';
import { environment } from '../../../environments/environment';
import {
  type PipelineConfigSubmit,
  STEP_PARAM_DEFAULTS,
} from '../../pages/pipeline-stepper/step-params.model';
import type {
  LoadStepResult,
  PipelineStepId,
  RunLevelConfig,
} from '../../models/pipeline-builder.model';

const SESSIONS_URL = `${environment.apiUrl}pipeline-builder/sessions`;
const STEP_URL = (sid: string, step: string) =>
  `${SESSIONS_URL}/${sid}/steps/${step}`;
const ACCEPTED = { status: 202, statusText: 'Accepted' };

// Fixture mirroring legacy `RunConfigPanelComponent.onSubmit()` defaults at
// frontend/src/app/pages/pipeline-stepper/run-config-panel.ts:91-114. Any
// drift between this fixture and either path is a contract break.
function legacySubmit(): PipelineConfigSubmit {
  const config: RunLevelConfig = {
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
  };
  return { config, steps: structuredClone(STEP_PARAM_DEFAULTS) };
}

function loadResult(): Record<string, unknown> {
  const r: LoadStepResult = {
    n_tickers: 42,
    n_trading_days: 1260,
    assembly_hash: 'h',
    base_currency: 'EUR',
    price_start: '2020-01-01',
    price_end: '2025-01-01',
  };
  return r as unknown as Record<string, unknown>;
}

function setup(): {
  fixture: ComponentFixture<PortfolioBuilderComponent>;
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
    ],
  });
  const http = TestBed.inject(HttpTestingController);
  const fixture = TestBed.createComponent(PortfolioBuilderComponent);
  fixture.detectChanges();
  return { fixture, http, tick };
}

describe('PortfolioBuilder integration — submit-shape parity', () => {
  it('when configSubmit fires from InputsPane, createSession receives the full 10-field RunLevelConfig and BuilderStore.setStepParams captures the full 7-field StepParamsConfig', () => {
    const { fixture, http } = setup();
    const runConfig = fixture.debugElement.query(
      (n) => n.name === 'app-run-config-panel',
    );
    // Universe accordion not expanded by default; expand to mount run-config.
    const sectionButtons = (fixture.nativeElement as HTMLElement).querySelectorAll<HTMLButtonElement>(
      'app-step-section > section > button',
    );
    sectionButtons[0].click();
    fixture.detectChanges();

    const mountedConfigPanel = fixture.debugElement.query(
      (n) => n.name === 'app-run-config-panel',
    );
    expect(mountedConfigPanel).toBeTruthy();
    expect(runConfig).toBeNull();

    const payload = legacySubmit();
    mountedConfigPanel.componentInstance.configSubmit.emit(payload);

    // BuilderStore receives both halves of the submit.
    const store = fixture.componentInstance.store;
    expect(store.config()).toEqual(payload.config);
    expect(Object.keys(store.config() ?? {}).sort()).toEqual(
      Object.keys(payload.config).sort(),
    );
    expect(store.stepParams()).toEqual(payload.steps);
    expect(Object.keys(store.stepParams() ?? {}).sort()).toEqual(
      ['build_history', 'coverage_gate', 'load', 'regime', 'screen'],
    );

    // API.createSession receives the RunLevelConfig verbatim.
    const req = http.expectOne(SESSIONS_URL);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(payload.config);
    req.flush({ sessionId: 'sid-int-1' });

    http.verify();
  });
});

describe('PortfolioBuilder integration — free navigation preserves state', () => {
  it('when one step completes and the user navigates across stages, stepResults retains the original entry', () => {
    const { fixture, http, tick } = setup();
    const store = fixture.componentInstance.store;

    // Hydrate session + dispatch one step.
    store.setConfig(legacySubmit().config);
    store.setStepParams(legacySubmit().steps);

    const inputsPane = fixture.debugElement.query(
      (n) => n.name === 'app-inputs-pane',
    ).componentInstance;
    inputsPane.sessionId.set('sid-int-2');

    const stepId: PipelineStepId = 'load';
    inputsPane.onPanelRunStep(stepId, {});
    const runReq = http.expectOne(STEP_URL('sid-int-2', stepId));
    runReq.flush({ job_id: 'j-1', status: 'pending', message: '' }, ACCEPTED);

    // Force a poll that completes with the load result.
    tick.next(0);
    const pollReq = http.expectOne(STEP_URL('sid-int-2', stepId));
    pollReq.flush({
      status: 'completed',
      progress: {},
      result: loadResult(),
      error: null,
      gateReason: null,
    });

    expect(store.stepResults().get('load')).toEqual(loadResult());

    // Navigate freely across stages.
    store.setStage('rebalance');
    store.setStage('universe');

    // State preserved — the original result is still there, unchanged.
    expect(store.stepResults().get('load')).toEqual(loadResult());
    expect(store.config()).toEqual(legacySubmit().config);
    expect(store.stepParams()).toEqual(legacySubmit().steps);

    http.verify();
  });
});
