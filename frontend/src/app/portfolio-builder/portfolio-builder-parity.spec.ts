import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { NEVER } from 'rxjs';

import { PortfolioBuilderComponent } from './portfolio-builder';
import { BuilderStore } from './state/builder.store';
import { InputsPaneComponent } from './inputs-pane/inputs-pane';
import { JOB_POLL_TICK } from '../shared/job-progress-tracker/job-progress-tracker';
import { environment } from '../../environments/environment';
import {
  STEP_PARAM_DEFAULTS,
  type PipelineConfigSubmit,
} from '../pipeline-stepper/models/step-params.model';
import type { RunLevelConfig } from '../core/models/pipeline-builder.model';

const SESSIONS_URL = `${environment.apiUrl}pipeline-builder/sessions`;

// Canonical fixture mirroring legacy `RunConfigPanelComponent.onSubmit()`
// defaults at frontend/src/app/pipeline-stepper/run-config-panel/run-config-panel.ts.
// Drift between this fixture and either dispatch path is a contract break.
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

function configure(): void {
  TestBed.configureTestingModule({
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
      { provide: JOB_POLL_TICK, useValue: NEVER },
      BuilderStore,
    ],
  });
}

describe('PortfolioBuilder parity — config shape', () => {
  let fixture: ComponentFixture<InputsPaneComponent>;
  let http: HttpTestingController;
  let store: BuilderStore;

  beforeEach(() => {
    configure();
    http = TestBed.inject(HttpTestingController);
    store = TestBed.inject(BuilderStore);
    fixture = TestBed.createComponent(InputsPaneComponent);
    fixture.detectChanges();
  });

  afterEach(() => {
    http.verify();
  });

  it('when onConfigSubmit is invoked with the legacy fixture, store.config deep-equals legacy.config', () => {
    const payload = legacySubmit();

    fixture.componentInstance.onConfigSubmit(payload);

    expect(store.config()).toEqual(payload.config);

    const req = http.expectOne(SESSIONS_URL);
    req.flush({ sessionId: 's-parity-cfg' });
  });

  it('when onConfigSubmit is invoked with the legacy fixture, store.stepParams deep-equals legacy.steps', () => {
    const payload = legacySubmit();

    fixture.componentInstance.onConfigSubmit(payload);

    expect(store.stepParams()).toEqual(payload.steps);

    const req = http.expectOne(SESSIONS_URL);
    req.flush({ sessionId: 's-parity-steps' });
  });

  it('when onConfigSubmit fires, the intercepted createSession POST body equals legacy.config verbatim', () => {
    const payload = legacySubmit();

    fixture.componentInstance.onConfigSubmit(payload);

    const req = http.expectOne(SESSIONS_URL);
    expect(req.request.method).toBe('POST');
    expect(req.request.url).toBe(SESSIONS_URL);
    expect(req.request.body).toEqual(payload.config);
    req.flush({ sessionId: 's-parity-body' });
  });
});

describe('PortfolioBuilder parity — dispatched actions', () => {
  let http: HttpTestingController;
  let store: BuilderStore;

  beforeEach(() => {
    configure();
    http = TestBed.inject(HttpTestingController);
    store = TestBed.inject(BuilderStore);
  });

  afterEach(() => {
    http.verify();
  });

  it('when store.optimize() is invoked with sessionId set, a POST is issued to a URL ending /sessions/s1/steps/optimize', () => {
    store.setSessionId('s1');

    store.optimize();

    const req = http.expectOne(
      (r) => r.url.endsWith('/sessions/s1/steps/optimize') && r.method === 'POST',
    );
    expect(req.request.body).toEqual({});
    req.flush({ job_id: 'j-opt', status: 'pending', message: '' });
  });

  it('when store.rebalance() is invoked with sessionId set, a POST is issued to a URL ending /sessions/s1/steps/rebalance_decision', () => {
    store.setSessionId('s1');

    store.rebalance();

    const req = http.expectOne(
      (r) =>
        r.url.endsWith('/sessions/s1/steps/rebalance_decision') &&
        r.method === 'POST',
    );
    expect(req.request.body).toEqual({});
    req.flush({ job_id: 'j-rb', status: 'pending', message: '' });
  });
});

describe('PortfolioBuilder parity — clean state on route re-entry', () => {
  let http: HttpTestingController;

  beforeEach(() => {
    // Clear localStorage to isolate portfolio context from any prior state.
    localStorage.clear();
    configure();
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
  });

  it('when PortfolioBuilderComponent is destroyed and re-mounted, the new BuilderStore is a fresh instance with default signals', () => {
    const fixture1 = TestBed.createComponent(PortfolioBuilderComponent);
    fixture1.detectChanges();
    // Mock the portfolio list fetch for the first component instance.
    http
      .expectOne(`${environment.apiUrl}portfolio/`)
      .flush({ items: [{ id: 't212', name: 't212', currency: 'EUR' }] });

    const store1 = fixture1.componentInstance.store;
    store1.setSessionId('s-first');
    store1.setConfig(legacySubmit().config);
    store1.setStepParams(legacySubmit().steps);
    expect(store1.sessionId()).toBe('s-first');
    expect(store1.config()).not.toBeNull();

    fixture1.destroy();

    const fixture2 = TestBed.createComponent(PortfolioBuilderComponent);
    fixture2.detectChanges();
    // Mock the portfolio list fetch for the second component instance.
    http
      .expectOne(`${environment.apiUrl}portfolio/`)
      .flush({ items: [{ id: 't212', name: 't212', currency: 'EUR' }] });

    const store2 = fixture2.componentInstance.store;

    expect(store2).not.toBe(store1);
    expect(store2.sessionId()).toBeNull();
    expect(store2.config()).toBeNull();
    expect(store2.stepParams()).toBeNull();
    expect(store2.stepResults().size).toBe(0);
    expect(store2.stepStatuses().size).toBe(0);
  });
});
