import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { BuilderStore } from './builder.store';
import {
  type RunLevelConfig,
  StepStatus,
} from '../../models/pipeline-builder.model';
import { STEP_PARAM_DEFAULTS } from '../../pages/pipeline-stepper/step-params.model';

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

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideZonelessChangeDetection(), BuilderStore],
    });
    store = TestBed.inject(BuilderStore);
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

  it('when setStage is called with "load", currentStage and summary.stage update', () => {
    store.setStage('load');
    expect(store.currentStage()).toBe('load');
    expect(store.summary().stage).toBe('load');
  });

  it('when setStatus is called with Running, summary.status propagates', () => {
    store.setStatus(StepStatus.Running);
    expect(store.status()).toBe(StepStatus.Running);
    expect(store.summary().status).toBe(StepStatus.Running);
  });

  it('when reset is called after mutations, all signals return to initial', () => {
    store.setConfig(defaultConfig());
    store.setStepParams(structuredClone(STEP_PARAM_DEFAULTS));
    store.setStage('optimize');
    store.setStatus(StepStatus.Completed);

    store.reset();

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
});
