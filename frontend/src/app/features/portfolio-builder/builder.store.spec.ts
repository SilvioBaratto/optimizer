import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { BuilderStore } from './builder.store';
import type { BuilderStage } from './builder-stage';
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

    store.reset();

    expect(store.config()).toBeNull();
    expect(store.stepParams()).toBeNull();
    expect(store.currentStage()).toBeNull();
    expect(store.status()).toBe(StepStatus.Pending);
    expect(store.stepResults().size).toBe(0);
    expect(store.stepStatuses().size).toBe(0);
    expect(store.summary()).toEqual({
      stage: null,
      status: StepStatus.Pending,
      hasConfig: false,
    });
  });
});
