import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { OptimizerPanelComponent, type OptimizerRunRequest } from './optimizer-panel';
import type {
  ObjectiveFunctionType,
  RiskMeasureType,
} from '../../models/optimization.model';

const EXPECTED_OBJECTIVES: readonly ObjectiveFunctionType[] = [
  'minimize_risk',
  'maximize_return',
  'maximize_utility',
  'maximize_ratio',
];

const EXPECTED_RISK_MEASURES: readonly RiskMeasureType[] = [
  'variance',
  'semi_variance',
  'standard_deviation',
  'semi_deviation',
  'mean_absolute_deviation',
  'first_lower_partial_moment',
  'cvar',
  'evar',
  'worst_realization',
  'cdar',
  'max_drawdown',
  'average_drawdown',
  'edar',
  'ulcer_index',
  'gini_mean_difference',
];

function buildPanel(): OptimizerPanelComponent {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(OptimizerPanelComponent).componentInstance;
}

describe('OptimizerPanelComponent', () => {
  it('exposes all 4 ObjectiveFunctionType values', () => {
    const panel = buildPanel();
    expect([...panel.objectiveFunctions].sort()).toEqual(
      [...EXPECTED_OBJECTIVES].sort(),
    );
  });

  it('exposes all 15 RiskMeasureType values', () => {
    const panel = buildPanel();
    expect([...panel.riskMeasures].sort()).toEqual(
      [...EXPECTED_RISK_MEASURES].sort(),
    );
  });

  it('exposes both robust_mean_risk and dr_cvar as optimizer types', () => {
    const panel = buildPanel();
    expect(panel.optimizerTypes).toContain('robust_mean_risk');
    expect(panel.optimizerTypes).toContain('dr_cvar');
  });

  it('emits the selected objective and risk measure in the run payload', () => {
    const panel = buildPanel();
    let emitted: OptimizerRunRequest | undefined;
    panel.runOptimization.subscribe((req) => (emitted = req));

    panel.setOptimizerType('mean_risk');
    panel.setObjective('maximize_utility');
    panel.setRiskMeasure('cvar');
    panel.emitRun();

    expect(emitted).toBeDefined();
    expect(emitted!.optimizerType).toBe('mean_risk');
    expect(emitted!.config['objective']).toBe('maximize_utility');
    expect(emitted!.config['risk_measure']).toBe('cvar');
  });

  it('posts kappa to the backend payload when optimizer is robust_mean_risk', () => {
    const panel = buildPanel();
    panel.setOptimizerType('robust_mean_risk');
    panel.robustKappa.set(1.5);

    let emitted: OptimizerRunRequest | undefined;
    panel.runOptimization.subscribe((req) => (emitted = req));
    panel.emitRun();

    expect(emitted!.optimizerType).toBe('robust_mean_risk');
    expect(emitted!.config['kappa']).toBe(1.5);
  });

  it('posts epsilon to the backend payload when optimizer is dr_cvar', () => {
    const panel = buildPanel();
    panel.setOptimizerType('dr_cvar');
    panel.drCvarEpsilon.set(0.25);

    let emitted: OptimizerRunRequest | undefined;
    panel.runOptimization.subscribe((req) => (emitted = req));
    panel.emitRun();

    expect(emitted!.optimizerType).toBe('dr_cvar');
    expect(emitted!.config['epsilon']).toBe(0.25);
  });

  it('does not include kappa when optimizer is not robust_mean_risk', () => {
    const panel = buildPanel();
    panel.setOptimizerType('mean_risk');
    panel.robustKappa.set(2.0);

    let emitted: OptimizerRunRequest | undefined;
    panel.runOptimization.subscribe((req) => (emitted = req));
    panel.emitRun();

    expect(Object.keys(emitted!.config)).not.toContain('kappa');
  });

  it('does not include epsilon when optimizer is not dr_cvar', () => {
    const panel = buildPanel();
    panel.setOptimizerType('mean_risk');
    panel.drCvarEpsilon.set(0.4);

    let emitted: OptimizerRunRequest | undefined;
    panel.runOptimization.subscribe((req) => (emitted = req));
    panel.emitRun();

    expect(Object.keys(emitted!.config)).not.toContain('epsilon');
  });

  it('hides objective+risk measure for optimizers that do not support them', () => {
    const panel = buildPanel();

    panel.setOptimizerType('hrp');
    expect(panel.showObjective()).toBe(false);
    expect(panel.showRiskMeasure()).toBe(false);

    panel.setOptimizerType('mean_risk');
    expect(panel.showObjective()).toBe(true);
    expect(panel.showRiskMeasure()).toBe(true);
  });

  it('shows CVaR beta input when a tail-risk measure is selected', () => {
    const panel = buildPanel();
    panel.setOptimizerType('mean_risk');
    panel.setRiskMeasure('cvar');
    expect(panel.showCvarBeta()).toBe(true);

    panel.setRiskMeasure('variance');
    expect(panel.showCvarBeta()).toBe(false);
  });
});
