import { Component, signal, output, computed, ChangeDetectionStrategy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HelpTooltipComponent } from '../../shared/help-tooltip/help-tooltip';
import type {
  ObjectiveFunctionType,
  OptimizerType,
  RiskMeasureType,
} from '../../models/optimization.model';

// All optimizer types exposed to the UI (matches backend OPTIMIZER_REGISTRY
// in api/app/services/optimization_service.py plus tracker/benchmark/max_sharpe
// aliases that the backend resolves internally).
const OPTIMIZER_TYPES: readonly OptimizerType[] = [
  'mean_risk',
  'risk_budgeting',
  'max_diversification',
  'equal_weighted',
  'inverse_volatility',
  'hrp',
  'herc',
  'nco',
  'stacking',
  'regime_blended',
  'benchmark_tracking',
  'max_sharpe',
  'min_variance',
  'robust_mean_risk',
  'dr_cvar',
];

const OPTIMIZER_LABELS: Record<OptimizerType, string> = {
  mean_risk: 'Mean-Risk (Markowitz)',
  max_sharpe: 'Max Sharpe Ratio',
  min_variance: 'Min Variance',
  risk_budgeting: 'Risk Budgeting',
  max_diversification: 'Max Diversification',
  equal_weighted: 'Equal Weight',
  inverse_volatility: 'Inverse Volatility',
  hrp: 'Hierarchical Risk Parity (HRP)',
  herc: 'Hierarchical Equal Risk Contribution (HERC)',
  nco: 'Nested Cluster Optimization (NCO)',
  benchmark_tracking: 'Benchmark Tracking',
  robust_mean_risk: 'Robust MeanRisk',
  dr_cvar: 'DR-CVaR (Wasserstein)',
  stacking: 'Stacking Optimizer',
  regime_blended: 'Regime-Blended Optimizer',
};

// All 4 ObjectiveFunctionType values from optimizer/optimization/_config.py.
// Applicable only to convex mean-risk optimizers.
const OBJECTIVE_FUNCTIONS: readonly ObjectiveFunctionType[] = [
  'minimize_risk',
  'maximize_return',
  'maximize_utility',
  'maximize_ratio',
];

const OBJECTIVE_LABELS: Record<ObjectiveFunctionType, string> = {
  minimize_risk: 'Minimize Risk',
  maximize_return: 'Maximize Return',
  maximize_utility: 'Maximize Utility',
  maximize_ratio: 'Maximize Ratio (Sharpe-like)',
};

// All 15 convex risk measures — mirrors RiskMeasureType enum.
const RISK_MEASURES: readonly RiskMeasureType[] = [
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

const RISK_MEASURE_LABELS: Record<RiskMeasureType, string> = {
  variance: 'Variance',
  semi_variance: 'Semi-Variance',
  standard_deviation: 'Standard Deviation',
  semi_deviation: 'Semi-Deviation',
  mean_absolute_deviation: 'Mean Absolute Deviation',
  first_lower_partial_moment: 'First Lower Partial Moment',
  cvar: 'Conditional Value at Risk (CVaR)',
  evar: 'Entropic Value at Risk (EVaR)',
  worst_realization: 'Worst Realization',
  cdar: 'Conditional Drawdown at Risk (CDaR)',
  max_drawdown: 'Max Drawdown',
  average_drawdown: 'Average Drawdown',
  edar: 'Entropic Drawdown at Risk (EDaR)',
  ulcer_index: 'Ulcer Index',
  gini_mean_difference: 'Gini Mean Difference',
};

export interface OptimizerRunRequest {
  optimizerType: OptimizerType;
  config: Record<string, unknown>;
}

@Component({
  selector: 'app-optimizer-panel',
  imports: [FormsModule, HelpTooltipComponent],
  templateUrl: './optimizer-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class OptimizerPanelComponent {
  readonly optimizerTypes = OPTIMIZER_TYPES;
  readonly optimizerLabels = OPTIMIZER_LABELS;
  readonly objectiveFunctions = OBJECTIVE_FUNCTIONS;
  readonly objectiveLabels = OBJECTIVE_LABELS;
  readonly riskMeasures = RISK_MEASURES;
  readonly riskMeasureLabels = RISK_MEASURE_LABELS;

  readonly optimizerType = signal<OptimizerType>('mean_risk');
  readonly objective = signal<ObjectiveFunctionType>('maximize_ratio');
  readonly riskMeasure = signal<RiskMeasureType>('variance');
  readonly riskAversion = signal<number>(1.0);
  readonly cvarBeta = signal<number>(0.95);
  readonly robustKappa = signal<number>(1.0);
  readonly drCvarEpsilon = signal<number>(0.1);

  readonly runOptimization = output<OptimizerRunRequest>();

  readonly showObjective = computed(() => this.supportsObjective(this.optimizerType()));
  readonly showRiskMeasure = computed(() => this.supportsRiskMeasure(this.optimizerType()));

  readonly showCvarBeta = computed(() => {
    const rm = this.riskMeasure();
    return (
      this.optimizerType() === 'dr_cvar' ||
      rm === 'cvar' ||
      rm === 'evar' ||
      rm === 'cdar' ||
      rm === 'edar'
    );
  });

  readonly showRobustKappa = computed(() => this.optimizerType() === 'robust_mean_risk');
  readonly showDrCvarEpsilon = computed(() => this.optimizerType() === 'dr_cvar');

  readonly activeConstraints = [
    'Long-only (weights ≥ 0)',
    'Fully invested (weights sum to 1)',
    'Max weight per asset: 30%',
  ];

  setOptimizerType(value: string): void {
    this.optimizerType.set(value as OptimizerType);
  }

  setObjective(value: string): void {
    this.objective.set(value as ObjectiveFunctionType);
  }

  setRiskMeasure(value: string): void {
    this.riskMeasure.set(value as RiskMeasureType);
  }

  emitRun(): void {
    this.runOptimization.emit({
      optimizerType: this.optimizerType(),
      config: this.buildConfig(),
    });
  }

  private buildConfig(): Record<string, unknown> {
    const type = this.optimizerType();
    const config: Record<string, unknown> = {};
    if (this.supportsObjective(type)) config['objective'] = this.objective();
    if (this.supportsRiskMeasure(type)) config['risk_measure'] = this.riskMeasure();
    if (type === 'mean_risk') config['l2_coef'] = this.riskAversion();
    if (type === 'robust_mean_risk') config['kappa'] = this.robustKappa();
    if (type === 'dr_cvar') config['epsilon'] = this.drCvarEpsilon();
    if (this.showCvarBeta()) config['beta'] = this.cvarBeta();
    return config;
  }

  private supportsObjective(type: OptimizerType): boolean {
    return (
      type === 'mean_risk' ||
      type === 'max_sharpe' ||
      type === 'min_variance' ||
      type === 'robust_mean_risk' ||
      type === 'benchmark_tracking'
    );
  }

  private supportsRiskMeasure(type: OptimizerType): boolean {
    return type === 'mean_risk' || type === 'robust_mean_risk' || type === 'risk_budgeting';
  }
}
