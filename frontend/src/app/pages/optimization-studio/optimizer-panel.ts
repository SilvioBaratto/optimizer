import { Component, signal, output, computed, ChangeDetectionStrategy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HelpTooltipComponent } from '../../shared/help-tooltip/help-tooltip';
import { RiskMeasureType } from '../../models/optimization.model';
import { MOCK_OPTIMIZATION_CONFIG } from '../../mocks/optimization-mocks';

type ObjectiveFunction =
  | 'max_sharpe'
  | 'min_variance'
  | 'min_cvar'
  | 'max_return'
  | 'risk_parity'
  | 'risk_budgeting'
  | 'max_diversification'
  | 'equal_weight'
  | 'inverse_volatility'
  | 'hrp'
  | 'herc'
  | 'nco'
  | 'benchmark_tracking'
  | 'robust_mean_risk'
  | 'dr_cvar';

const OBJECTIVE_LABELS: Record<ObjectiveFunction, string> = {
  max_sharpe: 'Max Sharpe Ratio',
  min_variance: 'Min Variance',
  min_cvar: 'Min CVaR',
  max_return: 'Max Return',
  risk_parity: 'Risk Parity',
  risk_budgeting: 'Risk Budgeting',
  max_diversification: 'Max Diversification',
  equal_weight: 'Equal Weight',
  inverse_volatility: 'Inverse Volatility',
  hrp: 'Hierarchical Risk Parity (HRP)',
  herc: 'Hierarchical Equal Risk Contribution (HERC)',
  nco: 'Nested Cluster Optimization (NCO)',
  benchmark_tracking: 'Benchmark Tracking',
  robust_mean_risk: 'Robust MeanRisk',
  dr_cvar: 'DR-CVaR (Wasserstein)',
};

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

@Component({
  selector: 'app-optimizer-panel',
  imports: [FormsModule, HelpTooltipComponent],
  templateUrl: './optimizer-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class OptimizerPanelComponent {
  readonly objectiveFunctions: ObjectiveFunction[] = [
    'max_sharpe', 'min_variance', 'min_cvar', 'max_return',
    'risk_parity', 'risk_budgeting', 'max_diversification',
    'equal_weight', 'inverse_volatility', 'hrp', 'herc', 'nco',
    'benchmark_tracking', 'robust_mean_risk', 'dr_cvar',
  ];
  readonly riskMeasures: RiskMeasureType[] = [
    'variance', 'semi_variance', 'standard_deviation', 'semi_deviation',
    'mean_absolute_deviation', 'first_lower_partial_moment', 'cvar', 'evar',
    'worst_realization', 'cdar', 'max_drawdown', 'average_drawdown',
    'edar', 'ulcer_index', 'gini_mean_difference',
  ];
  readonly objectiveLabels = OBJECTIVE_LABELS;
  readonly riskMeasureLabels = RISK_MEASURE_LABELS;

  objective = signal<ObjectiveFunction>(MOCK_OPTIMIZATION_CONFIG.strategy as ObjectiveFunction);
  riskMeasure = signal<RiskMeasureType>(MOCK_OPTIMIZATION_CONFIG.riskMeasure);
  riskAversion = signal<number>(MOCK_OPTIMIZATION_CONFIG.riskAversion);
  cvarBeta = signal<number>(MOCK_OPTIMIZATION_CONFIG.cvarBeta);
  robustKappa = signal<number>(MOCK_OPTIMIZATION_CONFIG.robustKappa);

  runOptimization = output<void>();

  readonly showCvarBeta = computed(() =>
    ['cvar', 'evar', 'min_cvar', 'dr_cvar'].includes(this.riskMeasure()) ||
    this.objective() === 'min_cvar' || this.objective() === 'dr_cvar',
  );

  readonly showRobustKappa = computed(() =>
    this.objective() === 'robust_mean_risk',
  );

  readonly activeConstraints = [
    'Long-only (weights ≥ 0)',
    'Fully invested (weights sum to 1)',
    'Max weight per asset: 30%',
  ];

  setObjective(value: string): void {
    this.objective.set(value as ObjectiveFunction);
  }

  setRiskMeasure(value: string): void {
    this.riskMeasure.set(value as RiskMeasureType);
  }
}
