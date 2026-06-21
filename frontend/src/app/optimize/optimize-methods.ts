/** The four curated optimization methods expressed as mean_risk config presets. */

export type MethodId = 'min_variance' | 'max_sharpe' | 'risk_parity_erc' | 'hrp';

export interface MethodConfig {
  /** Top-level optimizer_type for the POST body — NOT forwarded inside config. */
  readonly optimizer_type: string;
  readonly objective: string;
  readonly risk_measure: string;
  readonly l2_coef?: number;
  readonly [key: string]: unknown;
}

export interface OptimizeMethod {
  readonly id: MethodId;
  readonly label: string;
  readonly config: MethodConfig;
}

export const OPTIMIZE_METHODS: readonly OptimizeMethod[] = [
  {
    id: 'min_variance',
    label: 'Min-Variance',
    config: { optimizer_type: 'mean_risk', objective: 'minimize_risk', risk_measure: 'variance' },
  },
  {
    id: 'max_sharpe',
    label: 'Max-Sharpe',
    config: { optimizer_type: 'mean_risk', objective: 'maximize_ratio', risk_measure: 'variance' },
  },
  {
    id: 'risk_parity_erc',
    label: 'Risk-Parity (ERC)',
    config: { optimizer_type: 'mean_risk', objective: 'maximize_utility', risk_measure: 'variance', l2_coef: 1.0 },
  },
  {
    id: 'hrp',
    label: 'HRP (Low-Conc.)',
    config: { optimizer_type: 'mean_risk', objective: 'minimize_risk', risk_measure: 'variance', l2_coef: 0.5 },
  },
];
