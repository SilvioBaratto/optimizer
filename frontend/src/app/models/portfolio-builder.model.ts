export interface UniverseTicker {
  ticker: string;
  name: string;
  sector: string;
  marketCap: number;
  weight: number;
  selected: boolean;
}

export type ConstraintType =
  | 'weight_bounds'
  | 'sector_bounds'
  | 'cardinality'
  | 'turnover'
  | 'tracking_error';

export interface Constraint {
  id: string;
  type: ConstraintType;
  label: string;
  min?: number;
  max?: number;
  value?: number;
  target?: string;
  enabled: boolean;
}

export type RiskProfile = 'conservative' | 'moderate' | 'aggressive' | 'custom';

export interface IPS {
  name: string;
  riskProfile: RiskProfile;
  targetReturn: number;
  maxVolatility: number;
  maxDrawdown: number;
  rebalanceFrequency: 'monthly' | 'quarterly' | 'semi_annual' | 'annual';
  constraints: Constraint[];
}

export interface WeightAssignment {
  ticker: string;
  name: string;
  sector: string;
  currentWeight: number;
  targetWeight: number;
  difference: number;
}
