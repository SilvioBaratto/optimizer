export type FactorGroupType =
  | 'value'
  | 'profitability'
  | 'investment'
  | 'momentum'
  | 'low_risk'
  | 'liquidity'
  | 'dividend'
  | 'sentiment'
  | 'ownership';

export type FactorType =
  | 'book_to_price'
  | 'earnings_yield'
  | 'cash_flow_yield'
  | 'sales_to_price'
  | 'ebitda_to_ev'
  | 'gross_profitability'
  | 'roe'
  | 'operating_margin'
  | 'profit_margin'
  | 'asset_growth'
  | 'momentum_12_1'
  | 'volatility'
  | 'beta'
  | 'amihud_illiquidity'
  | 'dividend_yield'
  | 'recommendation_change'
  | 'net_insider_buying';

export type MacroRegime = 'expansion' | 'slowdown' | 'recession' | 'recovery';

export type HmmState = 'low_vol' | 'medium_vol' | 'high_vol';

export interface RegimeDetection {
  date: string;
  state: HmmState;
  probabilities: Record<HmmState, number>;
}

export interface TAASignal {
  factor: FactorGroupType;
  currentWeight: number;
  tiltedWeight: number;
  tiltReason: string;
  regime: MacroRegime;
}

export interface FactorReturnSeries {
  factor: FactorType;
  group: FactorGroupType;
  points: { date: string; cumReturn: number }[];
}

export interface CMASet {
  label: string;
  horizon: string;
  assets: {
    ticker: string;
    expectedReturn: number;
    expectedVol: number;
  }[];
}

export interface ScreenerFilter {
  factor: FactorType;
  operator: 'gt' | 'lt' | 'between';
  value: number;
  value2?: number;
}

export interface FactorICReport {
  factor: FactorType;
  group: FactorGroupType;
  ic: number;
  icir: number;
  tStat: number;
  pValue: number;
  vif: number;
  significant: boolean;
}
