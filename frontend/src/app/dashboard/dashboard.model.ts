export interface DashboardKPI {
  label: string;
  value: number;
  format: 'percent' | 'currency' | 'ratio' | 'number';
  change: number;
  changeLabel: string;
  sparkline: number[];
}

export interface EquityCurvePoint {
  date: string;
  portfolio: number;
  benchmark: number;
}

export interface MarketContext {
  vix: number;
  vixChange: number;
  sp500Return: number;
  tenYearYield: number;
  yieldChange: number;
  usdIndex: number;
  usdChange: number;
}

export interface AllocationNode {
  name: string;
  value: number;
  children?: AllocationNode[];
}

export interface AssetClassReturn {
  name: string;
  '1D': number;
  '1W': number;
  '1M': number;
  'YTD': number;
}

export interface PortfolioDataSnapshot {
  kpis: DashboardKPI[];
  nav: number;
  navChangePct: number;
  currency: string;
  equityCurvePoints: EquityCurvePoint[];
  allocationNodes: AllocationNode[];
  assetClassReturns: AssetClassReturn[];
}
