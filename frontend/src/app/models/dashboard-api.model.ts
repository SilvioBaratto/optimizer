/**
 * Raw API response types matching backend Pydantic schemas.
 * JSON arrives in camelCase via CamelCaseModel (alias_generator=to_camel).
 * AssetClassReturnRow uses explicit Field(alias=...) for "1D", "1W", "1M", "YTD".
 */

export interface ApiKpiItem {
  label: string;
  value: number;
  format: 'percent' | 'currency' | 'ratio' | 'number';
  change: number;
  changeLabel: string;
  sparkline: number[];
}

export interface ApiPerformanceMetricsResponse {
  kpis: ApiKpiItem[];
  nav: number;
  navChangePct: number;
  currency: string;
}

export interface ApiEquityCurvePoint {
  date: string;
  portfolio: number;
  benchmark: number;
}

export interface ApiEquityCurveResponse {
  points: ApiEquityCurvePoint[];
  portfolioTotalReturn: number;
  benchmarkTotalReturn: number;
}

export interface ApiAllocationChild {
  name: string;
  value: number;
}

export interface ApiAllocationNode {
  name: string;
  value: number;
  children: ApiAllocationChild[];
}

export interface ApiAllocationResponse {
  nodes: ApiAllocationNode[];
  totalPositions: number;
  totalSectors: number;
}

export interface ApiDriftEntry {
  ticker: string;
  name: string | null;
  target: number;
  actual: number;
  drift: number;
  breached: boolean;
}

export interface ApiDriftResponse {
  entries: ApiDriftEntry[];
  totalDrift: number;
  breachedCount: number;
  threshold: number;
}

export interface ApiActivityItem {
  id: string;
  type: string;
  title: string;
  description: string | null;
  timestamp: string;
}

export interface ApiActivityFeedResponse {
  items: ApiActivityItem[];
  total: number;
}

export interface ApiMarketSnapshotResponse {
  vix: number;
  vixChange: number;
  sp500Return: number;
  tenYearYield: number;
  yieldChange: number;
  usdIndex: number;
  usdChange: number;
  asOf: string;
}

export interface ApiHmmStateItem {
  regime: string;
  probability: number;
}

export interface ApiRegimeModelInfo {
  nStates: number;
  lastFitted: string;
}

export interface ApiMarketRegimeResponse {
  current: string;
  probability: number;
  since: string;
  hmmStates: ApiHmmStateItem[];
  modelInfo: ApiRegimeModelInfo;
}

export interface ApiAssetClassReturnRow {
  name: string;
  '1D': number;
  '1W': number;
  '1M': number;
  'YTD': number;
}

export interface ApiAssetClassReturnsResponse {
  returns: ApiAssetClassReturnRow[];
  asOf: string;
}
