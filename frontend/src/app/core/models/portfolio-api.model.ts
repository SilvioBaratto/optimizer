// DTOs for /api/v1/portfolio/*. Names and fields mirror the Pydantic schemas
// in api/app/schemas/portfolio.py — keep snake_case, do not rename.

export interface PortfolioDto {
  id: string;
  name: string;
  description: string | null;
  currency: string;
  benchmark_ticker: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface PortfolioListResponseDto {
  items: PortfolioDto[];
  total: number;
}

export interface CreatePortfolioDto {
  name: string;
  description?: string | null;
  currency?: string;
  benchmark_ticker?: string;
}

export interface UpdatePortfolioDto {
  name?: string;
  description?: string | null;
  currency?: string;
  benchmark_ticker?: string;
}

export interface SnapshotDto {
  id: string;
  portfolio_id: string;
  snapshot_date: string;
  snapshot_type: string;
  weights: Record<string, number>;
  sector_mapping: Record<string, string> | null;
  summary: Record<string, unknown> | null;
  optimizer_config: Record<string, unknown> | null;
  turnover: number | null;
  holding_count: number;
  created_at: string;
}

export interface SnapshotListResponseDto {
  items: SnapshotDto[];
  total: number;
}

export interface CreateSnapshotDto {
  snapshot_date: string;
  snapshot_type?: 'optimization' | 'rebalance' | 'manual';
  weights: Record<string, number>;
  sector_mapping?: Record<string, string>;
  summary?: Record<string, unknown>;
  optimizer_config?: Record<string, unknown>;
  turnover?: number;
}

export interface BrokerPositionDto {
  id: string;
  ticker: string;
  yfinance_ticker: string | null;
  name: string | null;
  quantity: number;
  average_price: number;
  current_price: number | null;
  ppl: number | null;
  fx_ppl: number | null;
  initial_fill_date: string | null;
  synced_at: string;
}

export interface BrokerAccountDto {
  id: string;
  total: number;
  free: number;
  invested: number;
  blocked: number | null;
  result: number | null;
  currency: string;
  synced_at: string;
}

export interface SyncJobResponseDto {
  job_id: string;
  status: string;
  message: string;
}

export interface SyncProgressResponseDto {
  job_id: string;
  status: string;
  current: number;
  total: number;
  result: Record<string, unknown> | null;
  error: string | null;
}
