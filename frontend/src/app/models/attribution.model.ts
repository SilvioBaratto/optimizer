export interface BrinsonSectorAttribution {
  sector: string;
  portfolioWeight: number;
  benchmarkWeight: number;
  portfolioReturn: number;
  benchmarkReturn: number;
  allocationEffect: number;
  selectionEffect: number;
  interactionEffect: number;
  totalEffect: number;
}

export interface BrinsonAttribution {
  totalAllocation: number;
  totalSelection: number;
  totalInteraction: number;
  totalActive: number;
  sectors: BrinsonSectorAttribution[];
}

export interface MultiLevelAttribution {
  level: string;
  name: string;
  contribution: number;
  weight: number;
  returnPct: number;
}

export interface FactorAttribution {
  factor: string;
  exposure: number;
  factorReturn: number;
  contribution: number;
  cumulative: number;
}

export interface HoldingsAttribution {
  ticker: string;
  name: string;
  sector: string;
  weight: number;
  returnPct: number;
  contribution: number;
}

// ── Backend API DTOs (mirror api/app/schemas/attribution.py) ────────────────

export interface BrinsonApiRequest {
  portfolio_weights: Record<string, number>;
  benchmark_weights: Record<string, number>;
  start_date: string;
  end_date: string;
}

export interface BrinsonSectorRowDto {
  sector: string;
  portfolioWeight: number;
  benchmarkWeight: number;
  portfolioReturn: number;
  benchmarkReturn: number;
  allocationEffect: number;
  selectionEffect: number;
  interactionEffect: number;
  totalEffect: number;
}

export interface BrinsonApiResponse {
  sectors: BrinsonSectorRowDto[];
  totalAllocation: number;
  totalSelection: number;
  totalInteraction: number;
  totalActiveReturn: number;
  portfolioReturn: number;
  benchmarkReturn: number;
}

export interface FactorAttributionApiRequest {
  portfolio_weights: Record<string, number>;
  start_date: string;
  end_date: string;
}

export interface FactorRowDto {
  factorName: string;
  exposure: number;
  factorReturn: number;
  contribution: number;
}

export interface FactorAttributionApiResponse {
  factors: FactorRowDto[];
  portfolioReturn: number;
  explainedReturn: number;
  residual: number;
}
