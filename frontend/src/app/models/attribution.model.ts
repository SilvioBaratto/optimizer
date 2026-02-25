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
