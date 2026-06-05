// Typed factory functions over src/app/models/* DTOs. Each returns a minimal
// schema-valid object and accepts a Partial `overrides` merged last. Plural
// factories (…Metrics/…Scenarios) return a one-element array; their `overrides`
// merge into that single element so callers tweak one field without rebuilding
// the row. For object factories that embed a nested array (sectors / trades /
// entries / assets), `overrides` patches top-level keys only — to change an
// inner row, pass the whole array key (e.g. `{ entries: [...] }`).
// Keep factories < 10 lines — extend by adding a field, not a branch.

import type { PortfolioDto, SnapshotDto } from '../app/models/portfolio-api.model';
import type {
  BrinsonApiResponse,
  FactorAttributionApiResponse,
} from '../app/models/attribution.model';
import type {
  ConcentrationMetric,
  CorrelationData,
  LiquidityMetric,
  StressScenario,
  VarApiResponse,
} from '../app/models/risk.model';
import type {
  DriftApiResponse,
  RebalanceDecideApiResponse,
  RebalancePreviewApiResponse,
} from '../app/models/rebalancing.model';
import type { CMASet, FactorICReport, TAASignal } from '../app/models/factor.model';

const ISO = '2026-01-01T00:00:00.000Z';

export function makePortfolioDto(overrides: Partial<PortfolioDto> = {}): PortfolioDto {
  return {
    id: 'pf-1',
    name: 'Test Portfolio',
    description: null,
    currency: 'EUR',
    benchmark_ticker: 'SPY',
    is_active: true,
    created_at: ISO,
    updated_at: ISO,
    ...overrides,
  };
}

export function makeSnapshotDto(overrides: Partial<SnapshotDto> = {}): SnapshotDto {
  return {
    id: 'snap-1',
    portfolio_id: 'pf-1',
    snapshot_date: '2026-01-01',
    snapshot_type: 'optimization',
    weights: { AAPL: 0.6, MSFT: 0.4 },
    sector_mapping: null,
    summary: null,
    optimizer_config: null,
    turnover: null,
    holding_count: 2,
    created_at: ISO,
    ...overrides,
  };
}

export function makeBrinsonResponse(
  overrides: Partial<BrinsonApiResponse> = {},
): BrinsonApiResponse {
  return {
    sectors: [
      {
        sector: 'Technology',
        portfolioWeight: 0.6,
        benchmarkWeight: 0.5,
        portfolioReturn: 0.1,
        benchmarkReturn: 0.08,
        allocationEffect: 0.002,
        selectionEffect: 0.001,
        interactionEffect: 0.0,
        totalEffect: 0.003,
      },
    ],
    totalAllocation: 0.002,
    totalSelection: 0.001,
    totalInteraction: 0.0,
    totalActiveReturn: 0.003,
    portfolioReturn: 0.1,
    benchmarkReturn: 0.08,
    ...overrides,
  };
}

export function makeFactorAttributionResponse(
  overrides: Partial<FactorAttributionApiResponse> = {},
): FactorAttributionApiResponse {
  return {
    factors: [
      { factorName: 'momentum', exposure: 0.5, factorReturn: 0.04, contribution: 0.02 },
    ],
    portfolioReturn: 0.1,
    explainedReturn: 0.08,
    residual: 0.02,
    ...overrides,
  };
}

export function makeVarApiResponse(overrides: Partial<VarApiResponse> = {}): VarApiResponse {
  return {
    var: { '0.95': 0.03 },
    cvar: { '0.95': 0.05 },
    method: 'historical',
    lookback: 252,
    nObservations: 252,
    ...overrides,
  };
}

export function makeCorrelationData(overrides: Partial<CorrelationData> = {}): CorrelationData {
  return {
    assets: ['AAPL', 'MSFT'],
    matrix: [
      [1, 0.5],
      [0.5, 1],
    ],
    ...overrides,
  };
}

export function makeConcentrationMetrics(
  overrides: Partial<ConcentrationMetric> = {},
): ConcentrationMetric[] {
  return [
    {
      ticker: 'AAPL',
      name: 'Apple Inc.',
      weight: 0.6,
      riskContribution: 0.7,
      componentVar: 0.04,
      ...overrides,
    },
  ];
}

export function makeLiquidityMetrics(
  overrides: Partial<LiquidityMetric> = {},
): LiquidityMetric[] {
  return [
    {
      ticker: 'AAPL',
      name: 'Apple Inc.',
      avgDailyVolume: 1_000_000,
      daysToLiquidate: 1.5,
      liquidityCost: 0.001,
      weight: 0.6,
      ...overrides,
    },
  ];
}

export function makeStressScenarios(
  overrides: Partial<StressScenario> = {},
): StressScenario[] {
  return [
    {
      id: 'scn-1',
      name: 'Rate Shock',
      description: '+100bps parallel shift',
      portfolioImpact: -0.08,
      benchmarkImpact: -0.06,
      worstAsset: 'TLT',
      worstAssetImpact: -0.15,
      ...overrides,
    },
  ];
}

export function makeDriftResponse(overrides: Partial<DriftApiResponse> = {}): DriftApiResponse {
  return {
    entries: [
      { ticker: 'AAPL', name: 'Apple Inc.', target: 0.6, actual: 0.65, drift: 0.05, breached: false },
    ],
    totalDrift: 0.05,
    breachedCount: 0,
    threshold: 0.05,
    ...overrides,
  };
}

export function makeRebalancePreview(
  overrides: Partial<RebalancePreviewApiResponse> = {},
): RebalancePreviewApiResponse {
  return {
    portfolioName: 'Test Portfolio',
    policyType: 'threshold',
    targetWeights: { AAPL: 0.6, MSFT: 0.4 },
    currentWeights: { AAPL: 0.65, MSFT: 0.35 },
    trades: [{ ticker: 'AAPL', weightDelta: -0.05, side: 'sell', shares: 10 }],
    portfolioValue: 100_000,
    ...overrides,
  };
}

export function makeRebalanceDecideResponse(
  overrides: Partial<RebalanceDecideApiResponse> = {},
): RebalanceDecideApiResponse {
  return {
    shouldRebalance: true,
    turnover: 0.1,
    estimatedCost: 0.0005,
    tradeWeights: { AAPL: -0.05, MSFT: 0.05 },
    ...overrides,
  };
}

export function makeFactorICReport(overrides: Partial<FactorICReport> = {}): FactorICReport {
  return {
    factor: 'momentum_12_1',
    group: 'momentum',
    ic: 0.05,
    icir: 0.8,
    tStat: 2.5,
    pValue: 0.01,
    vif: 1.2,
    significant: true,
    ...overrides,
  };
}

export function makeCMASet(overrides: Partial<CMASet> = {}): CMASet {
  return {
    label: 'Base CMA',
    horizon: '10Y',
    assets: [{ ticker: 'AAPL', expectedReturn: 0.07, expectedVol: 0.2 }],
    ...overrides,
  };
}

export function makeTAASignal(overrides: Partial<TAASignal> = {}): TAASignal {
  return {
    factor: 'momentum',
    currentWeight: 0.2,
    tiltedWeight: 0.25,
    tiltReason: 'Expansion regime favours momentum',
    regime: 'expansion',
    ...overrides,
  };
}
