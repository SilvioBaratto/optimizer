import {
  makeBrinsonResponse,
  makeCMASet,
  makeConcentrationAssetApi,
  makeConcentrationMetrics,
  makeCorrelationApiResponse,
  makeCorrelationData,
  makeDriftResponse,
  makeFactorAttributionResponse,
  makeFactorICReport,
  makeJobListResponse,
  makeJobSummary,
  makeLiquidityMetrics,
  makeMarketSnapshotResponse,
  makePortfolioDto,
  makePriceHistoryResponse,
  makeRebalanceDecideResponse,
  makeRebalancePreview,
  makeReportJobCreateResponse,
  makeSnapshotDto,
  makeStressScenarioApiResponse,
  makeStressScenarioItemApi,
  makeStressScenarios,
  makeTAASignal,
  makeUniverseScreenResponse,
  makeVarApiResponse,
  makeBacktestAsyncResponse,
  makeBacktestProgressResponse,
  makeBacktestRunResponse,
  makeDriftResponseRich,
  makeEntropyPoolingResponse,
  makeFactorCompositeResponse,
  makeFactorScoreDto,
  makeFactorSelectResponse,
  makeGenerateViewsResponse,
  makeMacroCalibrationApiResponse,
  makeOpinionPoolResponse,
  makeOptimizationRunListResponse,
  makeOptimizationRunResponse,
  makeStepPollResponse,
  makeStepRunWireResponse,
} from './index';

type Row = Record<string, unknown>;

interface ObjectCase {
  name: string;
  make: (o?: Row) => Row;
  key: string;
  value: unknown;
}

interface ArrayCase {
  name: string;
  make: (o?: Row) => Row[];
  key: string;
  value: unknown;
}

// Typed builders erase the per-factory Partial<T> generic so the cases tabulate
// uniformly without `any` and without 15 inline casts.
function objCase<T extends object>(
  name: string,
  make: (o?: Partial<T>) => T,
  key: keyof T,
  value: T[keyof T],
): ObjectCase {
  return { name, make: (o?: Row) => make(o as Partial<T>) as Row, key: key as string, value };
}

function arrCase<T extends object>(
  name: string,
  make: (o?: Partial<T>) => T[],
  key: keyof T,
  value: T[keyof T],
): ArrayCase {
  return { name, make: (o?: Row) => make(o as Partial<T>) as Row[], key: key as string, value };
}

const objectCases: readonly ObjectCase[] = [
  objCase('makePortfolioDto', makePortfolioDto, 'name', 'Custom PF'),
  objCase('makeSnapshotDto', makeSnapshotDto, 'holding_count', 7),
  objCase('makeBrinsonResponse', makeBrinsonResponse, 'portfolioReturn', 0.42),
  objCase('makeFactorAttributionResponse', makeFactorAttributionResponse, 'residual', 0.99),
  objCase('makeVarApiResponse', makeVarApiResponse, 'method', 'parametric'),
  objCase('makeCorrelationData', makeCorrelationData, 'assets', ['X']),
  objCase('makeCorrelationApiResponse', makeCorrelationApiResponse, 'clusterLabels', [1, 2]),
  objCase('makeConcentrationAssetApi', makeConcentrationAssetApi, 'weight', 0.9),
  objCase('makeMarketSnapshotResponse', makeMarketSnapshotResponse, 'vix', 30),
  objCase('makeDriftResponse', makeDriftResponse, 'breachedCount', 3),
  objCase('makeRebalancePreview', makeRebalancePreview, 'policyType', 'hybrid'),
  objCase('makeRebalanceDecideResponse', makeRebalanceDecideResponse, 'shouldRebalance', false),
  objCase('makeFactorICReport', makeFactorICReport, 'significant', false),
  objCase('makeCMASet', makeCMASet, 'label', 'Stress CMA'),
  objCase('makeTAASignal', makeTAASignal, 'regime', 'recession'),
  objCase('makeJobSummary', makeJobSummary, 'status', 'failed'),
  objCase('makeJobListResponse', makeJobListResponse, 'total', 9),
  objCase('makeReportJobCreateResponse', makeReportJobCreateResponse, 'status', 'completed'),
  objCase('makeUniverseScreenResponse', makeUniverseScreenResponse, 'totalScreened', 50),
  objCase('makeStressScenarioApiResponse', makeStressScenarioApiResponse, 'nScenarios', 3),
  objCase('makeStressScenarioItemApi', makeStressScenarioItemApi, 'name', 'FX Crisis'),
  objCase('makePriceHistoryResponse', makePriceHistoryResponse, 'close', 200),
  objCase('makeOptimizationRunResponse', makeOptimizationRunResponse, 'status', 'failed'),
  objCase('makeOptimizationRunListResponse', makeOptimizationRunListResponse, 'total', 5),
  objCase('makeBacktestRunResponse', makeBacktestRunResponse, 'status', 'failed'),
  objCase('makeBacktestProgressResponse', makeBacktestProgressResponse, 'current', 9),
  objCase('makeBacktestAsyncResponse', makeBacktestAsyncResponse, 'status', 'running'),
  objCase('makeFactorScoreDto', makeFactorScoreDto, 'ticker', 'MSFT'),
  objCase('makeFactorCompositeResponse', makeFactorCompositeResponse, 'score_date', '2026-02-01'),
  objCase('makeFactorSelectResponse', makeFactorSelectResponse, 'count', 5),
  objCase('makeGenerateViewsResponse', makeGenerateViewsResponse, 'nViews', 3),
  objCase('makeOpinionPoolResponse', makeOpinionPoolResponse, 'poolingType', 'geometric'),
  objCase('makeEntropyPoolingResponse', makeEntropyPoolingResponse, 'mu', [0.2]),
  objCase('makeMacroCalibrationApiResponse', makeMacroCalibrationApiResponse, 'phase', 'SLOWDOWN'),
  // makeCreateSessionResponse (single key `sessionId`) is covered by the parity
  // spec; it cannot satisfy the generic ≥2-key objectCases assertions below.
  objCase('makeStepPollResponse', makeStepPollResponse, 'status', 'completed'),
  objCase('makeStepRunWireResponse', makeStepRunWireResponse, 'status', 'completed'),
  objCase('makeDriftResponseRich', makeDriftResponseRich, 'request_id', 9),
];

describe('domain-fixtures object factories', () => {
  for (const { name, make } of objectCases) {
    it(`when ${name} is called with no args, a populated object is returned`, () => {
      const result = make();
      expect(result).toBeTruthy();
      expect(Object.keys(result).length).toBeGreaterThan(1);
    });
  }

  for (const { name, make, key, value } of objectCases) {
    it(`when ${name} is given an override, the override value wins`, () => {
      const base = make();
      const result = make({ [key]: value });
      expect(result[key]).toEqual(value);
      const untouched = Object.keys(base).filter((k) => k !== key);
      expect(untouched.length).toBeGreaterThan(0);
      expect(result[untouched[0]]).toEqual(base[untouched[0]]);
    });
  }
});

const arrayCases: readonly ArrayCase[] = [
  arrCase('makeConcentrationMetrics', makeConcentrationMetrics, 'weight', 0.5),
  arrCase('makeLiquidityMetrics', makeLiquidityMetrics, 'daysToLiquidate', 9),
  arrCase('makeStressScenarios', makeStressScenarios, 'name', 'FX Crisis'),
];

describe('domain-fixtures array factories', () => {
  for (const { name, make } of arrayCases) {
    it(`when ${name} is called with no args, a one-element array is returned`, () => {
      const result = make();
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(1);
      expect(Object.keys(result[0]).length).toBeGreaterThan(1);
    });
  }

  for (const { name, make, key, value } of arrayCases) {
    it(`when ${name} is given an override, the override merges into the element`, () => {
      const result = make({ [key]: value });
      expect(result[0][key]).toEqual(value);
    });
  }
});

describe('domain-fixtures nested shapes', () => {
  it('when makeBrinsonResponse is called, it carries one sector row', () => {
    expect(makeBrinsonResponse().sectors.length).toBe(1);
  });

  it('when makeRebalancePreview is called, it carries one trade item', () => {
    expect(makeRebalancePreview().trades.length).toBe(1);
  });

  it('when makeDriftResponse is called, it carries one drift entry', () => {
    expect(makeDriftResponse().entries.length).toBe(1);
  });

  it('when makeCMASet is called, it carries one asset row', () => {
    expect(makeCMASet().assets.length).toBe(1);
  });

  it('when makeVarApiResponse is called, var and cvar are keyed records', () => {
    expect(makeVarApiResponse().var['0.95']).toBeCloseTo(0.03);
    expect(makeVarApiResponse().cvar['0.95']).toBeCloseTo(0.05);
  });
});
