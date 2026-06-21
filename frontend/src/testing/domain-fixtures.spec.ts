import {
  makeDriftResponse,
  makeJobListResponse,
  makeJobSummary,
  makeMarketSnapshotResponse,
  makePortfolioDto,
  makePriceHistoryResponse,
  makeReportJobCreateResponse,
  makeSnapshotDto,
  makeStressScenarioApiResponse,
  makeStressScenarioItemApi,
  makeUniverseScreenResponse,
  makeBacktestAsyncResponse,
  makeBacktestProgressResponse,
  makeBacktestRunResponse,
  makeDriftResponseRich,
  makeEntropyPoolingResponse,
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

const objectCases: readonly ObjectCase[] = [
  objCase('makePortfolioDto', makePortfolioDto, 'name', 'Custom PF'),
  objCase('makeSnapshotDto', makeSnapshotDto, 'holding_count', 7),
  objCase('makeMarketSnapshotResponse', makeMarketSnapshotResponse, 'vix', 30),
  objCase('makeDriftResponse', makeDriftResponse, 'breachedCount', 3),
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

describe('domain-fixtures nested shapes', () => {
  it('when makeDriftResponse is called, it carries one drift entry', () => {
    expect(makeDriftResponse().entries.length).toBe(1);
  });
});
