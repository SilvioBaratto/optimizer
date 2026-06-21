// Contract-parity: each frontend fixture's key set must equal the wire property
// set declared in the committed API JSON-schema snapshot. A backend field rename
// (after the Python snapshot regenerates) fails the matching domain block here.
// Batch A — domains with existing fixture factories. Follow-up issues append
// their domains below using the same `schemaOf` + `assertParity` harness.

import backtestSnapshot from './contract-snapshots/backtest.json';
import dashboardSnapshot from './contract-snapshots/dashboard.json';
import jobsSnapshot from './contract-snapshots/jobs.json';
import macroSnapshot from './contract-snapshots/macro.json';
import marketDataSnapshot from './contract-snapshots/market_data.json';
import optimizationSnapshot from './contract-snapshots/optimization.json';
import portfolioSnapshot from './contract-snapshots/portfolio.json';
import reportsSnapshot from './contract-snapshots/reports.json';
import scenariosSnapshot from './contract-snapshots/scenarios.json';
import universeSnapshot from './contract-snapshots/universe.json';
import viewsSnapshot from './contract-snapshots/views.json';
import { assertParity, schemaOf, type JsonSchema } from './contract-parity';
import {
  makeAllocationResponse,
  makeAssetClassReturnsResponse,
  makeBacktestAsyncResponse,
  makeBacktestProgressResponse,
  makeBacktestRunResponse,
  makeDriftResponse,
  makeDriftResponseRich,
  makeEntropyPoolingResponse,
  makeEquityCurveResponse,
  makeGenerateViewsResponse,
  makeInstrumentListResponse,
  makeJobListResponse,
  makeJobSummary,
  makeMacroCalibrationApiResponse,
  makeMarketSnapshotResponse,
  makeOpinionPoolResponse,
  makeOptimizationRunListResponse,
  makeOptimizationRunResponse,
  makePerformanceMetricsResponse,
  makePortfolioDto,
  makePriceHistoryResponse,
  makeReportJobCreateResponse,
  makeRollingMetricsResponse,
  makeSnapshotDto,
  makeStressScenarioApiResponse,
  makeStressScenarioItemApi,
  makeTickerProfileResponse,
  makeUniverseScreenResponse,
  makeUniverseStatsResponse,
} from './domain-fixtures';

// `assertParity` is jasmine-free (it throws) so it compiles in the production
// build, which type-checks every non-spec file under src/. Wrap it here so each
// spec registers a real Jasmine expectation — otherwise every parity spec
// reports "has no expectations".
function expectParity(schema: JsonSchema, fixture: object): void {
  expect(() => assertParity(schema, fixture)).not.toThrow();
}

describe('contract-parity: dashboard', () => {
  it('when DriftResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(dashboardSnapshot, 'DriftResponse'), makeDriftResponse());
  });

  it('when MarketSnapshotResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(dashboardSnapshot, 'MarketSnapshotResponse'),
      makeMarketSnapshotResponse(),
    );
  });

  it('when PerformanceMetricsResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(dashboardSnapshot, 'PerformanceMetricsResponse'),
      makePerformanceMetricsResponse(),
    );
  });

  it('when EquityCurveResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(dashboardSnapshot, 'EquityCurveResponse'), makeEquityCurveResponse());
  });

  it('when AllocationResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(dashboardSnapshot, 'AllocationResponse'), makeAllocationResponse());
  });

  it('when AssetClassReturnsResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(dashboardSnapshot, 'AssetClassReturnsResponse'),
      makeAssetClassReturnsResponse(),
    );
  });

  it('when RollingMetricsResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(dashboardSnapshot, 'RollingMetricsResponse'),
      makeRollingMetricsResponse(),
    );
  });
});

describe('contract-parity: portfolio', () => {
  it('when PortfolioResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(portfolioSnapshot, 'PortfolioResponse'), makePortfolioDto());
  });

  it('when SnapshotResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(portfolioSnapshot, 'SnapshotResponse'), makeSnapshotDto());
  });

  // DriftResponse: the rich portfolio drift shape (holdings/target/drift/trades/totals/
  // diagnostics/request_id), distinct from the dashboard simple DriftResponse (entries[]).
  it('when the portfolio rich DriftResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(portfolioSnapshot, 'DriftResponse'), makeDriftResponseRich());
  });
});

// ── Batch B (#847): jobs, reports, universe, scenarios, market_data ──────────

describe('contract-parity: jobs', () => {
  it('when JobListResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(jobsSnapshot, 'JobListResponse'), makeJobListResponse());
  });

  it('when JobSummary is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(jobsSnapshot, 'JobSummary'), makeJobSummary());
  });
});

describe('contract-parity: reports', () => {
  it('when ReportJobCreateResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(reportsSnapshot, 'ReportJobCreateResponse'),
      makeReportJobCreateResponse(),
    );
  });
});

describe('contract-parity: universe', () => {
  it('when UniverseScreenResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(universeSnapshot, 'UniverseScreenResponse'),
      makeUniverseScreenResponse(),
    );
  });
});

describe('contract-parity: scenarios', () => {
  it('when StressScenarioResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(scenariosSnapshot, 'StressScenarioResponse'),
      makeStressScenarioApiResponse(),
    );
  });

  it('when StressScenarioItem is checked, the wire-only fixture matches the wire row', () => {
    expectParity(schemaOf(scenariosSnapshot, 'StressScenarioItem'), makeStressScenarioItemApi());
  });
});

describe('contract-parity: market_data', () => {
  it('when PriceHistoryResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(marketDataSnapshot, 'PriceHistoryResponse'),
      makePriceHistoryResponse(),
    );
  });
});

// ── Batch C (#848): optimization, backtest, factors, views, macro ─────────────

describe('contract-parity: optimization', () => {
  it('when OptimizationRunResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(optimizationSnapshot, 'OptimizationRunResponse'),
      makeOptimizationRunResponse(),
    );
  });

  it('when OptimizationRunListResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(optimizationSnapshot, 'OptimizationRunListResponse'),
      makeOptimizationRunListResponse(),
    );
  });
});

describe('contract-parity: backtest', () => {
  it('when BacktestRunResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(backtestSnapshot, 'BacktestRunResponse'), makeBacktestRunResponse());
  });

  it('when BacktestProgressResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(backtestSnapshot, 'BacktestProgressResponse'),
      makeBacktestProgressResponse(),
    );
  });

  // POST /api/v1/backtest returns a raw JSONResponse with no Pydantic
  // response_model, so it has no snapshot. Pin the documented manual shape by
  // key-set instead — a frontend rename of the async DTO fails here.
  it('when the schema-less POST /backtest async response is checked, its keys are pinned', () => {
    expect(Object.keys(makeBacktestAsyncResponse()).sort()).toEqual([
      'jobId',
      'message',
      'runId',
      'status',
    ]);
  });
});

describe('contract-parity: views', () => {
  it('when GenerateViewsResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(viewsSnapshot, 'GenerateViewsResponse'), makeGenerateViewsResponse());
  });

  it('when OpinionPoolResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(viewsSnapshot, 'OpinionPoolResponse'), makeOpinionPoolResponse());
  });

  it('when EntropyPoolingResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(viewsSnapshot, 'EntropyPoolingResponse'),
      makeEntropyPoolingResponse(),
    );
  });
});

describe('contract-parity: macro', () => {
  it('when MacroCalibrationResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(macroSnapshot, 'MacroCalibrationResponse'),
      makeMacroCalibrationApiResponse(),
    );
  });
});

// ── Batch D (#1029): remaining DTOs — closes out all uncovered response types ─
// Covers: InstrumentListResponse, UniverseStatsResponse, TickerProfileResponse.

describe('contract-parity: universe (extended #1029)', () => {
  it('when InstrumentListResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(universeSnapshot, 'InstrumentListResponse'),
      makeInstrumentListResponse(),
    );
  });

  it('when UniverseStatsResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(universeSnapshot, 'UniverseStatsResponse'),
      makeUniverseStatsResponse(),
    );
  });
});

describe('contract-parity: market_data (extended #1029)', () => {
  it('when TickerProfileResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(marketDataSnapshot, 'TickerProfileResponse'),
      makeTickerProfileResponse(),
    );
  });
});

