// Contract-parity: each frontend fixture's key set must equal the wire property
// set declared in the committed API JSON-schema snapshot. A backend field rename
// (after the Python snapshot regenerates) fails the matching domain block here.
// Batch A — domains with existing fixture factories. Follow-up issues append
// their domains below using the same `schemaOf` + `assertParity` harness.

import attributionSnapshot from './contract-snapshots/attribution.json';
import backtestSnapshot from './contract-snapshots/backtest.json';
import dashboardSnapshot from './contract-snapshots/dashboard.json';
import factorsSnapshot from './contract-snapshots/factors.json';
import jobsSnapshot from './contract-snapshots/jobs.json';
import macroSnapshot from './contract-snapshots/macro.json';
import marketDataSnapshot from './contract-snapshots/market_data.json';
import optimizationSnapshot from './contract-snapshots/optimization.json';
import pipelineBuilderSnapshot from './contract-snapshots/pipeline_builder.json';
import portfolioSnapshot from './contract-snapshots/portfolio.json';
import rebalancingSnapshot from './contract-snapshots/rebalancing.json';
import reportsSnapshot from './contract-snapshots/reports.json';
import riskSnapshot from './contract-snapshots/risk.json';
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
  makeBrinsonResponse,
  makeConcentrationApiResponse,
  makeConcentrationAssetApi,
  makeCorrelationApiResponse,
  makeCreateSessionResponse,
  makeDriftResponse,
  makeDriftResponseRich,
  makeEntropyPoolingResponse,
  makeEquityCurveResponse,
  makeFactorAttributionResponse,
  makeFactorCompositeResponse,
  makeFactorExposureResponse,
  makeFactorScoreDto,
  makeFactorSelectResponse,
  makeGenerateViewsResponse,
  makeInstrumentListResponse,
  makeJobListResponse,
  makeJobSummary,
  makeLiquidityApiResponse,
  makeLiquidityMetrics,
  makeMacroCalibrationApiResponse,
  makeMarketSnapshotResponse,
  makeOpinionPoolResponse,
  makeOptimizationRunListResponse,
  makeOptimizationRunResponse,
  makePerformanceMetricsResponse,
  makePortfolioDto,
  makePriceHistoryResponse,
  makeRebalanceDecideResponse,
  makeRebalancePreview,
  makeRebalancingPolicyListResponse,
  makeReportJobCreateResponse,
  makeRiskLimitListResponse,
  makeRollingMetricsResponse,
  makeSnapshotDto,
  makeStepPollResponse,
  makeStepRunWireResponse,
  makeStressScenarioApiResponse,
  makeStressScenarioItemApi,
  makeTickerProfileResponse,
  makeUniverseScreenResponse,
  makeUniverseStatsResponse,
  makeVarApiResponse,
} from './domain-fixtures';

// ── Batch D (#1029) inline fixtures ──────────────────────────────────────────
// Used for DTOs that lack a domain-fixtures factory.

const FACTOR_SCORE_LIST_FIXTURE = { items: [makeFactorScoreDto()], total: 1 };

// FactorValidationReportResponse — camelCase CamelCaseModel (distinct from the
// snake_case FactorValidateResponse used in earlier parity tasks).
const FACTOR_VALIDATION_REPORT_FIXTURE = {
  id: 'fvr-1',
  factorType: 'momentum_12_1',
  reportDate: '2026-01-01',
  validationType: 'in_sample',
  icMean: 0.05,
  icStd: 0.1,
  icir: 0.5,
  tStat: 2.4,
  pValue: 0.01,
  vif: 1.2,
  details: null,
  createdAt: '2026-01-01T00:00:00.000Z',
  updatedAt: '2026-01-01T00:00:00.000Z',
};

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

describe('contract-parity: risk', () => {
  it('when VarResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(riskSnapshot, 'VarResponse'), makeVarApiResponse());
  });

  it('when ConcentrationAsset is checked, the wire fixture matches the wire row', () => {
    expectParity(schemaOf(riskSnapshot, 'ConcentrationAsset'), makeConcentrationAssetApi());
  });

  it('when CorrelationResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(riskSnapshot, 'CorrelationResponse'), makeCorrelationApiResponse());
  });

  it('when LiquidityAsset is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(riskSnapshot, 'LiquidityAsset'), makeLiquidityMetrics()[0]);
  });

  it('when FactorExposureResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(riskSnapshot, 'FactorExposureResponse'),
      makeFactorExposureResponse(),
    );
  });

  it('when RiskLimitListResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(riskSnapshot, 'RiskLimitListResponse'),
      makeRiskLimitListResponse(),
    );
  });
});

describe('contract-parity: attribution', () => {
  it('when BrinsonResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(attributionSnapshot, 'BrinsonResponse'), makeBrinsonResponse());
  });

  it('when FactorAttributionResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(attributionSnapshot, 'FactorAttributionResponse'),
      makeFactorAttributionResponse(),
    );
  });
});

describe('contract-parity: rebalancing', () => {
  it('when RebalanceDecideResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(rebalancingSnapshot, 'RebalanceDecideResponse'),
      makeRebalanceDecideResponse(),
    );
  });

  it('when RebalancePreviewResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(rebalancingSnapshot, 'RebalancePreviewResponse'),
      makeRebalancePreview(),
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

// ── Batch C (#848): optimization, backtest, factors, views, macro,
// pipeline_builder — nested/mismatched shapes + real drift fixes. ────────────

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

describe('contract-parity: factors', () => {
  it('when FactorScoreResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(factorsSnapshot, 'FactorScoreResponse'), makeFactorScoreDto());
  });

  it('when FactorCompositeScoreResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(factorsSnapshot, 'FactorCompositeScoreResponse'),
      makeFactorCompositeResponse(),
    );
  });

  it('when FactorSelectResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(factorsSnapshot, 'FactorSelectResponse'), makeFactorSelectResponse());
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

describe('contract-parity: pipeline_builder', () => {
  it('when CreateSessionResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(pipelineBuilderSnapshot, 'CreateSessionResponse'),
      makeCreateSessionResponse(),
    );
  });

  it('when StepPollResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(pipelineBuilderSnapshot, 'StepPollResponse'), makeStepPollResponse());
  });

  it('when StepRunResponse is checked, the wire fixture matches (frontend remaps to jobId)', () => {
    expectParity(schemaOf(pipelineBuilderSnapshot, 'StepRunResponse'), makeStepRunWireResponse());
  });

  // DriftResponse collision: pipeline_builder/builder-drift consumes the rich
  // portfolio DriftResponse (holdings/target/drift/trades/totals/diagnostics/
  // request_id), distinct from the dashboard simple DriftResponse (entries[]).
  it('when the portfolio rich DriftResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(schemaOf(portfolioSnapshot, 'DriftResponse'), makeDriftResponseRich());
  });
});

// ── Batch D (#1029): remaining DTOs — closes out all uncovered response types ─
// Covers: RebalancingPolicyListResponse, FactorScoreListResponse,
// FactorValidationReportResponse, InstrumentListResponse, UniverseStatsResponse,
// TickerProfileResponse, ConcentrationResponse, LiquidityResponse.

describe('contract-parity: rebalancing (extended #1029)', () => {
  it('when RebalancingPolicyListResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(rebalancingSnapshot, 'RebalancingPolicyListResponse'),
      makeRebalancingPolicyListResponse(),
    );
  });
});

describe('contract-parity: factors (extended #1029)', () => {
  it('when FactorScoreListResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(factorsSnapshot, 'FactorScoreListResponse'),
      FACTOR_SCORE_LIST_FIXTURE,
    );
  });

  it('when FactorValidationReportResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(factorsSnapshot, 'FactorValidationReportResponse'),
      FACTOR_VALIDATION_REPORT_FIXTURE,
    );
  });
});

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

describe('contract-parity: risk (extended #1029)', () => {
  it('when ConcentrationResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(riskSnapshot, 'ConcentrationResponse'),
      makeConcentrationApiResponse(),
    );
  });

  it('when LiquidityResponse is checked, fixture keys equal the wire properties', () => {
    expectParity(
      schemaOf(riskSnapshot, 'LiquidityResponse'),
      makeLiquidityApiResponse(),
    );
  });
});
