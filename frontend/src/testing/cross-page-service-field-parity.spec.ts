/**
 * cross-page-service-field-parity.spec.ts — issue #968
 *
 * Field-level contract-parity for analytics-page and shared services not
 * covered by earlier parity specs (settings, portfolio-builder, research).
 *
 * Domains covered:
 *   attribution:    BrinsonResponse, FactorAttributionResponse
 *   risk:           ConcentrationResponse, CorrelationResponse, LiquidityResponse,
 *                   RiskLimitListResponse, VarResponse
 *   rebalancing:    RebalanceDecideResponse, RebalancePreviewResponse,
 *                   RebalancingPolicyListResponse
 *   dashboard:      AllocationResponse, AssetClassReturnsResponse, DriftResponse,
 *                   EquityCurveResponse, MarketSnapshotResponse, PerformanceMetricsResponse
 *   macro:          MacroCalibrationResponse
 *   market_data:    PriceHistoryResponse, TickerProfileResponse
 *   universe:       InstrumentListResponse, UniverseScreenResponse, UniverseStatsResponse
 *   reports:        ReportJobCreateResponse
 *   scenarios:      StressScenarioItem, StressScenarioResponse
 *
 * Convention: pass `schema` as the assertFieldParity root when the schema
 * has embedded `$defs` (so #/$defs/... refs resolve within the schema entry).
 * Pass `snapshot as never` when there are no embedded $defs.
 */

import { schemaOf } from './contract-parity';
import { assertFieldParity, requiredKeys } from './contract-field-parity';
import attributionSnapshot from './contract-snapshots/attribution.json';
import riskSnapshot from './contract-snapshots/risk.json';
import rebalancingSnapshot from './contract-snapshots/rebalancing.json';
import dashboardSnapshot from './contract-snapshots/dashboard.json';
import macroSnapshot from './contract-snapshots/macro.json';
import marketDataSnapshot from './contract-snapshots/market_data.json';
import universeSnapshot from './contract-snapshots/universe.json';
import reportsSnapshot from './contract-snapshots/reports.json';
import scenariosSnapshot from './contract-snapshots/scenarios.json';
import {
  makeAllocationResponse,
  makeAssetClassReturnsResponse,
  makeBrinsonResponse,
  makeConcentrationApiResponse,
  makeDriftResponse,
  makeEquityCurveResponse,
  makeFactorAttributionResponse,
  makeInstrumentListResponse,
  makeLiquidityApiResponse,
  makeMacroCalibrationApiResponse,
  makeMarketSnapshotResponse,
  makePerformanceMetricsResponse,
  makePriceHistoryResponse,
  makeRebalanceDecideResponse,
  makeRebalancePreview,
  makeRebalancingPolicyListResponse,
  makeReportJobCreateResponse,
  makeRiskLimitListResponse,
  makeStressScenarioApiResponse,
  makeStressScenarioItemApi,
  makeCorrelationApiResponse,
  makeTickerProfileResponse,
  makeUniverseScreenResponse,
  makeUniverseStatsResponse,
  makeVarApiResponse,
} from './domain-fixtures';

// ═══════════════════════════════════════════════════════════════════════════════
// Attribution
// ═══════════════════════════════════════════════════════════════════════════════

describe('BrinsonResponse — camelCase (attribution.json, issue #968)', () => {
  const schema = schemaOf(attributionSnapshot, 'BrinsonResponse');

  it('required fields: sectors, totalAllocation, totalSelection, totalInteraction, totalActiveReturn, portfolioReturn, benchmarkReturn', () => {
    const req = requiredKeys(schema);
    [
      'sectors', 'totalAllocation', 'totalSelection', 'totalInteraction',
      'totalActiveReturn', 'portfolioReturn', 'benchmarkReturn',
    ].forEach((k) => expect(req.has(k)).withContext(`expected '${k}' in required`).toBe(true));
  });

  it('camelCase casing: totalAllocation (not total_allocation), totalActiveReturn (not total_active_return)', () => {
    const props = schema.properties ?? {};
    expect('totalAllocation' in props).toBe(true);
    expect('total_allocation' in props).toBe(false);
    expect('totalActiveReturn' in props).toBe(true);
    expect('total_active_return' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot (schema as root for $defs)', () => {
    expect(() => assertFieldParity(schema, makeBrinsonResponse(), schema)).not.toThrow();
  });
});

describe('FactorAttributionResponse — camelCase (attribution.json, issue #968)', () => {
  const schema = schemaOf(attributionSnapshot, 'FactorAttributionResponse');

  it('required fields: factors, portfolioReturn, explainedReturn, residual', () => {
    const req = requiredKeys(schema);
    ['factors', 'portfolioReturn', 'explainedReturn', 'residual'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}' in required`).toBe(true),
    );
  });

  it('camelCase: explainedReturn (not explained_return), portfolioReturn (not portfolio_return)', () => {
    const props = schema.properties ?? {};
    expect('explainedReturn' in props).toBe(true);
    expect('explained_return' in props).toBe(false);
    expect('portfolioReturn' in props).toBe(true);
    expect('portfolio_return' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot (schema as root for $defs)', () => {
    expect(() => assertFieldParity(schema, makeFactorAttributionResponse(), schema)).not.toThrow();
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Risk
// ═══════════════════════════════════════════════════════════════════════════════

describe('VarResponse — camelCase (risk.json, issue #968)', () => {
  const schema = schemaOf(riskSnapshot, 'VarResponse');

  it('required fields: var, cvar, method, lookback, nObservations', () => {
    const req = requiredKeys(schema);
    ['var', 'cvar', 'method', 'lookback', 'nObservations'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('camelCase: nObservations (not n_observations)', () => {
    const props = schema.properties ?? {};
    expect('nObservations' in props).toBe(true);
    expect('n_observations' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot', () => {
    expect(() =>
      assertFieldParity(schema, makeVarApiResponse(), riskSnapshot as never),
    ).not.toThrow();
  });
});

describe('CorrelationResponse — camelCase (risk.json, issue #968)', () => {
  const schema = schemaOf(riskSnapshot, 'CorrelationResponse');

  it('required fields: assets, matrix, clusterLabels', () => {
    const req = requiredKeys(schema);
    ['assets', 'matrix', 'clusterLabels'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('camelCase: clusterLabels (not cluster_labels)', () => {
    const props = schema.properties ?? {};
    expect('clusterLabels' in props).toBe(true);
    expect('cluster_labels' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot', () => {
    expect(() =>
      assertFieldParity(schema, makeCorrelationApiResponse(), riskSnapshot as never),
    ).not.toThrow();
  });
});

describe('ConcentrationResponse — camelCase (risk.json, issue #968)', () => {
  const schema = schemaOf(riskSnapshot, 'ConcentrationResponse');

  it('required fields: assets, summary', () => {
    const req = requiredKeys(schema);
    ['assets', 'summary'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('summary has camelCase: effectiveN (not effective_n), topNRatio (not top_n_ratio)', () => {
    const summaryProps = schema.$defs?.['ConcentrationSummary']?.properties ?? {};
    expect('effectiveN' in summaryProps).toBe(true);
    expect('effective_n' in summaryProps).toBe(false);
    expect('topNRatio' in summaryProps).toBe(true);
    expect('top_n_ratio' in summaryProps).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot (schema as root for $defs)', () => {
    expect(() =>
      assertFieldParity(schema, makeConcentrationApiResponse(), schema),
    ).not.toThrow();
  });
});

describe('LiquidityResponse — camelCase (risk.json, issue #968)', () => {
  const schema = schemaOf(riskSnapshot, 'LiquidityResponse');

  it('required fields: assets, summary', () => {
    const req = requiredKeys(schema);
    ['assets', 'summary'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('asset rows: camelCase avgDailyVolume (not avg_daily_volume), daysToLiquidate (not days_to_liquidate)', () => {
    const assetProps = schema.$defs?.['LiquidityAsset']?.properties ?? {};
    expect('avgDailyVolume' in assetProps).toBe(true);
    expect('avg_daily_volume' in assetProps).toBe(false);
    expect('daysToLiquidate' in assetProps).toBe(true);
    expect('days_to_liquidate' in assetProps).toBe(false);
  });

  it('summary: camelCase weightedAvgDaysToLiquidate', () => {
    const summaryProps = schema.$defs?.['LiquiditySummary']?.properties ?? {};
    expect('weightedAvgDaysToLiquidate' in summaryProps).toBe(true);
    expect('weighted_avg_days_to_liquidate' in summaryProps).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot (schema as root for $defs)', () => {
    expect(() =>
      assertFieldParity(schema, makeLiquidityApiResponse(), schema),
    ).not.toThrow();
  });
});

describe('RiskLimitListResponse — camelCase (risk.json, issue #968)', () => {
  const schema = schemaOf(riskSnapshot, 'RiskLimitListResponse');

  it('required field: breachCount', () => {
    expect(requiredKeys(schema).has('breachCount')).toBe(true);
  });

  it('camelCase breachCount (not breach_count)', () => {
    const props = schema.properties ?? {};
    expect('breachCount' in props).toBe(true);
    expect('breach_count' in props).toBe(false);
  });

  it('inner RiskLimitResponse required: id, portfolioId, metric, limitType, threshold, isBreached, createdAt, updatedAt', () => {
    const innerReq = requiredKeys(schema.$defs?.['RiskLimitResponse'] ?? {});
    ['id', 'portfolioId', 'metric', 'limitType', 'threshold', 'isBreached', 'createdAt', 'updatedAt'].forEach(
      (k) => expect(innerReq.has(k)).withContext(`expected '${k}' in RiskLimitResponse required`).toBe(true),
    );
  });

  it('assertFieldParity: fixture matches snapshot (schema as root for $defs)', () => {
    expect(() =>
      assertFieldParity(schema, makeRiskLimitListResponse(), schema),
    ).not.toThrow();
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Rebalancing
// ═══════════════════════════════════════════════════════════════════════════════

describe('RebalanceDecideResponse — camelCase (rebalancing.json, issue #968)', () => {
  const schema = schemaOf(rebalancingSnapshot, 'RebalanceDecideResponse');

  it('required fields: shouldRebalance, turnover, estimatedCost, tradeWeights', () => {
    const req = requiredKeys(schema);
    ['shouldRebalance', 'turnover', 'estimatedCost', 'tradeWeights'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('camelCase: shouldRebalance (not should_rebalance), estimatedCost (not estimated_cost)', () => {
    const props = schema.properties ?? {};
    expect('shouldRebalance' in props).toBe(true);
    expect('should_rebalance' in props).toBe(false);
    expect('estimatedCost' in props).toBe(true);
    expect('estimated_cost' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot', () => {
    expect(() =>
      assertFieldParity(schema, makeRebalanceDecideResponse(), rebalancingSnapshot as never),
    ).not.toThrow();
  });
});

describe('RebalancePreviewResponse — camelCase (rebalancing.json, issue #968)', () => {
  const schema = schemaOf(rebalancingSnapshot, 'RebalancePreviewResponse');

  it('required field: portfolioName', () => {
    expect(requiredKeys(schema).has('portfolioName')).toBe(true);
  });

  it('camelCase: portfolioName (not portfolio_name), targetWeights (not target_weights)', () => {
    const props = schema.properties ?? {};
    expect('portfolioName' in props).toBe(true);
    expect('portfolio_name' in props).toBe(false);
    expect('targetWeights' in props).toBe(true);
    expect('target_weights' in props).toBe(false);
  });

  it('inner TradeItem required: ticker, weightDelta, side', () => {
    const itemReq = requiredKeys(schema.$defs?.['TradeItem'] ?? {});
    ['ticker', 'weightDelta', 'side'].forEach((k) =>
      expect(itemReq.has(k)).withContext(`expected '${k}' in TradeItem required`).toBe(true),
    );
  });

  it('assertFieldParity: fixture matches snapshot (schema as root for $defs)', () => {
    expect(() =>
      assertFieldParity(schema, makeRebalancePreview(), schema),
    ).not.toThrow();
  });
});

describe('RebalancingPolicyListResponse — camelCase (rebalancing.json, issue #968)', () => {
  const schema = schemaOf(rebalancingSnapshot, 'RebalancingPolicyListResponse');

  it('required field: total', () => {
    expect(requiredKeys(schema).has('total')).toBe(true);
  });

  it('inner RebalancingPolicyResponse required: id, portfolioId, name, policyType, config, isActive, createdAt, updatedAt', () => {
    const innerReq = requiredKeys(schema.$defs?.['RebalancingPolicyResponse'] ?? {});
    ['id', 'portfolioId', 'name', 'policyType', 'config', 'isActive', 'createdAt', 'updatedAt'].forEach(
      (k) => expect(innerReq.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('camelCase: portfolioId (not portfolio_id), policyType (not policy_type), isActive (not is_active)', () => {
    const innerProps = schema.$defs?.['RebalancingPolicyResponse']?.properties ?? {};
    expect('portfolioId' in innerProps).toBe(true);
    expect('portfolio_id' in innerProps).toBe(false);
    expect('policyType' in innerProps).toBe(true);
    expect('policy_type' in innerProps).toBe(false);
    expect('isActive' in innerProps).toBe(true);
    expect('is_active' in innerProps).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot (schema as root for $defs)', () => {
    expect(() =>
      assertFieldParity(schema, makeRebalancingPolicyListResponse(), schema),
    ).not.toThrow();
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Dashboard
// ═══════════════════════════════════════════════════════════════════════════════

describe('MarketSnapshotResponse — camelCase (dashboard.json, issue #968)', () => {
  const schema = schemaOf(dashboardSnapshot, 'MarketSnapshotResponse');

  it('required fields: vix, vixChange, sp500Return, tenYearYield, yieldChange, usdIndex, usdChange, asOf', () => {
    const req = requiredKeys(schema);
    ['vix', 'vixChange', 'sp500Return', 'tenYearYield', 'yieldChange', 'usdIndex', 'usdChange', 'asOf'].forEach(
      (k) => expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('camelCase: vixChange (not vix_change), sp500Return (not sp500_return)', () => {
    const props = schema.properties ?? {};
    expect('vixChange' in props).toBe(true);
    expect('vix_change' in props).toBe(false);
    expect('sp500Return' in props).toBe(true);
    expect('sp500_return' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot', () => {
    expect(() =>
      assertFieldParity(schema, makeMarketSnapshotResponse(), dashboardSnapshot as never),
    ).not.toThrow();
  });
});

describe('DriftResponse — camelCase (dashboard.json, issue #968)', () => {
  const schema = schemaOf(dashboardSnapshot, 'DriftResponse');

  it('required fields: entries, totalDrift, breachedCount, threshold', () => {
    const req = requiredKeys(schema);
    ['entries', 'totalDrift', 'breachedCount', 'threshold'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('camelCase: totalDrift (not total_drift), breachedCount (not breached_count)', () => {
    const props = schema.properties ?? {};
    expect('totalDrift' in props).toBe(true);
    expect('total_drift' in props).toBe(false);
    expect('breachedCount' in props).toBe(true);
    expect('breached_count' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot (schema as root for $defs)', () => {
    expect(() =>
      assertFieldParity(schema, makeDriftResponse(), schema),
    ).not.toThrow();
  });
});

describe('AllocationResponse — camelCase (dashboard.json, issue #968)', () => {
  const schema = schemaOf(dashboardSnapshot, 'AllocationResponse');

  it('required fields: nodes, totalPositions, totalSectors', () => {
    const req = requiredKeys(schema);
    ['nodes', 'totalPositions', 'totalSectors'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('camelCase: totalPositions (not total_positions), totalSectors (not total_sectors)', () => {
    const props = schema.properties ?? {};
    expect('totalPositions' in props).toBe(true);
    expect('total_positions' in props).toBe(false);
    expect('totalSectors' in props).toBe(true);
    expect('total_sectors' in props).toBe(false);
  });

  it('inner AllocationNode required: name, value, children', () => {
    const nodeReq = requiredKeys(schema.$defs?.['AllocationNode'] ?? {});
    ['name', 'value', 'children'].forEach((k) =>
      expect(nodeReq.has(k)).withContext(`expected '${k}' in AllocationNode`).toBe(true),
    );
  });

  it('assertFieldParity: fixture matches snapshot (schema as root for $defs)', () => {
    expect(() =>
      assertFieldParity(schema, makeAllocationResponse(), schema),
    ).not.toThrow();
  });
});

describe('AssetClassReturnsResponse — camelCase (dashboard.json, issue #968)', () => {
  const schema = schemaOf(dashboardSnapshot, 'AssetClassReturnsResponse');

  it('required fields: returns, asOf', () => {
    const req = requiredKeys(schema);
    ['returns', 'asOf'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('camelCase: asOf (not as_of)', () => {
    const props = schema.properties ?? {};
    expect('asOf' in props).toBe(true);
    expect('as_of' in props).toBe(false);
  });

  it('inner AssetClassReturnRow required: name, 1D, 1W, 1M, YTD', () => {
    const rowReq = requiredKeys(schema.$defs?.['AssetClassReturnRow'] ?? {});
    ['name', '1D', '1W', '1M', 'YTD'].forEach((k) =>
      expect(rowReq.has(k)).withContext(`expected '${k}' in AssetClassReturnRow`).toBe(true),
    );
  });

  it('assertFieldParity: fixture matches snapshot (schema as root for $defs)', () => {
    expect(() =>
      assertFieldParity(schema, makeAssetClassReturnsResponse(), schema),
    ).not.toThrow();
  });
});

describe('EquityCurveResponse — camelCase (dashboard.json, issue #968)', () => {
  const schema = schemaOf(dashboardSnapshot, 'EquityCurveResponse');

  it('required fields: points, portfolioTotalReturn, benchmarkTotalReturn', () => {
    const req = requiredKeys(schema);
    ['points', 'portfolioTotalReturn', 'benchmarkTotalReturn'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('camelCase: portfolioTotalReturn (not portfolio_total_return), benchmarkTotalReturn (not benchmark_total_return)', () => {
    const props = schema.properties ?? {};
    expect('portfolioTotalReturn' in props).toBe(true);
    expect('portfolio_total_return' in props).toBe(false);
    expect('benchmarkTotalReturn' in props).toBe(true);
    expect('benchmark_total_return' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot (schema as root for $defs)', () => {
    expect(() =>
      assertFieldParity(schema, makeEquityCurveResponse(), schema),
    ).not.toThrow();
  });
});

describe('PerformanceMetricsResponse — camelCase (dashboard.json, issue #968)', () => {
  const schema = schemaOf(dashboardSnapshot, 'PerformanceMetricsResponse');

  it('required fields: kpis, nav, navChangePct', () => {
    const req = requiredKeys(schema);
    ['kpis', 'nav', 'navChangePct'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('camelCase: navChangePct (not nav_change_pct)', () => {
    const props = schema.properties ?? {};
    expect('navChangePct' in props).toBe(true);
    expect('nav_change_pct' in props).toBe(false);
  });

  it('inner KpiItem required: label, value, format, change, changeLabel, sparkline', () => {
    const kpiReq = requiredKeys(schema.$defs?.['KpiItem'] ?? {});
    ['label', 'value', 'format', 'change', 'changeLabel', 'sparkline'].forEach((k) =>
      expect(kpiReq.has(k)).withContext(`expected '${k}' in KpiItem`).toBe(true),
    );
  });

  it('assertFieldParity: fixture matches snapshot (schema as root for $defs)', () => {
    expect(() =>
      assertFieldParity(schema, makePerformanceMetricsResponse(), schema),
    ).not.toThrow();
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Macro
// ═══════════════════════════════════════════════════════════════════════════════

describe('MacroCalibrationResponse — camelCase (macro.json, issue #968)', () => {
  const schema = schemaOf(macroSnapshot, 'MacroCalibrationResponse');

  it('required fields: phase, delta, tau, confidence, rationale, macroSummary, blConfig', () => {
    const req = requiredKeys(schema);
    ['phase', 'delta', 'tau', 'confidence', 'rationale', 'macroSummary', 'blConfig'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('camelCase: macroSummary (not macro_summary), blConfig (not bl_config)', () => {
    const props = schema.properties ?? {};
    expect('macroSummary' in props).toBe(true);
    expect('macro_summary' in props).toBe(false);
    expect('blConfig' in props).toBe(true);
    expect('bl_config' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot', () => {
    expect(() =>
      assertFieldParity(schema, makeMacroCalibrationApiResponse(), macroSnapshot as never),
    ).not.toThrow();
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Market data
// ═══════════════════════════════════════════════════════════════════════════════

describe('PriceHistoryResponse — snake_case (market_data.json, issue #968)', () => {
  const schema = schemaOf(marketDataSnapshot, 'PriceHistoryResponse');

  it('required fields: id, instrument_id, date, created_at, updated_at', () => {
    const req = requiredKeys(schema);
    ['id', 'instrument_id', 'date', 'created_at', 'updated_at'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('snake_case: instrument_id (not instrumentId), created_at (not createdAt)', () => {
    const props = schema.properties ?? {};
    expect('instrument_id' in props).toBe(true);
    expect('instrumentId' in props).toBe(false);
    expect('created_at' in props).toBe(true);
    expect('createdAt' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot', () => {
    expect(() =>
      assertFieldParity(schema, makePriceHistoryResponse(), marketDataSnapshot as never),
    ).not.toThrow();
  });
});

describe('TickerProfileResponse — snake_case (market_data.json, issue #968)', () => {
  const schema = schemaOf(marketDataSnapshot, 'TickerProfileResponse');

  it('required fields: id, instrument_id, created_at, updated_at', () => {
    const req = requiredKeys(schema);
    ['id', 'instrument_id', 'created_at', 'updated_at'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('snake_case: instrument_id (not instrumentId), created_at (not createdAt), updated_at (not updatedAt)', () => {
    const props = schema.properties ?? {};
    expect('instrument_id' in props).toBe(true);
    expect('instrumentId' in props).toBe(false);
    expect('created_at' in props).toBe(true);
    expect('createdAt' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot', () => {
    expect(() =>
      assertFieldParity(schema, makeTickerProfileResponse(), marketDataSnapshot as never),
    ).not.toThrow();
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Universe
// ═══════════════════════════════════════════════════════════════════════════════

describe('UniverseScreenResponse — camelCase (universe.json, issue #968)', () => {
  const schema = schemaOf(universeSnapshot, 'UniverseScreenResponse');

  it('required fields: passingTickers, totalScreened, diagnostics', () => {
    const req = requiredKeys(schema);
    ['passingTickers', 'totalScreened', 'diagnostics'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('camelCase: passingTickers (not passing_tickers), totalScreened (not total_screened)', () => {
    const props = schema.properties ?? {};
    expect('passingTickers' in props).toBe(true);
    expect('passing_tickers' in props).toBe(false);
    expect('totalScreened' in props).toBe(true);
    expect('total_screened' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot', () => {
    expect(() =>
      assertFieldParity(schema, makeUniverseScreenResponse(), universeSnapshot as never),
    ).not.toThrow();
  });
});

describe('UniverseStatsResponse — snake_case (universe.json, issue #968)', () => {
  const schema = schemaOf(universeSnapshot, 'UniverseStatsResponse');

  it('required fields: exchange_count, instrument_count', () => {
    const req = requiredKeys(schema);
    ['exchange_count', 'instrument_count'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('snake_case: exchange_count (not exchangeCount), instrument_count (not instrumentCount)', () => {
    const props = schema.properties ?? {};
    expect('exchange_count' in props).toBe(true);
    expect('exchangeCount' in props).toBe(false);
    expect('instrument_count' in props).toBe(true);
    expect('instrumentCount' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot', () => {
    expect(() =>
      assertFieldParity(schema, makeUniverseStatsResponse(), universeSnapshot as never),
    ).not.toThrow();
  });
});

describe('InstrumentListResponse — snake_case items (universe.json, issue #968)', () => {
  const schema = schemaOf(universeSnapshot, 'InstrumentListResponse');

  it('required fields: items, total', () => {
    const req = requiredKeys(schema);
    ['items', 'total'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('inner InstrumentResponse required: id, ticker, short_name, created_at, updated_at', () => {
    const innerReq = requiredKeys(schema.$defs?.['InstrumentResponse'] ?? {});
    ['id', 'ticker', 'short_name', 'created_at', 'updated_at'].forEach((k) =>
      expect(innerReq.has(k)).withContext(`expected '${k}' in InstrumentResponse`).toBe(true),
    );
  });

  it('inner InstrumentResponse: snake_case short_name (not shortName), created_at (not createdAt)', () => {
    const innerProps = schema.$defs?.['InstrumentResponse']?.properties ?? {};
    expect('short_name' in innerProps).toBe(true);
    expect('shortName' in innerProps).toBe(false);
    expect('created_at' in innerProps).toBe(true);
    expect('createdAt' in innerProps).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot (schema as root for $defs)', () => {
    expect(() =>
      assertFieldParity(schema, makeInstrumentListResponse(), schema),
    ).not.toThrow();
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Reports
// ═══════════════════════════════════════════════════════════════════════════════

describe('ReportJobCreateResponse — snake_case (reports.json, issue #968)', () => {
  const schema = schemaOf(reportsSnapshot, 'ReportJobCreateResponse');

  it('required field: job_id', () => {
    expect(requiredKeys(schema).has('job_id')).toBe(true);
  });

  it('snake_case: job_id (not jobId)', () => {
    const props = schema.properties ?? {};
    expect('job_id' in props).toBe(true);
    expect('jobId' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot', () => {
    expect(() =>
      assertFieldParity(schema, makeReportJobCreateResponse(), reportsSnapshot as never),
    ).not.toThrow();
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Scenarios
// ═══════════════════════════════════════════════════════════════════════════════

describe('StressScenarioItem — camelCase (scenarios.json, issue #968)', () => {
  const schema = schemaOf(scenariosSnapshot, 'StressScenarioItem');

  it('required fields: name, description, shocks, probability, horizonDays, syntheticDataArgs', () => {
    const req = requiredKeys(schema);
    ['name', 'description', 'shocks', 'probability', 'horizonDays', 'syntheticDataArgs'].forEach(
      (k) => expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('camelCase: horizonDays (not horizon_days), syntheticDataArgs (not synthetic_data_args)', () => {
    const props = schema.properties ?? {};
    expect('horizonDays' in props).toBe(true);
    expect('horizon_days' in props).toBe(false);
    expect('syntheticDataArgs' in props).toBe(true);
    expect('synthetic_data_args' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot', () => {
    expect(() =>
      assertFieldParity(schema, makeStressScenarioItemApi(), scenariosSnapshot as never),
    ).not.toThrow();
  });
});

describe('StressScenarioResponse — camelCase (scenarios.json, issue #968)', () => {
  const schema = schemaOf(scenariosSnapshot, 'StressScenarioResponse');

  it('required fields: nScenarios, tickers, scenarios', () => {
    const req = requiredKeys(schema);
    ['nScenarios', 'tickers', 'scenarios'].forEach((k) =>
      expect(req.has(k)).withContext(`expected '${k}'`).toBe(true),
    );
  });

  it('camelCase: nScenarios (not n_scenarios)', () => {
    const props = schema.properties ?? {};
    expect('nScenarios' in props).toBe(true);
    expect('n_scenarios' in props).toBe(false);
  });

  it('assertFieldParity: fixture matches snapshot (schema as root for $defs)', () => {
    expect(() =>
      assertFieldParity(schema, makeStressScenarioApiResponse(), schema),
    ).not.toThrow();
  });
});
