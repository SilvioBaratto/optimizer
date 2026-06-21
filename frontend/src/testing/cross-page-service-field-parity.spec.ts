/**
 * cross-page-service-field-parity.spec.ts — issue #968
 *
 * Field-level contract-parity for analytics-page and shared services not
 * covered by earlier parity specs (settings, portfolio-builder, research).
 *
 * Domains covered:
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
import dashboardSnapshot from './contract-snapshots/dashboard.json';
import macroSnapshot from './contract-snapshots/macro.json';
import marketDataSnapshot from './contract-snapshots/market_data.json';
import universeSnapshot from './contract-snapshots/universe.json';
import reportsSnapshot from './contract-snapshots/reports.json';
import scenariosSnapshot from './contract-snapshots/scenarios.json';
import {
  makeAllocationResponse,
  makeAssetClassReturnsResponse,
  makeDriftResponse,
  makeEquityCurveResponse,
  makeInstrumentListResponse,
  makeMacroCalibrationApiResponse,
  makeMarketSnapshotResponse,
  makePerformanceMetricsResponse,
  makePriceHistoryResponse,
  makeReportJobCreateResponse,
  makeStressScenarioApiResponse,
  makeStressScenarioItemApi,
  makeTickerProfileResponse,
  makeUniverseScreenResponse,
  makeUniverseStatsResponse,
} from './domain-fixtures';

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
