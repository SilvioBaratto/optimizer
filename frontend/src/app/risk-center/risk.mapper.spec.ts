import {
  toVarResults,
  toCorrelationData,
  toFactorExposures,
  toConcentrationMetrics,
  toLiquidityMetrics,
  toStressScenarios,
  toRiskLimitDisplay,
} from './risk.mapper';
import type {
  VarApiResponse,
  CorrelationApiResponse,
  FactorExposureApiResponse,
  ConcentrationApiResponse,
  LiquidityApiResponse,
  StressScenarioApiResponse,
  RiskLimitDto,
} from './risk.model';

const PORTFOLIO_VALUE = 10_000_000;

describe('risk.mapper', () => {
  describe('toVarResults', () => {
    it('when response is null, returns empty array', () => {
      expect(toVarResults(null, PORTFOLIO_VALUE)).toEqual([]);
    });

    it('when response has confidence keys, maps varDollar = varPct * portfolioValue', () => {
      const response: VarApiResponse = {
        var: { '95': 0.03, '99': 0.05 },
        cvar: { '95': 0.04, '99': 0.07 },
        method: 'historical',
        lookback: 252,
        nObservations: 250,
      };
      const results = toVarResults(response, PORTFOLIO_VALUE);
      expect(results.length).toBe(2);
      const r95 = results.find((r) => r.confidence === 0.95)!;
      expect(r95.varDollar).toBe(0.03 * PORTFOLIO_VALUE);
      expect(r95.cvarDollar).toBe(0.04 * PORTFOLIO_VALUE);
    });

    it('when method is null/undefined, defaults to historical', () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const response = { var: { '95': 0.02 }, cvar: { '95': 0.03 }, method: null, lookback: 252, nObservations: 100 } as any;
      const [r] = toVarResults(response, PORTFOLIO_VALUE);
      expect(r.method).toBe('historical');
    });

    it('when a confidence value is null and cvar lacks the key, both coalesce to 0', () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const response = { var: { '95': null }, cvar: {}, method: 'historical', lookback: 252, nObservations: 100 } as any;
      const [r] = toVarResults(response, PORTFOLIO_VALUE);
      expect(r.var).toBe(0);
      expect(r.cvar).toBe(0);
      expect(r.varDollar).toBe(0);
      expect(r.cvarDollar).toBe(0);
    });
  });

  describe('toCorrelationData', () => {
    it('when response is null, returns empty assets and matrix', () => {
      expect(toCorrelationData(null)).toEqual({ assets: [], matrix: [] });
    });

    it('when response has data, strips clusterLabels and keeps assets + matrix', () => {
      const response: CorrelationApiResponse = {
        assets: ['AAPL', 'MSFT'],
        matrix: [[1, 0.8], [0.8, 1]],
        clusterLabels: [0, 0],
      };
      expect(toCorrelationData(response)).toEqual({ assets: ['AAPL', 'MSFT'], matrix: [[1, 0.8], [0.8, 1]] });
    });
  });

  describe('toFactorExposures', () => {
    it('when response is null, returns empty array', () => {
      expect(toFactorExposures(null)).toEqual([]);
    });

    it('when response has exposures, sets contribution and marginalContribution equal to exposure', () => {
      const response: FactorExposureApiResponse = {
        exposures: { momentum: 0.4, value: -0.2 },
        assetExposures: {},
      };
      const [mom] = toFactorExposures(response).filter((e) => e.factor === 'momentum');
      expect(mom.contribution).toBe(0.4);
      expect(mom.marginalContribution).toBe(0.4);
    });
  });

  describe('toConcentrationMetrics', () => {
    it('when response is null, returns empty array', () => {
      expect(toConcentrationMetrics(null)).toEqual([]);
    });

    it('when response has assets, sets riskContribution to weight and componentVar to 0', () => {
      const response: ConcentrationApiResponse = {
        assets: [{ ticker: 'AAPL', name: 'Apple', weight: 0.4 }],
        summary: { hhi: 0.2, effectiveN: 5, topNRatio: 0.8 },
      };
      const [m] = toConcentrationMetrics(response);
      expect(m.riskContribution).toBe(0.4);
      expect(m.componentVar).toBe(0);
    });
  });

  describe('toLiquidityMetrics', () => {
    it('when response is null, returns empty array', () => {
      expect(toLiquidityMetrics(null)).toEqual([]);
    });

    it('when asset fields are null, coalesces to 0', () => {
      const response: LiquidityApiResponse = {
        assets: [{ ticker: 'AAPL', name: 'Apple', weight: 0.5, avgDailyVolume: null, daysToLiquidate: null, liquidityCost: null }],
        summary: { weightedAvgDaysToLiquidate: 0 },
      };
      const [m] = toLiquidityMetrics(response);
      expect(m.avgDailyVolume).toBe(0);
      expect(m.daysToLiquidate).toBe(0);
      expect(m.liquidityCost).toBe(0);
    });
  });

  describe('toStressScenarios', () => {
    it('when response is null, returns empty array', () => {
      expect(toStressScenarios(null)).toEqual([]);
    });

    it('when shocks are empty, meanShock guard returns portfolioImpact of 0', () => {
      const response: StressScenarioApiResponse = {
        nScenarios: 1,
        tickers: [],
        scenarios: [{ name: 'crash', description: 'test', shocks: {}, probability: 0.1, horizonDays: 5, syntheticDataArgs: {} }],
      };
      const [s] = toStressScenarios(response);
      expect(s.portfolioImpact).toBe(0);
    });

    it('when shocks are present, portfolioImpact is arithmetic mean of shock values', () => {
      const response: StressScenarioApiResponse = {
        nScenarios: 1,
        tickers: ['AAPL', 'MSFT'],
        scenarios: [{ name: 'crash', description: 'test', shocks: { AAPL: -0.1, MSFT: -0.3 }, probability: 0.1, horizonDays: 5, syntheticDataArgs: {} }],
      };
      const [s] = toStressScenarios(response);
      expect(s.portfolioImpact).toBeCloseTo((-0.1 + -0.3) / 2);
    });

    it('when shocks are present, benchmarkImpact is always 0', () => {
      const response: StressScenarioApiResponse = {
        nScenarios: 1,
        tickers: ['AAPL'],
        scenarios: [{ name: 'crash', description: 'test', shocks: { AAPL: -0.2 }, probability: 0.1, horizonDays: 5, syntheticDataArgs: {} }],
      };
      const [s] = toStressScenarios(response);
      expect(s.benchmarkImpact).toBe(0);
    });

    it('when multiple scenarios are present, ids are index-named s-0, s-1, …', () => {
      const response: StressScenarioApiResponse = {
        nScenarios: 2,
        tickers: ['AAPL'],
        scenarios: [
          { name: 'crash', description: 'first', shocks: { AAPL: -0.1 }, probability: 0.1, horizonDays: 5, syntheticDataArgs: {} },
          { name: 'rally', description: 'second', shocks: { AAPL: 0.1 }, probability: 0.05, horizonDays: 5, syntheticDataArgs: {} },
        ],
      };
      const result = toStressScenarios(response);
      expect(result[0].id).toBe('s-0');
      expect(result[1].id).toBe('s-1');
    });
  });

  describe('toRiskLimitDisplay', () => {
    function dto(overrides: Partial<RiskLimitDto> = {}): RiskLimitDto {
      return { id: 'l1', portfolioId: 'p1', metric: 'max_dd', limitType: 'upper', threshold: 0.2, currentValue: 0.1, isBreached: false, lastCheckedAt: '2026-01-01', createdAt: '', updatedAt: '', ...overrides };
    }

    it('when isBreached is true, status is breached regardless of current value', () => {
      expect(toRiskLimitDisplay(dto({ isBreached: true, currentValue: 0 })).status).toBe('breached');
    });

    it('when current is exactly threshold * 0.8, status is warning', () => {
      // Use values where threshold*0.8 is exact in IEEE 754 to avoid floating-point drift
      const d = dto({ threshold: 0.25, currentValue: 0.2, isBreached: false });
      expect(toRiskLimitDisplay(d).status).toBe('warning');
    });

    it('when current is below threshold * 0.8, status is ok', () => {
      const d = dto({ threshold: 0.2, currentValue: 0.1, isBreached: false });
      expect(toRiskLimitDisplay(d).status).toBe('ok');
    });

    it('when currentValue is null, coalesces to 0 and status is ok', () => {
      const d = dto({ threshold: 0.2, currentValue: null, isBreached: false });
      const result = toRiskLimitDisplay(d);
      expect(result.current).toBe(0);
      expect(result.status).toBe('ok');
    });

    it('when lastCheckedAt is null, falls back to updatedAt', () => {
      const d = dto({ lastCheckedAt: null, updatedAt: '2026-06-01' });
      expect(toRiskLimitDisplay(d).lastChecked).toBe('2026-06-01');
    });
  });
});
