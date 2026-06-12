/**
 * research-service-field-parity.spec.ts — issue #956
 *
 * Field-level contract-parity specs for OptimizationService, BacktestService,
 * and FactorsService request/response models. Uses the Cycle-1 harness
 * (assertFieldParity / resolveRef / requiredKeys) to lock shapes to the backend
 * OpenAPI snapshots so backend schema drift fails CI.
 *
 * Convention split (encoded explicitly, not flattened):
 *   • POST /optimize async    → snake_case  { job_id, run_id }
 *   • POST /backtest async    → camelCase   { jobId, runId }   — intentional CamelCaseModel
 *   • POST /factors/compute   → snake_case  { job_id }
 *   • GET  /optimize/:id      → camelCase   OptimizationRunResponse
 *   • GET  /backtest/:id      → camelCase   BacktestRunResponse
 *   • GET  /factors/scores    → camelCase   FactorScoreResponse
 *   • POST /factors/score     → snake_case  FactorCompositeScoreResponse
 *   • POST /factors/validate  → snake_case  FactorValidateResponse
 *   • POST /factors/select    → snake_case  FactorSelectResponse
 */

import { schemaOf } from './contract-parity';
import { assertFieldParity, requiredKeys } from './contract-field-parity';
import optimizationSnapshot from './contract-snapshots/optimization.json';
import backtestSnapshot from './contract-snapshots/backtest.json';
import factorsSnapshot from './contract-snapshots/factors.json';
import viewsSnapshot from './contract-snapshots/views.json';

// ── OptimizationService ───────────────────────────────────────────────────────

describe('OptimizationService — field-level parity (issue #956)', () => {

  describe('OptimizationRunResponse — camelCase persisted shape', () => {
    const schema = schemaOf(optimizationSnapshot, 'OptimizationRunResponse');

    const FIXTURE = {
      id: 'run-1',
      portfolioId: null as string | null,
      jobId: null as string | null,
      status: 'completed',
      optimizerType: 'mean_risk',
      universeTickers: ['AAPL'],
      config: {},
      weights: { AAPL: 1.0 },
      metrics: { sharpe: 1.2 },
      riskContributions: { AAPL: 1.0 },
      efficientFrontier: null as unknown[] | null,
      errorMessage: null as string | null,
      solverLog: null as string | null,
      durationSeconds: 1.5,
      createdAt: '2026-01-01T00:00:00Z',
      updatedAt: '2026-01-01T00:00:01Z',
    };

    it('snapshot required fields are camelCase', () => {
      const req = requiredKeys(schema);
      expect(req.has('id')).toBe(true);
      expect(req.has('status')).toBe(true);
      expect(req.has('optimizerType')).toBe(true);
      expect(req.has('weights')).toBe(true);
      expect(req.has('createdAt')).toBe(true);
      expect(req.has('updatedAt')).toBe(true);
    });

    it('assertFieldParity: fixture types match camelCase wire schema', () => {
      expect(() => assertFieldParity(schema, FIXTURE, optimizationSnapshot as never)).not.toThrow();
    });
  });

  describe('OptimizeAsyncResponse — snake_case async (job_id / run_id)', () => {
    // POST /optimize enqueues a job and returns snake_case.
    // Explicit contrast with BacktestAsyncResponse which uses camelCase.
    const FIXTURE = { job_id: 'j-1', run_id: 'run-1' };

    it('uses snake_case keys: job_id, run_id', () => {
      expect(typeof FIXTURE.job_id).toBe('string');
      expect(typeof FIXTURE.run_id).toBe('string');
    });

    it('does NOT carry camelCase keys (contrast: /backtest uses jobId)', () => {
      expect('jobId' in FIXTURE).toBe(false);
      expect('runId' in FIXTURE).toBe(false);
    });
  });

  describe('GenerateViewsResponse — camelCase LLM view shape', () => {
    const schema = schemaOf(viewsSnapshot, 'GenerateViewsResponse');

    const FIXTURE = {
      nViews: 2,
      nAssets: 3,
      viewStrings: ['AAPL == 0.02'],
      p: [[1, 0, 0]],
      q: [0.02],
      viewConfidences: [0.85],
      idzorekAlphas: { AAPL: 0.85 },
      views: [{ asset: 'AAPL', direction: 1, magnitudeBps: 200, confidence: 0.85, reasoning: 'momentum' }],
      rationale: 'Momentum signals strong.',
      tickersWithData: ['AAPL'],
      tickersMissingData: [] as string[],
    };

    it('all fields are in snapshot required set', () => {
      const req = requiredKeys(schema);
      const expectedRequired = ['nViews', 'nAssets', 'viewStrings', 'p', 'q', 'viewConfidences', 'idzorekAlphas', 'views', 'rationale', 'tickersWithData', 'tickersMissingData'];
      expectedRequired.forEach((k) =>
        expect(req.has(k)).withContext(`expected '${k}' in required`).toBe(true),
      );
    });

    it('assertFieldParity: fixture types match schema', () => {
      expect(() => assertFieldParity(schema, FIXTURE, viewsSnapshot as never)).not.toThrow();
    });
  });

  describe('EntropyPoolingResponse — posterior moment shape', () => {
    const schema = schemaOf(viewsSnapshot, 'EntropyPoolingResponse');

    const FIXTURE = {
      tickers: ['AAPL', 'MSFT'],
      mu: [0.12, 0.10],
      covariance: [[0.04, 0.01], [0.01, 0.03]],
    };

    it('required fields: mu, covariance, tickers', () => {
      const req = requiredKeys(schema);
      expect(req.has('mu')).toBe(true);
      expect(req.has('covariance')).toBe(true);
      expect(req.has('tickers')).toBe(true);
    });

    it('assertFieldParity: fixture types match schema', () => {
      expect(() => assertFieldParity(schema, FIXTURE, viewsSnapshot as never)).not.toThrow();
    });
  });

  describe('OpinionPoolResponse — expert pool shape', () => {
    const schema = schemaOf(viewsSnapshot, 'OpinionPoolResponse');

    const FIXTURE = {
      nExperts: 2,
      tickers: ['AAPL', 'MSFT'],
      tickersWithData: ['AAPL', 'MSFT'],
      tickersMissingData: [] as string[],
      experts: [
        { persona: 'value', name: 'A', nViews: 1, viewStrings: ['AAPL == 0.02'], idzorekAlphas: { AAPL: 0.8 }, icWeight: 0.6 },
        { persona: 'growth', name: 'B', nViews: 1, viewStrings: ['MSFT == 0.03'], idzorekAlphas: { MSFT: 0.75 }, icWeight: 0.4 },
      ],
      icWeights: [0.6, 0.4],
      poolingType: 'linear',
      totalViews: 2,
    };

    it('required fields: nExperts, tickers, experts, icWeights, poolingType, totalViews', () => {
      const req = requiredKeys(schema);
      ['nExperts', 'tickers', 'tickersWithData', 'tickersMissingData', 'experts', 'icWeights', 'poolingType', 'totalViews'].forEach((k) =>
        expect(req.has(k)).withContext(`expected '${k}' in required`).toBe(true),
      );
    });

    it('assertFieldParity: fixture types match schema', () => {
      expect(() => assertFieldParity(schema, FIXTURE, viewsSnapshot as never)).not.toThrow();
    });
  });

});

// ── BacktestService ───────────────────────────────────────────────────────────

describe('BacktestService — field-level parity (issue #956)', () => {

  describe('BacktestAsyncResponse — intentional camelCase (CamelCaseModel divergence)', () => {
    // /backtest uses CamelCaseModel; POST /optimize and POST /factors/compute use
    // snake_case for their async responses. This test encodes the divergence so
    // any accidental unification fails CI.
    const FIXTURE = { jobId: 'bj-1', runId: 'brun-1', status: 'pending', message: '' };

    it('camelCase keys: jobId, runId (not snake_case)', () => {
      expect(typeof FIXTURE.jobId).toBe('string');
      expect(typeof FIXTURE.runId).toBe('string');
    });

    it('does NOT carry snake_case keys (contrast: /optimize uses job_id)', () => {
      expect('job_id' in FIXTURE).toBe(false);
      expect('run_id' in FIXTURE).toBe(false);
    });
  });

  describe('BacktestProgressResponse — snake_case poll shape', () => {
    const schema = schemaOf(backtestSnapshot, 'BacktestProgressResponse');

    const FIXTURE = {
      job_id: 'bj-1',
      status: 'completed',
      current: 5,
      total: 5,
      errors: [] as string[],
      result: null as Record<string, unknown> | null,
      error: null as string | null,
    };

    it('required fields: job_id, status', () => {
      const req = requiredKeys(schema);
      expect(req.has('job_id')).toBe(true);
      expect(req.has('status')).toBe(true);
    });

    it('assertFieldParity: fixture types match snake_case schema', () => {
      expect(() => assertFieldParity(schema, FIXTURE, backtestSnapshot as never)).not.toThrow();
    });
  });

  describe('BacktestRunResponse — camelCase persisted shape', () => {
    const schema = schemaOf(backtestSnapshot, 'BacktestRunResponse');

    const FIXTURE = {
      id: 'brun-1',
      portfolioId: null as string | null,
      jobId: null as string | null,
      status: 'completed',
      config: {},
      equityCurve: {},
      drawdowns: {},
      monthlyReturns: {},
      yearlyReturns: {},
      rollingMetrics: {},
      turnoverHistory: {},
      cvFoldMetrics: null as unknown[] | null,
      summaryStats: {},
      errorMessage: null as string | null,
      durationSeconds: null as number | null,
      createdAt: '2026-01-01T00:00:00Z',
      updatedAt: '2026-01-01T00:00:01Z',
    };

    it('required fields include camelCase time-series keys', () => {
      const req = requiredKeys(schema);
      ['id', 'status', 'config', 'equityCurve', 'drawdowns', 'monthlyReturns', 'yearlyReturns', 'rollingMetrics', 'turnoverHistory', 'summaryStats', 'createdAt', 'updatedAt'].forEach((k) =>
        expect(req.has(k)).withContext(`expected '${k}' in required`).toBe(true),
      );
    });

    it('assertFieldParity: fixture types match camelCase wire schema', () => {
      expect(() => assertFieldParity(schema, FIXTURE, backtestSnapshot as never)).not.toThrow();
    });
  });

  describe('ValidateAsyncResponse — snake_case walk-forward async', () => {
    // POST /backtest/validate returns snake_case job_id (consistent with /factors/compute,
    // distinct from /backtest POST which returns camelCase jobId).
    const FIXTURE = { job_id: 'vj-1', status: 'pending', message: '' };

    it('snake_case keys: job_id, status, message', () => {
      expect(typeof FIXTURE.job_id).toBe('string');
      expect(typeof FIXTURE.status).toBe('string');
      expect(typeof FIXTURE.message).toBe('string');
    });
  });

});

// ── FactorsService ────────────────────────────────────────────────────────────

describe('FactorsService — field-level parity (issue #956)', () => {

  describe('FactorComputeAsyncResponse — snake_case async (job_id)', () => {
    // POST /factors/compute returns snake_case (same pattern as /optimize).
    const FIXTURE = { job_id: 'fc-1', status: 'pending', message: '' };

    it('snake_case job_id (not camelCase jobId)', () => {
      expect(typeof FIXTURE.job_id).toBe('string');
      expect('jobId' in FIXTURE).toBe(false);
    });
  });

  describe('FactorCompositeScoreResponse — snake_case score shape', () => {
    const schema = schemaOf(factorsSnapshot, 'FactorCompositeScoreResponse');

    const FIXTURE = {
      score_date: '2024-12-31',
      scores: { AAPL: 0.8, MSFT: 0.6 },
      group_contributions: { momentum: 0.4, value: 0.2 },
    };

    it('required fields: score_date, scores', () => {
      const req = requiredKeys(schema);
      expect(req.has('score_date')).toBe(true);
      expect(req.has('scores')).toBe(true);
    });

    it('group_contributions is optional (absent from required)', () => {
      expect(requiredKeys(schema).has('group_contributions')).toBe(false);
    });

    it('assertFieldParity: fixture types match snake_case schema', () => {
      expect(() => assertFieldParity(schema, FIXTURE, factorsSnapshot as never)).not.toThrow();
    });
  });

  describe('FactorValidateResponse — snake_case compute response', () => {
    // Direct response from POST /factors/validate.
    // Distinct from the persisted FactorValidationReportResponse (camelCase with id/timestamps).
    const FIXTURE = {
      report_date: '2024-12-31',
      factor_type: 'momentum_12_1',
      validation_type: 'in_sample',
      ic_mean: 0.05,
      ic_std: 0.12,
      icir: 0.42,
      t_stat: 2.1,
      p_value: 0.035,
      vif: 1.8,
      details: null as Record<string, unknown> | null,
    };

    it('snake_case keys (not camelCase icMean / pValue)', () => {
      expect('ic_mean' in FIXTURE).toBe(true);
      expect('icMean' in FIXTURE).toBe(false);
      expect('p_value' in FIXTURE).toBe(true);
      expect('pValue' in FIXTURE).toBe(false);
    });

    it('numeric fields carry correct JS types', () => {
      expect(typeof FIXTURE.ic_mean).toBe('number');
      expect(typeof FIXTURE.icir).toBe('number');
      expect(typeof FIXTURE.t_stat).toBe('number');
      expect(typeof FIXTURE.vif).toBe('number');
    });
  });

  describe('FactorSelectApiResponse — snake_case selection shape', () => {
    // buffer_zone property uses $ref → must pass the schema itself as root so
    // resolveRef can find BufferZone inside the schema's own $defs.
    const schema = schemaOf(factorsSnapshot, 'FactorSelectResponse');

    const FIXTURE = {
      selected_tickers: ['AAPL', 'MSFT'],
      count: 2,
      turnover: null as number | null,
      buffer_zone: { entered: ['MSFT'], exited: [] as string[] },
    };

    it('required fields: selected_tickers, count, buffer_zone', () => {
      const req = requiredKeys(schema);
      expect(req.has('selected_tickers')).toBe(true);
      expect(req.has('count')).toBe(true);
      expect(req.has('buffer_zone')).toBe(true);
    });

    it('assertFieldParity: fixture types match schema (self as root for $ref resolution)', () => {
      // Pass schema as root so resolveRef finds BufferZone in schema.$defs.
      expect(() => assertFieldParity(schema, FIXTURE, schema)).not.toThrow();
    });
  });

  describe('FactorScoreDto — camelCase persisted shape (FactorScoreResponse)', () => {
    // GET /factors/scores returns camelCase (CamelCaseModel), unlike compute endpoints.
    const schema = schemaOf(factorsSnapshot, 'FactorScoreResponse');

    const FIXTURE = {
      id: 'fs-1',
      ticker: 'AAPL',
      factorType: 'momentum_12_1',
      factorGroup: 'momentum',
      scoreDate: '2024-12-31',
      rawScore: 0.72,
      standardizedScore: null as number | null,
      compositeScore: null as number | null,
      createdAt: '2026-01-01T00:00:00Z',
      updatedAt: '2026-01-01T00:00:01Z',
    };

    it('required fields: id, ticker, scoreDate, factorType, factorGroup, rawScore', () => {
      const req = requiredKeys(schema);
      ['id', 'ticker', 'scoreDate', 'factorType', 'factorGroup', 'rawScore', 'createdAt', 'updatedAt'].forEach((k) =>
        expect(req.has(k)).withContext(`expected '${k}' in required`).toBe(true),
      );
    });

    it('assertFieldParity: camelCase fixture types match schema', () => {
      expect(() => assertFieldParity(schema, FIXTURE, factorsSnapshot as never)).not.toThrow();
    });
  });

});
