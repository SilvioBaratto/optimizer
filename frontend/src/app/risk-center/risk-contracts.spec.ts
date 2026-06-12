/**
 * Request-shape contract parity for RiskService (issue #999, Scope 12).
 *
 * Covers all 10 methods — read-only analytics (query params), the LLM POST, and
 * the limits CRUD (GET/POST/PUT/DELETE) — asserting HTTP method, exact
 * interpolated URL (path params `{name}`/`{limit_id}` encoded, no raw token),
 * query-parameter placement, and request-body shape via the #998 helper.
 */
import { TestBed } from '@angular/core/testing';
import type { HttpTestingController, TestRequest } from '@angular/common/http/testing';

import {
  assertBodyFieldTypes,
  assertBodyKeys,
  assertMethod,
  assertQueryParams,
  assertUrl,
  configureTestBed,
  injectHttp,
} from '../../testing';
import { RiskService } from './risk.service';
import type {
  RiskLimitCreatePayload,
  RiskLimitUpdatePayload,
  StressScenarioRequest,
} from './risk.model';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;
const NAME = 'My Port';
const ENC = encodeURIComponent(NAME);
const RISK = `${API}portfolio/${ENC}/risk`;
const LIMITS = `${API}risk/${ENC}/limits`;
const LIMIT_ID = 'lim 1';
const LIMIT_ENC = encodeURIComponent(LIMIT_ID);

describe('RiskService — request contract parity (issue #999)', () => {
  let svc: RiskService;
  let http: HttpTestingController;

  beforeEach(async () => {
    await configureTestBed({ withHttp: true });
    svc = TestBed.inject(RiskService);
    http = injectHttp();
  });

  afterEach(() => http.verify());

  function capture(url: string): TestRequest {
    const req = http.expectOne((r) => r.url === url);
    req.flush({});
    return req;
  }

  // ── Read-only analytics ──────────────────────────────────────────────────

  it('when getVar is called, a GET carries lookback and method in params', () => {
    svc.getVar(NAME, { lookback: 252, method: 'historical' }).subscribe({ error: () => {} });
    const req = capture(`${RISK}/var`);
    assertMethod(req, 'GET');
    assertUrl(req, `${RISK}/var`);
    assertQueryParams(req, { lookback: 252, method: 'historical' });
  });

  it('when getCorrelation is called, a GET carries lookback in params', () => {
    svc.getCorrelation(NAME, 252).subscribe({ error: () => {} });
    const req = capture(`${RISK}/correlation`);
    assertMethod(req, 'GET');
    assertUrl(req, `${RISK}/correlation`);
    assertQueryParams(req, { lookback: 252 });
  });

  it('when getFactorExposure is called, a GET to the interpolated factor-exposure URL is issued', () => {
    svc.getFactorExposure(NAME).subscribe({ error: () => {} });
    const req = capture(`${RISK}/factor-exposure`);
    assertMethod(req, 'GET');
    assertUrl(req, `${RISK}/factor-exposure`);
  });

  it('when getConcentration is called, a GET carries n in params', () => {
    svc.getConcentration(NAME, 10).subscribe({ error: () => {} });
    const req = capture(`${RISK}/concentration`);
    assertMethod(req, 'GET');
    assertUrl(req, `${RISK}/concentration`);
    assertQueryParams(req, { n: 10 });
  });

  it('when getLiquidity is called, a GET carries lookback_days and participation_rate in params', () => {
    svc.getLiquidity(NAME, { lookbackDays: 90, participationRate: 0.1 }).subscribe({ error: () => {} });
    const req = capture(`${RISK}/liquidity`);
    assertMethod(req, 'GET');
    assertUrl(req, `${RISK}/liquidity`);
    assertQueryParams(req, { lookback_days: 90, participation_rate: 0.1 });
  });

  // ── LLM POST ─────────────────────────────────────────────────────────────

  it('when generateStressScenarios is called, a POST carries the request body field-by-field', () => {
    const body: StressScenarioRequest = {
      current_portfolio: { AAPL: 1 },
      macro_context: 'recession',
      n_scenarios: 3,
    };
    svc.generateStressScenarios(body).subscribe({ error: () => {} });
    const req = capture(`${API}risk/stress-scenarios`);
    assertMethod(req, 'POST');
    assertUrl(req, `${API}risk/stress-scenarios`);
    assertBodyKeys(req, ['current_portfolio', 'macro_context', 'n_scenarios']);
    assertBodyFieldTypes(req, {
      current_portfolio: 'object',
      macro_context: 'string',
      n_scenarios: 'number',
    });
  });

  // ── Limits CRUD ──────────────────────────────────────────────────────────

  it('when listLimits is called, a GET to the interpolated limits URL is issued', () => {
    svc.listLimits(NAME).subscribe({ error: () => {} });
    const req = capture(LIMITS);
    assertMethod(req, 'GET');
    assertUrl(req, LIMITS);
  });

  it('when createLimit is called, a POST carries the create payload field-by-field', () => {
    const body: RiskLimitCreatePayload = { metric: 'var', limit_type: 'upper', threshold: 0.2 };
    svc.createLimit(NAME, body).subscribe({ error: () => {} });
    const req = capture(LIMITS);
    assertMethod(req, 'POST');
    assertUrl(req, LIMITS);
    assertBodyKeys(req, ['metric', 'limit_type', 'threshold']);
    assertBodyFieldTypes(req, { metric: 'string', limit_type: 'string', threshold: 'number' });
  });

  it('when updateLimit is called, a PUT to the interpolated limit URL carries the update payload', () => {
    const body: RiskLimitUpdatePayload = { current_value: 0.15, is_breached: false };
    svc.updateLimit(NAME, LIMIT_ID, body).subscribe({ error: () => {} });
    const req = capture(`${LIMITS}/${LIMIT_ENC}`);
    assertMethod(req, 'PUT');
    assertUrl(req, `${LIMITS}/${LIMIT_ENC}`);
    assertBodyKeys(req, ['current_value', 'is_breached']);
    assertBodyFieldTypes(req, { current_value: 'number', is_breached: 'boolean' });
  });

  it('when deleteLimit is called, a DELETE to the interpolated limit URL is issued', () => {
    svc.deleteLimit(NAME, LIMIT_ID).subscribe({ error: () => {} });
    const req = capture(`${LIMITS}/${LIMIT_ENC}`);
    assertMethod(req, 'DELETE');
    assertUrl(req, `${LIMITS}/${LIMIT_ENC}`);
  });
});
