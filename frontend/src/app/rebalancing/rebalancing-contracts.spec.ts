/**
 * Request-shape contract parity for RebalancingService (issue #1000, Scope 12).
 *
 * Asserts HTTP method, exact interpolated URL (path params `{name}`/`{policy_id}`
 * encoded, no raw token), query-parameter placement, and request-body shape for
 * all 7 methods, via the shared request-assertion helper (#998).
 */
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
  type TestRequest,
} from '@angular/common/http/testing';

import {
  assertBodyKeys,
  assertMethod,
  assertQueryParams,
  assertUrl,
} from '../../testing';
import { RebalancingService } from './rebalancing.service';
import type {
  RebalanceDecideRequest,
  RebalancingPolicyCreatePayload,
} from './rebalancing.model';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;
const NAME = 'My Port';
const ENC = encodeURIComponent(NAME);
const POLICY_ID = 'pol 1';
const POLICY_ENC = encodeURIComponent(POLICY_ID);

describe('RebalancingService — request contract parity (issue #1000)', () => {
  let svc: RebalancingService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    svc = TestBed.inject(RebalancingService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function capture(url: string): TestRequest {
    const req = http.expectOne((r) => r.url === url);
    req.flush({});
    return req;
  }

  it('when getDrift is called with a threshold, a GET carries threshold in params', () => {
    svc.getDrift(NAME, 0.05).subscribe({ error: () => {} });
    const req = capture(`${API}portfolio-analytics/${ENC}/drift`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}portfolio-analytics/${ENC}/drift`);
    assertQueryParams(req, { threshold: 0.05 });
  });

  it('when listPolicies is called, a GET to the interpolated rebalance-policy URL is issued', () => {
    svc.listPolicies(NAME).subscribe({ error: () => {} });
    const req = capture(`${API}portfolio/${ENC}/rebalance-policy`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}portfolio/${ENC}/rebalance-policy`);
  });

  it('when createPolicy is called, a POST carries the policy payload field-by-field', () => {
    const body: RebalancingPolicyCreatePayload = {
      name: 'Monthly',
      policy_type: 'calendar',
      config: {},
    };
    svc.createPolicy(NAME, body).subscribe({ error: () => {} });
    const req = capture(`${API}portfolio/${ENC}/rebalance-policy`);
    assertMethod(req, 'POST');
    assertUrl(req, `${API}portfolio/${ENC}/rebalance-policy`);
    assertBodyKeys(req, ['name', 'policy_type', 'config']);
  });

  it('when activatePolicy is called, a POST to the interpolated activate-policy URL is issued', () => {
    svc.activatePolicy(NAME, POLICY_ID).subscribe({ error: () => {} });
    const req = capture(`${API}portfolio/${ENC}/activate-policy/${POLICY_ENC}`);
    assertMethod(req, 'POST');
    assertUrl(req, `${API}portfolio/${ENC}/activate-policy/${POLICY_ENC}`);
  });

  it('when getPreview is called, a GET to the interpolated preview URL is issued', () => {
    svc.getPreview(NAME).subscribe({ error: () => {} });
    const req = capture(`${API}rebalance/preview/${ENC}`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}rebalance/preview/${ENC}`);
  });

  it('when getSnapshots is called, a GET to the interpolated snapshots URL is issued', () => {
    svc.getSnapshots(NAME).subscribe({ error: () => {} });
    const req = capture(`${API}portfolio/${ENC}/snapshots`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}portfolio/${ENC}/snapshots`);
  });

  it('when decide is called, a POST carries the decide request body field-by-field', () => {
    const body: RebalanceDecideRequest = {
      current_weights: { AAPL: 1 },
      target_weights: { AAPL: 1 },
      policy_type: 'threshold',
    };
    svc.decide(body).subscribe({ error: () => {} });
    const req = capture(`${API}rebalance/decide`);
    assertMethod(req, 'POST');
    assertUrl(req, `${API}rebalance/decide`);
    assertBodyKeys(req, ['current_weights', 'target_weights', 'policy_type']);
  });
});
