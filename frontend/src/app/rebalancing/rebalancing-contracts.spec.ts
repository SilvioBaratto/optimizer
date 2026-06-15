/**
 * Request-shape contract parity for RebalancingService (issue #1000, Scope 12).
 *
 * Asserts HTTP method, exact interpolated URL (path params `{name}`/`{policy_id}`
 * encoded, no raw token), query-parameter placement, and request-body shape for
 * all 7 methods, via the shared request-assertion helper (#998).
 *
 * Issue #1011 — activatePolicy must use Observable<void> + 204 No Content:
 *   the shared `capture` helper flushes `{}` which is wrong for a void endpoint.
 *   The activatePolicy block below uses its own flush to verify the 204 contract.
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

  // ---------------------------------------------------------------------------
  // activatePolicy — void / 204 contract (issue #1011)
  //
  // This block does NOT use the shared `capture()` helper because that helper
  // flushes `{}` (a DTO-shaped body), which is wrong for a void endpoint.
  // The correct contract is: POST + empty body {} → 204 No Content → Observable<void>.
  // ---------------------------------------------------------------------------

  describe('activatePolicy — void / 204 contract (issue #1011)', () => {
    it('when activatePolicy is called, sends POST to /portfolio/{name}/activate-policy/{policyId}', () => {
      svc.activatePolicy(NAME, POLICY_ID).subscribe({ error: () => {} });

      const req = http.expectOne(
        `${API}portfolio/${ENC}/activate-policy/${POLICY_ENC}`
      );
      assertMethod(req, 'POST');
      assertUrl(req, `${API}portfolio/${ENC}/activate-policy/${POLICY_ENC}`);
      req.flush(null, { status: 204, statusText: 'No Content' });
    });

    it('when activatePolicy is called, request body is an empty object {}', () => {
      svc.activatePolicy(NAME, POLICY_ID).subscribe({ error: () => {} });

      const req = http.expectOne(
        `${API}portfolio/${ENC}/activate-policy/${POLICY_ENC}`
      );
      expect(req.request.body).toEqual({});
      req.flush(null, { status: 204, statusText: 'No Content' });
    });

    it('when server flushes 204 No Content, the observable completes without error', (done) => {
      svc.activatePolicy(NAME, POLICY_ID).subscribe({
        error: (err) => done.fail(`unexpected error: ${String(err)}`),
        complete: () => done(),
      });

      const req = http.expectOne(
        `${API}portfolio/${ENC}/activate-policy/${POLICY_ENC}`
      );
      req.flush(null, { status: 204, statusText: 'No Content' });
    });

    it('when server returns 204 with null body, the result is void — no RebalancingPolicyDto emitted', (done) => {
      const emitted: unknown[] = [];

      svc.activatePolicy(NAME, POLICY_ID).subscribe({
        next: (v) => emitted.push(v),
        complete: () => {
          const dtoShaped = emitted.find(
            (v) =>
              v !== null &&
              v !== undefined &&
              typeof v === 'object' &&
              ('id' in (v as Record<string, unknown>) ||
                'portfolioName' in (v as Record<string, unknown>) ||
                'policyName' in (v as Record<string, unknown>))
          );
          expect(dtoShaped)
            .withContext(
              'activatePolicy is Observable<void>; it must not emit a DTO object'
            )
            .toBeUndefined();
          done();
        },
        error: (err) => done.fail(String(err)),
      });

      const req = http.expectOne(
        `${API}portfolio/${ENC}/activate-policy/${POLICY_ENC}`
      );
      req.flush(null, { status: 204, statusText: 'No Content' });
    });

    // URL-encoding invariant: both path segments must be encoded for any input
    const urlEncodingCases: ReadonlyArray<{
      portfolioName: string;
      policyId: string;
      description: string;
    }> = [
      {
        portfolioName: 'portfolio with spaces',
        policyId: 'policy-1',
        description: 'portfolio name containing spaces',
      },
      {
        portfolioName: 'my-portfolio',
        policyId: 'policy with spaces',
        description: 'policy ID containing spaces',
      },
      {
        portfolioName: 'portfolio/v2',
        policyId: 'policy/active',
        description: 'both segments containing slashes',
      },
      {
        portfolioName: 'portfolio&foo',
        policyId: 'policy&bar',
        description: 'both segments containing ampersands',
      },
    ];

    urlEncodingCases.forEach(({ portfolioName, policyId, description }) => {
      it(`when ${description}, both path segments appear URL-encoded in the request URL`, () => {
        const encName = encodeURIComponent(portfolioName);
        const encPolicy = encodeURIComponent(policyId);

        svc.activatePolicy(portfolioName, policyId).subscribe({ error: () => {} });

        const req = http.expectOne(
          `${API}portfolio/${encName}/activate-policy/${encPolicy}`
        );
        expect(req.request.url)
          .withContext('portfolio name segment must be URL-encoded')
          .toContain(encName);
        expect(req.request.url)
          .withContext('policy ID segment must be URL-encoded')
          .toContain(encPolicy);
        req.flush(null, { status: 204, statusText: 'No Content' });
      });
    });
  });

  // ---------------------------------------------------------------------------
  // Remaining methods (unchanged from issue #1000)
  // ---------------------------------------------------------------------------

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
