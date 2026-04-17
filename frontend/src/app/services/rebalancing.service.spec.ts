import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { RebalancingService } from './rebalancing.service';
import { environment } from '../../environments/environment';
import type {
  RebalanceDecideRequest,
  RebalancingPolicyCreatePayload,
  RebalancingPolicyDto,
} from '../models/rebalancing.model';

const API = environment.apiUrl;

function policy(overrides: Partial<RebalancingPolicyDto> = {}): RebalancingPolicyDto {
  return {
    id: 'p1',
    portfolioId: 'port-1',
    name: 'Quarterly',
    policyType: 'calendar',
    config: { frequency_days: 63 },
    isActive: false,
    createdAt: '',
    updatedAt: '',
    ...overrides,
  };
}

describe('RebalancingService', () => {
  let svc: RebalancingService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        RebalancingService,
      ],
    });
    svc = TestBed.inject(RebalancingService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  describe('getDrift()', () => {
    it('GETs /portfolio-analytics/{name}/drift with the threshold param', () => {
      svc.getDrift('myport', 0.05).subscribe();
      const req = http.expectOne(
        (r) => r.url === `${API}portfolio-analytics/myport/drift`,
      );
      expect(req.request.method).toBe('GET');
      expect(req.request.params.get('threshold')).toBe('0.05');
      req.flush({ entries: [], totalDrift: 0, breachedCount: 0, threshold: 0.05 });
    });

    it('omits threshold when not provided and URI-encodes the portfolio name', () => {
      svc.getDrift('my port').subscribe();
      const req = http.expectOne(
        (r) => r.url === `${API}portfolio-analytics/my%20port/drift`,
      );
      expect(req.request.params.keys().length).toBe(0);
      req.flush({ entries: [], totalDrift: 0, breachedCount: 0, threshold: 0.05 });
    });

    it('propagates 404 when the portfolio does not exist', () => {
      let err: HttpErrorResponse | undefined;
      svc.getDrift('nope').subscribe({ error: (e) => (err = e) });
      http
        .expectOne(`${API}portfolio-analytics/nope/drift`)
        .flush({ detail: 'not found' }, { status: 404, statusText: 'Not Found' });
      expect(err?.status).toBe(404);
    });
  });

  describe('Policy list/create/activate', () => {
    it('GETs /portfolio/{name}/rebalance-policy and returns list', () => {
      svc.listPolicies('myport').subscribe();
      const req = http.expectOne(`${API}portfolio/myport/rebalance-policy`);
      expect(req.request.method).toBe('GET');
      req.flush({ items: [policy()], total: 1 });
    });

    it('POSTs /portfolio/{name}/rebalance-policy to create a policy', () => {
      const payload: RebalancingPolicyCreatePayload = {
        name: 'Monthly hybrid',
        policy_type: 'hybrid',
        config: { frequency_days: 21, threshold: 0.05 },
      };
      svc.createPolicy('myport', payload).subscribe();
      const req = http.expectOne(`${API}portfolio/myport/rebalance-policy`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(payload);
      req.flush(policy({ policyType: 'hybrid' }));
    });

    it('POSTs /portfolio/{name}/activate-policy/{id} with empty body', () => {
      svc.activatePolicy('myport', 'p1').subscribe();
      const req = http.expectOne(`${API}portfolio/myport/activate-policy/p1`);
      expect(req.request.method).toBe('POST');
      req.flush(policy({ isActive: true }));
    });

    it('propagates 4xx from createPolicy', () => {
      let err: HttpErrorResponse | undefined;
      svc
        .createPolicy('myport', {
          name: 'bad',
          policy_type: 'calendar',
          config: {},
        })
        .subscribe({ error: (e) => (err = e) });
      http
        .expectOne(`${API}portfolio/myport/rebalance-policy`)
        .flush({ detail: 'invalid' }, { status: 422, statusText: 'Unprocessable' });
      expect(err?.status).toBe(422);
    });
  });

  describe('Trade preview', () => {
    it('GETs /rebalance/preview/{name} and returns the preview payload', () => {
      svc.getPreview('myport').subscribe();
      const req = http.expectOne(`${API}rebalance/preview/myport`);
      expect(req.request.method).toBe('GET');
      req.flush({
        portfolioName: 'myport',
        policyType: 'calendar',
        targetWeights: { AAPL: 0.5 },
        currentWeights: { AAPL: 0.48 },
        trades: [{ ticker: 'AAPL', weightDelta: 0.02, side: 'buy', shares: null }],
        portfolioValue: null,
      });
    });
  });

  describe('Snapshots feed', () => {
    it('GETs /portfolio/{name}/snapshots', () => {
      svc.getSnapshots('myport').subscribe();
      const req = http.expectOne(`${API}portfolio/myport/snapshots`);
      expect(req.request.method).toBe('GET');
      req.flush({ items: [], total: 0 });
    });
  });

  describe('Decide', () => {
    it('POSTs /rebalance/decide with the request body', () => {
      const body: RebalanceDecideRequest = {
        current_weights: { AAPL: 0.5, MSFT: 0.5 },
        target_weights: { AAPL: 0.6, MSFT: 0.4 },
        policy_type: 'threshold',
        policy_config: { threshold: 0.05, kind: 'absolute' },
        transaction_costs: 0.001,
      };
      svc.decide(body).subscribe();
      const req = http.expectOne(`${API}rebalance/decide`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body.policy_type).toBe('threshold');
      req.flush({
        shouldRebalance: true,
        turnover: 0.1,
        estimatedCost: 0.0001,
        tradeWeights: { AAPL: 0.1, MSFT: -0.1 },
      });
    });

    it('propagates 400 for invalid weights', () => {
      let err: HttpErrorResponse | undefined;
      svc
        .decide({
          current_weights: {},
          target_weights: {},
          policy_type: 'calendar',
        })
        .subscribe({ error: (e) => (err = e) });
      http
        .expectOne(`${API}rebalance/decide`)
        .flush({ detail: 'bad' }, { status: 400, statusText: 'Bad Request' });
      expect(err?.status).toBe(400);
    });
  });
});
