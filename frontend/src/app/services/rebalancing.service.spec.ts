import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { RebalancingService } from './rebalancing.service';
import { SUPPRESS_TOAST_STATUSES } from '../core/interceptors/api-http.interceptor';
import { environment } from '../../environments/environment';
import type {
  RebalanceDecideRequest,
  RebalancingPolicyCreatePayload,
  RebalancingPolicyDto,
  RebalancingPolicyListApiResponse,
  RebalancePreviewApiResponse,
} from '../models/rebalancing.model';
import type { SnapshotListResponseDto } from '../core/models/portfolio-api.model';

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

    it('when listPolicies is called, the returned list is delivered to the subscriber', () => {
      let result: RebalancingPolicyListApiResponse | undefined;
      svc.listPolicies('myport').subscribe((r) => (result = r));
      http.expectOne(`${API}portfolio/myport/rebalance-policy`).flush({ items: [policy()], total: 1 });
      expect(result?.total).toBe(1);
      expect(result?.items[0].id).toBe('p1');
    });

    it('URI-encodes the portfolio name in listPolicies URL', () => {
      svc.listPolicies('my port').subscribe();
      const req = http.expectOne(`${API}portfolio/my%20port/rebalance-policy`);
      expect(req.request.method).toBe('GET');
      req.flush({ items: [], total: 0 });
    });

    it('when listPolicies returns a 500, the error is propagated to the subscriber', () => {
      let err: HttpErrorResponse | undefined;
      svc.listPolicies('myport').subscribe({ error: (e) => (err = e) });
      http
        .expectOne(`${API}portfolio/myport/rebalance-policy`)
        .flush({ detail: 'db error' }, { status: 500, statusText: 'Internal Server Error' });
      expect(err?.status).toBe(500);
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

    it('URI-encodes the portfolio name in createPolicy URL', () => {
      svc.createPolicy('my port', { name: 'Q', policy_type: 'calendar', config: {} }).subscribe();
      const req = http.expectOne(`${API}portfolio/my%20port/rebalance-policy`);
      expect(req.request.method).toBe('POST');
      req.flush(policy());
    });

    it('POSTs /portfolio/{name}/activate-policy/{id} with empty body', () => {
      svc.activatePolicy('myport', 'p1').subscribe();
      const req = http.expectOne(`${API}portfolio/myport/activate-policy/p1`);
      expect(req.request.method).toBe('POST');
      req.flush(policy({ isActive: true }));
    });

    it('URI-encodes both portfolio name and policy id in activatePolicy URL', () => {
      svc.activatePolicy('my port', 'p/1').subscribe();
      const req = http.expectOne(`${API}portfolio/my%20port/activate-policy/p%2F1`);
      expect(req.request.method).toBe('POST');
      req.flush(policy({ isActive: true }));
    });

    it('when activatePolicy returns 404, the error is propagated to the subscriber', () => {
      let err: HttpErrorResponse | undefined;
      svc.activatePolicy('myport', 'gone').subscribe({ error: (e) => (err = e) });
      http
        .expectOne(`${API}portfolio/myport/activate-policy/gone`)
        .flush({ detail: 'policy not found' }, { status: 404, statusText: 'Not Found' });
      expect(err?.status).toBe(404);
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

    it('when getPreview is called, the response value is delivered to the subscriber', () => {
      const payload: RebalancePreviewApiResponse = {
        portfolioName: 'myport',
        policyType: 'threshold',
        targetWeights: { MSFT: 0.6 },
        currentWeights: { MSFT: 0.5 },
        trades: [{ ticker: 'MSFT', weightDelta: 0.1, side: 'buy', shares: null }],
        portfolioValue: 100000,
      };
      let result: RebalancePreviewApiResponse | undefined;
      svc.getPreview('myport').subscribe((r) => (result = r));
      http.expectOne(`${API}rebalance/preview/myport`).flush(payload);
      expect(result?.portfolioValue).toBe(100000);
      expect(result?.trades.length).toBe(1);
    });

    it('URI-encodes the portfolio name in getPreview URL', () => {
      svc.getPreview('my port').subscribe();
      const req = http.expectOne(`${API}rebalance/preview/my%20port`);
      expect(req.request.method).toBe('GET');
      req.flush({ portfolioName: 'my port', policyType: 'calendar', targetWeights: {}, currentWeights: {}, trades: [], portfolioValue: null });
    });

    it('sets SUPPRESS_TOAST_STATUSES context to include 404 on getPreview requests', () => {
      svc.getPreview('myport').subscribe();
      const req = http.expectOne(`${API}rebalance/preview/myport`);
      const suppressed = req.request.context.get(SUPPRESS_TOAST_STATUSES);
      expect(suppressed).toContain(404);
      req.flush({ portfolioName: 'myport', policyType: 'calendar', targetWeights: {}, currentWeights: {}, trades: [], portfolioValue: null });
    });

    it('when getPreview returns 404, the error is propagated to the subscriber', () => {
      let err: HttpErrorResponse | undefined;
      svc.getPreview('myport').subscribe({ error: (e) => (err = e) });
      http
        .expectOne(`${API}rebalance/preview/myport`)
        .flush({ detail: 'no preview' }, { status: 404, statusText: 'Not Found' });
      expect(err?.status).toBe(404);
    });
  });

  describe('Snapshots feed', () => {
    it('GETs /portfolio/{name}/snapshots', () => {
      svc.getSnapshots('myport').subscribe();
      const req = http.expectOne(`${API}portfolio/myport/snapshots`);
      expect(req.request.method).toBe('GET');
      req.flush({ items: [], total: 0 });
    });

    it('when snapshots are returned, the response value is delivered to the subscriber', () => {
      let result: SnapshotListResponseDto | undefined;
      svc.getSnapshots('myport').subscribe((r) => (result = r));
      http.expectOne(`${API}portfolio/myport/snapshots`).flush({ items: [], total: 0 });
      expect(result?.total).toBe(0);
    });

    it('URI-encodes the portfolio name in getSnapshots URL', () => {
      svc.getSnapshots('my port').subscribe();
      const req = http.expectOne(`${API}portfolio/my%20port/snapshots`);
      expect(req.request.method).toBe('GET');
      req.flush({ items: [], total: 0 });
    });

    it('when getSnapshots returns a 500, the error is propagated to the subscriber', () => {
      let err: HttpErrorResponse | undefined;
      svc.getSnapshots('myport').subscribe({ error: (e) => (err = e) });
      http
        .expectOne(`${API}portfolio/myport/snapshots`)
        .flush({ detail: 'db error' }, { status: 500, statusText: 'Internal Server Error' });
      expect(err?.status).toBe(500);
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
