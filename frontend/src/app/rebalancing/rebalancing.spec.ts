import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { of, throwError } from 'rxjs';

import { RebalancingComponent } from './rebalancing';
import { RebalancingService } from './rebalancing.service';
import { PortfolioApiService } from '../core/services/portfolio-api.service';
import {
  apiHttpInterceptor,
  RETRY_BACKOFF,
  SUPPRESS_TOAST_STATUSES,
  type BackoffFn,
} from '../core/interceptors/api-http.interceptor';
import { NotificationService } from '../shared/notification/notification.service';
import { environment } from '../../environments/environment';
import {
  configureTestBed,
  drainRequests,
  injectHttp,
  installResizeObserverStub,
  makePortfolioDto,
  makeDriftResponse,
  makeRebalanceDecideResponse,
  makeRebalancingPolicyDto,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';

const API = environment.apiUrl;
const PORTFOLIO = 'trading212';

describe('RebalancingComponent — preview error handling (issue #438)', () => {
  let http: HttpTestingController;
  let notifications: jasmine.SpyObj<NotificationService>;

  beforeEach(() => {
    notifications = jasmine.createSpyObj<NotificationService>(
      'NotificationService',
      ['success', 'error', 'warning', 'info'],
    );
    const immediateBackoff: BackoffFn = () => of(0);

    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(withInterceptors([apiHttpInterceptor])),
        provideHttpClientTesting(),
        { provide: NotificationService, useValue: notifications },
        { provide: RETRY_BACKOFF, useValue: immediateBackoff },
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function bootstrapWithSelectedPortfolio() {
    const fx = TestBed.createComponent(RebalancingComponent);
    fx.detectChanges();
    // Drain the portfolios bootstrap call (URL may carry a trailing slash)
    http
      .expectOne((r) => r.url.startsWith(`${API}portfolio`) && !r.url.includes('/snapshot') && !r.url.includes('/rebalance'))
      .flush({ items: [] });
    fx.componentInstance.onPortfolioSelect(PORTFOLIO);
    fx.detectChanges();
    return fx;
  }

  function flushNonPreviewRequests(): void {
    // Selecting a portfolio fires drift, policies, snapshots, preview.
    // We only care about the preview branch, so flush the others as success.
    const matches = http.match(
      (r) => !r.url.includes('/rebalance/preview/'),
    );
    for (const req of matches) {
      const url = req.request.url;
      if (url.includes('/drift')) req.flush({ entries: [], breachedCount: 0 });
      else if (url.includes('/rebalance-policy')) req.flush({ items: [] });
      else if (url.includes('/snapshots')) req.flush({ items: [] });
      else req.flush({});
    }
  }

  it('renders a friendly panel message on a 404 from /rebalance/preview AND suppresses the toast', () => {
    const fx = bootstrapWithSelectedPortfolio();
    flushNonPreviewRequests();

    const previewReq = http.expectOne(
      (r) => r.url === `${API}rebalance/preview/${PORTFOLIO}`,
    );
    previewReq.flush(
      { detail: `Portfolio '${PORTFOLIO}' not found` },
      { status: 404, statusText: 'Not Found' },
    );

    const message = fx.componentInstance.panelErrors()['preview'];
    expect(message).toBeTruthy();
    expect(message).not.toContain('Http failure response');
    expect(message?.toLowerCase()).toContain('preview');
    expect(notifications.error).not.toHaveBeenCalled();
  });

  it('renders a friendly panel message on a non-404 error from /rebalance/preview', () => {
    // 422 isn't retried by the interceptor and isn't suppressed by SUPPRESS_TOAST_STATUSES,
    // so it exercises the same code path as a 500 (toast + friendly panel) without
    // needing to flush 4 retry attempts.
    const fx = bootstrapWithSelectedPortfolio();
    flushNonPreviewRequests();

    const previewReq = http.expectOne(
      (r) => r.url === `${API}rebalance/preview/${PORTFOLIO}`,
    );
    previewReq.flush(
      { detail: 'Snapshot mismatch' },
      { status: 422, statusText: 'Unprocessable Entity' },
    );

    const message = fx.componentInstance.panelErrors()['preview'];
    expect(message).toBeTruthy();
    expect(message).not.toContain('Http failure response');
    expect(message).toContain('Snapshot mismatch');
  });

  it('opts the preview request out of the 404 toast via HttpContext', () => {
    const fx = bootstrapWithSelectedPortfolio();
    flushNonPreviewRequests();

    const previewReq = http.expectOne(
      (r) => r.url === `${API}rebalance/preview/${PORTFOLIO}`,
    );
    // The request must carry the suppress-toast context so the interceptor
    // skips notify.error() for 404. We assert the request context contains
    // 404 (the contract enforced by the rebalancing service).
    const suppressed = previewReq.request.context.get(SUPPRESS_TOAST_STATUSES);
    expect(suppressed).toContain(404);

    previewReq.flush(
      { detail: 'gone' },
      { status: 404, statusText: 'Not Found' },
    );
    void fx;
  });
});

// ── Full workflow coverage (issue #940) ──────────────────────────────────────
// Drives the component's methods through the standard drainRequests harness so
// the drift / policy / decide / activate flows (success + error) are covered.
describe('RebalancingComponent — workflow coverage (issue #940)', () => {
  let fixture: ComponentFixture<RebalancingComponent>;
  let comp: RebalancingComponent;
  let http: HttpTestingController;

  function stubFor(url: string): Record<string, unknown> {
    if (url.includes('portfolio-analytics') || url.includes('/drift')) {
      return { entries: [], totalDrift: 0, breachedCount: 0, threshold: 0.05 };
    }
    if (url.includes('rebalance-policy')) return { items: [] };
    if (url.includes('activate-policy')) return {};
    if (url.includes('rebalance/preview')) {
      return {
        portfolioName: 'Test Portfolio',
        policyType: 'threshold',
        targetWeights: {},
        currentWeights: {},
        trades: [],
        portfolioValue: 0,
        status: null,
      };
    }
    if (url.includes('/snapshots')) return { items: [] };
    if (url.includes('rebalance/decide')) {
      return { shouldRebalance: false, turnover: 0, estimatedCost: 0, tradeWeights: {} };
    }
    if (url.includes('portfolio')) return { items: [makePortfolioDto()], total: 1 };
    return {};
  }

  function settle(): void {
    fixture.detectChanges();
    drainRequests(http, stubFor);
    fixture.detectChanges();
    drainRequests(http, stubFor);
  }

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [RebalancingComponent],
      withHttp: true,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(RebalancingComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
  });

  afterEach(() => http.verify());

  it('initialises into loading then resolves with the first portfolio selected', () => {
    fixture.detectChanges();
    expect(comp.isLoading()).toBe(true);
    settle();
    expect(comp.isLoading()).toBe(false);
    expect(comp.selectedPortfolio()).toBe('Test Portfolio');
  });

  it('when the list request fails, the error state is shown', () => {
    fixture.detectChanges();
    http
      .expectOne((r) => r.url.includes('portfolio'))
      .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();
    expect(comp.hasError()).toBe(true);
  });

  it('onThresholdChange refetches drift with the new threshold', () => {
    settle();
    comp.onThresholdChange(0.1);
    http
      .expectOne((r) => r.url.includes('/drift'))
      .flush({ entries: [], totalDrift: 0, breachedCount: 0, threshold: 0.1 });
    expect(comp.driftThreshold()).toBe(0.1);
  });

  it('requestActivate then cancelActivate toggles the pending id', () => {
    settle();
    comp.requestActivate('p1');
    expect(comp.pendingActivateId()).toBe('p1');
    comp.cancelActivate();
    expect(comp.pendingActivateId()).toBeNull();
  });

  it('confirmActivate is a no-op when nothing is pending', () => {
    settle();
    comp.confirmActivate();
    expect(http.match((r) => r.url.includes('activate-policy')).length).toBe(0);
  });

  it('confirmActivate activates the policy and refreshes preview on success', () => {
    settle();
    comp.policies.set([
      makeRebalancingPolicyDto({ id: 'p1', isActive: false }),
      makeRebalancingPolicyDto({ id: 'p2', isActive: true }),
    ]);
    comp.requestActivate('p1');
    comp.confirmActivate();
    http.expectOne((r) => r.url.includes('activate-policy/p1')).flush({});
    expect(comp.pendingActivateId()).toBeNull();
    expect(comp.activePolicy()?.id).toBe('p1');
    drainRequests(http, stubFor);
  });

  it('confirmActivate records an error on failure', () => {
    settle();
    comp.requestActivate('p1');
    comp.confirmActivate();
    http
      .expectOne((r) => r.url.includes('activate-policy/p1'))
      .flush({ detail: 'no' }, { status: 500, statusText: 'Server Error' });
    expect(comp.panelErrors()['policy']).toBeTruthy();
  });

  it('onCreatePolicy posts then reloads the policy list', () => {
    settle();
    comp.onCreatePolicy({ name: 'P', policy_type: 'threshold', config: {} });
    http.expectOne((r) => r.url.includes('rebalance-policy') && r.method === 'POST').flush({});
    const created = makeRebalancingPolicyDto({ name: 'P' });
    http
      .expectOne((r) => r.url.includes('rebalance-policy') && r.method === 'GET')
      .flush({ items: [created] });
    expect(comp.policies()).toEqual([created]);
  });

  it('onCreatePolicy records an error on failure', () => {
    settle();
    comp.onCreatePolicy({ name: 'P', policy_type: 'threshold', config: {} });
    http
      .expectOne((r) => r.url.includes('rebalance-policy') && r.method === 'POST')
      .flush({ detail: 'bad' }, { status: 422, statusText: 'Unprocessable' });
    expect(comp.panelErrors()['policy']).toBeTruthy();
  });

  it('onRunDecide posts and stores the decision', () => {
    settle();
    comp.onRunDecide({
      current_weights: { AAPL: 0.6 },
      target_weights: { AAPL: 0.5 },
      policy_type: 'threshold',
    });
    http.expectOne((r) => r.url.includes('rebalance/decide')).flush(makeRebalanceDecideResponse());
    expect(comp.decideResponse()).not.toBeNull();
  });

  it('onRunDecide records an error on failure', () => {
    settle();
    comp.onRunDecide({
      current_weights: {},
      target_weights: {},
      policy_type: 'threshold',
    });
    http
      .expectOne((r) => r.url.includes('rebalance/decide'))
      .flush({ detail: 'bad' }, { status: 500, statusText: 'Server Error' });
    expect(comp.panelErrors()['whatif']).toBeTruthy();
  });

  it('a 5xx preview error yields a temporarily-unavailable message', () => {
    settle();
    comp.onPortfolioSelect('Other');
    fixture.detectChanges();
    http
      .expectOne((r) => r.url.includes('rebalance/preview'))
      .flush({ detail: 'down' }, { status: 500, statusText: 'Server Error' });
    expect(comp.panelErrors()['preview']).toContain('temporarily unavailable');
    drainRequests(http, stubFor);
  });

  it('kpiMaxDrift em-dashes on null/empty drift; driftEntries defaults to []', () => {
    settle();
    comp.driftResponse.set(null);
    expect(comp.kpiMaxDrift()).toBe('—');
    expect(comp.driftEntries()).toEqual([]); // covers ?.entries ?? []
    comp.driftResponse.set({ entries: [], totalDrift: 0, breachedCount: 0, threshold: 0.05 });
    expect(comp.kpiMaxDrift()).toBe('—'); // covers entries.length === 0
    comp.driftResponse.set(makeDriftResponse());
    expect(comp.kpiMaxDrift()).not.toBe('—');
    expect(comp.driftEntries().length).toBe(1);
  });

  it('retry reloads after an error', () => {
    fixture.detectChanges();
    http
      .expectOne((r) => r.url.includes('portfolio'))
      .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();
    expect(comp.hasError()).toBe(true);
    comp.retry();
    expect(comp.hasError()).toBe(false);
    drainRequests(http, stubFor);
    fixture.detectChanges();
    drainRequests(http, stubFor);
  });
});

// The subscribe error callbacks are typed `(err: Error)` and an HttpErrorResponse
// always carries a `.message`, so the `?? 'default'` / `|| 'Preview failed'`
// fallbacks are only reachable when the source errors with a message-less value.
// A mocked service supplies that, covering the otherwise-unreachable defaults.
describe('RebalancingComponent — default error messages', () => {
  function boomService(): RebalancingService {
    const boom = () => throwError(() => ({}));
    return {
      getDrift: boom, listPolicies: boom, createPolicy: boom, activatePolicy: boom,
      getPreview: boom, getSnapshots: boom, decide: boom,
    } as unknown as RebalancingService;
  }

  it('falls back to default panel messages when services error without a message', async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [RebalancingComponent],
      withHttp: true,
      providers: [
        ICON_PROVIDER,
        { provide: RebalancingService, useValue: boomService() },
        { provide: PortfolioApiService, useValue: { list: () => of({ items: [makePortfolioDto()], total: 1 }) } },
      ],
    });
    const fx = TestBed.createComponent(RebalancingComponent);
    fx.detectChanges();
    const c = fx.componentInstance;

    expect(c.panelErrors()['drift']).toBe('Drift failed');
    expect(c.panelErrors()['policy']).toBe('Policy load failed');
    expect(c.panelErrors()['history']).toBe('History failed');
    expect(c.panelErrors()['preview']).toBe('Preview failed');

    c.requestActivate('p1');
    c.confirmActivate();
    expect(c.panelErrors()['policy']).toBe('Activate failed');

    c.onCreatePolicy({ name: 'P', policy_type: 'threshold', config: {} });
    expect(c.panelErrors()['policy']).toBe('Create failed');

    c.onRunDecide({ current_weights: {}, target_weights: {}, policy_type: 'threshold' });
    expect(c.panelErrors()['whatif']).toBe('Decide failed');
  });

  it('falls back to the default error when portfolio loading errors without a message', async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [RebalancingComponent],
      withHttp: true,
      providers: [
        ICON_PROVIDER,
        { provide: RebalancingService, useValue: boomService() },
        { provide: PortfolioApiService, useValue: { list: () => throwError(() => ({})) } },
      ],
    });
    const fx = TestBed.createComponent(RebalancingComponent);
    fx.detectChanges();
    expect(fx.componentInstance.hasError()).toBe(true);
    expect(fx.componentInstance.errorMessage()).toBe('Failed to load portfolios');
  });
});
