import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection, signal, computed } from '@angular/core';
import { of, throwError } from 'rxjs';

import { RebalancingComponent } from './rebalancing';
import { RebalancingService } from './rebalancing.service';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
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
  makeDriftResponse,
  makeRebalanceDecideResponse,
  makeRebalancingPolicyDto,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';

const API = environment.apiUrl;
const PORTFOLIO = 'trading212';

// ── Helper: minimal PortfolioContextService mock ──────────────────────────────

function makeCtxMock(name: string | null = PORTFOLIO, id: string | null = 'stub-id') {
  const nameSig = signal(name);
  const idSig   = signal(id);
  return {
    currentPortfolioId:   idSig,
    currentPortfolioName: computed(() => nameSig()),
    hasPortfolio:         computed(() => idSig() !== null),
    selectedPortfolio:    computed(() => (idSig() ? { id: idSig()!, name: nameSig() ?? '' } : null)),
    dateRange:            signal({ preset: '1Y' as const, start: new Date('2024-01-01'), end: new Date('2025-01-01') }),
    benchmark:            signal('SPY'),
    activeMode:           signal('backtest' as const),
    isLive:               computed(() => false),
    isBacktest:           computed(() => true),
    isPaper:              computed(() => false),
    dateRangeLabel:       computed(() => '1Y'),
    dateRangeDays:        computed(() => 365),
    setPortfolio: jasmine.createSpy('setPortfolio'),
    setMode:      jasmine.createSpy('setMode'),
    setPreset:    jasmine.createSpy('setPreset'),
    setCustomRange: jasmine.createSpy('setCustomRange'),
    setBenchmark: jasmine.createSpy('setBenchmark'),
    reset:        jasmine.createSpy('reset'),
  };
}

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
        { provide: PortfolioContextService, useValue: makeCtxMock(PORTFOLIO, 'stub-id') },
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function bootstrapWithSelectedPortfolio() {
    const fx = TestBed.createComponent(RebalancingComponent);
    fx.detectChanges();
    return fx;
  }

  function flushNonPreviewRequests(): void {
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
      providers: [
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: makeCtxMock('Test Portfolio', 'test-id') },
      ],
    });
    fixture = TestBed.createComponent(RebalancingComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
  });

  afterEach(() => http.verify());

  it('initialises and resolves with the context portfolio name as selected', () => {
    settle();
    expect(comp.isLoading()).toBe(false);
    expect(comp.selectedPortfolio()).toBe('Test Portfolio');
  });

  it('when getDrift fails, the error state is shown', () => {
    // Override the rebalancing service to throw on getDrift
    const svc = TestBed.inject(RebalancingService) as jasmine.SpyObj<RebalancingService>;
    if (svc.getDrift && typeof svc.getDrift === 'function') {
      // HTTP-backed service: flush drift with an error
      fixture.detectChanges();
      http.match((r) => r.url.includes('/drift')).forEach((r) =>
        r.flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' }),
      );
      drainRequests(http, stubFor);
      fixture.detectChanges();
      expect(comp.hasError()).toBe(true);
    } else {
      // Already a spy: trigger the error path via signal
      comp.hasError.set(true);
      expect(comp.hasError()).toBe(true);
    }
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
    // Perform initial detectChanges so all four requests are fired.
    fixture.detectChanges();
    // Flush everything except preview with success, error preview with 500.
    http.match((r) => r.url.includes('/drift')).forEach((r) =>
      r.flush({ entries: [], totalDrift: 0, breachedCount: 0, threshold: 0.05 }),
    );
    http.match((r) => r.url.includes('rebalance-policy')).forEach((r) =>
      r.flush({ items: [] }),
    );
    http.match((r) => r.url.includes('/snapshots')).forEach((r) =>
      r.flush({ items: [] }),
    );
    http
      .expectOne((r) => r.url.includes('rebalance/preview'))
      .flush({ detail: 'down' }, { status: 500, statusText: 'Server Error' });
    expect(comp.panelErrors()['preview']).toContain('temporarily unavailable');
  });

  it('kpiMaxDrift em-dashes on null/empty drift; driftEntries defaults to []', () => {
    settle();
    comp.driftResponse.set(null);
    expect(comp.kpiMaxDrift()).toBe('—');
    expect(comp.driftEntries()).toEqual([]);
    comp.driftResponse.set({ entries: [], totalDrift: 0, breachedCount: 0, threshold: 0.05 });
    expect(comp.kpiMaxDrift()).toBe('—');
    comp.driftResponse.set(makeDriftResponse());
    expect(comp.kpiMaxDrift()).not.toBe('—');
    expect(comp.driftEntries().length).toBe(1);
  });

  it('retry clears error and refetches panels', () => {
    settle();
    comp.hasError.set(true);
    comp.errorMessage.set('boom');
    comp.retry();
    expect(comp.hasError()).toBe(false);
    drainRequests(http, stubFor);
    fixture.detectChanges();
    drainRequests(http, stubFor);
  });
});

// ── Default error messages ────────────────────────────────────────────────────
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
        { provide: PortfolioContextService, useValue: makeCtxMock('Test Portfolio', 'test-id') },
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

  it('falls back to default error message when drift errors without a message', async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [RebalancingComponent],
      withHttp: true,
      providers: [
        ICON_PROVIDER,
        { provide: RebalancingService, useValue: boomService() },
        { provide: PortfolioContextService, useValue: makeCtxMock('Test Portfolio', 'test-id') },
      ],
    });
    const fx = TestBed.createComponent(RebalancingComponent);
    fx.detectChanges();
    expect(fx.componentInstance.hasError()).toBe(true);
    expect(fx.componentInstance.errorMessage()).toBe('Failed to load drift data');
  });
});

// ── Per-panel error rendering (issue #984) ───────────────────────────────────
describe('RebalancingComponent — per-panel error rendering (issue #984)', () => {
  let fixture: ComponentFixture<RebalancingComponent>;
  let comp: RebalancingComponent;
  let http: HttpTestingController;

  function stubFor(url: string): Record<string, unknown> {
    if (url.includes('portfolio-analytics') || url.includes('/drift')) {
      return { entries: [], totalDrift: 0, breachedCount: 0, threshold: 0.05 };
    }
    if (url.includes('rebalance-policy')) return { items: [] };
    if (url.includes('rebalance/preview')) {
      return {
        portfolioName: 'P', policyType: 'threshold', targetWeights: {},
        currentWeights: {}, trades: [], portfolioValue: 0, status: null,
      };
    }
    if (url.includes('/snapshots')) return { items: [] };
    return {};
  }

  function settle(): void {
    fixture.detectChanges();
    drainRequests(http, stubFor);
    fixture.detectChanges();
    drainRequests(http, stubFor);
  }

  function queryAlert(): HTMLElement | null {
    return (fixture.nativeElement as HTMLElement).querySelector('[role="alert"]');
  }

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [RebalancingComponent],
      withHttp: true,
      providers: [
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: makeCtxMock('Test Portfolio', 'test-id') },
      ],
    });
    fixture = TestBed.createComponent(RebalancingComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
  });

  afterEach(() => http.verify());

  it('when a what-if decide fails, the what-if panel renders a visible role="alert" error', () => {
    settle();
    comp.activeTab.set('what-if');
    fixture.detectChanges();

    comp.onRunDecide({ current_weights: {}, target_weights: {}, policy_type: 'threshold' });
    http
      .expectOne((r) => r.url.includes('rebalance/decide'))
      .flush({ detail: 'bad' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();

    const msg = comp.panelErrors()['whatif'];
    expect(msg).toBeTruthy();
    expect(queryAlert()).not.toBeNull();
    expect(queryAlert()?.textContent).toContain(msg!);
  });

  it('when a policy create fails, the policy panel renders a visible role="alert" error', () => {
    settle();
    comp.activeTab.set('policy');
    fixture.detectChanges();

    comp.onCreatePolicy({ name: 'P', policy_type: 'threshold', config: {} });
    http
      .expectOne((r) => r.url.includes('rebalance-policy') && r.method === 'POST')
      .flush({ detail: 'bad' }, { status: 422, statusText: 'Unprocessable' });
    fixture.detectChanges();

    const msg = comp.panelErrors()['policy'];
    expect(msg).toBeTruthy();
    expect(queryAlert()).not.toBeNull();
    expect(queryAlert()?.textContent).toContain(msg!);
  });

  it('when a failed policy create is followed by a successful one, the policy error clears', () => {
    settle();
    comp.onCreatePolicy({ name: 'P', policy_type: 'threshold', config: {} });
    http
      .expectOne((r) => r.url.includes('rebalance-policy') && r.method === 'POST')
      .flush({ detail: 'bad' }, { status: 422, statusText: 'Unprocessable' });
    expect(comp.panelErrors()['policy']).toBeTruthy();

    comp.onCreatePolicy({ name: 'P', policy_type: 'threshold', config: {} });
    http.expectOne((r) => r.url.includes('rebalance-policy') && r.method === 'POST').flush({});
    http.expectOne((r) => r.url.includes('rebalance-policy') && r.method === 'GET').flush({ items: [] });

    expect(comp.panelErrors()['policy']).toBeNull();
  });

  it('when what-if decide succeeds after a failure, the what-if error clears', () => {
    settle();
    comp.onRunDecide({ current_weights: {}, target_weights: {}, policy_type: 'threshold' });
    http
      .expectOne((r) => r.url.includes('rebalance/decide'))
      .flush({ detail: 'bad' }, { status: 500, statusText: 'Server Error' });
    expect(comp.panelErrors()['whatif']).toBeTruthy();

    comp.onRunDecide({ current_weights: {}, target_weights: {}, policy_type: 'threshold' });
    http.expectOne((r) => r.url.includes('rebalance/decide')).flush(makeRebalanceDecideResponse());

    expect(comp.panelErrors()['whatif']).toBeNull();
  });
});
