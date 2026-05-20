import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { BuilderDriftService } from './builder-drift.service';
import { BuilderStore } from './builder.store';
import { environment } from '../../../environments/environment';
import type { DriftResponse } from '../../models/drift.model';

const API = environment.apiUrl;
const PORTFOLIO = 'core';
const DRIFT_URL = `${API}portfolio/${PORTFOLIO}/drift`;

function sampleDriftResponse(requestId = 1): DriftResponse {
  return {
    holdings: [],
    target: [
      { ticker: 'AAPL', weight: 0.5 },
      { ticker: 'MSFT', weight: 0.5 },
    ],
    drift: [
      {
        ticker: 'AAPL',
        current_weight: 0.55,
        target_weight: 0.5,
        delta_weight: 0.05,
        eur_value: 5500,
        flags: [],
      },
      {
        ticker: 'MSFT',
        current_weight: 0.45,
        target_weight: 0.5,
        delta_weight: -0.05,
        eur_value: 4500,
        flags: [],
      },
    ],
    trades: [
      {
        ticker: 'AAPL',
        action: 'sell',
        delta_eur: -500,
        est_shares: -3,
        est_cost_eur: 0.5,
      },
      {
        ticker: 'MSFT',
        action: 'buy',
        delta_eur: 500,
        est_shares: 4,
        est_cost_eur: 0.5,
      },
    ],
    totals: {
      deployable_eur: 10000,
      total_holdings_eur: 10000,
      total_drift_abs: 0.1,
      buy_eur: 500,
      sell_eur: 500,
    },
    diagnostics: {
      reconciliation_ok: true,
      reconciliation_delta_pct: 0.0001,
      unmapped_count: 0,
      fx_missing_count: 0,
      target_not_on_broker_count: 0,
      base_currency: 'EUR',
    },
    request_id: requestId,
  };
}

function flushEffects(): Promise<void> {
  TestBed.tick();
  return new Promise((resolve) => setTimeout(resolve, 5));
}

describe('BuilderDriftService', () => {
  let service: BuilderDriftService;
  let store: BuilderStore;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        BuilderStore,
        BuilderDriftService,
      ],
    });
    store = TestBed.inject(BuilderStore);
    service = TestBed.inject(BuilderDriftService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
  });

  function arm(): void {
    store.setPortfolioName(PORTFOLIO);
    store.setResultStatus('ok');
  }

  it('when freshly constructed, driftStatus stays "idle"', () => {
    expect(store.driftStatus()).toBe('idle');
  });

  it('when runExplicit is called before init, no HTTP fires', () => {
    arm();
    service.runExplicit();
    http.expectNone(() => true);
    expect(store.driftStatus()).toBe('idle');
  });

  it('when portfolioName is null and runExplicit is called after init, no HTTP fires', () => {
    service.init();
    store.setResultStatus('ok');
    service.runExplicit();
    http.expectNone(() => true);
    expect(store.driftStatus()).toBe('idle');
  });

  it('when resultStatus is not "ok" and runExplicit is called, no HTTP fires', () => {
    service.init();
    store.setPortfolioName(PORTFOLIO);
    store.setResultStatus('running');
    service.runExplicit();
    http.expectNone(() => true);
    expect(store.driftStatus()).toBe('idle');
  });

  it('when guards pass and runExplicit is called, GET fires with portfolio name and base param', () => {
    service.init();
    arm();
    service.runExplicit();
    const req = http.expectOne((r) => r.url === DRIFT_URL);
    expect(req.request.method).toBe('GET');
    expect(req.request.params.get('base')).toBe('invested');
    req.flush(sampleDriftResponse());
  });

  it('when guards pass and runExplicit is called, X-Drift-Request-Id header is sent with seq', () => {
    service.init();
    arm();
    service.runExplicit();
    const req = http.expectOne((r) => r.url === DRIFT_URL);
    expect(req.request.headers.get('X-Drift-Request-Id')).toBe('1');
    req.flush(sampleDriftResponse());
    service.runExplicit();
    const req2 = http.expectOne((r) => r.url === DRIFT_URL);
    expect(req2.request.headers.get('X-Drift-Request-Id')).toBe('2');
    req2.flush(sampleDriftResponse(2));
  });

  it('when runExplicit fires, driftStatus optimistically becomes "running"', () => {
    service.init();
    arm();
    service.runExplicit();
    expect(store.driftStatus()).toBe('running');
    const req = http.expectOne((r) => r.url === DRIFT_URL);
    req.flush(sampleDriftResponse());
  });

  it('when GET succeeds, store.drift, trades, driftDiagnostics, status are populated', () => {
    service.init();
    arm();
    service.runExplicit();
    const req = http.expectOne((r) => r.url === DRIFT_URL);
    req.flush(sampleDriftResponse(7));

    expect(store.driftStatus()).toBe('ok');
    expect(store.drift()).not.toBeNull();
    expect(store.drift()!.portfolioName).toBe(PORTFOLIO);
    expect(store.drift()!.drift.length).toBe(2);
    expect(store.trades()).not.toBeNull();
    expect(store.trades()!.trades.length).toBe(2);
    expect(store.driftDiagnostics()).not.toBeNull();
    expect(store.driftDiagnostics()!.requestId).toBe(7);
  });

  it('when GET succeeds, BuilderTrade rows carry delta_weight joined from DriftRow on ticker', () => {
    service.init();
    arm();
    service.runExplicit();
    const req = http.expectOne((r) => r.url === DRIFT_URL);
    req.flush(sampleDriftResponse());

    const trades = store.trades()!.trades;
    const aapl = trades.find((t) => t.ticker === 'AAPL')!;
    const msft = trades.find((t) => t.ticker === 'MSFT')!;
    expect(aapl.delta_weight).toBe(0.05);
    expect(msft.delta_weight).toBe(-0.05);
  });

  it('when GET fails with 500, driftStatus transitions to "error"', () => {
    service.init();
    arm();
    service.runExplicit();
    const req = http.expectOne((r) => r.url === DRIFT_URL);
    req.flush('boom', { status: 500, statusText: 'Server Error' });
    expect(store.driftStatus()).toBe('error');
  });

  it('when base toggle changes mid-flight, prior request is cancelled and only one pending request remains', async () => {
    service.init();
    arm();
    service.runExplicit();
    http.expectOne((r) => r.url === DRIFT_URL);

    store.setDeployableBase('total');
    await flushEffects();

    const pending = http.match((r) => r.url === DRIFT_URL);
    expect(pending.length).toBe(1);
    expect(pending[0].request.params.get('base')).toBe('total');
    pending[0].flush(sampleDriftResponse(2));
  });

  it('when a base toggle change supersedes a pending request, switchMap cancels it and only the latest response lands in the store', async () => {
    service.init();
    arm();
    service.runExplicit();
    const first = http.expectOne((r) => r.url === DRIFT_URL);

    store.setDeployableBase('total');
    await flushEffects();

    expect(first.cancelled).toBe(true);
    const second = http.expectOne((r) => r.url === DRIFT_URL);
    second.flush(sampleDriftResponse(2));
    expect(store.driftDiagnostics()!.requestId).toBe(2);
  });

  it('when currentView becomes "drift" and resultStatus is "ok", auto-fetch fires exactly once', async () => {
    service.init();
    store.setPortfolioName(PORTFOLIO);
    store.setResultStatus('ok');
    store.setCurrentView('drift');
    await flushEffects();

    const reqs = http.match((r) => r.url === DRIFT_URL);
    expect(reqs.length).toBe(1);
    reqs[0].flush(sampleDriftResponse());

    await flushEffects();
    expect(http.match((r) => r.url === DRIFT_URL).length).toBe(0);
  });

  it('when currentView leaves "drift" and returns, auto-fetch fires again', async () => {
    service.init();
    store.setPortfolioName(PORTFOLIO);
    store.setResultStatus('ok');
    store.setCurrentView('drift');
    await flushEffects();
    const first = http.match((r) => r.url === DRIFT_URL);
    expect(first.length).toBe(1);
    first[0].flush(sampleDriftResponse());

    store.setCurrentView('donut');
    await flushEffects();
    store.setCurrentView('drift');
    await flushEffects();

    const second = http.match((r) => r.url === DRIFT_URL);
    expect(second.length).toBe(1);
    second[0].flush(sampleDriftResponse(2));
  });

  it('when resultStatus is not "ok" and currentView becomes "drift", auto-fetch does not fire', async () => {
    service.init();
    store.setPortfolioName(PORTFOLIO);
    store.setResultStatus('idle');
    store.setCurrentView('drift');
    await flushEffects();
    expect(http.match((r) => r.url === DRIFT_URL).length).toBe(0);
  });

  it('when base toggle is set to the same value, no retrigger fires', async () => {
    service.init();
    arm();
    store.setDeployableBase('invested');
    await flushEffects();
    expect(http.match((r) => r.url === DRIFT_URL).length).toBe(0);
  });

  it('when runExplicit is called twice in succession, prior request is cancelled and only the latest remains pending', () => {
    service.init();
    arm();
    service.runExplicit();
    const first = http.expectOne((r) => r.url === DRIFT_URL);
    service.runExplicit();
    expect(first.cancelled).toBe(true);
    const pending = http.match((r) => r.url === DRIFT_URL);
    expect(pending.length).toBe(1);
    expect(pending[0].request.headers.get('X-Drift-Request-Id')).toBe('2');
    pending[0].flush(sampleDriftResponse(2));
  });

  it('when prior request is cancelled and the latest response lands, store reflects only the latest packet', () => {
    service.init();
    arm();
    service.runExplicit();
    const first = http.expectOne((r) => r.url === DRIFT_URL);
    service.runExplicit();
    expect(first.cancelled).toBe(true);
    const second = http.expectOne((r) => r.url === DRIFT_URL);
    second.flush(sampleDriftResponse(42));
    expect(store.driftDiagnostics()!.requestId).toBe(42);
    expect(store.driftStatus()).toBe('ok');
  });

  it('when GET fails with 502, driftStatus becomes "error" and no drift/trades/diagnostics setters are called', () => {
    service.init();
    arm();
    service.runExplicit();
    const req = http.expectOne((r) => r.url === DRIFT_URL);
    req.flush('upstream broken', { status: 502, statusText: 'Bad Gateway' });
    expect(store.driftStatus()).toBe('error');
    expect(store.drift()).toBeNull();
    expect(store.trades()).toBeNull();
    expect(store.driftDiagnostics()).toBeNull();
  });

  it('when currentView is "drift" and base toggle changes, a new fetch fires with updated base param', async () => {
    service.init();
    store.setPortfolioName(PORTFOLIO);
    store.setResultStatus('ok');
    store.setCurrentView('drift');
    await flushEffects();
    const auto = http.expectOne((r) => r.url === DRIFT_URL);
    auto.flush(sampleDriftResponse());

    store.setDeployableBase('total');
    await flushEffects();
    const after = http.expectOne((r) => r.url === DRIFT_URL);
    expect(after.request.params.get('base')).toBe('total');
    after.flush(sampleDriftResponse(2));
  });
});
