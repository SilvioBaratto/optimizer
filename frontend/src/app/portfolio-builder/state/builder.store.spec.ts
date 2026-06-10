import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { BUILDER_DRIFT_SERVICE, BuilderStore, type CanvasView } from './builder.store';
import { BuilderResultService } from '../builder-result.service';
import type { BuilderStage } from '../models/builder-stage';
import type {
  BuilderBacktest,
  BuilderDrift,
  BuilderDriftDiagnostics,
  BuilderFrontier,
  BuilderResult,
  BuilderTrades,
} from '../models/builder-result.model';
import {
  type RunLevelConfig,
  StepStatus,
} from '../../core/models/pipeline-builder.model';
import { STEP_PARAM_DEFAULTS } from '../../pipeline-stepper/models/step-params.model';
import { environment } from '../../../environments/environment';

function sampleResult(): BuilderResult {
  return {
    runId: 'run-1',
    weights: { AAPL: 0.5, MSFT: 0.5 },
    metrics: { sharpe: 1.2 },
    riskContributions: { AAPL: 0.5, MSFT: 0.5 },
    optimizerType: 'mean_risk',
    createdAt: '2026-05-19T10:00:00Z',
  };
}

function sampleFrontier(): BuilderFrontier {
  return {
    points: [
      { risk: 0.1, return: 0.05, sharpe: 0.5 },
      { risk: 0.2, return: 0.12, sharpe: 0.6 },
    ],
  };
}

function sampleBacktest(): BuilderBacktest {
  return {
    runId: 'bt-1',
    summaryStats: { totalReturn: 0.12, sharpe: 0.9, maxDrawdown: -0.08 },
    equityCurve: { '2026-01-01': 1.0, '2026-02-01': 1.05 },
    createdAt: '2026-05-19T10:01:00Z',
  };
}

function sampleDrift(): BuilderDrift {
  return {
    drift: [
      {
        ticker: 'AAPL',
        current_weight: 0.55,
        target_weight: 0.5,
        delta_weight: 0.05,
        eur_value: 5500,
        flags: [],
      },
    ],
    totals: {
      deployable_eur: 10000,
      total_holdings_eur: 10000,
      total_drift_abs: 0.05,
      buy_eur: 0,
      sell_eur: 500,
    },
    portfolioName: 'core',
  };
}

function sampleTrades(): BuilderTrades {
  return {
    trades: [
      {
        ticker: 'AAPL',
        action: 'sell',
        delta_eur: -500,
        est_shares: -3,
        est_cost_eur: 0.5,
        delta_weight: 0.05,
        flags: [],
      },
    ],
  };
}

function sampleDriftDiagnostics(): BuilderDriftDiagnostics {
  return {
    diagnostics: {
      reconciliation_ok: true,
      reconciliation_delta_pct: 0.0001,
      unmapped_count: 0,
      fx_missing_count: 0,
      target_not_on_broker_count: 0,
      base_currency: 'EUR',
      sum_eur: 10000,
      invested_eur: 10000,
      delta_eur: 0,
      tolerance_pct: 0.015,
      stale_price_count: 0,
      entries: [],
    },
    requestId: 7,
  };
}

const SESSIONS_URL = `${environment.apiUrl}pipeline-builder/sessions`;
const STEP_URL = (sid: string, step: string) =>
  `${SESSIONS_URL}/${sid}/steps/${step}`;
const ACCEPTED = { status: 202, statusText: 'Accepted' };

function acceptResponse() {
  return { job_id: 'j-1', status: 'pending', message: '' };
}

function defaultConfig(): RunLevelConfig {
  return {
    rebalance_freq: 63,
    n_selected: 20,
    cost_bps: 10,
    tax_rate: 0.26,
    base_currency: 'EUR',
    robust: false,
    persist: false,
    start_date: null,
    end_date: null,
    seed: 42,
  };
}

describe('BuilderStore', () => {
  let store: BuilderStore;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        BuilderStore,
      ],
    });
    store = TestBed.inject(BuilderStore);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
  });

  it('when freshly constructed, initial summary has no config and status Pending', () => {
    expect(store.config()).toBeNull();
    expect(store.stepParams()).toBeNull();
    expect(store.currentStage()).toBeNull();
    expect(store.status()).toBe(StepStatus.Pending);
    expect(store.summary()).toEqual({
      stage: null,
      status: StepStatus.Pending,
      hasConfig: false,
    });
  });

  it('when freshly constructed, sessionId is null', () => {
    expect(store.sessionId()).toBeNull();
  });

  it('when setConfig is called, hasConfig flips to true', () => {
    const cfg = defaultConfig();
    store.setConfig(cfg);
    expect(store.config()).toBe(cfg);
    expect(store.summary().hasConfig).toBe(true);
  });

  it('when setStepParams is called, stepParams signal exposes the value', () => {
    const params = structuredClone(STEP_PARAM_DEFAULTS);
    store.setStepParams(params);
    expect(store.stepParams()).toBe(params);
  });

  it('when setStage is called with "universe", currentStage and summary.stage update', () => {
    store.setStage('universe');
    expect(store.currentStage()).toBe('universe');
    expect(store.summary().stage).toBe('universe');
  });

  it('when setStage cycles through all 6 BuilderStage ids, each is accepted and exposed', () => {
    const stages: readonly BuilderStage[] = [
      'universe',
      'objective',
      'constraints',
      'optimize',
      'review',
      'rebalance',
    ];
    for (const stage of stages) {
      store.setStage(stage);
      expect(store.currentStage()).toBe(stage);
      expect(store.summary().stage).toBe(stage);
    }
  });

  it('when setStage is called with null, currentStage clears back to null', () => {
    store.setStage('optimize');
    store.setStage(null);
    expect(store.currentStage()).toBeNull();
    expect(store.summary().stage).toBeNull();
  });

  it('when setStatus is called with Running, summary.status propagates', () => {
    store.setStatus(StepStatus.Running);
    expect(store.status()).toBe(StepStatus.Running);
    expect(store.summary().status).toBe(StepStatus.Running);
  });

  it('when setSessionId is called with a string, sessionId exposes that value', () => {
    store.setSessionId('sid-xyz');
    expect(store.sessionId()).toBe('sid-xyz');
  });

  it('when setSessionId is called with null after being set, sessionId returns to null', () => {
    store.setSessionId('sid-xyz');
    store.setSessionId(null);
    expect(store.sessionId()).toBeNull();
  });

  it('when freshly constructed, stepResults and stepStatuses are empty maps', () => {
    expect(store.stepResults().size).toBe(0);
    expect(store.stepStatuses().size).toBe(0);
  });

  it('when setStepResult is called, stepResults exposes the entry under the step id', () => {
    const result = { n_tickers: 42 } as Record<string, unknown>;
    store.setStepResult('load', result);
    expect(store.stepResults().get('load')).toBe(result);
  });

  it('when setStepResult is called twice for different steps, both entries are preserved', () => {
    const a = { n_tickers: 1 } as Record<string, unknown>;
    const b = { n_investable: 2 } as Record<string, unknown>;
    store.setStepResult('load', a);
    store.setStepResult('screen', b);
    expect(store.stepResults().get('load')).toBe(a);
    expect(store.stepResults().get('screen')).toBe(b);
    expect(store.stepResults().size).toBe(2);
  });

  it('when setStepResult is called with null, the entry is stored as null', () => {
    store.setStepResult('load', null);
    expect(store.stepResults().has('load')).toBe(true);
    expect(store.stepResults().get('load')).toBeNull();
  });

  it('when setStepStatus is called, stepStatuses exposes the entry under the step id', () => {
    store.setStepStatus('optimize', StepStatus.Running);
    expect(store.stepStatuses().get('optimize')).toBe(StepStatus.Running);
  });

  it('when setStepStatus mutates the same step twice, the latest value wins', () => {
    store.setStepStatus('load', StepStatus.Running);
    store.setStepStatus('load', StepStatus.Completed);
    expect(store.stepStatuses().get('load')).toBe(StepStatus.Completed);
  });

  it('when reset is called after mutations, all signals return to initial', () => {
    store.setConfig(defaultConfig());
    store.setStepParams(structuredClone(STEP_PARAM_DEFAULTS));
    store.setStage('optimize');
    store.setStatus(StepStatus.Completed);
    store.setStepResult('load', { n_tickers: 1 } as Record<string, unknown>);
    store.setStepStatus('load', StepStatus.Completed);
    store.setSessionId('sid-old');

    store.reset();

    expect(store.config()).toBeNull();
    expect(store.stepParams()).toBeNull();
    expect(store.currentStage()).toBeNull();
    expect(store.status()).toBe(StepStatus.Pending);
    expect(store.stepResults().size).toBe(0);
    expect(store.stepStatuses().size).toBe(0);
    expect(store.sessionId()).toBeNull();
    expect(store.summary()).toEqual({
      stage: null,
      status: StepStatus.Pending,
      hasConfig: false,
    });
  });

  describe('optimize()', () => {
    it('when sessionId is null, optimize() is a no-op and no HTTP request is made', () => {
      store.optimize();
      http.expectNone(() => true);
      expect(store.status()).toBe(StepStatus.Pending);
    });

    it('when status is already Running, optimize() is a no-op and no HTTP request is made', () => {
      store.setSessionId('sid-1');
      store.setStatus(StepStatus.Running);
      store.optimize();
      http.expectNone(() => true);
      expect(store.status()).toBe(StepStatus.Running);
    });

    it('when sessionId is set and status is not Running, optimize() POSTs to /sessions/{sid}/steps/optimize with empty body', () => {
      store.setSessionId('sid-1');
      store.optimize();
      const req = http.expectOne(STEP_URL('sid-1', 'optimize'));
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({});
      req.flush(acceptResponse(), ACCEPTED);
    });

    it('when optimize() dispatches, status is set to Running optimistically', () => {
      store.setSessionId('sid-1');
      store.optimize();
      expect(store.status()).toBe(StepStatus.Running);
      const req = http.expectOne(STEP_URL('sid-1', 'optimize'));
      req.flush(acceptResponse(), ACCEPTED);
    });

    it('when optimize() POST fails, status transitions to Error', () => {
      store.setSessionId('sid-1');
      store.optimize();
      const req = http.expectOne(STEP_URL('sid-1', 'optimize'));
      req.flush('boom', { status: 500, statusText: 'Server Error' });
      expect(store.status()).toBe(StepStatus.Error);
    });
  });

  describe('rebalance()', () => {
    it('when sessionId is null, rebalance() is a no-op and no HTTP request is made', () => {
      store.rebalance();
      http.expectNone(() => true);
      expect(store.status()).toBe(StepStatus.Pending);
    });

    it('when status is already Running, rebalance() is a no-op and no HTTP request is made', () => {
      store.setSessionId('sid-1');
      store.setStatus(StepStatus.Running);
      store.rebalance();
      http.expectNone(() => true);
      expect(store.status()).toBe(StepStatus.Running);
    });

    it('when sessionId is set and status is not Running, rebalance() POSTs to /sessions/{sid}/steps/rebalance_decision with empty body', () => {
      store.setSessionId('sid-2');
      store.rebalance();
      const req = http.expectOne(STEP_URL('sid-2', 'rebalance_decision'));
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({});
      req.flush(acceptResponse(), ACCEPTED);
    });

    it('when rebalance() dispatches, status is set to Running optimistically', () => {
      store.setSessionId('sid-2');
      store.rebalance();
      expect(store.status()).toBe(StepStatus.Running);
      const req = http.expectOne(STEP_URL('sid-2', 'rebalance_decision'));
      req.flush(acceptResponse(), ACCEPTED);
    });

    it('when rebalance() POST fails, status transitions to Error', () => {
      store.setSessionId('sid-2');
      store.rebalance();
      const req = http.expectOne(STEP_URL('sid-2', 'rebalance_decision'));
      req.flush('boom', { status: 500, statusText: 'Server Error' });
      expect(store.status()).toBe(StepStatus.Error);
    });
  });

  describe('result signals', () => {
    it('when freshly constructed, resultStatus is "idle" and result/frontier/backtest are null', () => {
      expect(store.resultStatus()).toBe('idle');
      expect(store.result()).toBeNull();
      expect(store.frontier()).toBeNull();
      expect(store.backtest()).toBeNull();
    });

    it('when setResultStatus is called, resultStatus exposes the new value', () => {
      store.setResultStatus('running');
      expect(store.resultStatus()).toBe('running');
      store.setResultStatus('ok');
      expect(store.resultStatus()).toBe('ok');
      store.setResultStatus('error');
      expect(store.resultStatus()).toBe('error');
      store.setResultStatus('stale');
      expect(store.resultStatus()).toBe('stale');
      store.setResultStatus('idle');
      expect(store.resultStatus()).toBe('idle');
    });

    it('when setResult is called, result exposes the same object', () => {
      const r = sampleResult();
      store.setResult(r);
      expect(store.result()).toBe(r);
    });

    it('when setResult is called with null after being set, result returns to null', () => {
      store.setResult(sampleResult());
      store.setResult(null);
      expect(store.result()).toBeNull();
    });

    it('when setFrontier is called, frontier exposes the same object', () => {
      const f = sampleFrontier();
      store.setFrontier(f);
      expect(store.frontier()).toBe(f);
    });

    it('when setFrontier is called with null after being set, frontier returns to null', () => {
      store.setFrontier(sampleFrontier());
      store.setFrontier(null);
      expect(store.frontier()).toBeNull();
    });

    it('when setBacktest is called, backtest exposes the same object', () => {
      const b = sampleBacktest();
      store.setBacktest(b);
      expect(store.backtest()).toBe(b);
    });

    it('when setBacktest is called with null after being set, backtest returns to null', () => {
      store.setBacktest(sampleBacktest());
      store.setBacktest(null);
      expect(store.backtest()).toBeNull();
    });

    it('when reset is called after mutating result signals, all four return to initial values', () => {
      store.setResultStatus('ok');
      store.setResult(sampleResult());
      store.setFrontier(sampleFrontier());
      store.setBacktest(sampleBacktest());

      store.reset();

      expect(store.resultStatus()).toBe('idle');
      expect(store.result()).toBeNull();
      expect(store.frontier()).toBeNull();
      expect(store.backtest()).toBeNull();
    });
  });

  describe('currentView signal', () => {
    it('when freshly constructed, currentView defaults to "donut"', () => {
      expect(store.currentView()).toBe('donut');
    });

    it('when setCurrentView is called with "donut", currentView exposes "donut"', () => {
      store.setCurrentView('bars');
      store.setCurrentView('donut');
      expect(store.currentView()).toBe('donut');
    });

    it('when setCurrentView is called with "bars", currentView exposes "bars"', () => {
      store.setCurrentView('bars');
      expect(store.currentView()).toBe('bars');
    });

    it('when setCurrentView is called with "frontier", currentView exposes "frontier"', () => {
      store.setCurrentView('frontier');
      expect(store.currentView()).toBe('frontier');
    });

    it('when setCurrentView is called with "backtest", currentView exposes "backtest"', () => {
      store.setCurrentView('backtest');
      expect(store.currentView()).toBe('backtest');
    });

    it('when setCurrentView is called with "drift", currentView exposes "drift"', () => {
      store.setCurrentView('drift');
      expect(store.currentView()).toBe('drift');
    });

    it('when reset is called after setCurrentView, currentView reverts to "donut"', () => {
      store.setCurrentView('frontier');
      store.reset();
      expect(store.currentView()).toBe('donut');
    });
  });

  describe('drift signals', () => {
    it('when freshly constructed, drift/trades/driftDiagnostics are null', () => {
      expect(store.drift()).toBeNull();
      expect(store.trades()).toBeNull();
      expect(store.driftDiagnostics()).toBeNull();
    });

    it('when freshly constructed, deployableBase defaults to "invested"', () => {
      expect(store.deployableBase()).toBe('invested');
    });

    it('when freshly constructed, driftStatus defaults to "idle"', () => {
      expect(store.driftStatus()).toBe('idle');
    });

    it('when freshly constructed, portfolioName defaults to null', () => {
      expect(store.portfolioName()).toBeNull();
    });

    it('when setDrift is called, drift exposes the same object', () => {
      const d = sampleDrift();
      store.setDrift(d);
      expect(store.drift()).toBe(d);
    });

    it('when setDrift is called with null after being set, drift returns to null', () => {
      store.setDrift(sampleDrift());
      store.setDrift(null);
      expect(store.drift()).toBeNull();
    });

    it('when setTrades is called, trades exposes the same object', () => {
      const t = sampleTrades();
      store.setTrades(t);
      expect(store.trades()).toBe(t);
    });

    it('when setTrades is called with null after being set, trades returns to null', () => {
      store.setTrades(sampleTrades());
      store.setTrades(null);
      expect(store.trades()).toBeNull();
    });

    it('when setDriftDiagnostics is called, driftDiagnostics exposes the same object', () => {
      const d = sampleDriftDiagnostics();
      store.setDriftDiagnostics(d);
      expect(store.driftDiagnostics()).toBe(d);
    });

    it('when setDeployableBase is called with "total", deployableBase exposes "total"', () => {
      store.setDeployableBase('total');
      expect(store.deployableBase()).toBe('total');
    });

    it('when setDeployableBase is called with "invested" after "total", deployableBase reverts to "invested"', () => {
      store.setDeployableBase('total');
      store.setDeployableBase('invested');
      expect(store.deployableBase()).toBe('invested');
    });

    it('when setDriftStatus is called, driftStatus exposes the new value', () => {
      store.setDriftStatus('running');
      expect(store.driftStatus()).toBe('running');
      store.setDriftStatus('ok');
      expect(store.driftStatus()).toBe('ok');
      store.setDriftStatus('error');
      expect(store.driftStatus()).toBe('error');
      store.setDriftStatus('stale');
      expect(store.driftStatus()).toBe('stale');
      store.setDriftStatus('idle');
      expect(store.driftStatus()).toBe('idle');
    });

    it('when setPortfolioName is called, portfolioName exposes the value', () => {
      store.setPortfolioName('core');
      expect(store.portfolioName()).toBe('core');
    });

    it('when setPortfolioName is called with null after being set, portfolioName returns to null', () => {
      store.setPortfolioName('core');
      store.setPortfolioName(null);
      expect(store.portfolioName()).toBeNull();
    });

    it('when reset is called after mutating drift signals, all return to initial values', () => {
      store.setDrift(sampleDrift());
      store.setTrades(sampleTrades());
      store.setDriftDiagnostics(sampleDriftDiagnostics());
      store.setDeployableBase('total');
      store.setDriftStatus('ok');
      store.setPortfolioName('core');

      store.reset();

      expect(store.drift()).toBeNull();
      expect(store.trades()).toBeNull();
      expect(store.driftDiagnostics()).toBeNull();
      expect(store.deployableBase()).toBe('invested');
      expect(store.driftStatus()).toBe('idle');
      expect(store.portfolioName()).toBeNull();
    });
  });

  describe('triggerDriftRun()', () => {
    let runner: { runExplicit: jasmine.Spy };

    beforeEach(() => {
      runner = { runExplicit: jasmine.createSpy('runExplicit') };
      TestBed.resetTestingModule();
      TestBed.configureTestingModule({
        providers: [
          provideZonelessChangeDetection(),
          provideHttpClient(),
          provideHttpClientTesting(),
          BuilderStore,
          { provide: BUILDER_DRIFT_SERVICE, useValue: runner },
        ],
      });
      store = TestBed.inject(BuilderStore);
      http = TestBed.inject(HttpTestingController);
    });

    it('when triggerDriftRun is called once, BUILDER_DRIFT_SERVICE.runExplicit is invoked exactly once', () => {
      store.triggerDriftRun();
      expect(runner.runExplicit).toHaveBeenCalledTimes(1);
    });

    it('when triggerDriftRun is called three times, runExplicit is invoked three times', () => {
      store.triggerDriftRun();
      store.triggerDriftRun();
      store.triggerDriftRun();
      expect(runner.runExplicit).toHaveBeenCalledTimes(3);
    });

    it('when triggerDriftRun is never called, runExplicit is not invoked', () => {
      expect(runner.runExplicit).not.toHaveBeenCalled();
    });
  });

  describe('triggerResultRun()', () => {
    let resultRunner: { runExplicit: jasmine.Spy };

    beforeEach(() => {
      resultRunner = { runExplicit: jasmine.createSpy('runExplicit') };
      TestBed.resetTestingModule();
      TestBed.configureTestingModule({
        providers: [
          provideZonelessChangeDetection(),
          provideHttpClient(),
          provideHttpClientTesting(),
          BuilderStore,
          { provide: BuilderResultService, useValue: resultRunner },
        ],
      });
      store = TestBed.inject(BuilderStore);
      http = TestBed.inject(HttpTestingController);
    });

    it('when triggerResultRun is called once, BuilderResultService.runExplicit is invoked exactly once', () => {
      store.triggerResultRun();
      expect(resultRunner.runExplicit).toHaveBeenCalledTimes(1);
    });

    it('when triggerResultRun is never called, runExplicit is not invoked', () => {
      expect(resultRunner.runExplicit).not.toHaveBeenCalled();
    });
  });

  describe('CanvasView union narrowing', () => {
    it('when "drift" is assigned to a CanvasView variable, the assignment is type-safe', () => {
      const view: CanvasView = 'drift';
      expect(view).toBe('drift');
    });

    it('when "donut" | "bars" | "frontier" | "backtest" | "drift" are assigned, all members satisfy the union', () => {
      const all: readonly CanvasView[] = [
        'donut',
        'bars',
        'frontier',
        'backtest',
        'drift',
      ];
      expect(all.length).toBe(5);
    });

    it('when an invalid string literal is assigned, the compiler rejects it (ts-expect-error marker)', () => {
      // @ts-expect-error 'invalid' is not part of the CanvasView union
      const bad: CanvasView = 'invalid';
      expect(bad as unknown as string).toBe('invalid');
    });
  });

  describe('export()', () => {
    const ARTIFACTS = ['report.md', 'weights.csv', 'metrics.json', 'checklist.json'];

    function spyOnAnchors(): { href: string; download: string }[] {
      const created: { href: string; download: string }[] = [];
      const realCreate = document.createElement.bind(document);
      spyOn(document, 'createElement').and.callFake((tag: string) => {
        const el = realCreate(tag) as HTMLElement;
        if (tag === 'a') {
          spyOn(el as HTMLAnchorElement, 'click').and.callFake(() =>
            created.push({
              href: (el as HTMLAnchorElement).getAttribute('href') ?? '',
              download: (el as HTMLAnchorElement).download,
            }),
          );
        }
        return el;
      });
      return created;
    }

    it('when export() is invoked with no sessionId, it does not throw and downloads nothing', () => {
      const clicks = spyOnAnchors();
      expect(() => store.export()).not.toThrow();
      expect(clicks.length).toBe(0);
      http.expectNone(() => true);
    });

    it('when export() is invoked with a sessionId, each of the four artifacts is downloaded', () => {
      const clicks = spyOnAnchors();
      store.setSessionId('sid-3');

      store.export();

      expect(clicks.map((c) => c.download)).toEqual(ARTIFACTS);
      for (const artifact of ARTIFACTS) {
        const hit = clicks.find((c) => c.download === artifact);
        expect(hit?.href).toContain(`sessions/sid-3/artifacts/${artifact}`);
      }
      http.expectNone(() => true);
    });

    it('when export() is invoked, status is unchanged', () => {
      spyOnAnchors();
      store.setSessionId('sid-3');
      store.setStatus(StepStatus.Completed);
      store.export();
      expect(store.status()).toBe(StepStatus.Completed);
    });
  });
});
