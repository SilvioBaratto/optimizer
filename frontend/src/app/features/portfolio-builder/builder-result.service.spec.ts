import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { Subject } from 'rxjs';

import {
  BUILDER_RESULT_DEBOUNCE_MS,
  BuilderResultService,
} from './builder-result.service';
import { BuilderStore } from './builder.store';
import { JOB_POLL_TICK } from '../../shared/job-progress-tracker/job-progress-tracker';
import { environment } from '../../../environments/environment';
import type { RunLevelConfig } from '../../models/pipeline-builder.model';
import type {
  OptimizationRunResponse,
  OptimizeAsyncResponse,
} from '../../models/optimization.model';
import type {
  BacktestProgressResponse,
  BacktestRunResponse,
} from '../../models/backtest.model';

const API = environment.apiUrl;
const OPT_URL = `${API}optimize`;
const BT_URL = `${API}backtest`;
const optRunUrl = (id: string) => `${OPT_URL}/${id}`;
const btJobUrl = (id: string) => `${BT_URL}/${id}`;
const btRunUrl = (id: string) => `${BT_URL}/runs/${id}`;

function defaultConfig(): RunLevelConfig {
  return {
    rebalance_freq: 63,
    n_selected: 20,
    cost_bps: 10,
    tax_rate: 0.26,
    base_currency: 'EUR',
    robust: false,
    persist: false,
    start_date: '2024-01-01',
    end_date: '2024-12-31',
    seed: 42,
  };
}

function tickers(): Record<string, unknown> {
  return { tickers: ['AAPL', 'MSFT'], n_tickers: 2 };
}

function optRun(
  id: string,
  status: OptimizationRunResponse['status'] = 'completed',
): OptimizationRunResponse {
  return {
    id,
    portfolioId: null,
    jobId: null,
    status,
    optimizerType: 'mean_risk',
    universeTickers: ['AAPL', 'MSFT'],
    config: {},
    weights: { AAPL: 0.5, MSFT: 0.5 },
    metrics: { sharpe: 1.2 },
    riskContributions: { AAPL: 0.5, MSFT: 0.5 },
    efficientFrontier: [{ risk: 0.1, return: 0.05, sharpe: 0.5 }],
    errorMessage: null,
    solverLog: null,
    durationSeconds: 1,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:01Z',
  };
}

function optAsync(jobId: string, runId: string): OptimizeAsyncResponse {
  return { job_id: jobId, run_id: runId };
}

function btProgress(
  status: BacktestProgressResponse['status'],
  error: string | null = null,
): BacktestProgressResponse {
  return {
    job_id: 'job-1',
    status,
    current: 1,
    total: 1,
    errors: [],
    result: null,
    error,
  };
}

function btRun(id: string): BacktestRunResponse {
  return {
    id,
    portfolioId: null,
    jobId: 'job-1',
    status: 'completed',
    config: {},
    equityCurve: { '2024-01-01': 1.0 },
    drawdowns: {},
    monthlyReturns: {},
    yearlyReturns: {},
    rollingMetrics: {},
    turnoverHistory: {},
    cvFoldMetrics: null,
    summaryStats: { totalReturn: 0.1 },
    errorMessage: null,
    durationSeconds: 1,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:01Z',
  };
}

describe('BuilderResultService', () => {
  let service: BuilderResultService;
  let store: BuilderStore;
  let http: HttpTestingController;
  let tick$: Subject<number>;

  beforeEach(() => {
    tick$ = new Subject<number>();
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: JOB_POLL_TICK, useValue: tick$.asObservable() },
        { provide: BUILDER_RESULT_DEBOUNCE_MS, useValue: 0 },
        BuilderStore,
        BuilderResultService,
      ],
    });
    store = TestBed.inject(BuilderStore);
    service = TestBed.inject(BuilderResultService);
    http = TestBed.inject(HttpTestingController);
  });

  function flushDebounce(): Promise<void> {
    TestBed.tick();
    return new Promise((resolve) => setTimeout(resolve, 5));
  }

  afterEach(() => {
    http.verify();
  });

  function arm(): void {
    store.setSessionId('sid-1');
    store.setConfig(defaultConfig());
    store.setStepResult('load', tickers());
  }

  it('when freshly constructed, no HTTP fires and resultStatus stays "idle"', () => {
    expect(store.resultStatus()).toBe('idle');
  });

  it('when runExplicit is called before init, no HTTP fires and status stays "idle"', () => {
    arm();
    service.runExplicit();
    expect(store.resultStatus()).toBe('idle');
  });

  it('when sessionId is null and runExplicit is called, status stays "idle" and no HTTP fires', () => {
    service.init();
    store.setConfig(defaultConfig());
    store.setStepResult('load', tickers());
    service.runExplicit();
    expect(store.resultStatus()).toBe('idle');
  });

  it('when config is null and runExplicit is called, status stays "idle" and no HTTP fires', () => {
    service.init();
    store.setSessionId('sid-1');
    store.setStepResult('load', tickers());
    service.runExplicit();
    expect(store.resultStatus()).toBe('idle');
  });

  it('when tickers are missing from step results, status transitions to "error" without firing HTTP', () => {
    service.init();
    store.setSessionId('sid-1');
    store.setConfig(defaultConfig());
    service.runExplicit();
    expect(store.resultStatus()).toBe('error');
  });

  it('when optimize is sync and backtest poll completes, result/frontier/backtest are populated and status becomes "ok"', () => {
    service.init();
    arm();
    service.runExplicit();

    http.expectOne(OPT_URL).flush(optRun('run-1'));
    http
      .expectOne(BT_URL)
      .flush({ jobId: 'job-1', runId: 'bt-1', status: 'pending', message: '' });
    tick$.next(0);
    http.expectOne(btJobUrl('job-1')).flush(btProgress('completed'));
    http.expectOne(btRunUrl('bt-1')).flush(btRun('bt-1'));

    expect(store.resultStatus()).toBe('ok');
    expect(store.result()?.runId).toBe('run-1');
    expect(store.frontier()?.points.length).toBe(1);
    expect(store.backtest()?.runId).toBe('bt-1');
  });

  it('when optimize returns 202 async response, the run is polled until completed and the chain proceeds', () => {
    service.init();
    arm();
    service.runExplicit();

    http.expectOne(OPT_URL).flush(optAsync('oj-1', 'opt-1'));
    tick$.next(0);
    http.expectOne(optRunUrl('opt-1')).flush(optRun('opt-1', 'running'));
    tick$.next(1);
    http.expectOne(optRunUrl('opt-1')).flush(optRun('opt-1', 'completed'));

    http
      .expectOne(BT_URL)
      .flush({ jobId: 'job-1', runId: 'bt-1', status: 'pending', message: '' });
    tick$.next(2);
    http.expectOne(btJobUrl('job-1')).flush(btProgress('completed'));
    http.expectOne(btRunUrl('bt-1')).flush(btRun('bt-1'));

    expect(store.resultStatus()).toBe('ok');
    expect(store.result()?.runId).toBe('opt-1');
  });

  it('when POST /optimize fails with 500, status transitions to "error" and backtest is not called', () => {
    service.init();
    arm();
    service.runExplicit();

    http.expectOne(OPT_URL).flush('boom', { status: 500, statusText: 'Server Error' });

    expect(store.resultStatus()).toBe('error');
    expect(store.result()).toBeNull();
    expect(store.backtest()).toBeNull();
    http.expectNone(BT_URL);
  });

  it('when backtest poll returns failed, status transitions to "error" and no run is fetched', () => {
    service.init();
    arm();
    service.runExplicit();

    http.expectOne(OPT_URL).flush(optRun('run-1'));
    http
      .expectOne(BT_URL)
      .flush({ jobId: 'job-1', runId: 'bt-1', status: 'pending', message: '' });
    tick$.next(0);
    http.expectOne(btJobUrl('job-1')).flush(btProgress('failed', 'panic'));

    expect(store.resultStatus()).toBe('error');
    expect(store.backtest()).toBeNull();
    http.expectNone(btRunUrl('bt-1'));
  });

  it('when runExplicit is called twice rapidly, the first chain is cancelled and only the second response writes to the store', () => {
    service.init();
    arm();
    service.runExplicit();
    service.runExplicit();

    const live = http.match(OPT_URL).filter((r) => !r.cancelled);
    expect(live.length).toBe(1);
    live[0].flush(optRun('second-run'));

    http
      .expectOne(BT_URL)
      .flush({ jobId: 'job-1', runId: 'bt-2', status: 'pending', message: '' });
    tick$.next(0);
    http.expectOne(btJobUrl('job-1')).flush(btProgress('completed'));
    http.expectOne(btRunUrl('bt-2')).flush(btRun('bt-2'));

    expect(store.result()?.runId).toBe('second-run');
    expect(store.resultStatus()).toBe('ok');
  });

  it('when init is called twice, the trigger subscription is registered only once', () => {
    const setSpy = spyOn(store, 'setResultStatus').and.callThrough();
    service.init();
    service.init();
    store.setSessionId('sid-1');
    store.setConfig(defaultConfig());
    service.runExplicit();
    // Missing tickers => running + error per subscription. One sub => 2 calls.
    expect(setSpy).toHaveBeenCalledTimes(2);
  });

  it('when BuilderStore.triggerResultRun is called, it delegates to BuilderResultService.runExplicit', () => {
    const runSpy = spyOn(service, 'runExplicit');
    store.triggerResultRun();
    expect(runSpy).toHaveBeenCalledTimes(1);
  });

  it('when sessionId is null and setConfig is called repeatedly, no HTTP fires and status stays "idle"', async () => {
    service.init();
    store.setConfig(defaultConfig());
    store.setConfig({ ...defaultConfig(), seed: 99 });
    store.setConfig({ ...defaultConfig(), seed: 100 });
    await flushDebounce();
    expect(store.resultStatus()).toBe('idle');
    http.expectNone(OPT_URL);
  });

  it('when sessionId is set and config changes, the debounced auto-run fires a single optimize POST', async () => {
    service.init();
    arm();
    store.setConfig({ ...defaultConfig(), seed: 99 });
    await flushDebounce();

    const optReq = http.expectOne(OPT_URL);
    optReq.flush(optRun('auto-1'));
    http
      .expectOne(BT_URL)
      .flush({ jobId: 'job-1', runId: 'bt-1', status: 'pending', message: '' });
    tick$.next(0);
    http.expectOne(btJobUrl('job-1')).flush(btProgress('completed'));
    http.expectOne(btRunUrl('bt-1')).flush(btRun('bt-1'));

    expect(store.resultStatus()).toBe('ok');
    expect(store.result()?.runId).toBe('auto-1');
  });

  it('when resultStatus is "ok" and config changes, status immediately becomes "stale" before the debounced run starts', async () => {
    service.init();
    arm();
    store.setResultStatus('ok');
    store.setConfig({ ...defaultConfig(), seed: 99 });
    TestBed.tick();
    expect(store.resultStatus()).toBe('stale');

    await new Promise((resolve) => setTimeout(resolve, 5));
    const optReq = http.expectOne(OPT_URL);
    expect(store.resultStatus()).toBe('running');
    optReq.flush(optRun('auto-2'));
    http
      .expectOne(BT_URL)
      .flush({ jobId: 'job-1', runId: 'bt-2', status: 'pending', message: '' });
    tick$.next(0);
    http.expectOne(btJobUrl('job-1')).flush(btProgress('completed'));
    http.expectOne(btRunUrl('bt-2')).flush(btRun('bt-2'));

    expect(store.resultStatus()).toBe('ok');
  });

  it('when setConfig is called multiple times rapidly, debounce collapses them into a single chain', async () => {
    service.init();
    arm();
    store.setConfig({ ...defaultConfig(), seed: 1 });
    store.setConfig({ ...defaultConfig(), seed: 2 });
    store.setConfig({ ...defaultConfig(), seed: 3 });
    await flushDebounce();

    const live = http.match(OPT_URL).filter((r) => !r.cancelled);
    expect(live.length).toBe(1);
    live[0].flush(optRun('collapsed'));
    http
      .expectOne(BT_URL)
      .flush({ jobId: 'job-1', runId: 'bt-3', status: 'pending', message: '' });
    tick$.next(0);
    http.expectOne(btJobUrl('job-1')).flush(btProgress('completed'));
    http.expectOne(btRunUrl('bt-3')).flush(btRun('bt-3'));

    expect(store.result()?.runId).toBe('collapsed');
  });

  it('when runExplicit is called, it bypasses the debounce window and fires immediately', () => {
    service.init();
    arm();
    service.runExplicit();
    http.expectOne(OPT_URL).flush(optRun('explicit-1'));
    http
      .expectOne(BT_URL)
      .flush({ jobId: 'job-1', runId: 'bt-4', status: 'pending', message: '' });
    tick$.next(0);
    http.expectOne(btJobUrl('job-1')).flush(btProgress('completed'));
    http.expectOne(btRunUrl('bt-4')).flush(btRun('bt-4'));
    expect(store.result()?.runId).toBe('explicit-1');
  });

  describe('stale and explicit retry transitions', () => {
    it('when resultStatus is "ok" and a config update fires, resultStatus becomes "stale" synchronously before the debounce flushes', () => {
      service.init();
      arm();
      store.setResultStatus('ok');

      store.setConfig({ ...defaultConfig(), seed: 7 });
      TestBed.tick();

      expect(store.resultStatus()).toBe('stale');
      http.expectNone(OPT_URL);
    });

    it('when resultStatus is "error" and store.triggerResultRun is invoked with sessionId and config present, resultStatus becomes "running" and a fresh optimize chain starts', () => {
      service.init();
      arm();
      store.setResultStatus('error');

      store.triggerResultRun();

      expect(store.resultStatus()).toBe('running');
      const optReq = http.expectOne(OPT_URL);
      optReq.flush(optRun('retry-1'));
      http
        .expectOne(BT_URL)
        .flush({ jobId: 'job-1', runId: 'bt-retry', status: 'pending', message: '' });
      tick$.next(0);
      http.expectOne(btJobUrl('job-1')).flush(btProgress('completed'));
      http.expectOne(btRunUrl('bt-retry')).flush(btRun('bt-retry'));

      expect(store.resultStatus()).toBe('ok');
      expect(store.result()?.runId).toBe('retry-1');
    });

    it('when sessionId is null, calling store.triggerResultRun does not change resultStatus from "idle" and no chain starts', () => {
      service.init();
      store.setConfig(defaultConfig());
      store.setStepResult('load', tickers());

      store.triggerResultRun();

      expect(store.resultStatus()).toBe('idle');
      http.expectNone(OPT_URL);
    });
  });
});
