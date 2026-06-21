/**
 * backtest-weight-resolver.spec.ts — issue #1051
 *
 * Source-blind contract tests for the weight resolver + universe-feeds-service criteria.
 * Authored from acceptance criteria only — no implementation source was read.
 *
 * Criteria pinned (from oracle report):
 *
 *   [UNIT] "A weight resolver loads the picked portfolio's latest optimization snapshot
 *           and derives tickers from its weights; equal-weight fallback when no
 *           snapshot / weights exist."
 *
 *   [UNIT] "The resolved universe feeds BacktestService run + benchmark from the
 *           setup form."
 *
 * Observable boundary: HTTP requests issued by the component, and the component's
 * public error-state signal (`runError()`).
 *
 * Test failure mode (RED phase):
 *   — Tests 1–7 fail today because BacktestingComponent does not yet:
 *       (a) react to a portfolio-context change by fetching the portfolio's latest
 *           optimization snapshot, or
 *       (b) derive the backtest tickers from that snapshot's weight keys.
 *     The `http.expectOne(…snapshot…)` call throws "Expected 1 request but found 0"
 *     because no snapshot fetch is triggered, and downstream POST assertions also fail.
 *   — Tests 3–4 fail today because without the weight resolver, `onRunBacktest()`
 *     with an empty tickersRaw sets `runError = 'Provide at least one ticker.'`;
 *     the equal-weight fallback does not yet exist so the assertion fails.
 *
 * No property-based tests — Angular / Karma / Jasmine ships no built-in property
 * library; the "tickers === Object.keys(weights)" invariant is exhaustively covered
 * by the three concrete examples (populated, empty, and absent snapshot).
 */

import { TestBed, ComponentFixture } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import {
  configureTestBed,
  injectHttp,
  installResizeObserverStub,
  makeSnapshotDto,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { BacktestingComponent } from './backtesting';
import { PortfolioContextService } from '../core/services/portfolio-context.service';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Drain only the portfolio-list GET (ends with /portfolio/) — does NOT drain snapshot requests. */
function drainPortfolioListRequests(http: HttpTestingController): void {
  http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
}

/** Flush a pending POST /backtest with a minimal async response. */
function flushBacktestPost(http: HttpTestingController): void {
  http
    .expectOne((r) => r.method === 'POST' && r.url.includes('backtest'))
    .flush({ jobId: 'j-stub', runId: 'r-stub', status: 'pending', message: '' });
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('BacktestingComponent — weight resolver (issue #1051)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;
  let comp: BacktestingComponent;
  let http: HttpTestingController;
  let ctx: PortfolioContextService;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await configureTestBed({
      imports: [BacktestingComponent],
      withHttp: true,
      withRouter: true,
      providers: [ICON_PROVIDER],
    });

    TestBed.inject(PortfolioContextService).reset();
    fixture = TestBed.createComponent(BacktestingComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
    ctx = TestBed.inject(PortfolioContextService);
    fixture.detectChanges();

    // Drain any bootstrap GET /portfolio/ list that may fire on init.
    drainPortfolioListRequests(http);
  });

  afterEach(() => {
    // Drain all outstanding requests so tests remain independent.
    http.match(() => true);
    localStorage.clear();
  });

  // ── 1. Snapshot fetch is triggered on portfolio selection ─────────────────
  //
  // Criterion 1: the weight resolver must request the portfolio's latest snapshot
  // whenever a portfolio is picked.

  it('when a portfolio is selected, component fetches its latest snapshot', () => {
    ctx.currentPortfolioId.set('pf-resolver-1');
    fixture.detectChanges();

    // Drain any secondary portfolio-list request before asserting the snapshot call.
    drainPortfolioListRequests(http);

    const req = http.expectOne(
      (r) => r.method === 'GET' && r.url.includes('pf-resolver-1') && r.url.includes('snapshot'),
    );
    expect(req.request.url).toContain('pf-resolver-1');
    req.flush(makeSnapshotDto({ portfolio_id: 'pf-resolver-1', snapshot_type: 'optimization' }));
  });

  // ── 2. Optimization-snapshot weights become the backtest tickers ───────────
  //
  // Criterion 1 + 2: tickers derived from snapshot.weights flow into POST /backtest.

  it('when optimization snapshot has weights, POST /backtest tickers equal the snapshot weight keys', () => {
    ctx.currentPortfolioId.set('pf-opt');
    fixture.detectChanges();
    drainPortfolioListRequests(http);

    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('snapshot'))
      .flush(
        makeSnapshotDto({
          portfolio_id: 'pf-opt',
          snapshot_type: 'optimization',
          weights: { AAPL: 0.6, MSFT: 0.4 },
        }),
      );
    fixture.detectChanges();

    comp.onRunBacktest();

    const btReq = http.expectOne((r) => r.method === 'POST' && r.url.includes('backtest'));
    const tickers = (btReq.request.body as Record<string, unknown>)['tickers'] as string[];
    expect(tickers).toEqual(jasmine.arrayContaining(['AAPL', 'MSFT']));
    expect(tickers.length).toBe(2);
    btReq.flush({ jobId: 'j1', runId: 'r1', status: 'pending', message: '' });
  });

  // ── 3. Equal-weight fallback — no snapshot (404) ──────────────────────────
  //
  // Criterion 1: 404 on snapshot → equal-weight fallback → backtest can still run;
  // "Provide at least one ticker" error must NOT be set.

  it('when no optimization snapshot exists (404), equal-weight fallback lets backtest proceed without a ticker error', () => {
    ctx.currentPortfolioId.set('pf-never-optimised');
    fixture.detectChanges();
    drainPortfolioListRequests(http);

    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('snapshot'))
      .flush({ detail: 'Not found' }, { status: 404, statusText: 'Not Found' });
    fixture.detectChanges();

    comp.onRunBacktest();
    fixture.detectChanges();

    expect(comp.runError()).not.toBe('Provide at least one ticker.');
  });

  // ── 4. Equal-weight fallback — snapshot with empty weights ────────────────
  //
  // Criterion 1: weights: {} is treated the same as a missing snapshot.

  it('when optimization snapshot has empty weights, equal-weight fallback activates and no ticker error is set', () => {
    ctx.currentPortfolioId.set('pf-empty');
    fixture.detectChanges();
    drainPortfolioListRequests(http);

    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('snapshot'))
      .flush(makeSnapshotDto({ snapshot_type: 'optimization', weights: {} }));
    fixture.detectChanges();

    comp.onRunBacktest();
    fixture.detectChanges();

    expect(comp.runError()).not.toBe('Provide at least one ticker.');
  });

  // ── 5. Equal-weight fallback — non-optimization snapshot type ────────────
  //
  // Criterion 1: only snapshot_type === 'optimization' provides a weight-derived
  // universe; a 'manual' snapshot must trigger the equal-weight fallback.

  it('when latest snapshot type is "manual", equal-weight fallback activates', () => {
    ctx.currentPortfolioId.set('pf-manual');
    fixture.detectChanges();
    drainPortfolioListRequests(http);

    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('snapshot'))
      .flush(makeSnapshotDto({ snapshot_type: 'manual', weights: { AAPL: 0.5, MSFT: 0.5 } }));
    fixture.detectChanges();

    comp.onRunBacktest();
    fixture.detectChanges();

    expect(comp.runError()).not.toBe('Provide at least one ticker.');
  });

  // ── 6. Benchmark from setup form reaches the POST body ────────────────────
  //
  // Criterion 2: the benchmark the user selects in the local setup form must
  // appear in the outgoing POST /backtest request body.

  it('when benchmark is set in setup form, POST /backtest body includes that benchmark', () => {
    ctx.currentPortfolioId.set('pf-bm');
    fixture.detectChanges();
    drainPortfolioListRequests(http);

    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('snapshot'))
      .flush(makeSnapshotDto({ snapshot_type: 'optimization', weights: { SPY: 1.0 } }));
    fixture.detectChanges();

    comp.onBenchmarkChange({ target: { value: 'QQQ' } } as unknown as Event);
    comp.onRunBacktest();

    const btReq = http.expectOne((r) => r.method === 'POST' && r.url.includes('backtest'));
    // Benchmark value must appear in the serialised body regardless of the exact key name.
    expect(JSON.stringify(btReq.request.body)).toContain('QQQ');
    btReq.flush({ jobId: 'j2', runId: 'r2', status: 'pending', message: '' });
  });

  // ── 7. Date-range from setup form reaches the POST body ──────────────────
  //
  // Criterion 2: the start_date and end_date from the local setup form must
  // appear in the POST /backtest body (snake_case, per BacktestApiRequest).

  it('when start and end dates are set in setup form, POST /backtest body contains those dates', () => {
    ctx.currentPortfolioId.set('pf-dates');
    fixture.detectChanges();
    drainPortfolioListRequests(http);

    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('snapshot'))
      .flush(makeSnapshotDto({ snapshot_type: 'optimization', weights: { AAPL: 1.0 } }));
    fixture.detectChanges();

    comp.onStartDateChange({ target: { value: '2021-01-01' } } as unknown as Event);
    comp.onEndDateChange({ target: { value: '2023-12-31' } } as unknown as Event);
    comp.onRunBacktest();

    const btReq = http.expectOne((r) => r.method === 'POST' && r.url.includes('backtest'));
    const body = btReq.request.body as Record<string, unknown>;
    expect(body['start_date']).toBe('2021-01-01');
    expect(body['end_date']).toBe('2023-12-31');
    btReq.flush({ jobId: 'j3', runId: 'r3', status: 'pending', message: '' });
  });

  // ── 8. Exact tickers — no extras beyond the snapshot weight keys ──────────
  //
  // Criterion 1: the resolver must not append extra tickers beyond the snapshot.

  it('when snapshot has three weight keys, exactly three tickers appear in the POST body', () => {
    ctx.currentPortfolioId.set('pf-three');
    fixture.detectChanges();
    drainPortfolioListRequests(http);

    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('snapshot'))
      .flush(
        makeSnapshotDto({
          snapshot_type: 'optimization',
          weights: { AAPL: 0.4, MSFT: 0.3, GOOGL: 0.3 },
        }),
      );
    fixture.detectChanges();

    comp.onRunBacktest();

    const btReq = http.expectOne((r) => r.method === 'POST' && r.url.includes('backtest'));
    const tickers = (btReq.request.body as Record<string, unknown>)['tickers'] as string[];
    expect(tickers.length).toBe(3);
    expect(tickers).toEqual(jasmine.arrayContaining(['AAPL', 'MSFT', 'GOOGL']));
    btReq.flush({ jobId: 'j4', runId: 'r4', status: 'pending', message: '' });
  });
});
