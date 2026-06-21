/**
 * backtesting-setup-form-picker-wiring.spec.ts — issue #1054
 *
 * Contract tests asserting that the setup-form portfolio picker drives the
 * resolved ticker universe for a backtest run — not the global context.
 *
 * Gap this spec closes: backtesting-wiring.spec.ts and
 * backtest-weight-resolver.spec.ts exercise the resolver in isolation and the
 * global-context path, but no spec asserts that `onSetupFormRun(config)` uses
 * `config.portfolio` to resolve weights and feeds them into POST /backtest.
 *
 * Criteria pinned:
 *
 *   [WIRING-A] When onSetupFormRun is called with portfolio:"pf-B", the component
 *              fetches the snapshot for pf-B, NOT the global-context portfolio.
 *
 *   [WIRING-B] When pf-B's snapshot has weights {NVDA:0.5, TSLA:0.5}, POST /backtest
 *              body tickers equal exactly Object.keys(snapshot.weights).
 *
 *   [WIRING-C] When pf-B snapshot returns 404, POST /backtest body uses the
 *              resolvedTickers() fallback (no ticker error; backtest still runs).
 *
 *   [WIRING-D] benchmark from config.benchmark appears in the POST /backtest body.
 *
 *   [WIRING-E] start_date / end_date derived from config.period appear in POST body.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import {
  configureTestBed,
  injectHttp,
  installResizeObserverStub,
  makeSnapshotDto,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { BacktestingComponent } from './backtesting';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import type { BacktestRunConfig } from './backtesting-setup-form/backtesting-setup-form';
import type { HttpTestingController } from '@angular/common/http/testing';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Drain portfolio-list GETs (ends with /portfolio/) without asserting them. */
function drainPortfolioListRequests(http: HttpTestingController): void {
  http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
}

/** Drain snapshot GETs without asserting them. */
function drainSnapshotRequests(http: HttpTestingController): void {
  http.match((r) => r.method === 'GET' && r.url.toLowerCase().includes('snapshot'));
}

/** Build a minimal BacktestRunConfig. */
function makeRunConfig(overrides: Partial<BacktestRunConfig> = {}): BacktestRunConfig {
  return { portfolio: 'pf-picked', period: '1Y', benchmark: 'SPY', ...overrides };
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('BacktestingComponent — setup-form picker drives run universe (issue #1054)', () => {
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
      providers: [ICON_PROVIDER],
    });

    TestBed.inject(PortfolioContextService).reset();
    fixture = TestBed.createComponent(BacktestingComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
    ctx = TestBed.inject(PortfolioContextService);
    fixture.detectChanges();

    // Drain bootstrap requests: portfolio list + global-context snapshot.
    drainPortfolioListRequests(http);
    drainSnapshotRequests(http);
  });

  afterEach(() => {
    http.match(() => true);
    localStorage.clear();
  });

  // ── WIRING-A: snapshot fetch targets the picked portfolio, not the global one ──

  it('when onSetupFormRun is called with portfolio "pf-B", component fetches the snapshot for "pf-B"', () => {
    // Global context has a different portfolio so the test distinguishes them.
    ctx.currentPortfolioId.set('pf-A');
    fixture.detectChanges();
    drainPortfolioListRequests(http);
    drainSnapshotRequests(http);  // drain global-context snapshot for pf-A

    comp.onSetupFormRun(makeRunConfig({ portfolio: 'pf-B' }));
    fixture.detectChanges();

    const req = http.expectOne(
      (r) => r.method === 'GET' && r.url.includes('pf-B') && r.url.toLowerCase().includes('snapshot'),
    );
    expect(req.request.url).toContain('pf-B');
    req.flush(makeSnapshotDto({ portfolio_id: 'pf-B', snapshot_type: 'optimization' }));

    // Drain the POST /backtest that follows.
    http.match((r) => r.method === 'POST' && r.url.includes('backtest'));
  });

  // ── WIRING-B: snapshot weights become the POST /backtest tickers ─────────────

  it('when onSetupFormRun is called and snapshot has weights, POST /backtest tickers equal snapshot weight keys', () => {
    comp.onSetupFormRun(makeRunConfig({ portfolio: 'pf-weights' }));
    fixture.detectChanges();

    http
      .expectOne(
        (r) => r.method === 'GET' && r.url.includes('pf-weights') && r.url.toLowerCase().includes('snapshot'),
      )
      .flush(
        makeSnapshotDto({
          portfolio_id: 'pf-weights',
          snapshot_type: 'optimization',
          weights: { NVDA: 0.5, TSLA: 0.5 },
        }),
      );
    fixture.detectChanges();

    const btReq = http.expectOne((r) => r.method === 'POST' && r.url.includes('backtest'));
    const tickers = (btReq.request.body as Record<string, unknown>)['tickers'] as string[];
    expect(tickers).toEqual(jasmine.arrayContaining(['NVDA', 'TSLA']));
    expect(tickers.length).toBe(2);
    btReq.flush({ jobId: 'j-b', runId: 'r-b', status: 'pending', message: '' });
  });

  // ── WIRING-C: 404 → equal-weight fallback, no ticker error ──────────────────

  it('when snapshot returns 404 for the picked portfolio, backtest still runs without a ticker error', () => {
    // Seed resolvedTickers with something so the fallback is non-empty.
    comp.resolvedTickers.set(['AAPL', 'MSFT', 'GOOGL']);

    comp.onSetupFormRun(makeRunConfig({ portfolio: 'pf-never-optimised' }));
    fixture.detectChanges();

    http
      .expectOne(
        (r) => r.method === 'GET' && r.url.includes('pf-never-optimised') && r.url.toLowerCase().includes('snapshot'),
      )
      .flush({ detail: 'Not found' }, { status: 404, statusText: 'Not Found' });
    fixture.detectChanges();

    // A POST /backtest must have been attempted (not blocked by ticker error).
    const btReq = http.expectOne((r) => r.method === 'POST' && r.url.includes('backtest'));
    expect(comp.runError()).not.toBe('Provide at least one ticker.');
    btReq.flush({ jobId: 'j-c', runId: 'r-c', status: 'pending', message: '' });
  });

  // ── WIRING-D: config.benchmark flows into the POST /backtest body ────────────

  it('when onSetupFormRun is called with benchmark "QQQ", POST /backtest body contains "QQQ"', () => {
    comp.onSetupFormRun(makeRunConfig({ portfolio: 'pf-bm', benchmark: 'QQQ' }));
    fixture.detectChanges();

    http
      .expectOne(
        (r) => r.method === 'GET' && r.url.includes('pf-bm') && r.url.toLowerCase().includes('snapshot'),
      )
      .flush(makeSnapshotDto({ snapshot_type: 'optimization', weights: { SPY: 1.0 } }));
    fixture.detectChanges();

    const btReq = http.expectOne((r) => r.method === 'POST' && r.url.includes('backtest'));
    expect(JSON.stringify(btReq.request.body)).toContain('QQQ');
    btReq.flush({ jobId: 'j-d', runId: 'r-d', status: 'pending', message: '' });
  });

  // ── WIRING-E: config.period → ISO date range in POST /backtest body ──────────

  it('when onSetupFormRun is called with period "1Y", POST /backtest body contains start_date and end_date', () => {
    comp.onSetupFormRun(makeRunConfig({ portfolio: 'pf-dates', period: '1Y' }));
    fixture.detectChanges();

    http
      .expectOne(
        (r) => r.method === 'GET' && r.url.includes('pf-dates') && r.url.toLowerCase().includes('snapshot'),
      )
      .flush(makeSnapshotDto({ snapshot_type: 'optimization', weights: { AAPL: 1.0 } }));
    fixture.detectChanges();

    const btReq = http.expectOne((r) => r.method === 'POST' && r.url.includes('backtest'));
    const body = btReq.request.body as Record<string, unknown>;
    // Both dates must be present and in ISO yyyy-mm-dd format.
    expect(body['start_date']).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(body['end_date']).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    // The end date must be after the start date for a valid range.
    expect(body['end_date'] as string >= (body['start_date'] as string)).toBeTrue();
    btReq.flush({ jobId: 'j-e', runId: 'r-e', status: 'pending', message: '' });
  });

  // ── WIRING-B2: globally-different context does NOT bleed into picker run ─────

  it('when global context is pf-A but picker selects pf-B, POST /backtest uses pf-B snapshot tickers', () => {
    ctx.currentPortfolioId.set('pf-A');
    fixture.detectChanges();
    drainPortfolioListRequests(http);
    // Flush pf-A global snapshot with different tickers so we can confirm they are NOT used.
    http.match((r) => r.method === 'GET' && r.url.includes('pf-A') && r.url.toLowerCase().includes('snapshot'))
        .forEach((req) => req.flush(makeSnapshotDto({ portfolio_id: 'pf-A', snapshot_type: 'optimization', weights: { AAPL: 1.0 } })));
    fixture.detectChanges();

    comp.onSetupFormRun(makeRunConfig({ portfolio: 'pf-B', period: '1Y', benchmark: 'SPY' }));
    fixture.detectChanges();

    http
      .expectOne(
        (r) => r.method === 'GET' && r.url.includes('pf-B') && r.url.toLowerCase().includes('snapshot'),
      )
      .flush(makeSnapshotDto({ portfolio_id: 'pf-B', snapshot_type: 'optimization', weights: { NVDA: 0.6, MSFT: 0.4 } }));
    fixture.detectChanges();

    const btReq = http.expectOne((r) => r.method === 'POST' && r.url.includes('backtest'));
    const tickers = (btReq.request.body as Record<string, unknown>)['tickers'] as string[];
    expect(tickers).toEqual(jasmine.arrayContaining(['NVDA', 'MSFT']));
    // AAPL (from pf-A global context) must NOT appear.
    expect(tickers).not.toContain('AAPL');
    btReq.flush({ jobId: 'j-f', runId: 'r-f', status: 'pending', message: '' });
  });
});
