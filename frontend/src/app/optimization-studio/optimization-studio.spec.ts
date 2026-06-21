import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { OptimizationStudioComponent } from './optimization-studio';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { environment } from '../../environments/environment';

const OPTIMIZE_URL = `${environment.apiUrl}optimize`;
const RUN: { optimizerType: 'mean_risk'; config: Record<string, unknown> } = {
  optimizerType: 'mean_risk',
  config: {},
};

describe('OptimizationStudioComponent', () => {
  let fixture: ComponentFixture<OptimizationStudioComponent>;
  let comp: OptimizationStudioComponent;
  let http: HttpTestingController;
  let ctx: PortfolioContextService;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await configureTestBed({ imports: [OptimizationStudioComponent], withHttp: true, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(OptimizationStudioComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
    ctx = TestBed.inject(PortfolioContextService);
    fixture.detectChanges(); // runs the dateRange effect + seeding subscription
  });

  afterEach(() => {
    // Drain the PortfolioContextService auto-select bootstrap (GET /portfolio/)
    // that fires when no portfolio id is stored, so verify() only asserts on the
    // requests each test explicitly exercises.
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  it('when a sync optimize resolves, the run result is set and polling is off', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush({ id: 'run-1', weights: {} });
    expect(comp.hasResult()).toBe(true);
    expect(comp.isPolling()).toBe(false);
  });

  it('when optimize returns a job_id, polling begins', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush({ job_id: 'j1', run_id: 'r1', status: 'pending' });
    expect(comp.isPolling()).toBe(true);
  });

  it('when the job completes, the run is fetched and polling stops', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush({ job_id: 'j1', run_id: 'r1', status: 'pending' });
    comp.onJobCompleted('run-1');
    http.expectOne(`${OPTIMIZE_URL}/run-1`).flush({ id: 'run-1', weights: {} });
    expect(comp.hasResult()).toBe(true);
    expect(comp.isPolling()).toBe(false);
  });

  it('when no portfolio is selected, applying weights errors without a request', () => {
    // Drain the context auto-select bootstrap so expectNone asserts only on
    // requests triggered by onApplyWeights.
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    expect(ctx.currentPortfolioId()).toBeNull();
    comp.onApplyWeights({ AAPL: 1 });
    expect(comp.applyStatus()).toBe('error');
    expect(comp.applyError()).toContain('No active portfolio');
    http.expectNone(() => true);
  });

it('when optimize errors, the run error is recorded and running stops', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush({ detail: 'bad' }, { status: 500, statusText: 'Server Error' });
    expect(comp.runError()).toBeTruthy();
    expect(comp.isRunning()).toBe(false);
  });

  it('onRunPipeline is a no-op while a run is already in flight', () => {
    comp.isRunning.set(true);
    comp.onRunPipeline(RUN);
    expect(http.match(OPTIMIZE_URL).length).toBe(0);
  });

  it('onJobFailed records the error with a default fallback', () => {
    comp.onJobFailed('boom');
    expect(comp.runError()).toBe('boom');
    expect(comp.isRunning()).toBe(false);
    comp.onJobFailed('');
    expect(comp.runError()).toBe('Job failed');
  });

  it('when fetching a completed run fails, the error is recorded', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush({ job_id: 'j1', run_id: 'r1', status: 'pending' });
    comp.onJobCompleted('r1');
    http
      .expectOne(`${OPTIMIZE_URL}/r1`)
      .flush({ detail: 'bad' }, { status: 500, statusText: 'Server Error' });
    expect(comp.runError()).toBeTruthy();
  });

  it('applying weights to a selected portfolio saves a snapshot and reports success', () => {
    ctx.currentPortfolioId.set('pf-1');
    comp.onApplyWeights({ AAPL: 1 });
    expect(comp.applyStatus()).toBe('saving');
    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('portfolio') && !r.url.includes('snapshot'))
      .flush({ items: [{ id: 'pf-1', name: 'My Port' }] });
    http
      .expectOne((r) => r.method === 'POST' && r.url.includes('snapshot'))
      .flush({ id: 'snap-1' });
    expect(comp.applyStatus()).toBe('success');
    expect(comp.appliedPortfolioName()).toBe('My Port');
  });

  it('applying weights surfaces an error when the snapshot fails', () => {
    ctx.currentPortfolioId.set('pf-1');
    comp.onApplyWeights({ AAPL: 1 });
    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('portfolio') && !r.url.includes('snapshot'))
      .flush({ items: [{ id: 'pf-1', name: 'My Port' }] });
    http
      .expectOne((r) => r.method === 'POST' && r.url.includes('snapshot'))
      .flush({ detail: 'bad' }, { status: 500, statusText: 'Server Error' });
    expect(comp.applyStatus()).toBe('error');
    expect(comp.applyError()).toBeTruthy();
  });

  it('retry clears the run error and stops loading', () => {
    comp.runError.set('x');
    comp.retry();
    expect(comp.runError()).toBeNull();
    expect(comp.isLoading()).toBe(false);
  });

  it('loadData clears the error and loading flags', () => {
    comp.hasError.set(true);
    comp.loadData();
    expect(comp.hasError()).toBe(false);
    expect(comp.isLoading()).toBe(false);
  });

  it('openReportModal does not throw', () => {
    expect(() => comp.openReportModal()).not.toThrow();
  });

  // ── Ticker seeding (issue #950) ───────────────────────────────────────────────
  //
  // T3 criterion: "every wired page has specs covering
  //   (a) loads data on init,
  //   (b) handles API error,
  //   (c) reacts to portfolio context change."
  //
  // DEFAULT_TICKERS taken from optimization-studio.ts:37.
  // All HTTP assertions use the same http.expectOne pattern as the rest of
  // this file so that afterEach's http.verify() catches any leaked request.

  describe('ticker seeding', () => {
    const DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V'];
    const PORTFOLIO_LIST_URL = `${environment.apiUrl}portfolio/`;

    function snapshotUrl(name: string): string {
      return `${environment.apiUrl}portfolio/${encodeURIComponent(name)}/snapshots/latest`;
    }

    function buildSnapshotDto(weights: Record<string, number>) {
      return {
        id: 'snap-1',
        portfolio_id: 'pf-1',
        snapshot_date: '2026-01-01',
        snapshot_type: 'manual',
        weights,
        sector_mapping: null,
        summary: null,
        optimizer_config: null,
        turnover: null,
        holding_count: Object.keys(weights).length,
        created_at: '2026-01-01T00:00:00Z',
      };
    }

    function buildPortfolioList(
      portfolios: { id: string; name: string }[],
    ) {
      return {
        items: portfolios.map((p) => ({
          id: p.id,
          name: p.name,
          description: null,
          currency: 'USD',
          benchmark_ticker: 'SPY',
        })),
        total: portfolios.length,
      };
    }

    // ── (a) loads data on init ──────────────────────────────────────────────────

    it('when no portfolio is selected at startup, tickers() equals DEFAULT_TICKERS', () => {
      // No portfolio → no HTTP, no seed.
      expect(comp.tickers()).toEqual(DEFAULT_TICKERS);
      http.expectNone((r) => r.url.includes('snapshot'));
    });

    it('when a portfolio is selected at startup, snapshot is fetched', () => {
      ctx.currentPortfolioId.set('pf-1');
      fixture.detectChanges();

      // portfolio/ list fires because id transitioned null → non-null
      http
        .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
        .flush(buildPortfolioList([{ id: 'pf-1', name: 'growth-fund' }]));
      fixture.detectChanges();

      // snapshot fetch fires because currentPortfolioName() became 'growth-fund'
      http
        .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('growth-fund'))
        .flush(buildSnapshotDto({ TSLA: 0.6, NVDA: 0.4 }));
      fixture.detectChanges();

      expect(comp.tickers()).toEqual(jasmine.arrayContaining(['TSLA', 'NVDA']));
      expect(comp.tickers().length).toBe(2);
    });

    it('when portfolio selected at startup, DEFAULT_TICKERS are replaced by snapshot keys', () => {
      ctx.currentPortfolioId.set('pf-1');
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
        .flush(buildPortfolioList([{ id: 'pf-1', name: 'growth-fund' }]));
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('growth-fund'))
        .flush(buildSnapshotDto({ TSLA: 1.0 }));
      fixture.detectChanges();

      expect(comp.tickers()).not.toEqual(DEFAULT_TICKERS);
    });

    // ── (b) handles API error ───────────────────────────────────────────────────

    it('when snapshot API returns an error, tickers() falls back to DEFAULT_TICKERS', () => {
      ctx.currentPortfolioId.set('pf-1');
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
        .flush(buildPortfolioList([{ id: 'pf-1', name: 'growth-fund' }]));
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('growth-fund'))
        .flush({ detail: 'not found' }, { status: 404, statusText: 'Not Found' });
      fixture.detectChanges();

      expect(comp.tickers()).toEqual(DEFAULT_TICKERS);
    });

    it('when snapshot.weights is empty, tickers() falls back to DEFAULT_TICKERS', () => {
      ctx.currentPortfolioId.set('pf-1');
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
        .flush(buildPortfolioList([{ id: 'pf-1', name: 'growth-fund' }]));
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('growth-fund'))
        .flush(buildSnapshotDto({}));
      fixture.detectChanges();

      expect(comp.tickers()).toEqual(DEFAULT_TICKERS);
    });

    it('when API errors, runError() is not polluted by the seeding path', () => {
      ctx.currentPortfolioId.set('pf-1');
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
        .flush(buildPortfolioList([{ id: 'pf-1', name: 'growth-fund' }]));
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('growth-fund'))
        .flush({ detail: 'oops' }, { status: 500, statusText: 'Server Error' });
      fixture.detectChanges();

      // Ticker seeding errors are silent — runError belongs to optimization, not seeding.
      expect(comp.runError()).toBeNull();
    });

    // ── (c) reacts to portfolio context change ──────────────────────────────────

    it('when portfolio switches from null to selected, snapshot is fetched and tickers updated', () => {
      // Start: no portfolio, no HTTP.
      expect(comp.tickers()).toEqual(DEFAULT_TICKERS);

      // Select a portfolio.
      ctx.currentPortfolioId.set('pf-1');
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
        .flush(buildPortfolioList([
          { id: 'pf-1', name: 'fund-a' },
          { id: 'pf-2', name: 'fund-b' },
        ]));
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('fund-a'))
        .flush(buildSnapshotDto({ MSFT: 1.0 }));
      fixture.detectChanges();

      expect(comp.tickers()).toEqual(['MSFT']);
    });

    it('when portfolio id changes, snapshot is fetched for the new portfolio name', () => {
      // Select portfolio A — flush list (includes B so the switch works without a new list fetch).
      ctx.currentPortfolioId.set('pf-1');
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
        .flush(buildPortfolioList([
          { id: 'pf-1', name: 'fund-a' },
          { id: 'pf-2', name: 'fund-b' },
        ]));
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('fund-a'))
        .flush(buildSnapshotDto({ TSLA: 1.0 }));
      fixture.detectChanges();

      expect(comp.tickers()).toEqual(['TSLA']);

      // Switch to portfolio B — no new list fetch (id was already non-null).
      ctx.currentPortfolioId.set('pf-2');
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('fund-b'))
        .flush(buildSnapshotDto({ NVDA: 0.7, AMZN: 0.3 }));
      fixture.detectChanges();

      expect(comp.tickers()).toEqual(jasmine.arrayContaining(['NVDA', 'AMZN']));
      expect(comp.tickers()).not.toContain('TSLA');
    });

    // ── Re-seed guard ───────────────────────────────────────────────────────────

    it('when user edits tickers after seeding, a portfolio change does not overwrite the edit', () => {
      ctx.currentPortfolioId.set('pf-1');
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
        .flush(buildPortfolioList([
          { id: 'pf-1', name: 'fund-a' },
          { id: 'pf-2', name: 'fund-b' },
        ]));
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('fund-a'))
        .flush(buildSnapshotDto({ TSLA: 1.0 }));
      fixture.detectChanges();

      // User edits tickers — now diverges from the last seed.
      comp.tickers.set(['MY_CUSTOM_TICKER']);

      // Switch portfolio.
      ctx.currentPortfolioId.set('pf-2');
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('fund-b'))
        .flush(buildSnapshotDto({ NVDA: 1.0 }));
      fixture.detectChanges();

      // Guard must preserve the user's edit.
      expect(comp.tickers()).toEqual(['MY_CUSTOM_TICKER']);
    });

    it('when tickers still match the last seed, a portfolio change re-seeds normally', () => {
      ctx.currentPortfolioId.set('pf-1');
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
        .flush(buildPortfolioList([
          { id: 'pf-1', name: 'fund-a' },
          { id: 'pf-2', name: 'fund-b' },
        ]));
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('fund-a'))
        .flush(buildSnapshotDto({ TSLA: 1.0 }));
      fixture.detectChanges();

      // No user edit — tickers still match the seed.
      expect(comp.tickers()).toEqual(['TSLA']);

      // Switch portfolio.
      ctx.currentPortfolioId.set('pf-2');
      fixture.detectChanges();

      http
        .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('fund-b'))
        .flush(buildSnapshotDto({ SPY: 1.0 }));
      fixture.detectChanges();

      expect(comp.tickers()).toEqual(['SPY']);
    });
  });
});
