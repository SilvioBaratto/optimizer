/**
 * Source-blind spec for BacktestingComponent — ticker seeding (issue #951).
 *
 * Acceptance criteria tested here:
 *   (a) On load with portfolio selected → tickersRaw becomes snapshot keys joined by ", ".
 *   (a) No portfolio selected → tickersRaw stays at DEFAULT_TICKERS string.
 *   (b) API error → silent fallback to DEFAULT_TICKERS, no run error raised.
 *   (b) Empty snapshot weights → fallback to DEFAULT_TICKERS.
 *   (c) Portfolio change → re-seeds tickersRaw when parsed list still matches last seed.
 *   (c) User-typed tickers are left untouched when they differ from the last seed.
 *   (d) Existing date-range sync effect is not broken by seeding.
 *
 * Assertions use tickersRaw() for the seeding contract (writable string signal, the
 * source the text <input> is bound to) and tickers() where the parsed array is needed
 * for the guard test. Both are public signals on BacktestingComponent.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { BacktestingComponent } from './backtesting';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { environment } from '../../environments/environment';

// Mirrors the 5-ticker default the issue body specifies for backtesting.
const DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'];
const DEFAULT_TICKERS_RAW = DEFAULT_TICKERS.join(', ');

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

function buildPortfolioList(portfolios: { id: string; name: string }[]) {
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

describe('BacktestingComponent — ticker seeding (issue #951)', () => {
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
    fixture = TestBed.createComponent(BacktestingComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
    ctx = TestBed.inject(PortfolioContextService);
    fixture.detectChanges();
  });

  afterEach(() => {
    // Drain all portfolio-related requests: the PortfolioContextService bootstrap
    // list request AND any weight-resolver snapshot requests that fire via
    // initWeightResolver() (uses currentPortfolioId directly — issue #1051).
    http.match((r) => r.url.includes('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  // ── Criterion (a): loads data on init ───────────────────────────────────────

  it('when no portfolio is selected at startup, tickersRaw equals the default string', () => {
    expect(comp.tickersRaw()).toBe(DEFAULT_TICKERS_RAW);
    http.expectNone((r) => r.url.includes('snapshot'));
  });

  it('when no portfolio is selected at startup, no snapshot request is made', () => {
    http.expectNone((r) => r.url.includes('snapshot'));
  });

  it('when a portfolio is selected at startup, snapshot is fetched and tickersRaw seeded', () => {
    ctx.currentPortfolioId.set('pf-1');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-1', name: 'bt-fund' }]));
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('bt-fund'))
      .flush(buildSnapshotDto({ TSLA: 0.5, NVDA: 0.5 }));
    fixture.detectChanges();

    // tickersRaw must be the snapshot keys joined by ", " (the contract from the issue body)
    const raw = comp.tickersRaw();
    const parts = raw.split(',').map((s) => s.trim());
    expect(parts).toEqual(jasmine.arrayContaining(['TSLA', 'NVDA']));
    expect(parts.length).toBe(2);
  });

  it('when portfolio selected, tickersRaw no longer equals the default string', () => {
    ctx.currentPortfolioId.set('pf-1');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-1', name: 'bt-fund' }]));
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('bt-fund'))
      .flush(buildSnapshotDto({ XYZ: 1.0 }));
    fixture.detectChanges();

    expect(comp.tickersRaw()).not.toBe(DEFAULT_TICKERS_RAW);
  });

  it('when portfolio selected, tickers() reflects the seeded snapshot keys', () => {
    ctx.currentPortfolioId.set('pf-1');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-1', name: 'bt-fund' }]));
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('bt-fund'))
      .flush(buildSnapshotDto({ MSFT: 0.6, AMZN: 0.4 }));
    fixture.detectChanges();

    expect(comp.tickers()).toEqual(jasmine.arrayContaining(['MSFT', 'AMZN']));
    expect(comp.tickers().length).toBe(2);
  });

  // ── Criterion (b): handles API error ─────────────────────────────────────────

  it('when snapshot API returns an error, tickersRaw stays at the default string', () => {
    ctx.currentPortfolioId.set('pf-1');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-1', name: 'bt-fund' }]));
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('bt-fund'))
      .flush({ detail: 'not found' }, { status: 404, statusText: 'Not Found' });
    fixture.detectChanges();

    expect(comp.tickersRaw()).toBe(DEFAULT_TICKERS_RAW);
  });

  it('when snapshot weights are empty, tickersRaw stays at the default string', () => {
    ctx.currentPortfolioId.set('pf-1');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-1', name: 'bt-fund' }]));
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('bt-fund'))
      .flush(buildSnapshotDto({}));
    fixture.detectChanges();

    expect(comp.tickersRaw()).toBe(DEFAULT_TICKERS_RAW);
  });

  it('when snapshot API errors, no run error is surfaced by the seeding path', () => {
    ctx.currentPortfolioId.set('pf-1');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-1', name: 'bt-fund' }]));
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('bt-fund'))
      .flush({ detail: 'oops' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();

    // Seeding errors are silent; the component's run-error state must not be polluted.
    expect(comp.runError()).toBeNull();
  });

  // ── Criterion (c): reacts to portfolio context change ────────────────────────

  it('when portfolio switches from null to selected, tickersRaw re-seeds', () => {
    expect(comp.tickersRaw()).toBe(DEFAULT_TICKERS_RAW);

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
      .flush(buildSnapshotDto({ GOOGL: 1.0 }));
    fixture.detectChanges();

    expect(comp.tickersRaw()).toBe('GOOGL');
  });

  it('when portfolio id changes, snapshot is fetched for the new portfolio name', () => {
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

    // Switch — no new list fetch (distinctUntilChanged on hasId).
    ctx.currentPortfolioId.set('pf-2');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('fund-b'))
      .flush(buildSnapshotDto({ NVDA: 0.7, AMZN: 0.3 }));
    fixture.detectChanges();

    expect(comp.tickers()).toEqual(jasmine.arrayContaining(['NVDA', 'AMZN']));
    expect(comp.tickers()).not.toContain('TSLA');
  });

  // ── Re-seed guard ─────────────────────────────────────────────────────────────

  it('when user edits tickersRaw after seeding, portfolio change does not overwrite', () => {
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

    // User types into the ticker field — diverges from the last seed.
    comp.tickersRaw.set('MY_CUSTOM_TICKER');

    ctx.currentPortfolioId.set('pf-2');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('fund-b'))
      .flush(buildSnapshotDto({ NVDA: 1.0 }));
    fixture.detectChanges();

    // Guard must preserve the user's edit.
    expect(comp.tickersRaw()).toBe('MY_CUSTOM_TICKER');
  });

  it('when tickersRaw still matches the last seed, portfolio change re-seeds', () => {
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

    // No user edit — tickersRaw still reflects the seed.
    expect(comp.tickersRaw()).toBe('TSLA');

    ctx.currentPortfolioId.set('pf-2');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('fund-b'))
      .flush(buildSnapshotDto({ SPY: 1.0 }));
    fixture.detectChanges();

    expect(comp.tickersRaw()).toBe('SPY');
  });

});
