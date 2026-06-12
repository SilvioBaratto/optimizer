/**
 * Source-blind spec for FactorResearchComponent — ticker seeding (issue #952).
 *
 * Authored from acceptance criteria only. No implementation source was read.
 *
 * Criteria covered (T3-verifiable only — per oracle report):
 *
 * T3-a — "loads data on init":
 *   • no portfolio → tickers() equals DEFAULT_TICKERS (unchanged)
 *   • portfolio selected → snapshot fetched, tickers() seeded from weights keys
 *
 * T3-b — "handles API error":
 *   • snapshot API error → tickers() stays at DEFAULT_TICKERS (silent)
 *   • empty snapshot weights → tickers() stays at DEFAULT_TICKERS (silent)
 *   • error must not surface as a run/compute error (seeding path is silent)
 *
 * T3-c — "reacts to portfolio context change":
 *   • portfolio id change → new snapshot fetched, tickers() re-seeded
 *   • user edit detected (tickers diverged from last seed) → edit preserved
 *   • tickers still matching seed → re-seeded normally on portfolio change
 *
 * Assumption: FactorResearchComponent exposes a writable `tickers` signal
 * (string[]) that the seeding writes to directly — same surface as
 * OptimizationStudioComponent. If the component uses a string `tickersRaw`
 * signal instead, the assertions on tickers() should be adapted accordingly.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { FactorResearchComponent } from './factor-research';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { environment } from '../../environments/environment';

const PORTFOLIO_LIST_URL = `${environment.apiUrl}portfolio/`;

function snapshotUrl(name: string): string {
  return `${environment.apiUrl}portfolio/${encodeURIComponent(name)}/snapshots/latest`;
}

function buildSnapshotDto(weights: Record<string, number>) {
  return {
    id: 'snap-fr-1',
    portfolio_id: 'pf-fr-1',
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

describe('FactorResearchComponent — ticker seeding (issue #952)', () => {
  let fixture: ComponentFixture<FactorResearchComponent>;
  let comp: FactorResearchComponent;
  let http: HttpTestingController;
  let ctx: PortfolioContextService;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await configureTestBed({
      imports: [FactorResearchComponent],
      withHttp: true,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(FactorResearchComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
    ctx = TestBed.inject(PortfolioContextService);
    fixture.detectChanges();
  });

  afterEach(() => {
    http.verify();
    localStorage.clear();
  });

  // ── T3-a: loads data on init ─────────────────────────────────────────────────

  it('when no portfolio selected at startup, tickers are non-empty defaults', () => {
    // DEFAULT_TICKERS must be non-empty (requirements: "fallback to DEFAULT_TICKERS")
    expect(comp.tickers().length).toBeGreaterThan(0);
    http.expectNone((r) => r.url.includes('snapshot'));
  });

  it('when no portfolio selected at startup, no snapshot request is made', () => {
    expect(ctx.currentPortfolioId()).toBeNull();
    http.expectNone((r) => r.url.includes('snapshot'));
  });

  it('when a portfolio is selected at startup, snapshot is fetched', () => {
    ctx.currentPortfolioId.set('pf-1');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-1', name: 'factor-fund' }]));
    fixture.detectChanges();

    // snapshot request must fire once currentPortfolioName() resolves
    const req = http.expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('factor-fund'));
    req.flush(buildSnapshotDto({ TSLA: 0.4, NVDA: 0.4, MSFT: 0.2 }));
    fixture.detectChanges();

    expect(comp.tickers()).toEqual(jasmine.arrayContaining(['TSLA', 'NVDA', 'MSFT']));
    expect(comp.tickers().length).toBe(3);
  });

  it('when portfolio selected, snapshot keys replace the default tickers', () => {
    const initial = [...comp.tickers()];

    ctx.currentPortfolioId.set('pf-1');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-1', name: 'factor-fund' }]));
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('factor-fund'))
      .flush(buildSnapshotDto({ XYZ: 1.0 }));
    fixture.detectChanges();

    expect(comp.tickers()).not.toEqual(initial);
    expect(comp.tickers()).toEqual(['XYZ']);
  });

  // ── T3-b: handles API error ──────────────────────────────────────────────────

  it('when snapshot API returns an error, tickers fall back to DEFAULT_TICKERS', () => {
    const initial = [...comp.tickers()];

    ctx.currentPortfolioId.set('pf-1');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-1', name: 'factor-fund' }]));
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('factor-fund'))
      .flush({ detail: 'not found' }, { status: 404, statusText: 'Not Found' });
    fixture.detectChanges();

    expect(comp.tickers()).toEqual(initial);
  });

  it('when snapshot weights are empty, tickers fall back to DEFAULT_TICKERS', () => {
    const initial = [...comp.tickers()];

    ctx.currentPortfolioId.set('pf-1');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-1', name: 'factor-fund' }]));
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('factor-fund'))
      .flush(buildSnapshotDto({}));
    fixture.detectChanges();

    expect(comp.tickers()).toEqual(initial);
  });

  it('when snapshot API errors, the seeding path does not raise a compute error', () => {
    // Requirements: "API-error fallback occurs silently"
    // The compute/run error state must not be polluted by a snapshot fetch failure.
    ctx.currentPortfolioId.set('pf-1');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-1', name: 'factor-fund' }]));
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('factor-fund'))
      .flush({ detail: 'oops' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();

    // The component must not surface a visible error from ticker seeding alone.
    expect(comp.computeError?.() ?? null).toBeNull();
  });

  // ── T3-c: reacts to portfolio context change ─────────────────────────────────

  it('when portfolio switches from null to selected, tickers re-seed', () => {
    const initial = [...comp.tickers()];
    expect(initial.length).toBeGreaterThan(0);

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

    expect(comp.tickers()).toEqual(['GOOGL']);
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

    // Switch — list not re-fetched (distinctUntilChanged on hasId).
    ctx.currentPortfolioId.set('pf-2');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('fund-b'))
      .flush(buildSnapshotDto({ NVDA: 0.6, AMZN: 0.4 }));
    fixture.detectChanges();

    expect(comp.tickers()).toEqual(jasmine.arrayContaining(['NVDA', 'AMZN']));
    expect(comp.tickers()).not.toContain('TSLA');
  });

  // ── Re-seed guard ─────────────────────────────────────────────────────────────

  it('when user edits tickers after seeding, portfolio change does not overwrite the edit', () => {
    // Requirements: "re-seed on portfolio change only when field is still at its
    // previous seed value, and a user-edited field is left untouched."
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

    // User manually edits — tickers now diverge from the last seed.
    comp.tickers.set(['MY_FACTOR_TICKER']);

    ctx.currentPortfolioId.set('pf-2');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('fund-b'))
      .flush(buildSnapshotDto({ NVDA: 1.0 }));
    fixture.detectChanges();

    // Guard must preserve the user's edit.
    expect(comp.tickers()).toEqual(['MY_FACTOR_TICKER']);
  });

  it('when tickers still match the last seed, portfolio change re-seeds normally', () => {
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

    ctx.currentPortfolioId.set('pf-2');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('fund-b'))
      .flush(buildSnapshotDto({ SPY: 1.0 }));
    fixture.detectChanges();

    expect(comp.tickers()).toEqual(['SPY']);
  });
});
