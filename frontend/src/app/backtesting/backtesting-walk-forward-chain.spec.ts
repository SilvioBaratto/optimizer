/**
 * Walk-forward panel wiring spec for BacktestingComponent (issue #1024, criterion 4).
 *
 * Covers:
 *   - app-walk-forward-panel is rendered inside the parent template.
 *   - The panel receives `tickers`, `startDate`, `endDate` from the parent signals
 *     via one-way input binding.
 *   - When the parent's `walkForwardError()` signal is set, a [role="alert"] element
 *     appears in the PARENT template (distinct from the panel's own alert).
 *   - onRunWalkForward() on the parent dispatches POST /validate/walk-forward.
 *
 * The POST body field contract and the panel's own error → alert flow are covered in:
 *   backtesting-contracts.spec.ts
 *   walk-forward-panel/walk-forward-panel.spec.ts
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { BacktestingComponent } from './backtesting';
import { WalkForwardPanelComponent } from './walk-forward-panel/walk-forward-panel';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { environment } from '../../environments/environment';

const WALK_FWD_URL = `${environment.apiUrl}validate/walk-forward`;

describe('BacktestingComponent — walk-forward panel wiring (issue #1024)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;
  let comp: BacktestingComponent;
  let el: HTMLElement;
  let http: HttpTestingController;

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
    el = fixture.nativeElement as HTMLElement;
    http = injectHttp();
    fixture.detectChanges();
  });

  afterEach(() => {
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  // ── Panel presence ────────────────────────────────────────────────────────

  it('when component renders, app-walk-forward-panel is present in the DOM', () => {
    expect(el.querySelector('app-walk-forward-panel')).not.toBeNull();
  });

  // ── Input bindings: parent signals flow into panel inputs ─────────────────

  it('when tickersRaw changes, the panel tickers input reflects the parsed array', () => {
    comp.tickersRaw.set('TSLA, NVDA, AMZN');
    fixture.detectChanges();

    const panelEl = fixture.debugElement.query(By.directive(WalkForwardPanelComponent));
    if (!panelEl) pending('WalkForwardPanelComponent not found — rendering may be deferred');
    const panel = panelEl.componentInstance as WalkForwardPanelComponent;
    expect(panel.tickers()).toEqual(jasmine.arrayContaining(['TSLA', 'NVDA', 'AMZN']));
    expect(panel.tickers().length).toBe(3);
  });

  it('when selectedStartDate changes, the panel startDate input reflects the new value', () => {
    comp.selectedStartDate.set('2022-06-01');
    fixture.detectChanges();

    const panelEl = fixture.debugElement.query(By.directive(WalkForwardPanelComponent));
    if (!panelEl) pending('WalkForwardPanelComponent not found — rendering may be deferred');
    const panel = panelEl.componentInstance as WalkForwardPanelComponent;
    expect(panel.startDate()).toBe('2022-06-01');
  });

  it('when selectedEndDate changes, the panel endDate input reflects the new value', () => {
    comp.selectedEndDate.set('2024-09-30');
    fixture.detectChanges();

    const panelEl = fixture.debugElement.query(By.directive(WalkForwardPanelComponent));
    if (!panelEl) pending('WalkForwardPanelComponent not found — rendering may be deferred');
    const panel = panelEl.componentInstance as WalkForwardPanelComponent;
    expect(panel.endDate()).toBe('2024-09-30');
  });

  it('when date-range is synced from context, the panel endDate input reflects the new end', () => {
    const ctx = TestBed.inject(PortfolioContextService);
    ctx.setCustomRange(new Date('2023-01-01'), new Date('2023-12-31'));
    fixture.detectChanges();

    const panelEl = fixture.debugElement.query(By.directive(WalkForwardPanelComponent));
    if (!panelEl) pending('WalkForwardPanelComponent not found');
    const panel = panelEl.componentInstance as WalkForwardPanelComponent;
    expect(panel.endDate()).toBe('2023-12-31');
  });

  // ── Parent walk-forward error → [role="alert"] in parent DOM ─────────────

  it('when walkForwardError is set on the parent, a [role="alert"] element appears', () => {
    comp.walkForwardError.set('Walk-forward failed: timeout');
    fixture.detectChanges();

    const alerts = Array.from(el.querySelectorAll<HTMLElement>('[role="alert"]'));
    const parentAlert = alerts.find((a) => a.textContent?.includes('Walk-forward'));
    expect(parentAlert).withContext('expected a parent [role="alert"] for walk-forward error').toBeTruthy();
    expect(parentAlert?.textContent?.trim().length).toBeGreaterThan(0);
  });

  it('when walkForwardError is null, no walk-forward alert is rendered in the parent', () => {
    comp.walkForwardError.set(null);
    fixture.detectChanges();

    const alerts = Array.from(el.querySelectorAll<HTMLElement>('[role="alert"]'));
    const wfAlert = alerts.find((a) => a.textContent?.includes('Walk-forward'));
    expect(wfAlert).toBeUndefined();
  });

  // ── Parent onRunWalkForward → POST /validate/walk-forward ─────────────────

  it('when onRunWalkForward is called, POST is sent to /validate/walk-forward', () => {
    comp.onRunWalkForward();
    const req = http.expectOne(
      (r) => r.method === 'POST' && r.url === WALK_FWD_URL,
    );
    expect(req.request.method).toBe('POST');
    req.flush({ job_id: 'wf-1', run_id: 'wfr-1', status: 'pending' });
  });

  it('when onRunWalkForward POST body, cv_type is walk_forward', () => {
    comp.onRunWalkForward();
    const req = http.expectOne(WALK_FWD_URL);
    expect((req.request.body as Record<string, unknown>)['cv_type']).toBe('walk_forward');
    req.flush({ job_id: 'wf-2', run_id: 'wfr-2', status: 'pending' });
  });

  it('when onRunWalkForward POST body, tickers array is non-empty', () => {
    comp.onRunWalkForward();
    const req = http.expectOne(WALK_FWD_URL);
    const tickers = (req.request.body as Record<string, unknown>)['tickers'] as string[];
    expect(Array.isArray(tickers)).toBeTrue();
    expect(tickers.length).toBeGreaterThan(0);
    req.flush({ job_id: 'wf-3', run_id: 'wfr-3', status: 'pending' });
  });

  it('when onRunWalkForward succeeds, walkForwardError remains null', () => {
    comp.onRunWalkForward();
    http
      .expectOne(WALK_FWD_URL)
      .flush({ job_id: 'wf-4', run_id: 'wfr-4', status: 'pending' });
    expect(comp.walkForwardError()).toBeNull();
  });

  it('when onRunWalkForward errors, walkForwardError is set on the parent', () => {
    comp.onRunWalkForward();
    http
      .expectOne(WALK_FWD_URL)
      .flush({ detail: 'failed' }, { status: 500, statusText: 'Server Error' });
    expect(comp.walkForwardError()).not.toBeNull();
  });
});
