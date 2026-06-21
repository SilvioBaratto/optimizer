/**
 * Explicit Save contract tests for Issue #1045:
 * "feat(optimize): explicit Save persists optimized weights onto the selected portfolio"
 *
 * Covers the UNIT-verifiable AC criteria:
 *
 *   AC-1 / AC-5  applyWeightsToPortfolio → createSnapshot payload shape
 *                (snapshot_type: 'optimization', weights present)
 *   AC-2         runOptimization does NOT auto-persist (save is opt-in)
 *   AC-3         canSave is false when hasPortfolio() is false or no result exists
 *
 * HTTP contract (snapshot_type, weights field-by-field) is also pinned in:
 *   - optimization-studio-contracts.spec.ts  (component integration)
 *   - optimization.service.spec.ts           (service unit)
 *
 * Criteria NOT covered here (oracle: not runtime-verifiable):
 *   AC-4  Success/error states surfaced to user (UI prose)
 *   AC-6  All tests pass (suite gate)
 *   AC-7  SOLID/clean code (subjective quality prose)
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import {
  configureTestBed,
  injectHttp,
  installResizeObserverStub,
  makeOptimizationRunResponse,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { OptimizationStudioComponent } from './optimization-studio';
import { ResultsPanelComponent } from './results-panel/results-panel';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { environment } from '../../environments/environment';

const PORTFOLIO_LIST_URL = `${environment.apiUrl}portfolio/`;

// ── OptimizationStudioComponent — canSave guard ─────────────────────────────

describe('OptimizationStudioComponent — canSave guard (Issue #1045)', () => {
  let fixture: ComponentFixture<OptimizationStudioComponent>;
  let comp: OptimizationStudioComponent;
  let http: HttpTestingController;
  let ctx: PortfolioContextService;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await configureTestBed({
      imports: [OptimizationStudioComponent],
      withHttp: true,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(OptimizationStudioComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
    ctx = TestBed.inject(PortfolioContextService);
    fixture.detectChanges();
  });

  afterEach(() => {
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  // AC-3: no portfolio, no result

  it('when no portfolio is selected and no result exists, canSave is false', () => {
    expect(ctx.currentPortfolioId()).toBeNull();
    expect(comp.hasResult()).toBe(false);
    expect(comp.canSave()).toBe(false);
  });

  it('when portfolio is selected but no optimization has run, canSave is false', () => {
    ctx.currentPortfolioId.set('pf-1');
    http.match(() => true);
    expect(comp.hasResult()).toBe(false);
    expect(comp.canSave()).toBe(false);
  });

  it('when no portfolio selected but a result exists, canSave is false', () => {
    comp.runResult.set(makeOptimizationRunResponse());
    expect(comp.hasResult()).toBe(true);
    expect(ctx.currentPortfolioId()).toBeNull();
    expect(comp.canSave()).toBe(false);
  });

  it('when portfolio is selected and a result exists, canSave is true', () => {
    ctx.currentPortfolioId.set('pf-1');
    comp.runResult.set(makeOptimizationRunResponse());
    expect(comp.canSave()).toBe(true);
  });

  // AC-2: save is opt-in — running optimization must NOT auto-persist

  it('when optimization completes, no snapshot POST is made automatically', () => {
    ctx.currentPortfolioId.set('pf-1');
    comp.onRunPipeline({ optimizerType: 'mean_risk', config: {} });
    http
      .expectOne((r) => r.method === 'POST' && r.url.includes('optimize'))
      .flush({ id: 'run-1', weights: { AAPL: 0.6, MSFT: 0.4 } });
    http.expectNone((r) => r.method === 'POST' && r.url.includes('snapshot'));
  });

  // AC-1 & AC-5: snapshot payload shape

  it('when hasPortfolio is false, onApplyWeights makes no snapshot POST', () => {
    expect(ctx.currentPortfolioId()).toBeNull();
    comp.onApplyWeights({ AAPL: 0.6, MSFT: 0.4 });
    http.expectNone((r) => r.method === 'POST' && r.url.includes('snapshot'));
  });

  it('when portfolio selected, snapshot POST carries snapshot_type "optimization"', () => {
    ctx.currentPortfolioId.set('pf-1');
    comp.onApplyWeights({ AAPL: 0.6, MSFT: 0.4 });

    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('portfolio') && !r.url.includes('snapshot'))
      .flush({ items: [{ id: 'pf-1', name: 'my-portfolio' }] });

    const snap = http.expectOne((r) => r.method === 'POST' && r.url.includes('snapshot'));
    expect(snap.request.body).toEqual(jasmine.objectContaining({ snapshot_type: 'optimization' }));
    snap.flush({ id: 'snap-1' });
  });

  it('when portfolio selected, snapshot POST contains the submitted weights', () => {
    ctx.currentPortfolioId.set('pf-1');
    const weights = { AAPL: 0.6, MSFT: 0.4 };
    comp.onApplyWeights(weights);

    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('portfolio') && !r.url.includes('snapshot'))
      .flush({ items: [{ id: 'pf-1', name: 'my-portfolio' }] });

    const snap = http.expectOne((r) => r.method === 'POST' && r.url.includes('snapshot'));
    expect(snap.request.body).toEqual(jasmine.objectContaining({ weights }));
    snap.flush({ id: 'snap-1' });
  });
});

// ── ResultsPanelComponent — Save button disabled state ──────────────────────

describe('ResultsPanelComponent — Save button disabled state (Issue #1045)', () => {
  let fixture: ComponentFixture<ResultsPanelComponent>;
  let el: HTMLElement;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [ResultsPanelComponent],
      withHttp: false,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(ResultsPanelComponent);
    el = fixture.nativeElement as HTMLElement;
    // Provide a result so the Save button is rendered (inside @else hasResult branch).
    fixture.componentRef.setInput('result', makeOptimizationRunResponse());
    fixture.componentRef.setInput('hasResult', true);
  });

  function saveButton(): HTMLButtonElement | null {
    return (
      Array.from(el.querySelectorAll<HTMLButtonElement>('button')).find((b) =>
        b.textContent?.toLowerCase().includes('save'),
      ) ?? null
    );
  }

  // AC-3: disabled when no portfolio

  it('when canSave is false, the Save button is disabled', () => {
    fixture.componentRef.setInput('canSave', false);
    fixture.detectChanges();
    expect(saveButton()?.disabled).toBe(true);
  });

  it('when canSave is false, the Save button shows a tooltip hint mentioning portfolio', () => {
    fixture.componentRef.setInput('canSave', false);
    fixture.detectChanges();
    const title = saveButton()?.title ?? '';
    expect(title.length).toBeGreaterThan(0);
    expect(title.toLowerCase()).toContain('portfolio');
  });

  // AC-3 positive: enabled when canSave is true

  it('when canSave is true, the Save button is enabled', () => {
    fixture.componentRef.setInput('canSave', true);
    fixture.detectChanges();
    expect(saveButton()?.disabled).toBe(false);
  });

  it('when canSave is true, the Save button has no tooltip hint', () => {
    fixture.componentRef.setInput('canSave', true);
    fixture.detectChanges();
    expect(saveButton()?.title ?? '').toBe('');
  });
});
