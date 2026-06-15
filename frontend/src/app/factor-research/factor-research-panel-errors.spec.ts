/**
 * factor-research-panel-errors.spec.ts — issue #1025
 *
 * Table-driven spec: every panel surfaces a visible non-blank [role="alert"]
 * on its primary API error.
 *
 * Strategy: test through FactorResearchComponent, setting panelErrors() directly
 * and activating the relevant tab. Sub-panels receive errorMessage as an input;
 * all sub-panel error divs must have role="alert" to pass these assertions.
 *
 * The backgroundStubInterceptor in testbed-factory automatically fires a 204
 * error for the BACKGROUND_REQUEST-tagged macro-calibration call on init, which
 * pre-sets panelErrors['macro-calibration']. All other keys are set explicitly.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { FactorResearchComponent } from './factor-research';

function expectVisibleAlert(el: HTMLElement, context: string): void {
  const alert = el.querySelector('[role="alert"]');
  expect(alert).withContext(`[role="alert"] must exist in the DOM — ${context}`).not.toBeNull();
  const text = (alert as HTMLElement | null)?.textContent?.trim() ?? '';
  expect(text.length)
    .withContext(`[role="alert"] text must not be blank — ${context}`)
    .toBeGreaterThan(0);
}

describe('FactorResearchComponent — panel errors (issue #1025)', () => {
  let fixture: ComponentFixture<FactorResearchComponent>;
  let comp: FactorResearchComponent;
  let http: HttpTestingController;
  let el: HTMLElement;

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
    el = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
    // background stub already fired macro-calibration error; other panelErrors still empty
  });

  afterEach(() => {
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  // ── regime tab: macro-calibration error ──────────────────────────────────────
  // The backgroundStubInterceptor already triggered a 204 error on init for the
  // BACKGROUND_REQUEST-tagged macro-calibration call, setting panelErrors['macro-calibration'].
  // Explicitly setting it here ensures a deterministic non-blank message.

  it('regime tab: [role="alert"] is visible when panelErrors["macro-calibration"] is set', () => {
    comp.panelErrors.update((p) => ({ ...p, 'macro-calibration': 'Macro calibration unavailable' }));
    comp.activeTab.set('regime');
    fixture.detectChanges();

    expectVisibleAlert(el, 'regime tab / macro-calibration error');
  });

  // ── taa tab ──────────────────────────────────────────────────────────────────

  it('taa tab: [role="alert"] is visible when panelErrors["taa"] is set', () => {
    comp.panelErrors.update((p) => ({ ...p, taa: 'TAA weights unavailable' }));
    comp.activeTab.set('taa');
    fixture.detectChanges();

    expectVisibleAlert(el, 'taa tab / taa error');
  });

  // ── factor-analysis tab ──────────────────────────────────────────────────────

  it('factor-analysis tab: [role="alert"] is visible when panelErrors["validate"] is set', () => {
    comp.panelErrors.update((p) => ({ ...p, validate: 'Factor validation failed' }));
    comp.activeTab.set('factor-analysis');
    fixture.detectChanges();

    expectVisibleAlert(el, 'factor-analysis tab / validate error');
  });

  it('factor-analysis tab: [role="alert"] is visible when panelErrors["quintile"] is set', () => {
    comp.panelErrors.update((p) => ({ ...p, quintile: 'Quintile spread failed' }));
    comp.activeTab.set('factor-analysis');
    fixture.detectChanges();

    expectVisibleAlert(el, 'factor-analysis tab / quintile error');
  });

  // ── cma-builder tab ──────────────────────────────────────────────────────────

  it('cma-builder tab: [role="alert"] is visible when panelErrors["cma-builder"] is set', () => {
    comp.panelErrors.update((p) => ({ ...p, 'cma-builder': 'CMA build failed' }));
    comp.activeTab.set('cma-builder');
    fixture.detectChanges();

    expectVisibleAlert(el, 'cma-builder tab / cma-builder error');
  });

  // ── screener tab ─────────────────────────────────────────────────────────────

  it('screener tab: [role="alert"] is visible when panelErrors["te"] is set', () => {
    comp.panelErrors.update((p) => ({ ...p, te: 'TE observations fetch failed' }));
    comp.activeTab.set('screener');
    fixture.detectChanges();

    expectVisibleAlert(el, 'screener tab / te error');
  });

  // ── score tab ────────────────────────────────────────────────────────────────

  it('score tab: [role="alert"] is visible when panelErrors["score"] is set', () => {
    comp.panelErrors.update((p) => ({ ...p, score: 'Score run failed' }));
    comp.activeTab.set('score');
    fixture.detectChanges();

    expectVisibleAlert(el, 'score tab / score error');
  });

  // ── select tab ───────────────────────────────────────────────────────────────

  it('select tab: [role="alert"] is visible when panelErrors["select"] is set', () => {
    comp.panelErrors.update((p) => ({ ...p, select: 'Selection failed' }));
    comp.activeTab.set('select');
    fixture.detectChanges();

    expectVisibleAlert(el, 'select tab / select error');
  });

  // ── exposure-constraints tab ─────────────────────────────────────────────────

  it('exposure-constraints tab: [role="alert"] is visible when panelErrors["exposure-constraints"] is set', () => {
    comp.panelErrors.update((p) => ({ ...p, 'exposure-constraints': 'Constraints run failed' }));
    comp.activeTab.set('exposure-constraints');
    fixture.detectChanges();

    expectVisibleAlert(el, 'exposure-constraints tab / exposure-constraints error');
  });

  // ── table-driven: all non-blank panelErrors produce visible alerts ────────────

  const TABLE: Array<{ tab: string; errorKey: string }> = [
    { tab: 'regime', errorKey: 'macro-calibration' },
    { tab: 'taa', errorKey: 'taa' },
    { tab: 'factor-analysis', errorKey: 'validate' },
    { tab: 'cma-builder', errorKey: 'cma-builder' },
    { tab: 'screener', errorKey: 'te' },
    { tab: 'score', errorKey: 'score' },
    { tab: 'select', errorKey: 'select' },
    { tab: 'exposure-constraints', errorKey: 'exposure-constraints' },
  ];

  TABLE.forEach(({ tab, errorKey }) => {
    it(`[table] ${tab} tab: non-blank panelErrors["${errorKey}"] renders [role="alert"]`, () => {
      comp.panelErrors.update((p) => ({ ...p, [errorKey]: `${errorKey} error — non-blank` }));
      comp.activeTab.set(tab);
      fixture.detectChanges();

      const alert = el.querySelector('[role="alert"]');
      expect(alert)
        .withContext(`${tab}: [role="alert"] must exist for key "${errorKey}"`)
        .not.toBeNull();
      const text = (alert as HTMLElement | null)?.textContent?.trim() ?? '';
      expect(text.length)
        .withContext(`${tab}: [role="alert"] text must not be blank`)
        .toBeGreaterThan(0);
    });
  });

  // ── null panelErrors never produce alerts ─────────────────────────────────────

  it('when all panelErrors are null, no [role="alert"] is shown on score tab', () => {
    comp.panelErrors.update((p) => ({ ...p, score: null }));
    comp.activeTab.set('score');
    fixture.detectChanges();

    const alert = el.querySelector('[role="alert"]');
    expect(alert).withContext('null panelErrors must not render [role="alert"]').toBeNull();
  });
});
