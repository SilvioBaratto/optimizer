/**
 * factor-research-macro-cma.spec.ts — issue #1025
 *
 * Tests:
 *   - GET /views/macro-calibration on init populates macroCalibration() and
 *     the regime/TAA card renders correctly.
 *   - GET /macro-data/te-observations (via onFetchTe()) populates teObservations()
 *     and the CMA/screener renders.
 *   - Both error paths set the correct panelErrors key and show [role="alert"] banners.
 *
 * Implementation note:
 *   macroCalibration() uses backgroundContext() so the backgroundStubInterceptor in
 *   testbed-factory intercepts it with a 204 error in the default test setup. To test
 *   the success path we spy on FactorsService.macroCalibration() before component
 *   creation so the spy returns the desired observable directly without hitting HTTP.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { FactorResearchComponent } from './factor-research';
import { FactorsService } from './factors.service';
import type { TradingEconomicsObservation } from './factor.model';
import type { MacroCalibrationResponse } from '../core/models/macro-intelligence.model';

const MACRO_CAL_RESPONSE: MacroCalibrationResponse = {
  phase: 'EARLY_EXPANSION',
  delta: 3.0,
  tau: 0.05,
  confidence: 0.8,
  rationale: 'positive growth indicators',
  macro_summary: 'Expansion phase confirmed',
  timestamp: '2024-01-01T00:00:00Z',
  bl_config: {
    views: [],
    tau: 0.05,
    prior_config: {
      mu_estimator: 'shrunk',
      risk_aversion: 3.0,
      cov_estimator: 'ledoit_wolf',
    },
  },
};

const TE_OBS_RESPONSE: TradingEconomicsObservation[] = [
  {
    id: 'te-1',
    country: 'USA',
    indicator_key: 'gdp_growth',
    date: '2024-01-01',
    value: 2.5,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
];

describe('FactorResearchComponent — macro-calibration + TE observations (issue #1025)', () => {
  let fixture: ComponentFixture<FactorResearchComponent>;
  let comp: FactorResearchComponent;
  let http: HttpTestingController;
  let factorsSvc: FactorsService;
  let el: HTMLElement;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await configureTestBed({
      imports: [FactorResearchComponent],
      withHttp: true,
      providers: [ICON_PROVIDER],
    });
    factorsSvc = TestBed.inject(FactorsService);
    http = injectHttp();
  });

  afterEach(() => {
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  function createComponent(macroResult: 'success' | 'error' = 'success'): void {
    if (macroResult === 'success') {
      spyOn(factorsSvc, 'macroCalibration').and.returnValue(of(MACRO_CAL_RESPONSE));
    } else {
      spyOn(factorsSvc, 'macroCalibration').and.returnValue(
        throwError(() => new Error('Macro calibration unavailable')),
      );
    }
    fixture = TestBed.createComponent(FactorResearchComponent);
    comp = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
  }

  // ── macro-calibration: success path ─────────────────────────────────────────

  it('on init, macroCalibration() is populated when API call succeeds', () => {
    createComponent('success');

    expect(comp.macroCalibration()).not.toBeNull();
    expect(comp.macroCalibration()?.phase).toBe('EARLY_EXPANSION');
  });

  it('on init success, panelErrors["macro-calibration"] is null', () => {
    createComponent('success');

    expect(comp.panelErrors()['macro-calibration'] ?? null).toBeNull();
  });

  it('when macroCalibration() is set, macroPhase() computed signal returns the phase', () => {
    createComponent('success');

    expect(comp.macroPhase()).toBe('EARLY_EXPANSION');
  });

  it('when macroCalibration() is set, macroConfidence() computed signal includes "80"', () => {
    createComponent('success');

    expect(comp.macroConfidence()).toContain('80');
  });

  it('when macroCalibration() is set, macroDelta() computed signal includes "3.00"', () => {
    createComponent('success');

    expect(comp.macroDelta()).toContain('3.00');
  });

  it('when macroCalibration() is set, regime tab renders the phase value', () => {
    createComponent('success');

    comp.activeTab.set('regime');
    fixture.detectChanges();

    expect(el.textContent).toContain('EARLY_EXPANSION');
  });

  it('when macroCalibration() is set, taa tab renders app-taa-panel', () => {
    createComponent('success');

    comp.activeTab.set('taa');
    fixture.detectChanges();

    expect(el.querySelector('app-taa-panel'))
      .withContext('app-taa-panel must render when taa tab is active')
      .not.toBeNull();
  });

  it('when macroCalibration() is set, regime tab does not show [role="alert"]', () => {
    createComponent('success');

    comp.activeTab.set('regime');
    fixture.detectChanges();

    const alert = el.querySelector('[role="alert"]');
    expect(alert)
      .withContext('no [role="alert"] should appear on regime tab when macro-cal succeeds')
      .toBeNull();
  });

  // ── macro-calibration: error path ────────────────────────────────────────────

  it('when GET /views/macro-calibration errors, panelErrors["macro-calibration"] is set', () => {
    createComponent('error');

    expect(comp.panelErrors()['macro-calibration'])
      .withContext('panelErrors["macro-calibration"] must be truthy on error')
      .toBeTruthy();
  });

  it('when macro-calibration errors, macroCalibration() stays null', () => {
    createComponent('error');

    expect(comp.macroCalibration()).toBeNull();
  });

  it('when macro-calibration errors, macroPhase() returns dash placeholder', () => {
    createComponent('error');

    expect(comp.macroPhase()).toBe('—');
  });

  it('when macro-calibration errors, regime tab shows [role="alert"] banner', () => {
    createComponent('error');

    comp.activeTab.set('regime');
    fixture.detectChanges();

    const alert = el.querySelector('[role="alert"]');
    expect(alert)
      .withContext('regime tab must render [role="alert"] when macro-cal errors')
      .not.toBeNull();
    expect((alert as HTMLElement | null)?.textContent?.trim().length)
      .withContext('[role="alert"] text must not be blank')
      .toBeGreaterThan(0);
  });

  // ── TE observations: success path ─────────────────────────────────────────────

  it('when onFetchTe() is called and succeeds, teObservations() is populated', () => {
    createComponent('success');
    spyOn(factorsSvc, 'teObservations').and.returnValue(of(TE_OBS_RESPONSE));

    comp.onFetchTe();

    expect(comp.teObservations().length).toBe(1);
    expect(comp.teObservations()[0].indicator_key).toBe('gdp_growth');
    expect(comp.teObservations()[0].country).toBe('USA');
  });

  it('when teObservations() is populated, screener tab renders app-asset-screener-panel', () => {
    createComponent('success');
    spyOn(factorsSvc, 'teObservations').and.returnValue(of(TE_OBS_RESPONSE));

    comp.onFetchTe();
    comp.activeTab.set('screener');
    fixture.detectChanges();

    expect(el.querySelector('app-asset-screener-panel'))
      .withContext('app-asset-screener-panel must render on screener tab')
      .not.toBeNull();
  });

  it('when onFetchTe() succeeds, panelErrors["te"] stays null', () => {
    createComponent('success');
    spyOn(factorsSvc, 'teObservations').and.returnValue(of(TE_OBS_RESPONSE));

    comp.onFetchTe();

    expect(comp.panelErrors()['te'] ?? null)
      .withContext('no te error on success')
      .toBeNull();
  });

  it('when onFetchTe() succeeds, teObservations() carries all returned rows', () => {
    createComponent('success');
    const multiRow: TradingEconomicsObservation[] = [
      { ...TE_OBS_RESPONSE[0], id: 'te-1' },
      { ...TE_OBS_RESPONSE[0], id: 'te-2', indicator_key: 'cpi' },
    ];
    spyOn(factorsSvc, 'teObservations').and.returnValue(of(multiRow));

    comp.onFetchTe();

    expect(comp.teObservations().length).toBe(2);
  });

  // ── TE observations: error path ───────────────────────────────────────────────

  it('when onFetchTe() errors, panelErrors["te"] is set', () => {
    createComponent('success');
    spyOn(factorsSvc, 'teObservations').and.returnValue(
      throwError(() => new Error('TE service unavailable')),
    );

    comp.onFetchTe();

    expect(comp.panelErrors()['te'])
      .withContext('panelErrors["te"] must be truthy on error')
      .toBeTruthy();
  });

  it('when onFetchTe() errors, teObservations() stays empty', () => {
    createComponent('success');
    spyOn(factorsSvc, 'teObservations').and.returnValue(
      throwError(() => new Error('TE fetch failed')),
    );

    comp.onFetchTe();

    expect(comp.teObservations()).toEqual([]);
  });

  it('when TE observations error, screener tab shows [role="alert"] banner', () => {
    createComponent('success');
    spyOn(factorsSvc, 'teObservations').and.returnValue(
      throwError(() => new Error('TE service unavailable')),
    );

    comp.onFetchTe();
    comp.activeTab.set('screener');
    fixture.detectChanges();

    const alert = el.querySelector('[role="alert"]');
    expect(alert)
      .withContext('screener tab must render [role="alert"] when TE errors')
      .not.toBeNull();
    expect((alert as HTMLElement | null)?.textContent?.trim().length)
      .withContext('[role="alert"] text must not be blank')
      .toBeGreaterThan(0);
  });

  it('TE error does not pollute panelErrors["macro-calibration"]', () => {
    createComponent('success');
    spyOn(factorsSvc, 'teObservations').and.returnValue(
      throwError(() => new Error('TE failed')),
    );

    comp.onFetchTe();

    expect(comp.panelErrors()['macro-calibration'] ?? null)
      .withContext('TE error must not pollute macro-calibration error key')
      .toBeNull();
  });
});
