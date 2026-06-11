/**
 * factor-research-render-coverage.spec.ts — issue #941
 *
 * Template-branch coverage for FactorResearchComponent. Mounts the template and
 * drives the three-way gate, the 8 tab arms, the regime nested @if, the
 * compute-job tracker / error banner and the macro KPI cards by setting STATE
 * signals directly. The macro-calibration HTTP fired in ngOnInit is left
 * pending (no http.verify()).
 */

import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { TestBed, ComponentFixture } from '@angular/core/testing';
import { of } from 'rxjs';

import { installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { FactorResearchComponent } from './factor-research';
import { FactorsService } from './factors.service';
import type { MacroCalibrationResponse } from '../core/models/macro-intelligence.model';

const CAL: MacroCalibrationResponse = {
  phase: 'EARLY_EXPANSION',
  delta: 3.5,
  tau: 0.05,
  confidence: 0.78,
  rationale: 'Leading indicators positive',
  macro_summary: 'PMI above 55',
  timestamp: '2026-04-28T00:00:00Z',
  bl_config: {
    views: [],
    tau: 0.05,
    prior_config: { mu_estimator: 'shrunk', risk_aversion: 3.5, cov_estimator: 'ledoit_wolf' },
  },
};

describe('FactorResearchComponent — render-coverage (issue #941)', () => {
  let fixture: ComponentFixture<FactorResearchComponent>;
  let comp: FactorResearchComponent;
  let el: HTMLElement;

  beforeEach(async () => {
    installResizeObserverStub();
    await TestBed.configureTestingModule({
      imports: [FactorResearchComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(FactorResearchComponent);
    comp = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
  });

  describe('loading / error / success three-way', () => {
    it('when isLoading is true, skeleton tiles are shown', () => {
      comp.isLoading.set(true);
      fixture.detectChanges();
      expect(el.querySelectorAll('.skeleton').length).toBeGreaterThan(0);
    });

    it('when hasError is true, the error block and message are shown', () => {
      comp.hasError.set(true);
      comp.errorMessage.set('factor boom');
      fixture.detectChanges();
      expect(el.textContent).toContain('Something went wrong');
      expect(el.textContent).toContain('factor boom');
    });

    it('when not loading and no error, the tab group is rendered', () => {
      expect(el.querySelector('app-tab-group')).not.toBeNull();
    });
  });

  describe('regime tab nested @if', () => {
    it('when macroCalibration is null, the unavailable message is shown', () => {
      comp.activeTab.set('regime');
      comp.macroCalibration.set(null);
      fixture.detectChanges();
      expect(el.textContent).toContain('Macro calibration data unavailable');
    });

    it('when macroCalibration is set, the calibration grid is shown', () => {
      comp.activeTab.set('regime');
      comp.macroCalibration.set(CAL);
      fixture.detectChanges();
      expect(el.textContent).toContain('EARLY_EXPANSION');
      expect(el.textContent).toContain('Leading indicators positive');
    });
  });

  describe('child panel tab arms', () => {
    const cases = [
      { tab: 'taa', selector: 'app-taa-panel' },
      { tab: 'factor-analysis', selector: 'app-factor-analysis-panel' },
      { tab: 'cma-builder', selector: 'app-cma-builder-panel' },
      { tab: 'screener', selector: 'app-asset-screener-panel' },
      { tab: 'score', selector: 'app-score-panel' },
      { tab: 'select', selector: 'app-select-panel' },
      { tab: 'exposure-constraints', selector: 'app-exposure-constraints-panel' },
    ];

    cases.forEach(({ tab, selector }) => {
      it(`when activeTab is "${tab}", ${selector} is rendered`, () => {
        comp.activeTab.set(tab);
        fixture.detectChanges();
        expect(el.querySelector(selector)).not.toBeNull();
      });
    });
  });

  describe('macro KPI cards', () => {
    it('when macroCalibration is null, the Cycle Phase card shows an em dash', () => {
      comp.macroCalibration.set(null);
      fixture.detectChanges();
      const card = Array.from(el.querySelectorAll('app-stat-card')).find((c) =>
        c.textContent?.includes('Cycle Phase'),
      );
      expect(card?.textContent).toContain('—');
    });

    it('when macroCalibration is set, the macro KPI cards show computed values', () => {
      comp.macroCalibration.set(CAL);
      fixture.detectChanges();
      const card = Array.from(el.querySelectorAll('app-stat-card')).find((c) =>
        c.textContent?.includes('Cycle Phase'),
      );
      expect(card?.textContent).toContain('EARLY_EXPANSION');
    });
  });

  describe('compute job tracker and error banner', () => {
    it('when computeJobId is null, no job tracker is rendered', () => {
      comp.computeJobId.set(null);
      fixture.detectChanges();
      expect(el.querySelector('app-job-progress-tracker')).toBeNull();
    });

    it('when computeJobId is set, the job tracker renders and the button is disabled', () => {
      comp.computeJobId.set('job-1');
      fixture.detectChanges();
      expect(el.querySelector('app-job-progress-tracker')).not.toBeNull();
      const btn = Array.from(el.querySelectorAll('button')).find((b) =>
        b.textContent?.includes('Running'),
      ) as HTMLButtonElement | undefined;
      expect(btn?.disabled).toBe(true);
    });

    it('when computeError is set, the compute-failed banner is shown', () => {
      comp.computeError.set('compute boom');
      fixture.detectChanges();
      expect(el.textContent).toContain('Compute failed: compute boom');
    });

    it('when a compute job is already running, onRunCompute fires no new request', () => {
      const factors = TestBed.inject(FactorsService);
      const spy = spyOn(factors, 'compute').and.returnValue(of({ job_id: 'x', status: 'pending', message: '' }));
      comp.computeJobId.set('job-1');
      comp.onRunCompute();
      expect(spy).not.toHaveBeenCalled();
    });
  });
});
