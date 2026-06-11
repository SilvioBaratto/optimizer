/**
 * optimization-studio-render-coverage.spec.ts — issue #941
 *
 * Template-branch coverage for OptimizationStudioComponent. Mounts the template
 * and drives the three-way gate, run/apply/pipeline status banners, the polling
 * job tracker and the @switch (activeNode) cases (p2..p6 + default + no-node)
 * by setting STATE signals directly. PortfolioContextService is reset so the
 * date-range effect is deterministic.
 */

import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { TestBed, ComponentFixture } from '@angular/core/testing';

import { installResizeObserverStub, makeOptimizationRunResponse } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { OptimizationStudioComponent } from './optimization-studio';
import { PortfolioContextService } from '../core/services/portfolio-context.service';

describe('OptimizationStudioComponent — render-coverage (issue #941)', () => {
  let fixture: ComponentFixture<OptimizationStudioComponent>;
  let comp: OptimizationStudioComponent;
  let el: HTMLElement;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await TestBed.configureTestingModule({
      imports: [OptimizationStudioComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
      ],
    }).compileComponents();

    TestBed.inject(PortfolioContextService).reset();
    fixture = TestBed.createComponent(OptimizationStudioComponent);
    comp = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
  });

  afterEach(() => localStorage.clear());

  describe('loading / error / success three-way', () => {
    it('when isLoading is true, skeleton tiles are shown', () => {
      comp.isLoading.set(true);
      fixture.detectChanges();
      expect(el.querySelectorAll('.skeleton').length).toBeGreaterThan(0);
    });

    it('when hasError is true, the error block and message are shown', () => {
      comp.hasError.set(true);
      comp.errorMessage.set('opt boom');
      fixture.detectChanges();
      expect(el.textContent).toContain('Something went wrong');
      expect(el.textContent).toContain('opt boom');
    });

    it('when not loading and no error, the pipeline builder is rendered', () => {
      expect(el.querySelector('app-pipeline-builder')).not.toBeNull();
    });
  });

  describe('status banners', () => {
    it('when runError is set, the optimization-failed banner is shown', () => {
      comp.runError.set('run boom');
      fixture.detectChanges();
      expect(el.textContent).toContain('Optimization failed: run boom');
    });

    it('when polling, the run job tracker is shown', () => {
      comp.runJobId.set('job-1'); // runResult null → isPolling true
      fixture.detectChanges();
      expect(el.textContent).toContain('Pipeline running');
      expect(el.querySelector('app-job-progress-tracker')).not.toBeNull();
    });

    it('when applyStatus is "saving", the saving banner is shown', () => {
      comp.applyStatus.set('saving');
      fixture.detectChanges();
      expect(el.textContent).toContain('Saving optimization snapshot');
    });

    it('when applyStatus is "success", the snapshot-saved banner names the portfolio', () => {
      comp.applyStatus.set('success');
      comp.appliedPortfolioName.set('Alpha');
      fixture.detectChanges();
      expect(el.textContent).toContain("Snapshot saved to 'Alpha'");
    });

    it('when applyError is set, the apply-failed banner is shown', () => {
      comp.applyError.set('apply boom');
      fixture.detectChanges();
      expect(el.textContent).toContain('Apply weights failed: apply boom');
    });

    it('when pipelineStatus is set, the status message is shown', () => {
      comp.pipelineStatus.set('Pipeline saved.');
      fixture.detectChanges();
      expect(el.textContent).toContain('Pipeline saved.');
    });
  });

  describe('@switch (activeNode) cases', () => {
    const cases = [
      { node: 'p2', selector: 'app-preprocessing-panel' },
      { node: 'p3', selector: 'app-moment-panel' },
      { node: 'p4', selector: 'app-view-panel' },
      { node: 'p5', selector: 'app-optimizer-panel' },
      { node: 'p6', selector: 'app-results-panel' },
      { node: 'p9', selector: 'app-results-panel' }, // @default arm
    ];

    cases.forEach(({ node, selector }) => {
      it(`when activeNode is "${node}", ${selector} is rendered`, () => {
        comp.activeNode.set(node);
        fixture.detectChanges();
        expect(el.querySelector(selector)).not.toBeNull();
      });
    });

    it('when no node is active, the results panel is rendered', () => {
      comp.activeNode.set(null);
      fixture.detectChanges();
      expect(el.querySelector('app-results-panel')).not.toBeNull();
    });
  });

  describe('save-pipeline button enablement', () => {
    it('is disabled when there is no result and no tickers', () => {
      comp.tickers.set([]);
      fixture.detectChanges();
      const btn = Array.from(el.querySelectorAll('button')).find((b) =>
        b.textContent?.includes('Save Pipeline'),
      ) as HTMLButtonElement | undefined;
      expect(btn?.disabled).toBe(true);
    });

    it('is enabled once a result exists', () => {
      comp.runResult.set(makeOptimizationRunResponse());
      comp.tickers.set([]);
      fixture.detectChanges();
      const btn = Array.from(el.querySelectorAll('button')).find((b) =>
        b.textContent?.includes('Save Pipeline'),
      ) as HTMLButtonElement | undefined;
      expect(btn?.disabled).toBe(false);
    });
  });
});
