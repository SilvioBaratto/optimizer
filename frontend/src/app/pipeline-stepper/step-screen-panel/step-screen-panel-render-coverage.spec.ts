/**
 * Render-coverage spec for StepScreenPanelComponent (issue #901).
 *
 * Drives every template branch at the DOM:
 *   - Ready  → [data-testid="run-step"]
 *   - Running → [data-testid="spinner"]
 *   - Completed + screenResult → app-stat-card elements (structural marker)
 *   - lastError → [data-testid="error-block"]
 *
 * Secondary branch:
 *   - @if (r.band_warning) — badge present when true, absent when false
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { StepScreenPanelComponent } from './step-screen-panel';
import { StepStatus, type ScreenStepResult } from '../../core/models/pipeline-builder.model';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface Harness {
  fixture: ComponentFixture<StepScreenPanelComponent>;
  el: HTMLElement;
}

async function build(): Promise<Harness> {
  await configureTestBed({ imports: [StepScreenPanelComponent], withHttp: false });
  const fixture = TestBed.createComponent(StepScreenPanelComponent);
  fixture.componentRef.setInput('stepStatus', StepStatus.Pending);
  fixture.detectChanges();
  return { fixture, el: fixture.nativeElement as HTMLElement };
}

// ---------------------------------------------------------------------------
// Minimal result fixtures
// ---------------------------------------------------------------------------

const RESULT_NO_WARNING: ScreenStepResult = {
  n_investable: 40,
  preset: 'developed_markets',
  band_warning: false,
  band_low: 20,
  band_high: 60,
};

const RESULT_WITH_WARNING: ScreenStepResult = {
  ...RESULT_NO_WARNING,
  band_warning: true,
};

// ---------------------------------------------------------------------------
// Base status-arm tests
// ---------------------------------------------------------------------------

describe('StepScreenPanelComponent (render coverage)', () => {
  it('when Ready shows run button', async () => {
    const { fixture, el } = await build();
    fixture.componentRef.setInput('stepStatus', StepStatus.Ready);
    fixture.detectChanges();
    expect(el.querySelector('[data-testid="run-step"]')).not.toBeNull();
  });

  it('when Running shows spinner', async () => {
    const { fixture, el } = await build();
    fixture.componentRef.setInput('stepStatus', StepStatus.Running);
    fixture.detectChanges();
    expect(el.querySelector('[data-testid="spinner"]')).not.toBeNull();
  });

  it('when Completed with result, stat-card grid renders', async () => {
    const { fixture, el } = await build();
    fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
    fixture.componentRef.setInput('result', RESULT_NO_WARNING);
    fixture.detectChanges();
    expect(el.querySelector('app-stat-card')).not.toBeNull();
  });

  it('when lastError is set, error-block renders', async () => {
    const { fixture, el } = await build();
    fixture.componentRef.setInput('stepStatus', StepStatus.Error);
    fixture.componentRef.setInput('lastError', 'oops');
    fixture.detectChanges();
    expect(el.querySelector('[data-testid="error-block"]')).not.toBeNull();
  });

  // ---- Secondary: @if (r.band_warning) badge --------------------------------

  describe('band_warning badge', () => {
    it('when band_warning is true, the band-warning badge renders', async () => {
      const { fixture, el } = await build();
      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_WITH_WARNING);
      fixture.detectChanges();
      expect(el.querySelector('[data-testid="band-warning"]')).not.toBeNull();
    });

    it('when band_warning is false, the band-warning badge is absent', async () => {
      const { fixture, el } = await build();
      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_NO_WARNING);
      fixture.detectChanges();
      expect(el.querySelector('[data-testid="band-warning"]')).toBeNull();
    });
  });
});
