import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import {
  SCREEN_PRESETS,
  StepScreenPanelComponent,
  type ScreenPreset,
} from './step-screen-panel';
import {
  StepStatus,
  type ScreenStepResult,
} from '../../models/pipeline-builder.model';

const RESULT_OK: ScreenStepResult = {
  n_investable: 850,
  preset: 'developed_markets',
  band_warning: false,
  band_low: 300,
  band_high: 1500,
};

const RESULT_OUT_OF_BAND: ScreenStepResult = {
  ...RESULT_OK,
  n_investable: 120,
  band_warning: true,
};

function setup(): ComponentFixture<StepScreenPanelComponent> {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(StepScreenPanelComponent);
}

function host(fix: ComponentFixture<StepScreenPanelComponent>): HTMLElement {
  return fix.nativeElement as HTMLElement;
}

describe('StepScreenPanelComponent', () => {
  describe('SCREEN_PRESETS constant', () => {
    it('exports the four valid presets', () => {
      expect(SCREEN_PRESETS).toEqual([
        'developed_markets',
        'broad_universe',
        'small_cap',
        'large_cap',
      ]);
    });
  });

  describe('form defaults', () => {
    it('preset defaults to developed_markets and form is valid', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      expect(fix.componentInstance.form.controls.preset.value).toBe('developed_markets');
      expect(fix.componentInstance.form.valid).toBe(true);
    });
  });

  describe('preset validation', () => {
    it('when preset is an unknown value, form is invalid', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      fix.componentInstance.form.controls.preset.setValue('bogus' as ScreenPreset);
      expect(fix.componentInstance.form.valid).toBe(false);
    });

    for (const preset of SCREEN_PRESETS) {
      it(`when preset is "${preset}", form is valid`, async () => {
        const fix = setup();
        fix.componentRef.setInput('stepStatus', StepStatus.Ready);
        await fix.whenStable();

        fix.componentInstance.form.controls.preset.setValue(preset);
        expect(fix.componentInstance.form.valid).toBe(true);
      });
    }
  });

  describe('submit', () => {
    it('when status is Ready and form valid, submit emits runStep({ preset })', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const emitted: Record<string, unknown>[] = [];
      fix.componentInstance.runStep.subscribe((p) => emitted.push(p));

      fix.componentInstance.form.controls.preset.setValue('small_cap');
      const btn = host(fix).querySelector<HTMLButtonElement>(
        'button[data-testid="run-step"]',
      );
      btn!.click();

      expect(emitted).toEqual([{ preset: 'small_cap' }]);
    });

    it('when form invalid, submit does NOT emit', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const emitted: unknown[] = [];
      fix.componentInstance.runStep.subscribe((p) => emitted.push(p));

      fix.componentInstance.form.controls.preset.setValue('nope' as ScreenPreset);
      fix.componentInstance.onSubmit();

      expect(emitted).toEqual([]);
    });
  });

  describe('Running state', () => {
    it('when status is Running, spinner renders and submit is hidden', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Running);
      await fix.whenStable();

      const el = host(fix);
      expect(el.querySelector('[data-testid="spinner"]')).not.toBeNull();
      expect(el.querySelector('button[data-testid="run-step"]')).toBeNull();
    });
  });

  describe('Completed render', () => {
    it('when result.band_warning false, no warning chip and stats render', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_OK);
      await fix.whenStable();

      const el = host(fix);
      const text = el.textContent ?? '';
      expect(text).toContain('850');
      expect(text).toContain('developed_markets');
      expect(text).toContain('300');
      expect(text).toContain('1,500');
      expect(el.querySelector('[data-testid="band-warning"]')).toBeNull();
    });

    it('when result.band_warning true, warning chip renders next to universe count', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_OUT_OF_BAND);
      await fix.whenStable();

      const chip = host(fix).querySelector('[data-testid="band-warning"]');
      expect(chip).not.toBeNull();
      expect(chip!.textContent).toMatch(/outside/i);
    });
  });

  describe('Error render', () => {
    it('when lastError non-null, error block renders', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Error);
      fix.componentRef.setInput('lastError', 'universe size below floor');
      await fix.whenStable();

      const errEl = host(fix).querySelector('[data-testid="error-block"]');
      expect(errEl).not.toBeNull();
      expect(errEl!.textContent).toContain('universe size below floor');
    });
  });
});
