import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { RunConfigPanelComponent } from './run-config-panel';
import type { PipelineConfigSubmit } from '../models/step-params.model';

function buildPanel(): RunConfigPanelComponent {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(RunConfigPanelComponent).componentInstance;
}

describe('RunConfigPanelComponent', () => {
  describe('defaults', () => {
    it('when constructed, matches Pydantic defaults', () => {
      const panel = buildPanel();
      const raw = panel.form.getRawValue();
      expect(raw.rebalance_freq).toBe(21);
      expect(raw.n_selected).toBe(20);
      expect(raw.cost_bps).toBe(10);
      expect(raw.tax_rate).toBe(0.26);
      expect(raw.base_currency).toBe('EUR');
      expect(raw.robust).toBe(false);
      expect(raw.persist).toBe(false);
      expect(raw.start_date).toBeNull();
      expect(raw.end_date).toBeNull();
      expect(raw.seed).toBe(42);
    });

    it('when default, form is valid', () => {
      expect(buildPanel().form.valid).toBe(true);
    });
  });

  // API band is [15, 30] (RunLevelConfig Field(ge=15, le=30)); issue #855.
  describe('n_selected range', () => {
    it('when n_selected = 14, form is invalid', () => {
      const panel = buildPanel();
      panel.form.controls.n_selected.setValue(14);
      expect(panel.form.controls.n_selected.valid).toBe(false);
    });

    it('when n_selected = 31, form is invalid', () => {
      const panel = buildPanel();
      panel.form.controls.n_selected.setValue(31);
      expect(panel.form.controls.n_selected.valid).toBe(false);
    });

    it('when n_selected = 15, form is valid', () => {
      const panel = buildPanel();
      panel.form.controls.n_selected.setValue(15);
      expect(panel.form.controls.n_selected.valid).toBe(true);
    });

    it('when n_selected = 30, form is valid', () => {
      const panel = buildPanel();
      panel.form.controls.n_selected.setValue(30);
      expect(panel.form.controls.n_selected.valid).toBe(true);
    });
  });

  describe('date cross-validator', () => {
    it('when end_date is before start_date, form has dateOrder error', () => {
      const panel = buildPanel();
      panel.form.patchValue({ start_date: '2024-06-01', end_date: '2024-01-01' });
      expect(panel.form.errors?.['dateOrder']).toBe(true);
    });

    it('when end_date equals start_date, form is valid', () => {
      const panel = buildPanel();
      panel.form.patchValue({ start_date: '2024-06-01', end_date: '2024-06-01' });
      expect(panel.form.errors).toBeNull();
    });

    it('when end_date is after start_date, form is valid', () => {
      const panel = buildPanel();
      panel.form.patchValue({ start_date: '2024-01-01', end_date: '2024-06-01' });
      expect(panel.form.errors).toBeNull();
    });

    it('when only one of the two dates is set, no cross-field error', () => {
      const panel = buildPanel();
      panel.form.patchValue({ start_date: '2024-01-01', end_date: null });
      expect(panel.form.errors).toBeNull();
    });
  });

  describe('base_currency', () => {
    it('exposes exactly EUR, GBP, USD', () => {
      expect([...buildPanel().currencies].sort()).toEqual(['EUR', 'GBP', 'USD']);
    });
  });

  describe('step-param defaults', () => {
    it('matches the Pydantic *StepRequest defaults', () => {
      const raw = buildPanel().form.getRawValue();
      expect(raw.include_delisted).toBe(true);
      expect(raw.macro_country).toBe('USA');
      expect(raw.preset).toBe('developed_markets');
      expect(raw.market_proxy_ticker).toBe('URTH');
      expect(raw.min_factors).toBe(2);
      expect(raw.enable_tilts).toBe(true);
      expect(raw.persist_regime).toBe(false);
    });
  });

  describe('onSubmit()', () => {
    it('when form is valid, configSubmit emits { config, steps }', () => {
      const panel = buildPanel();
      let emitted: PipelineConfigSubmit | undefined;
      panel.configSubmit.subscribe((v) => (emitted = v));

      panel.onSubmit();
      expect(emitted).toBeDefined();
      expect(emitted!.config.n_selected).toBe(20);
      expect(emitted!.config.base_currency).toBe('EUR');
      // run-level keys must not leak into config (no step params)
      expect(
        (emitted!.config as unknown as Record<string, unknown>)['preset'],
      ).toBeUndefined();
      expect(emitted!.steps.screen.preset).toBe('developed_markets');
      expect(emitted!.steps.load).toEqual({
        include_delisted: true,
        macro_country: 'USA',
      });
      expect(emitted!.steps.coverage_gate.min_factors).toBe(2);
      expect(emitted!.steps.regime).toEqual({
        enable_tilts: true,
        persist_regime: false,
      });
    });

    it('when form is invalid, configSubmit does not emit', () => {
      const panel = buildPanel();
      panel.form.controls.n_selected.setValue(14); // below the API band [15, 30]
      let emitted: PipelineConfigSubmit | undefined;
      panel.configSubmit.subscribe((v) => (emitted = v));

      panel.onSubmit();
      expect(emitted).toBeUndefined();
    });
  });
});
