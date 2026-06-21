import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { OptimizerPanelComponent, type OptimizerRunRequest } from './optimizer-panel';

const METHOD_IDS = ['min_variance', 'max_sharpe', 'risk_parity_erc', 'hrp'] as const;

function buildPanel(): OptimizerPanelComponent {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(OptimizerPanelComponent).componentInstance;
}

function captureEmit(panel: OptimizerPanelComponent): () => OptimizerRunRequest | undefined {
  let emitted: OptimizerRunRequest | undefined;
  panel.runOptimization.subscribe(r => (emitted = r));
  return () => emitted;
}

describe('OptimizerPanelComponent — 4-method picker', () => {

  // ── cardinality ──────────────────────────────────────────────────────────────

  it('when methods are listed, exactly four are available', () => {
    const panel = buildPanel();
    expect(panel.methods.length).toBe(4);
  });

  // ── membership ───────────────────────────────────────────────────────────────

  for (const id of METHOD_IDS) {
    it(`when methods are listed, "${id}" is among them`, () => {
      const panel = buildPanel();
      expect(panel.methods.map(m => m.id)).toContain(id);
    });
  }

  // ── default selection ─────────────────────────────────────────────────────────

  it('when panel initialises, max-sharpe method is selected by default', () => {
    const panel = buildPanel();
    expect(panel.selectedMethodId()).toBe('max_sharpe');
  });

  // ── method selection ──────────────────────────────────────────────────────────

  it('when selectMethod is called with min-variance, selectedMethodId updates', () => {
    const panel = buildPanel();
    panel.selectMethod('min_variance');
    expect(panel.selectedMethodId()).toBe('min_variance');
  });

  it('when selectMethod is called with hrp, selectedMethodId updates', () => {
    const panel = buildPanel();
    panel.selectMethod('hrp');
    expect(panel.selectedMethodId()).toBe('hrp');
  });

  // ── optimizer_type in emitted request ────────────────────────────────────────

  it('when emitRun is called, optimizerType in the request is always mean_risk', () => {
    const panel = buildPanel();
    const get = captureEmit(panel);
    panel.emitRun();
    expect(get()!.optimizerType).toBe('mean_risk');
  });

  for (const id of METHOD_IDS) {
    it(`when emitRun is called with method "${id}", optimizerType is mean_risk`, () => {
      const panel = buildPanel();
      panel.selectMethod(id);
      const get = captureEmit(panel);
      panel.emitRun();
      expect(get()!.optimizerType).toBe('mean_risk');
    });
  }

  // ── optimizer_type NOT in config (top-level body field only) ─────────────────

  it('when emitRun is called, emitted config does NOT contain optimizer_type', () => {
    const panel = buildPanel();
    const get = captureEmit(panel);
    panel.emitRun();
    expect((get()!.config as Record<string, unknown>)['optimizer_type']).toBeUndefined();
  });

  // ── constraints → config.max_weights ─────────────────────────────────────────

  it('when stock cap is 0.10 (default), emitted config includes max_weights of 0.10', () => {
    const panel = buildPanel();
    const get = captureEmit(panel);
    panel.emitRun();
    expect(get()!.config['max_weights']).toBe(0.10);
  });

  it('when stock cap is set to 0.20, emitted config includes max_weights of 0.20', () => {
    const panel = buildPanel();
    panel.constraintsForm.patchValue({ stockCap: 0.20 });
    const get = captureEmit(panel);
    panel.emitRun();
    expect(get()!.config['max_weights']).toBe(0.20);
  });

  it('when stock cap is set to 0.05, emitted config includes max_weights of 0.05', () => {
    const panel = buildPanel();
    panel.constraintsForm.patchValue({ stockCap: 0.05 });
    const get = captureEmit(panel);
    panel.emitRun();
    expect(get()!.config['max_weights']).toBe(0.05);
  });

  // ── budget / fully-invested → config.budget ──────────────────────────────────

  it('when budget is enabled, emitted config includes a truthy budget value', () => {
    const panel = buildPanel();
    panel.constraintsForm.patchValue({ budget: true });
    const get = captureEmit(panel);
    panel.emitRun();
    expect((get()!.config as Record<string, unknown>)['budget']).toBeTruthy();
  });

  it('when budget is disabled, emitted config omits budget', () => {
    const panel = buildPanel();
    panel.constraintsForm.patchValue({ budget: false });
    const get = captureEmit(panel);
    panel.emitRun();
    expect((get()!.config as Record<string, unknown>)['budget']).toBeFalsy();
  });

  // ── min_weights NEVER in config (minPositionSize is display-only) ─────────────

  it('when emitRun is called, emitted config never contains min_weights', () => {
    const panel = buildPanel();
    panel.constraintsForm.patchValue({ minPositionSize: 0.02 });
    const get = captureEmit(panel);
    panel.emitRun();
    expect((get()!.config as Record<string, unknown>)['min_weights']).toBeUndefined();
  });

  // ── method config fields are fixed per method (no raw risk-measure selection) ─

  it('when max-sharpe is selected and run emitted, objective is maximize_ratio', () => {
    const panel = buildPanel();
    panel.selectMethod('max_sharpe');
    const get = captureEmit(panel);
    panel.emitRun();
    expect(get()!.config['objective']).toBe('maximize_ratio');
  });

  it('when min-variance is selected and run emitted, objective is minimize_risk', () => {
    const panel = buildPanel();
    panel.selectMethod('min_variance');
    const get = captureEmit(panel);
    panel.emitRun();
    expect(get()!.config['objective']).toBe('minimize_risk');
  });

  for (const id of METHOD_IDS) {
    it(`when method "${id}" is selected, emitted config has a fixed risk_measure string`, () => {
      const panel = buildPanel();
      panel.selectMethod(id);
      const get = captureEmit(panel);
      panel.emitRun();
      expect(typeof get()!.config['risk_measure']).toBe('string');
    });
  }

  // ── default constraints ────────────────────────────────────────────────────────

  it('when panel initialises, long-only is enabled by default', () => {
    const panel = buildPanel();
    expect(panel.constraintsForm.getRawValue().longOnly).toBe(true);
  });

  it('when panel initialises, stock cap defaults to 0.10', () => {
    const panel = buildPanel();
    expect(panel.constraintsForm.getRawValue().stockCap).toBe(0.10);
  });

  it('when panel initialises, min position size defaults to 0.02', () => {
    const panel = buildPanel();
    expect(panel.constraintsForm.getRawValue().minPositionSize).toBe(0.02);
  });

  it('when panel initialises, budget toggle is off by default', () => {
    const panel = buildPanel();
    expect(panel.constraintsForm.getRawValue().budget).toBe(false);
  });
});
