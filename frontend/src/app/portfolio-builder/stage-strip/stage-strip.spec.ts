import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { StageStripComponent } from './stage-strip';
import {
  BUILDER_STAGES,
  type BuilderStage,
  STAGE_LABELS,
} from '../models/builder-stage';
import { BuilderStore } from '../state/builder.store';
import type {
  LoadStepResult,
  OptimizeStepResult,
  PipelineStepId,
  RunLevelConfig,
} from '../../core/models/pipeline-builder.model';

type ResultMap = Map<PipelineStepId, Record<string, unknown> | null>;

function emptyResults(): ResultMap {
  return new Map();
}

function loadResult(): Record<string, unknown> {
  const r: LoadStepResult = {
    n_tickers: 42,
    n_trading_days: 1260,
    assembly_hash: 'h',
    base_currency: 'EUR',
    price_start: '2020-01-01',
    price_end: '2025-01-01',
  };
  return r as unknown as Record<string, unknown>;
}

function optimizeResult(): Record<string, unknown> {
  const r: OptimizeStepResult = {
    weights: [],
    n_selected: 25,
    is_sharpe: 1.2,
    net_sharpe: 1.1,
    hockey_stick_warning: false,
    sector_breakdown: {},
    country_breakdown: {},
  };
  return r as unknown as Record<string, unknown>;
}

function defaultConfig(): RunLevelConfig {
  return {
    rebalance_freq: 21,
    n_selected: 25,
    cost_bps: 10,
    tax_rate: 0.26,
    base_currency: 'EUR',
    robust: false,
    persist: false,
    start_date: null,
    end_date: null,
    seed: 42,
  };
}

function tabButtons(host: HTMLElement): HTMLButtonElement[] {
  return Array.from(host.querySelectorAll<HTMLButtonElement>('button[role="tab"]'));
}

function chips(host: HTMLElement): HTMLElement[] {
  return Array.from(
    host.querySelectorAll<HTMLElement>('[data-region="stage-chips"] [data-stage]'),
  );
}

describe('StageStripComponent', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideZonelessChangeDetection()],
    });
  });

  it('when rendered with currentStage="universe", 6 tabs appear in BUILDER_STAGES order with STAGE_LABELS labels', () => {
    const fixture = TestBed.createComponent(StageStripComponent);
    fixture.componentRef.setInput('currentStage', 'universe' as BuilderStage);
    fixture.componentRef.setInput('stepResults', emptyResults());
    fixture.detectChanges();

    const buttons = tabButtons(fixture.nativeElement as HTMLElement);
    expect(buttons.length).toBe(6);
    BUILDER_STAGES.forEach((stage, i) => {
      expect(buttons[i].textContent?.trim()).toBe(STAGE_LABELS[stage]);
    });
  });

  it('when currentStage is set to "optimize", the matching tab is aria-selected', () => {
    const fixture = TestBed.createComponent(StageStripComponent);
    fixture.componentRef.setInput('currentStage', 'optimize' as BuilderStage);
    fixture.componentRef.setInput('stepResults', emptyResults());
    fixture.detectChanges();

    const buttons = tabButtons(fixture.nativeElement as HTMLElement);
    const idx = BUILDER_STAGES.indexOf('optimize');
    expect(buttons[idx].getAttribute('aria-selected')).toBe('true');
    buttons.forEach((btn, i) => {
      if (i !== idx) {
        expect(btn.getAttribute('aria-selected')).toBe('false');
      }
    });
  });

  it('when currentStage is null, the "universe" tab is aria-selected by default', () => {
    const fixture = TestBed.createComponent(StageStripComponent);
    fixture.componentRef.setInput('currentStage', null);
    fixture.componentRef.setInput('stepResults', emptyResults());
    fixture.detectChanges();

    const buttons = tabButtons(fixture.nativeElement as HTMLElement);
    const idx = BUILDER_STAGES.indexOf('universe');
    expect(buttons[idx].getAttribute('aria-selected')).toBe('true');
  });

  it('when a tab is clicked, stageSelect emits the matching BuilderStage', () => {
    const fixture = TestBed.createComponent(StageStripComponent);
    fixture.componentRef.setInput('currentStage', 'universe' as BuilderStage);
    fixture.componentRef.setInput('stepResults', emptyResults());
    const emitted: BuilderStage[] = [];
    fixture.componentInstance.stageSelect.subscribe((s) => emitted.push(s));
    fixture.detectChanges();

    const buttons = tabButtons(fixture.nativeElement as HTMLElement);
    buttons[BUILDER_STAGES.indexOf('review')].click();
    expect(emitted).toEqual(['review']);
  });

  it('when each tab is clicked in turn, stageSelect emits each BuilderStage id in order', () => {
    const fixture = TestBed.createComponent(StageStripComponent);
    fixture.componentRef.setInput('currentStage', 'universe' as BuilderStage);
    fixture.componentRef.setInput('stepResults', emptyResults());
    const emitted: BuilderStage[] = [];
    fixture.componentInstance.stageSelect.subscribe((s) => emitted.push(s));
    fixture.detectChanges();

    const buttons = tabButtons(fixture.nativeElement as HTMLElement);
    buttons.forEach((b) => b.click());
    expect(emitted).toEqual([...BUILDER_STAGES]);
  });

  it('when stepResults is empty, no chip is rendered', () => {
    const fixture = TestBed.createComponent(StageStripComponent);
    fixture.componentRef.setInput('currentStage', 'universe' as BuilderStage);
    fixture.componentRef.setInput('stepResults', emptyResults());
    fixture.detectChanges();

    expect(chips(fixture.nativeElement as HTMLElement).length).toBe(0);
  });

  it('when stepResults has a load entry, a single "universe" chip is rendered with the stageSummary text', () => {
    const fixture = TestBed.createComponent(StageStripComponent);
    fixture.componentRef.setInput('currentStage', 'universe' as BuilderStage);
    const results: ResultMap = new Map([['load', loadResult()]]);
    fixture.componentRef.setInput('stepResults', results);
    fixture.detectChanges();

    const ch = chips(fixture.nativeElement as HTMLElement);
    expect(ch.length).toBe(1);
    expect(ch[0].getAttribute('data-stage')).toBe('universe');
    expect(ch[0].textContent?.trim()).toBe('42 tickers · 1260 days · EUR');
  });

  it('when stepResults has load and optimize entries, two chips render in BUILDER_STAGES order with matching text', () => {
    const fixture = TestBed.createComponent(StageStripComponent);
    fixture.componentRef.setInput('currentStage', 'universe' as BuilderStage);
    const results: ResultMap = new Map([
      ['load', loadResult()],
      ['optimize', optimizeResult()],
    ]);
    fixture.componentRef.setInput('stepResults', results);
    fixture.detectChanges();

    const ch = chips(fixture.nativeElement as HTMLElement);
    expect(ch.map((c) => c.getAttribute('data-stage'))).toEqual([
      'universe',
      'optimize',
    ]);
    expect(ch[0].textContent?.trim()).toBe('42 tickers · 1260 days · EUR');
    expect(ch[1].textContent?.trim()).toBe('25 holdings · net Sharpe 1.10');
  });
});

@Component({
  selector: 'test-strip-host',
  imports: [StageStripComponent],
  providers: [BuilderStore],
  template: `
    <app-stage-strip
      [currentStage]="store.currentStage()"
      [stepResults]="store.stepResults()"
      (stageSelect)="store.setStage($event)"
    />
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
class StripHostComponent {
  readonly store = inject(BuilderStore);
}

describe('StageStripComponent state preservation', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideZonelessChangeDetection()],
    });
  });

  it('when a tab is clicked while BuilderStore.config is set, config is preserved', () => {
    const fixture = TestBed.createComponent(StripHostComponent);
    const store = fixture.componentInstance.store;
    const cfg = defaultConfig();
    store.setConfig(cfg);
    store.setStage('universe');
    fixture.detectChanges();

    const host = fixture.nativeElement as HTMLElement;
    const buttons = tabButtons(host);
    buttons[BUILDER_STAGES.indexOf('optimize')].click();
    fixture.detectChanges();

    expect(store.currentStage()).toBe('optimize');
    expect(store.config()).toBe(cfg);
  });
});
