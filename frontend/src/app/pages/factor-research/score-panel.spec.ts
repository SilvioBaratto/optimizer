import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { ScorePanelComponent } from './score-panel';
import { ICON_PROVIDER } from '../../icons';
import type {
  CompositeMethod,
  FactorScoreApiResponse,
  FactorScoreRequest,
} from '../../models/factor.model';

function last(emitted: FactorScoreRequest[]): FactorScoreRequest {
  if (emitted.length === 0) throw new Error('runScore never fired');
  return emitted[emitted.length - 1];
}

describe('ScorePanelComponent', () => {
  let fixture: ComponentFixture<ScorePanelComponent>;
  let component: ScorePanelComponent;
  let root: HTMLElement;
  let emitted: FactorScoreRequest[];

  const TICKERS = ['AAPL', 'MSFT', 'GOOG'];

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [ScorePanelComponent],
      providers: [provideZonelessChangeDetection(), ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(ScorePanelComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('tickers', TICKERS);
    fixture.detectChanges();
    root = fixture.nativeElement as HTMLElement;
    emitted = [];
    component.runScore.subscribe((req) => emitted.push(req));
  });

  function clickRun(): void {
    const btn = root.querySelector<HTMLButtonElement>(
      '[data-testid="run-score-btn"]',
    );
    if (!btn) throw new Error('Run button not found');
    btn.click();
    fixture.detectChanges();
  }

  function selectMethod(method: CompositeMethod): void {
    component.method.set(method);
    fixture.detectChanges();
  }

  it("renders a <select> with the five CompositeMethod options (default 'equal_weight')", () => {
    const select = root.querySelector<HTMLSelectElement>(
      '[data-testid="composite-method"]',
    );
    expect(select).not.toBeNull();
    const values = Array.from(select!.options).map((o) => o.value);
    expect(values).toEqual([
      'equal_weight',
      'ic_weighted',
      'icir_weighted',
      'ridge_weighted',
      'gbt_weighted',
    ]);
    expect(component.method()).toBe('equal_weight');
  });

  it("emits a payload with today's ISO score_date and the tickers input on Run click", () => {
    clickRun();
    const req = last(emitted);
    expect(req.tickers).toEqual(TICKERS);
    expect(req.composite_method).toBe('equal_weight');
    expect(req.score_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    // today must be within a 2-day window of "now" (timezone tolerance)
    const today = new Date();
    const parsed = new Date(req.score_date);
    expect(Math.abs(today.getTime() - parsed.getTime())).toBeLessThan(
      2 * 24 * 60 * 60 * 1000,
    );
  });

  it("does NOT include training_start_date or training_end_date for equal_weight", () => {
    selectMethod('equal_weight');
    clickRun();
    const req = last(emitted);
    expect(req.training_start_date).toBeUndefined();
    expect(req.training_end_date).toBeUndefined();
  });

  it("does NOT include training dates for ic_weighted", () => {
    selectMethod('ic_weighted');
    clickRun();
    const req = last(emitted);
    expect(req.training_start_date).toBeUndefined();
    expect(req.training_end_date).toBeUndefined();
  });

  it("does NOT include training dates for icir_weighted", () => {
    selectMethod('icir_weighted');
    clickRun();
    const req = last(emitted);
    expect(req.training_start_date).toBeUndefined();
    expect(req.training_end_date).toBeUndefined();
  });

  it("includes training_start_date / training_end_date for ridge_weighted (approx last 252d window)", () => {
    selectMethod('ridge_weighted');
    clickRun();
    const req = last(emitted);
    expect(req.training_start_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(req.training_end_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    const start = new Date(req.training_start_date!).getTime();
    const end = new Date(req.training_end_date!).getTime();
    const dayMs = 24 * 60 * 60 * 1000;
    const spanDays = (end - start) / dayMs;
    expect(spanDays).toBeGreaterThanOrEqual(364);
    expect(spanDays).toBeLessThanOrEqual(366);
  });

  it("includes training dates for gbt_weighted", () => {
    selectMethod('gbt_weighted');
    clickRun();
    const req = last(emitted);
    expect(req.training_start_date).toBeDefined();
    expect(req.training_end_date).toBeDefined();
  });

  it("never includes group_weights in the payload (YAGNI)", () => {
    for (const m of [
      'equal_weight',
      'ic_weighted',
      'icir_weighted',
      'ridge_weighted',
      'gbt_weighted',
    ] as CompositeMethod[]) {
      selectMethod(m);
      clickRun();
      const req = last(emitted);
      expect(req.group_weights).toBeUndefined();
    }
  });

  it("maps scoreResult to sortable rows with quintile derived from descending rank", () => {
    // 10 tickers → buckets of 2 each.
    const scores: Record<string, number> = {
      A: 10,
      B: 9,
      C: 8,
      D: 7,
      E: 6,
      F: 5,
      G: 4,
      H: 3,
      I: 2,
      J: 1,
    };
    const result: FactorScoreApiResponse = {
      score_date: '2026-04-18',
      scores,
      group_contributions: {},
    };
    fixture.componentRef.setInput('scoreResult', result);
    fixture.detectChanges();

    const rows = component.scoreRows();
    expect(rows.length).toBe(10);
    // Top two (A, B) are quintile 1, bottom two (I, J) are quintile 5.
    expect(rows[0].ticker).toBe('A');
    expect(rows[0].quintile).toBe(1);
    expect(rows[1].ticker).toBe('B');
    expect(rows[1].quintile).toBe(1);
    expect(rows[8].ticker).toBe('I');
    expect(rows[8].quintile).toBe(5);
    expect(rows[9].ticker).toBe('J');
    expect(rows[9].quintile).toBe(5);
  });

  it("disables the Run button while loading=true", () => {
    fixture.componentRef.setInput('loading', true);
    fixture.detectChanges();
    const btn = root.querySelector<HTMLButtonElement>(
      '[data-testid="run-score-btn"]',
    );
    expect(btn!.disabled).toBe(true);
  });

  it("renders the inline error banner with the provided error message", () => {
    fixture.componentRef.setInput('errorMessage', 'score backend down');
    fixture.detectChanges();
    const banner = root.querySelector('[data-testid="score-error"]');
    expect(banner).not.toBeNull();
    expect(banner!.textContent).toContain('score backend down');
  });

  it("shows zero rows and an enabled Run button before the first run", () => {
    const btn = root.querySelector<HTMLButtonElement>(
      '[data-testid="run-score-btn"]',
    );
    expect(btn!.disabled).toBe(false);
    expect(component.scoreRows().length).toBe(0);
  });
});
