import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { SelectPanelComponent } from './select-panel';
import { ICON_PROVIDER } from '../../icons';
import { deriveSelectRows } from '../../models/factor.model';
import type {
  FactorSelectApiResponse,
  FactorSelectRequest,
} from '../../models/factor.model';

function last(emitted: FactorSelectRequest[]): FactorSelectRequest {
  if (emitted.length === 0) throw new Error('runSelect never fired');
  return emitted[emitted.length - 1];
}

function sampleResponse(): FactorSelectApiResponse {
  return {
    selected_tickers: ['AAPL', 'MSFT', 'NVDA'],
    count: 3,
    turnover: 0.125,
    buffer_zone: { entered: ['NVDA'], exited: ['GOOG'] },
  };
}

describe('SelectPanelComponent', () => {
  let fixture: ComponentFixture<SelectPanelComponent>;
  let component: SelectPanelComponent;
  let root: HTMLElement;
  let emitted: FactorSelectRequest[];

  const TICKERS = ['AAPL', 'MSFT', 'GOOG', 'NVDA'];

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [SelectPanelComponent],
      providers: [provideZonelessChangeDetection(), ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(SelectPanelComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('tickers', TICKERS);
    fixture.detectChanges();
    root = fixture.nativeElement as HTMLElement;
    emitted = [];
    component.runSelect.subscribe((req) => emitted.push(req));
  });

  function clickRun(): void {
    const btn = root.querySelector<HTMLButtonElement>(
      '[data-testid="run-select-btn"]',
    );
    if (!btn) throw new Error('Run button not found');
    btn.click();
    fixture.detectChanges();
  }

  // ------------------- Form defaults & visibility -------------------

  it("uses the pinned form defaults", () => {
    const f = component.form;
    expect(f.controls.method.value).toBe('fixed_count');
    expect(f.controls.targetCount.value).toBe(30);
    expect(f.controls.targetQuantile.value).toBe(0.2);
    expect(f.controls.bufferEnabled.value).toBe(false);
    expect(f.controls.bufferFraction.value).toBe(0.1);
    expect(f.controls.sectorBalance.value).toBe(false);
  });

  it("shows targetCount input and hides targetQuantile when method === 'fixed_count'", () => {
    component.form.controls.method.setValue('fixed_count');
    fixture.detectChanges();
    expect(root.querySelector('[data-testid="target-count"]')).not.toBeNull();
    expect(root.querySelector('[data-testid="target-quantile"]')).toBeNull();
  });

  it("shows targetQuantile input and hides targetCount when method === 'quantile'", () => {
    component.form.controls.method.setValue('quantile');
    fixture.detectChanges();
    expect(root.querySelector('[data-testid="target-count"]')).toBeNull();
    expect(root.querySelector('[data-testid="target-quantile"]')).not.toBeNull();
  });

  it("shows bufferFraction only when bufferEnabled === true", () => {
    expect(root.querySelector('[data-testid="buffer-fraction"]')).toBeNull();
    component.form.controls.bufferEnabled.setValue(true);
    fixture.detectChanges();
    expect(root.querySelector('[data-testid="buffer-fraction"]')).not.toBeNull();
  });

  // ------------------- Payload rules on Run click -------------------

  it("emits tickers=parent input, start_date=today-365d, end_date=today ISO", () => {
    clickRun();
    const req = last(emitted);
    expect(req.tickers).toEqual(TICKERS);
    expect(req.start_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(req.end_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    const dayMs = 24 * 60 * 60 * 1000;
    const spanDays =
      (new Date(req.end_date).getTime() - new Date(req.start_date).getTime()) / dayMs;
    expect(spanDays).toBeGreaterThanOrEqual(364);
    expect(spanDays).toBeLessThanOrEqual(366);
  });

  it("includes target_count but NOT target_quantile when method === 'fixed_count'", () => {
    component.form.controls.method.setValue('fixed_count');
    component.form.controls.targetCount.setValue(25);
    fixture.detectChanges();
    clickRun();
    const req = last(emitted);
    expect(req.target_count).toBe(25);
    expect(req.target_quantile).toBeUndefined();
  });

  it("includes target_quantile but NOT target_count when method === 'quantile'", () => {
    component.form.controls.method.setValue('quantile');
    component.form.controls.targetQuantile.setValue(0.33);
    fixture.detectChanges();
    clickRun();
    const req = last(emitted);
    expect(req.target_quantile).toBe(0.33);
    expect(req.target_count).toBeUndefined();
  });

  it("includes buffer_fraction only when bufferEnabled === true", () => {
    clickRun();
    expect(last(emitted).buffer_fraction).toBeUndefined();

    component.form.controls.bufferEnabled.setValue(true);
    component.form.controls.bufferFraction.setValue(0.15);
    fixture.detectChanges();
    clickRun();
    expect(last(emitted).buffer_fraction).toBe(0.15);
  });

  it("always sends sector_balance as the current form boolean", () => {
    clickRun();
    expect(last(emitted).sector_balance).toBe(false);

    component.form.controls.sectorBalance.setValue(true);
    fixture.detectChanges();
    clickRun();
    expect(last(emitted).sector_balance).toBe(true);
  });

  it("never includes current_members in the payload (YAGNI)", () => {
    component.form.controls.method.setValue('fixed_count');
    component.form.controls.bufferEnabled.setValue(true);
    fixture.detectChanges();
    clickRun();
    expect(last(emitted).current_members).toBeUndefined();
  });

  // ------------------- Loading / error / empty -------------------

  it("disables the Run button while loading=true", () => {
    fixture.componentRef.setInput('loading', true);
    fixture.detectChanges();
    const btn = root.querySelector<HTMLButtonElement>(
      '[data-testid="run-select-btn"]',
    );
    expect(btn!.disabled).toBe(true);
  });

  it("renders an inline error banner when errorMessage is set", () => {
    fixture.componentRef.setInput('errorMessage', 'select backend down');
    fixture.detectChanges();
    const banner = root.querySelector('[data-testid="select-error"]');
    expect(banner).not.toBeNull();
    expect(banner!.textContent).toContain('select backend down');
  });

  it("shows zero rows and an enabled Run button before the first run", () => {
    const btn = root.querySelector<HTMLButtonElement>(
      '[data-testid="run-select-btn"]',
    );
    expect(btn!.disabled).toBe(false);
    expect(component.selectRows().length).toBe(0);
  });

  // ------------------- Summary row + turnover formatting -------------------

  it("summary row renders count and turnover as percent when non-null", () => {
    fixture.componentRef.setInput('selectResult', sampleResponse());
    fixture.detectChanges();

    const summary = root.querySelector('[data-testid="select-summary"]')!;
    expect(summary.textContent).toContain('3');
    expect(summary.textContent).toContain('12.50%');
  });

  it("summary row renders '—' when turnover is null", () => {
    const resp: FactorSelectApiResponse = {
      ...sampleResponse(),
      turnover: null,
    };
    fixture.componentRef.setInput('selectResult', resp);
    fixture.detectChanges();

    const summary = root.querySelector('[data-testid="select-summary"]')!;
    expect(summary.textContent).toContain('—');
  });

  // ------------------- Status derivation (via deriveSelectRows) -------------------

  it("derives 'Held' for tickers in both selected_tickers and current_members", () => {
    const rows = deriveSelectRows(sampleResponse(), ['AAPL', 'MSFT']);
    const byTicker = new Map(rows.map((r) => [r.ticker, r.status]));
    expect(byTicker.get('AAPL')).toBe('Held');
    expect(byTicker.get('MSFT')).toBe('Held');
  });

  it("derives 'Entered' for tickers in selected_tickers AND buffer_zone.entered", () => {
    const rows = deriveSelectRows(sampleResponse(), []);
    const nvda = rows.find((r) => r.ticker === 'NVDA');
    expect(nvda?.status).toBe('Entered');
  });

  it("derives 'Exited' for tickers in buffer_zone.exited but not selected_tickers", () => {
    const rows = deriveSelectRows(sampleResponse(), ['AAPL', 'GOOG']);
    const goog = rows.find((r) => r.ticker === 'GOOG');
    expect(goog?.status).toBe('Exited');
  });

  it("derives 'Selected' for remaining selected tickers with no other signal", () => {
    const rows = deriveSelectRows(sampleResponse(), []);
    const aapl = rows.find((r) => r.ticker === 'AAPL');
    expect(aapl?.status).toBe('Selected');
  });

  it("includes the union of selected_tickers and buffer_zone.exited so exited tickers stay visible", () => {
    const rows = deriveSelectRows(sampleResponse(), []);
    const tickers = new Set(rows.map((r) => r.ticker));
    expect(tickers.has('GOOG')).toBe(true); // exited
    expect(tickers.has('AAPL')).toBe(true); // selected
  });

  it("panel's selectRows binds the derivation through the component signal", () => {
    fixture.componentRef.setInput('selectResult', sampleResponse());
    fixture.detectChanges();
    const rows = component.selectRows();
    expect(rows.length).toBeGreaterThan(0);
    expect(new Set(rows.map((r) => r.ticker)).has('GOOG')).toBe(true);
  });
});
