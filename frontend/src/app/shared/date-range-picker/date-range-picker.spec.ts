import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { DateRangePickerComponent } from './date-range-picker';
import {
  PortfolioContextService,
  DateRangePreset,
} from '../../core/services/portfolio-context.service';

describe('DateRangePickerComponent', () => {
  let ctx: PortfolioContextService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [DateRangePickerComponent],
      providers: [provideZonelessChangeDetection()],
    });
    ctx = TestBed.inject(PortfolioContextService);
  });

  function presetButtons(el: HTMLElement): HTMLButtonElement[] {
    return Array.from(
      el.querySelectorAll<HTMLButtonElement>('[data-testid="preset-btn"]'),
    );
  }

  function findPresetButton(
    el: HTMLElement,
    preset: DateRangePreset,
  ): HTMLButtonElement {
    const btn = presetButtons(el).find(
      (b) => b.textContent?.trim() === preset,
    );
    if (!btn) throw new Error(`preset button '${preset}' not rendered`);
    return btn;
  }

  it('renders a button for each of the 10 presets', () => {
    const fixture = TestBed.createComponent(DateRangePickerComponent);
    fixture.detectChanges();

    const labels = presetButtons(fixture.nativeElement).map((b) =>
      b.textContent?.trim(),
    );
    expect(labels).toEqual([
      '1D', '1W', '1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'Max',
    ]);
  });

  it('calls setPreset when a preset button is clicked', () => {
    const spy = spyOn(ctx, 'setPreset').and.callThrough();
    const fixture = TestBed.createComponent(DateRangePickerComponent);
    fixture.detectChanges();

    findPresetButton(fixture.nativeElement, '3M').click();

    expect(spy).toHaveBeenCalledWith('3M');
  });

  it('highlights the preset that matches the current dateRange', () => {
    ctx.setPreset('1M');
    const fixture = TestBed.createComponent(DateRangePickerComponent);
    fixture.detectChanges();

    const btn = findPresetButton(fixture.nativeElement, '1M');
    expect(btn.className).toContain('bg-accent');
  });

  it('shows the custom panel only after clicking the Custom button', () => {
    const fixture = TestBed.createComponent(DateRangePickerComponent);
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;

    expect(root.querySelector('[data-testid="custom-panel"]')).toBeNull();

    const toggle = root.querySelector(
      '[data-testid="custom-toggle"]',
    ) as HTMLButtonElement;
    toggle.click();
    fixture.detectChanges();

    expect(root.querySelector('[data-testid="custom-panel"]')).not.toBeNull();
  });

  it('disables Apply when dates are missing or start >= end', () => {
    const fixture = TestBed.createComponent(DateRangePickerComponent);
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;

    (root.querySelector('[data-testid="custom-toggle"]') as HTMLButtonElement).click();
    fixture.detectChanges();

    const apply = root.querySelector(
      '[data-testid="custom-apply"]',
    ) as HTMLButtonElement;
    const startInput = root.querySelector(
      '[data-testid="custom-start"]',
    ) as HTMLInputElement;
    const endInput = root.querySelector(
      '[data-testid="custom-end"]',
    ) as HTMLInputElement;

    // Empty inputs
    expect(apply.disabled).toBe(true);

    // start >= end
    startInput.value = '2026-02-01';
    startInput.dispatchEvent(new Event('input'));
    endInput.value = '2026-01-01';
    endInput.dispatchEvent(new Event('input'));
    fixture.detectChanges();
    expect(apply.disabled).toBe(true);

    // Valid range
    startInput.value = '2026-01-01';
    startInput.dispatchEvent(new Event('input'));
    endInput.value = '2026-02-01';
    endInput.dispatchEvent(new Event('input'));
    fixture.detectChanges();
    expect(apply.disabled).toBe(false);
  });

  it('calls setCustomRange with Date objects when Apply is clicked', () => {
    const spy = spyOn(ctx, 'setCustomRange').and.callThrough();
    const fixture = TestBed.createComponent(DateRangePickerComponent);
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;

    (root.querySelector('[data-testid="custom-toggle"]') as HTMLButtonElement).click();
    fixture.detectChanges();

    const startInput = root.querySelector(
      '[data-testid="custom-start"]',
    ) as HTMLInputElement;
    const endInput = root.querySelector(
      '[data-testid="custom-end"]',
    ) as HTMLInputElement;

    startInput.value = '2026-01-01';
    startInput.dispatchEvent(new Event('input'));
    endInput.value = '2026-02-01';
    endInput.dispatchEvent(new Event('input'));
    fixture.detectChanges();

    (root.querySelector('[data-testid="custom-apply"]') as HTMLButtonElement).click();

    expect(spy).toHaveBeenCalledTimes(1);
    const [start, end] = spy.calls.mostRecent().args;
    expect(start instanceof Date).toBe(true);
    expect(end instanceof Date).toBe(true);
    expect(start.toISOString().slice(0, 10)).toBe('2026-01-01');
    expect(end.toISOString().slice(0, 10)).toBe('2026-02-01');
  });
});
