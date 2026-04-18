import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { ExposureConstraintsPanelComponent } from './exposure-constraints-panel';
import { ICON_PROVIDER } from '../../icons';
import type {
  FactorExposureConstraintsApiResponse,
  FactorExposureConstraintsRequest,
  FactorType,
} from '../../models/factor.model';

function last(emitted: FactorExposureConstraintsRequest[]): FactorExposureConstraintsRequest {
  if (emitted.length === 0) throw new Error('runConstraints never fired');
  return emitted[emitted.length - 1];
}

function sampleResponse(): FactorExposureConstraintsApiResponse {
  return {
    left_inequality: [
      [1, 0, -1],
      [0, 1, 0],
    ],
    right_inequality: [0.5, 0.5],
  };
}

describe('ExposureConstraintsPanelComponent', () => {
  let fixture: ComponentFixture<ExposureConstraintsPanelComponent>;
  let component: ExposureConstraintsPanelComponent;
  let root: HTMLElement;
  let emitted: FactorExposureConstraintsRequest[];

  const TICKERS = ['AAPL', 'MSFT', 'GOOG'];

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [ExposureConstraintsPanelComponent],
      providers: [provideZonelessChangeDetection(), ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(ExposureConstraintsPanelComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('tickers', TICKERS);
    fixture.detectChanges();
    root = fixture.nativeElement as HTMLElement;
    emitted = [];
    component.runConstraints.subscribe((req) => emitted.push(req));
  });

  function clickSubmit(): void {
    const btn = root.querySelector<HTMLButtonElement>(
      '[data-testid="submit-constraints-btn"]',
    );
    if (!btn) throw new Error('Submit button not found');
    btn.click();
    fixture.detectChanges();
  }

  function setRow(index: number, factor: FactorType, min: number, max: number): void {
    const row = component.rows.at(index);
    row.patchValue({ factor, min, max });
  }

  // ---------------- FormArray add / remove ----------------

  it('starts with exactly one row (rowsMinOne initial state)', () => {
    expect(component.rows.length).toBe(1);
  });

  it('adds a new row when addRow() is called', () => {
    component.addRow();
    fixture.detectChanges();
    expect(component.rows.length).toBe(2);
  });

  it('removes a row at the given index', () => {
    component.addRow();
    component.addRow();
    fixture.detectChanges();
    expect(component.rows.length).toBe(3);

    component.removeRow(1);
    fixture.detectChanges();
    expect(component.rows.length).toBe(2);
  });

  it('removeRow() refuses to drop the last remaining row', () => {
    expect(component.rows.length).toBe(1);
    component.removeRow(0);
    fixture.detectChanges();
    expect(component.rows.length).toBe(1);
  });

  // ---------------- Row-level validator ----------------

  it('maxGtMin: row invalid when max <= min', () => {
    setRow(0, 'momentum_12_1', 0.5, 0.5);
    fixture.detectChanges();
    expect(component.rows.at(0).errors?.['maxGtMin']).toBe(true);
    expect(component.form.valid).toBe(false);
  });

  it('maxGtMin: row valid when max > min', () => {
    setRow(0, 'momentum_12_1', -0.5, 0.5);
    fixture.detectChanges();
    expect(component.rows.at(0).errors?.['maxGtMin']).toBeUndefined();
  });

  // ---------------- Form-level validators ----------------

  it('uniqueFactor: form invalid when two rows share the same factor', () => {
    component.addRow();
    fixture.detectChanges();
    setRow(0, 'momentum_12_1', -0.5, 0.5);
    setRow(1, 'momentum_12_1', -0.2, 0.2);
    fixture.detectChanges();
    expect(component.form.errors?.['uniqueFactor']).toBe(true);
    expect(component.form.valid).toBe(false);
  });

  it('uniqueFactor: form valid when all factors are distinct', () => {
    component.addRow();
    fixture.detectChanges();
    setRow(0, 'momentum_12_1', -0.5, 0.5);
    setRow(1, 'volatility', -0.3, 0.3);
    fixture.detectChanges();
    expect(component.form.errors?.['uniqueFactor']).toBeUndefined();
    expect(component.form.valid).toBe(true);
  });

  it('rowsMinOne: removing all rows keeps at least one, so the form can still become valid', () => {
    setRow(0, 'momentum_12_1', -0.5, 0.5);
    fixture.detectChanges();
    expect(component.rows.length).toBe(1);
    expect(component.form.valid).toBe(true);
  });

  it('Submit button is disabled while form is invalid', () => {
    setRow(0, 'momentum_12_1', 0.5, 0.5); // max <= min → invalid
    fixture.detectChanges();
    const btn = root.querySelector<HTMLButtonElement>(
      '[data-testid="submit-constraints-btn"]',
    );
    expect(btn!.disabled).toBe(true);
  });

  // ---------------- Payload shape on Submit ----------------

  it('Submit emits the expected payload with Record-form bounds', () => {
    setRow(0, 'momentum_12_1', -0.5, 0.5);
    component.addRow();
    fixture.detectChanges();
    setRow(1, 'volatility', -0.3, 0.3);
    fixture.detectChanges();
    clickSubmit();

    const req = last(emitted);
    expect(req.tickers).toEqual(TICKERS);
    expect(req.start_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(req.end_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    const dayMs = 24 * 60 * 60 * 1000;
    const spanDays =
      (new Date(req.end_date).getTime() - new Date(req.start_date).getTime()) / dayMs;
    expect(spanDays).toBeGreaterThanOrEqual(364);
    expect(spanDays).toBeLessThanOrEqual(366);

    // Record-form bounds (never the single tuple form)
    expect(Array.isArray(req.bounds)).toBe(false);
    const bounds = req.bounds as Record<string, [number, number]>;
    expect(bounds['momentum_12_1']).toEqual([-0.5, 0.5]);
    expect(bounds['volatility']).toEqual([-0.3, 0.3]);
  });

  // ---------------- Response rendering ----------------

  it('renders matrix A (left_inequality) and vector b (right_inequality) when result is set', () => {
    fixture.componentRef.setInput('result', sampleResponse());
    fixture.detectChanges();

    const matrixA = root.querySelector('[data-testid="matrix-a"]');
    const vectorB = root.querySelector('[data-testid="vector-b"]');
    expect(matrixA).not.toBeNull();
    expect(vectorB).not.toBeNull();
    // At least one row in matrix A
    expect(matrixA!.textContent).toContain('1');
    expect(vectorB!.textContent).toContain('0.5');
  });

  it('does NOT render result sections before the first submit (empty state)', () => {
    expect(root.querySelector('[data-testid="matrix-a"]')).toBeNull();
    expect(root.querySelector('[data-testid="vector-b"]')).toBeNull();
  });

  // ---------------- Clipboard + copied signal ----------------

  it('Copy JSON calls navigator.clipboard.writeText with the exact JSON string', () => {
    fixture.componentRef.setInput('result', sampleResponse());
    fixture.detectChanges();

    const spy = spyOn(navigator.clipboard, 'writeText').and.returnValue(
      Promise.resolve(),
    );

    const btn = root.querySelector<HTMLButtonElement>(
      '[data-testid="copy-json-btn"]',
    );
    expect(btn).not.toBeNull();
    btn!.click();
    fixture.detectChanges();

    const expected = JSON.stringify(
      {
        left_inequality: sampleResponse().left_inequality,
        right_inequality: sampleResponse().right_inequality,
      },
      null,
      2,
    );
    expect(spy).toHaveBeenCalledWith(expected);
  });

  it('copied signal flips to true after Copy JSON and resets to false after 2 seconds', () => {
    jasmine.clock().install();
    try {
      fixture.componentRef.setInput('result', sampleResponse());
      fixture.detectChanges();
      spyOn(navigator.clipboard, 'writeText').and.returnValue(Promise.resolve());

      const btn = root.querySelector<HTMLButtonElement>(
        '[data-testid="copy-json-btn"]',
      );
      btn!.click();
      fixture.detectChanges();

      expect(component.copied()).toBe(true);
      jasmine.clock().tick(2001);
      expect(component.copied()).toBe(false);
    } finally {
      jasmine.clock().uninstall();
    }
  });

  // ---------------- Loading / error ----------------

  it('disables Submit button while loading=true', () => {
    fixture.componentRef.setInput('loading', true);
    fixture.detectChanges();
    const btn = root.querySelector<HTMLButtonElement>(
      '[data-testid="submit-constraints-btn"]',
    );
    expect(btn!.disabled).toBe(true);
  });

  it('renders inline error banner when errorMessage is set', () => {
    fixture.componentRef.setInput('errorMessage', 'bounds rejected');
    fixture.detectChanges();
    const banner = root.querySelector('[data-testid="constraints-error"]');
    expect(banner).not.toBeNull();
    expect(banner!.textContent).toContain('bounds rejected');
  });

  // ---------------- Factor dropdown ----------------

  it('renders the full 17-value FactorType whitelist in the factor dropdown', () => {
    const select = root.querySelector<HTMLSelectElement>(
      '[data-testid="factor-select-0"]',
    );
    expect(select).not.toBeNull();
    const values = Array.from(select!.options).map((o) => o.value);
    expect(values).toEqual([
      'book_to_price',
      'earnings_yield',
      'cash_flow_yield',
      'sales_to_price',
      'ebitda_to_ev',
      'gross_profitability',
      'roe',
      'operating_margin',
      'profit_margin',
      'asset_growth',
      'momentum_12_1',
      'volatility',
      'beta',
      'amihud_illiquidity',
      'dividend_yield',
      'recommendation_change',
      'net_insider_buying',
    ]);
  });
});
