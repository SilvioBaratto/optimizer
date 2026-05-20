import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import { DiagnosticsStripComponent } from './diagnostics-strip.component';
import type { DriftDiagnostics } from '../../../../models/drift.model';

function makeDiagnostics(overrides: Partial<DriftDiagnostics> = {}): DriftDiagnostics {
  return {
    reconciliation_ok: true,
    reconciliation_delta_pct: 0,
    unmapped_count: 0,
    fx_missing_count: 0,
    target_not_on_broker_count: 0,
    base_currency: 'EUR',
    sum_eur: 0,
    invested_eur: 0,
    delta_eur: 0,
    tolerance_pct: 0.015,
    stale_price_count: 0,
    entries: [],
    ...overrides,
  };
}

describe('DiagnosticsStripComponent', () => {
  let fixture: ComponentFixture<DiagnosticsStripComponent>;
  let host: HTMLElement;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideZonelessChangeDetection()],
    });
  });

  function render(d: DriftDiagnostics): void {
    fixture = TestBed.createComponent(DiagnosticsStripComponent);
    fixture.componentRef.setInput('diagnostics', d);
    fixture.detectChanges();
    host = fixture.nativeElement as HTMLElement;
  }

  it('when all counts are zero and reconciliation_ok, no count items or reconciliation block render', () => {
    render(makeDiagnostics());
    expect(host.querySelectorAll('[data-region="diagnostics-count-item"]').length).toBe(0);
    expect(host.querySelector('[data-region="diagnostics-reconciliation"]')).toBeNull();
  });

  it('when unmapped_count > 0, an unmapped count item renders with the right value', () => {
    render(makeDiagnostics({ unmapped_count: 3 }));
    const items = host.querySelectorAll('[data-region="diagnostics-count-item"]');
    expect(items.length).toBe(1);
    const item = host.querySelector('[data-cat-code="unmapped"]')!;
    expect(item.querySelector('app-diagnostic-badge')).not.toBeNull();
    expect(item.querySelector('[data-cell="count-value"]')!.textContent!.trim()).toBe('3');
  });

  it('when multiple counts are non-zero, one item per category renders in defined order', () => {
    render(
      makeDiagnostics({
        unmapped_count: 1,
        fx_missing_count: 2,
        stale_price_count: 4,
        target_not_on_broker_count: 5,
      }),
    );
    const codes = Array.from(
      host.querySelectorAll('[data-region="diagnostics-count-item"]'),
    ).map((el) => el.getAttribute('data-cat-code'));
    expect(codes).toEqual([
      'unmapped',
      'fx_missing',
      'stale_price',
      'target_not_on_broker',
    ]);
  });

  it('when one category has zero count, only the non-zero categories render', () => {
    render(makeDiagnostics({ unmapped_count: 0, stale_price_count: 7 }));
    const codes = Array.from(
      host.querySelectorAll('[data-region="diagnostics-count-item"]'),
    ).map((el) => el.getAttribute('data-cat-code'));
    expect(codes).toEqual(['stale_price']);
  });

  it('when reconciliation_ok is false, reconciliation block renders with sum/invested/delta/tolerance', () => {
    render(
      makeDiagnostics({
        reconciliation_ok: false,
        reconciliation_delta_pct: 0.05,
        sum_eur: 9500,
        invested_eur: 10000,
        delta_eur: 500,
        tolerance_pct: 0.015,
      }),
    );
    const block = host.querySelector('[data-region="diagnostics-reconciliation"]');
    expect(block).not.toBeNull();
    const label = block!.querySelector('[data-cell="reconciliation-label"]')!.textContent!;
    expect(label).toContain('€9,500');
    expect(label).toContain('€10,000');
    expect(label).toContain('€500');
    expect(label).toContain('5.00%');
    expect(label).toContain('±1.50%');
    expect(block!.querySelector('app-diagnostic-badge')).not.toBeNull();
  });

  it('when reconciliation_ok is true, reconciliation block is absent', () => {
    render(makeDiagnostics({ reconciliation_ok: true }));
    expect(host.querySelector('[data-region="diagnostics-reconciliation"]')).toBeNull();
  });

  it('when reconciliation_delta_pct is null, label still renders with n/a substitution', () => {
    render(
      makeDiagnostics({
        reconciliation_ok: false,
        reconciliation_delta_pct: null,
        sum_eur: 100,
        invested_eur: 0,
        delta_eur: 100,
      }),
    );
    const label = host
      .querySelector('[data-cell="reconciliation-label"]')!
      .textContent!;
    expect(label).toContain('n/a');
  });

  it('when counts present AND reconciliation mismatch, both regions render', () => {
    render(
      makeDiagnostics({
        unmapped_count: 1,
        reconciliation_ok: false,
        reconciliation_delta_pct: 0.02,
        sum_eur: 9800,
        invested_eur: 10000,
        delta_eur: 200,
      }),
    );
    expect(host.querySelector('[data-cat-code="unmapped"]')).not.toBeNull();
    expect(host.querySelector('[data-region="diagnostics-reconciliation"]')).not.toBeNull();
  });
});
