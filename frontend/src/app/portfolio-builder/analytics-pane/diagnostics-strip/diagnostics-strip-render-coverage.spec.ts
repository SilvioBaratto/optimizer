import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed } from '@angular/core/testing';

import { DiagnosticsStripComponent } from './diagnostics-strip';
import type { DriftDiagnostics } from '../../../core/models/drift.model';

function makeDiagnostics(overrides: Partial<DriftDiagnostics> = {}): DriftDiagnostics {
  return {
    reconciliation_ok: true, reconciliation_delta_pct: 0,
    unmapped_count: 0, fx_missing_count: 0, target_not_on_broker_count: 0,
    base_currency: 'EUR', sum_eur: 100, invested_eur: 100, delta_eur: 0,
    tolerance_pct: 0, stale_price_count: 0, entries: [],
    ...overrides,
  };
}

function render(d: DriftDiagnostics): HTMLElement {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  const fixture = TestBed.createComponent(DiagnosticsStripComponent);
  fixture.componentRef.setInput('diagnostics', d);
  fixture.detectChanges();
  return fixture.nativeElement as HTMLElement;
}

describe('DiagnosticsStripComponent — render coverage', () => {
  it('when a category count is non-zero, the counts block renders', () => {
    const host = render(makeDiagnostics({ unmapped_count: 2 }));
    expect(host.querySelector('[data-region="diagnostics-counts"]')).not.toBeNull();
    expect(host.querySelector('[data-cat-code="unmapped"]')).not.toBeNull();
    expect(
      host.querySelector('[data-cat-code="unmapped"] [data-cell="count-value"]')!
        .textContent!.trim(),
    ).toContain('2');
  });

  it('when all counts are zero and reconciliation_ok, the counts block and reconciliation are hidden (empty state)', () => {
    const host = render(makeDiagnostics());
    expect(host.querySelector('[data-region="diagnostics-counts"]')).toBeNull();
    expect(host.querySelector('[data-region="diagnostics-reconciliation"]')).toBeNull();
  });

  it('when only some categories are non-zero, only those render', () => {
    const host = render(makeDiagnostics({ unmapped_count: 2, stale_price_count: 1 }));
    const items = host.querySelectorAll('[data-region="diagnostics-count-item"]');
    expect(items.length).toBe(2);
    expect(host.querySelector('[data-cat-code="unmapped"]')).not.toBeNull();
    expect(host.querySelector('[data-cat-code="stale_price"]')).not.toBeNull();
    expect(host.querySelector('[data-cat-code="fx_missing"]')).toBeNull();
    expect(host.querySelector('[data-cat-code="target_not_on_broker"]')).toBeNull();
  });

  it('when reconciliation_ok is false, the reconciliation row renders', () => {
    const host = render(makeDiagnostics({ reconciliation_ok: false }));
    expect(host.querySelector('[data-region="diagnostics-reconciliation"]')).not.toBeNull();
  });
});
