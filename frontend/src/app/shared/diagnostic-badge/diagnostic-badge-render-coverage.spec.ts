import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed } from '@angular/core/testing';

import { DiagnosticBadgeComponent } from './diagnostic-badge';
import type { FlagInstance, PositionFlag } from '../../core/models/drift.model';

const CASES = [
  { code: 'unmapped' as PositionFlag, label: 'UNMAPPED' },
  { code: 'fx_missing' as PositionFlag, label: 'FX' },
  { code: 'stale_price' as PositionFlag, label: 'STALE' },
  { code: 'target_not_on_broker' as PositionFlag, label: 'NO BROKER' },
  { code: 'reconciliation_mismatch' as PositionFlag, label: 'RECON' },
] as const;

function render(flag: FlagInstance): HTMLElement {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  const fixture = TestBed.createComponent(DiagnosticBadgeComponent);
  fixture.componentRef.setInput('flag', flag);
  fixture.detectChanges();
  return fixture.nativeElement as HTMLElement;
}

describe('DiagnosticBadgeComponent — render coverage', () => {
  CASES.forEach(({ code, label }) => {
    it(`when flag.code is "${code}", the data-flag-code attribute is set and badge-label is "${label}"`, () => {
      const host = render({ code, reason: 'x', reference: null });
      expect(host.querySelector(`[data-flag-code="${code}"]`)).not.toBeNull();
      expect(
        host.querySelector('[data-region="badge-label"]')!.textContent!.trim(),
      ).toBe(label);
    });
  });

  it('when reference is set, the title is "reason (ref)"', () => {
    const host = render({ code: 'unmapped', reason: '2 affected', reference: 'AAPL' });
    expect(
      host.querySelector('[data-flag-code="unmapped"]')!.getAttribute('title'),
    ).toBe('2 affected (AAPL)');
  });

  it('when reference is null, the title is just the reason', () => {
    const host = render({ code: 'unmapped', reason: '2 affected', reference: null });
    expect(
      host.querySelector('[data-flag-code="unmapped"]')!.getAttribute('title'),
    ).toBe('2 affected');
  });
});
