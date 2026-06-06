import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import { DiagnosticBadgeComponent } from './diagnostic-badge';
import type { FlagInstance, PositionFlag } from '../../core/models/drift.model';

function _flag(
  code: PositionFlag,
  reason = 'reason text',
  reference: string | null = null,
): FlagInstance {
  return { code, reason, reference };
}

describe('DiagnosticBadgeComponent', () => {
  let fixture: ComponentFixture<DiagnosticBadgeComponent>;
  let host: HTMLElement;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideZonelessChangeDetection()],
    });
  });

  function render(flag: FlagInstance): void {
    fixture = TestBed.createComponent(DiagnosticBadgeComponent);
    fixture.componentRef.setInput('flag', flag);
    fixture.detectChanges();
    host = fixture.nativeElement as HTMLElement;
  }

  it('when flag.code is unmapped, container carries text-flag-unmapped class', () => {
    render(_flag('unmapped'));
    const span = host.querySelector('[data-flag-code="unmapped"]');
    expect(span).not.toBeNull();
    expect(span!.className).toContain('text-flag-unmapped');
    expect(span!.className).toContain('bg-flag-unmapped-bg');
  });

  it('when flag.code is fx_missing, container carries text-flag-fx-missing class', () => {
    render(_flag('fx_missing'));
    const span = host.querySelector('[data-flag-code="fx_missing"]');
    expect(span!.className).toContain('text-flag-fx-missing');
  });

  it('when flag.code is stale_price, container carries text-flag-stale class', () => {
    render(_flag('stale_price'));
    const span = host.querySelector('[data-flag-code="stale_price"]');
    expect(span!.className).toContain('text-flag-stale');
  });

  it('when flag.code is target_not_on_broker, container carries text-flag-target-gap class', () => {
    render(_flag('target_not_on_broker'));
    const span = host.querySelector('[data-flag-code="target_not_on_broker"]');
    expect(span!.className).toContain('text-flag-target-gap');
  });

  it('when flag.code is reconciliation_mismatch, container carries text-flag-reconciliation class', () => {
    render(_flag('reconciliation_mismatch'));
    const span = host.querySelector('[data-flag-code="reconciliation_mismatch"]');
    expect(span!.className).toContain('text-flag-reconciliation');
  });

  it('when flag.reason is set, tooltip text matches reason', () => {
    render(_flag('unmapped', 'No yfinance mapping'));
    const span = host.querySelector('[data-flag-code="unmapped"]')!;
    expect(span.getAttribute('title')).toBe('No yfinance mapping');
  });

  it('when flag.reference is set, reference is appended to tooltip', () => {
    render(_flag('fx_missing', 'No FX', 'USD'));
    const span = host.querySelector('[data-flag-code="fx_missing"]')!;
    expect(span.getAttribute('title')).toBe('No FX (USD)');
  });

  it('when flag.reference is null, tooltip omits parentheses suffix', () => {
    render(_flag('stale_price', 'tick too old', null));
    const span = host.querySelector('[data-flag-code="stale_price"]')!;
    expect(span.getAttribute('title')).toBe('tick too old');
    expect(span.getAttribute('title')!.includes('(')).toBe(false);
  });

  it('when rendered, an inline SVG icon is present', () => {
    render(_flag('unmapped'));
    expect(host.querySelector('svg')).not.toBeNull();
    expect(host.querySelector('svg path')).not.toBeNull();
  });

  for (const [code, expected] of [
    ['unmapped', 'UNMAPPED'],
    ['fx_missing', 'FX'],
    ['stale_price', 'STALE'],
    ['target_not_on_broker', 'NO BROKER'],
    ['reconciliation_mismatch', 'RECON'],
  ] as Array<[PositionFlag, string]>) {
    it(`when flag.code is ${code}, the short label is "${expected}"`, () => {
      render(_flag(code));
      const label = host.querySelector('[data-region="badge-label"]');
      expect(label?.textContent?.trim()).toBe(expected);
    });
  }
});
