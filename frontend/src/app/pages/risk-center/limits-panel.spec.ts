import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { LimitsPanelComponent } from './limits-panel';
import type { RiskLimit, RiskLimitCreatePayload } from '../../models/risk.model';

function limit(overrides: Partial<RiskLimit> = {}): RiskLimit {
  return {
    id: 'l1',
    name: 'max_drawdown',
    metric: 'max_drawdown',
    limit: 0.15,
    current: 0.08,
    status: 'ok',
    lastChecked: '2026-04-17T00:00:00Z',
    ...overrides,
  };
}

describe('LimitsPanelComponent', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideZonelessChangeDetection()],
    });
  });

  function createPanel(limits: RiskLimit[] = []) {
    const fx = TestBed.createComponent(LimitsPanelComponent);
    fx.componentRef.setInput('limits', limits);
    fx.componentRef.setInput('alerts', []);
    fx.detectChanges();
    return fx;
  }

  it('renders a breach badge when any limit is breached', () => {
    const fx = createPanel([
      limit({ id: 'a', status: 'ok' }),
      limit({ id: 'b', status: 'breached' }),
      limit({ id: 'c', status: 'breached' }),
    ]);

    expect(fx.componentInstance.breachCount()).toBe(2);
    const el = fx.nativeElement.querySelector('[data-testid="breach-badge"]');
    expect(el?.textContent).toContain('2 breached');
  });

  it('does not render the breach badge when no limit is breached', () => {
    const fx = createPanel([limit({ status: 'ok' }), limit({ id: 'l2', status: 'warning' })]);
    expect(fx.componentInstance.breachCount()).toBe(0);
    expect(fx.nativeElement.querySelector('[data-testid="breach-badge"]')).toBeNull();
  });

  it('emits createLimit when the create form is valid and submit fires', () => {
    const fx = createPanel();
    let emitted: RiskLimitCreatePayload | undefined;
    fx.componentInstance.createLimit.subscribe((p) => (emitted = p));

    fx.componentInstance.updateFormField('metric', 'max_drawdown');
    fx.componentInstance.updateFormField('limit_type', 'upper');
    fx.componentInstance.updateFormField('threshold', 0.2);
    fx.componentInstance.submitCreate();

    expect(emitted).toEqual({
      metric: 'max_drawdown',
      limit_type: 'upper',
      threshold: 0.2,
    });
  });

  it('rejects create submissions with empty metric or non-positive threshold', () => {
    const fx = createPanel();
    let emittedCount = 0;
    fx.componentInstance.createLimit.subscribe(() => emittedCount++);

    fx.componentInstance.updateFormField('metric', '');
    fx.componentInstance.updateFormField('threshold', 0.1);
    fx.componentInstance.submitCreate();
    expect(emittedCount).toBe(0);

    fx.componentInstance.updateFormField('metric', 'cvar');
    fx.componentInstance.updateFormField('threshold', 0);
    fx.componentInstance.submitCreate();
    expect(emittedCount).toBe(0);
  });

  it('gates deleteLimit emission on confirmation', () => {
    const fx = createPanel([limit({ id: 'abc' })]);
    let emitted: string | undefined;
    fx.componentInstance.deleteLimit.subscribe((id) => (emitted = id));

    fx.componentInstance.requestDelete('abc');
    expect(fx.componentInstance.pendingDeleteId()).toBe('abc');
    expect(emitted).toBeUndefined();

    fx.componentInstance.confirmDelete();
    expect(emitted).toBe('abc');
    expect(fx.componentInstance.pendingDeleteId()).toBeNull();
  });

  it('does not emit deleteLimit when the user cancels the confirmation', () => {
    const fx = createPanel([limit({ id: 'abc' })]);
    let emitted: string | undefined;
    fx.componentInstance.deleteLimit.subscribe((id) => (emitted = id));

    fx.componentInstance.requestDelete('abc');
    fx.componentInstance.cancelDelete();

    expect(emitted).toBeUndefined();
    expect(fx.componentInstance.pendingDeleteId()).toBeNull();
  });
});
