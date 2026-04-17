import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { PolicyPanelComponent } from './policy-panel';
import type {
  RebalancingPolicyCreatePayload,
  RebalancingPolicyDto,
} from '../../models/rebalancing.model';

function policy(overrides: Partial<RebalancingPolicyDto> = {}): RebalancingPolicyDto {
  return {
    id: 'p1', portfolioId: 'port-1', name: 'Quarterly',
    policyType: 'calendar', config: { frequency_days: 63 },
    isActive: false, createdAt: '', updatedAt: '',
    ...overrides,
  };
}

describe('PolicyPanelComponent', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideZonelessChangeDetection()],
    });
  });

  function create(policies: RebalancingPolicyDto[] = []) {
    const fx = TestBed.createComponent(PolicyPanelComponent);
    fx.componentRef.setInput('policies', policies);
    fx.detectChanges();
    return fx;
  }

  it('emits requestActivate (not a direct activate call) when the button is clicked', () => {
    const fx = create([policy({ id: 'abc' })]);
    let emitted: string | undefined;
    fx.componentInstance.requestActivate.subscribe((id) => (emitted = id));

    fx.componentInstance.activate('abc');
    expect(emitted).toBe('abc');
  });

  it('emits createPolicy with a calendar config when the form is valid', () => {
    const fx = create();
    let emitted: RebalancingPolicyCreatePayload | undefined;
    fx.componentInstance.createPolicy.subscribe((p) => (emitted = p));

    fx.componentInstance.updateField('name', 'Monthly');
    fx.componentInstance.updateField('policy_type', 'calendar');
    fx.componentInstance.updateField('frequency_days', 21);
    fx.componentInstance.submit();

    expect(emitted).toEqual({
      name: 'Monthly',
      policy_type: 'calendar',
      config: { frequency_days: 21 },
    });
  });

  it('emits createPolicy with threshold config for threshold policy', () => {
    const fx = create();
    let emitted: RebalancingPolicyCreatePayload | undefined;
    fx.componentInstance.createPolicy.subscribe((p) => (emitted = p));

    fx.componentInstance.updateField('name', 'Threshold 5%');
    fx.componentInstance.updateField('policy_type', 'threshold');
    fx.componentInstance.updateField('threshold', 0.05);
    fx.componentInstance.updateField('kind', 'absolute');
    fx.componentInstance.submit();

    expect(emitted?.config).toEqual({ threshold: 0.05, kind: 'absolute' });
    expect(emitted?.policy_type).toBe('threshold');
  });

  it('emits createPolicy with combined hybrid config', () => {
    const fx = create();
    let emitted: RebalancingPolicyCreatePayload | undefined;
    fx.componentInstance.createPolicy.subscribe((p) => (emitted = p));

    fx.componentInstance.updateField('name', 'Hybrid');
    fx.componentInstance.updateField('policy_type', 'hybrid');
    fx.componentInstance.updateField('frequency_days', 63);
    fx.componentInstance.updateField('threshold', 0.1);
    fx.componentInstance.submit();

    expect(emitted?.config).toEqual({
      frequency_days: 63,
      threshold: 0.1,
      kind: 'absolute',
    });
  });

  it('rejects submissions with empty name or invalid threshold', () => {
    const fx = create();
    let count = 0;
    fx.componentInstance.createPolicy.subscribe(() => count++);

    fx.componentInstance.updateField('name', 'A');
    fx.componentInstance.submit();
    expect(count).toBe(0);

    fx.componentInstance.updateField('name', 'OK');
    fx.componentInstance.updateField('policy_type', 'threshold');
    fx.componentInstance.updateField('threshold', 1.5);
    fx.componentInstance.submit();
    expect(count).toBe(0);
  });
});
