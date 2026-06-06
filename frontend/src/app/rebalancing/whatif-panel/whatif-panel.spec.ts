import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeRebalanceDecideResponse } from '../../../testing';
import { WhatifPanelComponent } from './whatif-panel';
import type { RebalanceDecideRequest } from '../rebalancing.model';

describe('WhatifPanelComponent', () => {
  let fixture: ComponentFixture<WhatifPanelComponent>;
  let comp: WhatifPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [WhatifPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(WhatifPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when weights are malformed, parseWeights drops the bad entries', () => {
    comp.currentRaw.set('AAPL:abc, MSFT:0.5');
    expect(comp.currentWeights()).toEqual({ MSFT: 0.5 });
  });

  it('when both weight sets sum near one, the form is valid', () => {
    // defaults: current 0.5+0.5, target 0.6+0.4
    expect(comp.isFormValid()).toBe(true);
    comp.currentRaw.set('AAPL:0.5'); // sum 0.5
    expect(comp.isFormValid()).toBe(false);
  });

  it('when the form is invalid, submit emits nothing', () => {
    comp.currentRaw.set('AAPL:0.5'); // invalid
    let emitted = false;
    comp.runDecide.subscribe(() => (emitted = true));
    comp.submit();
    expect(emitted).toBe(false);
  });

  it('when valid, submit emits the decide request via runDecide', () => {
    let body: RebalanceDecideRequest | undefined;
    comp.runDecide.subscribe((b) => (body = b));
    comp.submit();
    expect(body?.policy_type).toBe('threshold');
    expect(body?.current_date).toBeUndefined();
  });

  it('when policy is hybrid, the body carries both current_date and last_review_date', () => {
    comp.policyType.set('hybrid');
    let body: RebalanceDecideRequest | undefined;
    comp.runDecide.subscribe((b) => (body = b));
    comp.submit();
    expect(body?.current_date).toBeTruthy();
    expect(body?.last_review_date).toBeTruthy();
  });

  it('when a decide response is present, the KPIs map from its fields', () => {
    fixture.componentRef.setInput('decideResponse', makeRebalanceDecideResponse());
    expect(comp.kpiShouldRebalance()).toBe('YES');
    expect(comp.kpiTurnover()).toBe('10.00%'); // turnover 0.1
    expect(comp.kpiEstCost()).toBe('0.050%'); // estimatedCost 0.0005
  });

  it('when no decide response, the KPIs are dashes', () => {
    expect(comp.kpiShouldRebalance()).toBe('—');
    expect(comp.kpiTurnover()).toBe('—');
  });
});
