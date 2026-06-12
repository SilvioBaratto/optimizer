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

  it('when shouldRebalance is false, kpiShouldRebalance returns NO', () => {
    fixture.componentRef.setInput('decideResponse', makeRebalanceDecideResponse({ shouldRebalance: false }));
    expect(comp.kpiShouldRebalance()).toBe('NO');
  });

  it('when a decide response is present, tradeRows sign-prefixes positives only; when null it returns empty', () => {
    // null guard
    expect(comp.tradeRows()).toEqual([]);

    fixture.componentRef.setInput('decideResponse', makeRebalanceDecideResponse());
    const rows = comp.tradeRows();
    // default: MSFT +0.05, AAPL -0.05 — sorted by abs descending (equal → stable)
    const msft = rows.find((r) => r['ticker'] === 'MSFT')!;
    const aapl = rows.find((r) => r['ticker'] === 'AAPL')!;
    expect(msft['delta']).toBe('+5.00%');
    expect(aapl['delta']).toBe('-5.00%');
  });

  it('when currentRaw is empty, weightsSumTolerant returns false so isFormValid is false', () => {
    comp.currentRaw.set('');
    expect(comp.isFormValid()).toBe(false);
  });

  it('when parseWeights receives a no-colon entry, that entry is dropped', () => {
    // empty string → all entries have no colon → result is {}
    comp.currentRaw.set('');
    expect(comp.currentWeights()).toEqual({});

    // 'AAPL' has no colon: key='AAPL', value=undefined → Number(undefined)=NaN → dropped
    // 'MSFT:0.5' is valid
    comp.currentRaw.set('AAPL, MSFT:0.5');
    expect(comp.currentWeights()).toEqual({ MSFT: 0.5 });
  });

  it('when error is set, a non-blank role="alert" region with the message is rendered', () => {
    fixture.componentRef.setInput('error', 'Decide failed');
    fixture.detectChanges();

    const alert = (fixture.nativeElement as HTMLElement).querySelector('[role="alert"]');
    expect(alert).not.toBeNull();
    expect(alert?.textContent?.trim()).toContain('Decide failed');
  });

  it('when error is null, no alert region is rendered', () => {
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).querySelector('[role="alert"]')).toBeNull();
  });
});
