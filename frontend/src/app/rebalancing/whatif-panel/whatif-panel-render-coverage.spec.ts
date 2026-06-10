import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeRebalanceDecideResponse } from '../../../testing';
import { WhatifPanelComponent } from './whatif-panel';
import type { RebalanceDecideRequest } from '../rebalancing.model';

// ---------------------------------------------------------------------------
// DOM accessors
// ---------------------------------------------------------------------------
function queryWhatifThreshold(el: HTMLElement): HTMLInputElement | null {
  return el.querySelector('#whatifThreshold');
}

function queryWhatifFreq(el: HTMLElement): HTMLInputElement | null {
  return el.querySelector('#whatifFreq');
}

function queryWarning(el: HTMLElement): HTMLParagraphElement | null {
  return el.querySelector('p.text-xs');
}

function queryStatCards(el: HTMLElement): NodeListOf<Element> {
  return el.querySelectorAll('app-stat-card');
}

function queryDataTable(el: HTMLElement): Element | null {
  return el.querySelector('app-data-table');
}

function querySubmitButton(el: HTMLElement): HTMLButtonElement {
  return el.querySelector('button[type="button"]') as HTMLButtonElement;
}

// ---------------------------------------------------------------------------
// Render-coverage spec — drives every @if branch in whatif-panel.html
// ---------------------------------------------------------------------------
describe('WhatifPanelComponent (render coverage)', () => {
  let fixture: ComponentFixture<WhatifPanelComponent>;
  let comp: WhatifPanelComponent;
  let el: HTMLElement;

  beforeEach(async () => {
    await configureTestBed({ imports: [WhatifPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(WhatifPanelComponent);
    comp = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
  });

  // -------------------------------------------------------------------------
  // 1. policy_type conditional inputs — @if threshold/hybrid and calendar/hybrid
  // -------------------------------------------------------------------------
  describe('policy_type conditional inputs', () => {
    it('when policyType is threshold, whatifThreshold is present and whatifFreq is absent', () => {
      comp.policyType.set('threshold');
      fixture.detectChanges();
      expect(queryWhatifThreshold(el)).toBeTruthy();
      expect(queryWhatifFreq(el)).toBeNull();
    });

    it('when policyType is calendar, whatifFreq is present and whatifThreshold is absent', () => {
      comp.policyType.set('calendar');
      fixture.detectChanges();
      expect(queryWhatifFreq(el)).toBeTruthy();
      expect(queryWhatifThreshold(el)).toBeNull();
    });

    it('when policyType is hybrid, both whatifThreshold and whatifFreq are present', () => {
      comp.policyType.set('hybrid');
      fixture.detectChanges();
      expect(queryWhatifThreshold(el)).toBeTruthy();
      expect(queryWhatifFreq(el)).toBeTruthy();
    });
  });

  // -------------------------------------------------------------------------
  // 2. Validation warning — @if (!isFormValid())
  // -------------------------------------------------------------------------
  describe('validation warning paragraph', () => {
    it('when defaults are set (valid weights), the warning paragraph is absent', () => {
      // defaults: currentRaw='AAPL:0.5, MSFT:0.5', targetRaw='AAPL:0.6, MSFT:0.4' — both sum to 1
      expect(queryWarning(el)).toBeNull();
    });

    it('when currentRaw is cleared (invalid), the warning paragraph appears', () => {
      comp.currentRaw.set('');
      fixture.detectChanges();
      expect(queryWarning(el)).toBeTruthy();
    });

    it('when weights do not sum to 1, the warning paragraph appears', () => {
      comp.currentRaw.set('AAPL:0.3, MSFT:0.3');
      fixture.detectChanges();
      expect(queryWarning(el)).toBeTruthy();
    });
  });

  // -------------------------------------------------------------------------
  // 3. decideResponse grid — @if (decideResponse())
  // -------------------------------------------------------------------------
  describe('decide response result grid', () => {
    it('when decideResponse is null (default), no app-stat-card elements are rendered', () => {
      expect(queryStatCards(el).length).toBe(0);
    });

    it('when decideResponse is provided, three app-stat-card elements are rendered', () => {
      fixture.componentRef.setInput('decideResponse', makeRebalanceDecideResponse());
      fixture.detectChanges();
      expect(queryStatCards(el).length).toBe(3);
    });

    it('when decideResponse is provided, an app-data-table is rendered', () => {
      fixture.componentRef.setInput('decideResponse', makeRebalanceDecideResponse());
      fixture.detectChanges();
      expect(queryDataTable(el)).toBeTruthy();
    });

    it('when decideResponse is null, no app-data-table is rendered', () => {
      expect(queryDataTable(el)).toBeNull();
    });
  });

  // -------------------------------------------------------------------------
  // 4. Submit button disabled state — covers isFormValid() binding
  // -------------------------------------------------------------------------
  describe('submit button disabled state', () => {
    it('when form is valid (defaults), the submit button is enabled', () => {
      expect(querySubmitButton(el).disabled).toBe(false);
    });

    it('when form is invalid, the submit button is disabled', () => {
      comp.currentRaw.set('');
      fixture.detectChanges();
      expect(querySubmitButton(el).disabled).toBe(true);
    });
  });

  // -------------------------------------------------------------------------
  // 5. buildPolicyConfig() via submit() — three-way policy_config shape
  // -------------------------------------------------------------------------
  describe('buildPolicyConfig via submit()', () => {
    it('when policyType is calendar, emits policy_config with only frequency_days and no current_date', () => {
      comp.policyType.set('calendar');
      comp.frequencyDays.set(63);
      // ensure form stays valid: defaults are already valid
      let emitted: RebalanceDecideRequest | undefined;
      comp.runDecide.subscribe((body) => (emitted = body));
      comp.submit();
      expect(emitted).toBeTruthy();
      expect(emitted!.policy_config).toEqual({ frequency_days: 63 });
      expect(emitted!.policy_config!['threshold']).toBeUndefined();
      expect(emitted!.current_date).toBeUndefined();
    });

    it('when policyType is threshold, emits policy_config with threshold and kind but no frequency_days', () => {
      comp.policyType.set('threshold');
      comp.thresholdValue.set(0.05);
      let emitted: RebalanceDecideRequest | undefined;
      comp.runDecide.subscribe((body) => (emitted = body));
      comp.submit();
      expect(emitted).toBeTruthy();
      expect(emitted!.policy_config).toEqual({ threshold: 0.05, kind: 'absolute' });
      expect(emitted!.policy_config!['frequency_days']).toBeUndefined();
      expect(emitted!.current_date).toBeUndefined();
    });

    it('when policyType is hybrid, emits policy_config with both frequency_days and threshold, plus current_date and last_review_date', () => {
      comp.policyType.set('hybrid');
      comp.frequencyDays.set(63);
      comp.thresholdValue.set(0.05);
      let emitted: RebalanceDecideRequest | undefined;
      comp.runDecide.subscribe((body) => (emitted = body));
      comp.submit();
      expect(emitted).toBeTruthy();
      expect(emitted!.policy_config!['frequency_days']).toBe(63);
      expect(emitted!.policy_config!['threshold']).toBe(0.05);
      expect(emitted!.policy_config!['kind']).toBe('absolute');
      expect(typeof emitted!.current_date).toBe('string');
      expect(typeof emitted!.last_review_date).toBe('string');
    });
  });
});
