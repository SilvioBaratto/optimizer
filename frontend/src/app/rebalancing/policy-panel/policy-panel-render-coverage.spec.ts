import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { PolicyPanelComponent } from './policy-panel';
import type { RebalancingPolicyDto } from '../rebalancing.model';

// ---------------------------------------------------------------------------
// Local builder — full DTO shape from rebalancing.model.ts
// ---------------------------------------------------------------------------
function makePolicy(overrides: Partial<RebalancingPolicyDto> = {}): RebalancingPolicyDto {
  return {
    id: 'p1',
    portfolioId: 'port-1',
    name: 'Quarterly',
    policyType: 'calendar',
    config: { frequency_days: 63 },
    isActive: false,
    createdAt: '',
    updatedAt: '',
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// DOM accessors
// ---------------------------------------------------------------------------
function queryNameInput(el: HTMLElement): HTMLInputElement | null {
  return el.querySelector('#newPolicyName');
}

function queryFreqDays(el: HTMLElement): HTMLSelectElement | null {
  return el.querySelector('#freqDays');
}

function queryThreshold(el: HTMLElement): HTMLInputElement | null {
  return el.querySelector('#thresholdValue');
}

function queryEmptyState(el: HTMLElement): HTMLParagraphElement | null {
  return el.querySelector('p');
}

function queryCreateButton(el: HTMLElement): HTMLButtonElement | null {
  return el.querySelector('button[type="button"]:last-of-type') as HTMLButtonElement | null;
}

function queryCreatePolicyButton(el: HTMLElement): HTMLButtonElement | null {
  // The Create policy button is the last button inside the create-form section
  const buttons = el.querySelectorAll('button[type="button"]');
  for (let i = buttons.length - 1; i >= 0; i--) {
    const btn = buttons[i] as HTMLButtonElement;
    if (btn.textContent?.trim() === 'Create policy') return btn;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Render-coverage spec — drives every @if branch in policy-panel.html
// ---------------------------------------------------------------------------
describe('PolicyPanelComponent (render coverage)', () => {
  let fixture: ComponentFixture<PolicyPanelComponent>;
  let comp: PolicyPanelComponent;
  let el: HTMLElement;

  beforeEach(async () => {
    await configureTestBed({ imports: [PolicyPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(PolicyPanelComponent);
    comp = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
  });

  // -------------------------------------------------------------------------
  // 1. showCreate toggle — @if (showCreate())
  // -------------------------------------------------------------------------
  describe('showCreate toggle', () => {
    it('when hidden by default, the name input is absent', () => {
      expect(queryNameInput(el)).toBeNull();
    });

    it('when toggleCreate() is called, the name input appears', () => {
      comp.toggleCreate();
      fixture.detectChanges();
      expect(queryNameInput(el)).toBeTruthy();
    });

    it('when toggleCreate() is called again, the name input disappears', () => {
      comp.toggleCreate();
      fixture.detectChanges();
      comp.toggleCreate();
      fixture.detectChanges();
      expect(queryNameInput(el)).toBeNull();
    });
  });

  // -------------------------------------------------------------------------
  // 2. policy_type conditional sections (freqDays / thresholdValue)
  // -------------------------------------------------------------------------
  describe('policy_type conditional inputs', () => {
    beforeEach(() => {
      comp.toggleCreate();
      fixture.detectChanges();
    });

    it('when policy_type is calendar, freqDays is present and thresholdValue is absent', () => {
      comp.updateField('policy_type', 'calendar');
      fixture.detectChanges();
      expect(queryFreqDays(el)).toBeTruthy();
      expect(queryThreshold(el)).toBeNull();
    });

    it('when policy_type is threshold, thresholdValue is present and freqDays is absent', () => {
      comp.updateField('policy_type', 'threshold');
      fixture.detectChanges();
      expect(queryThreshold(el)).toBeTruthy();
      expect(queryFreqDays(el)).toBeNull();
    });

    it('when policy_type is hybrid, both freqDays and thresholdValue are present', () => {
      comp.updateField('policy_type', 'hybrid');
      fixture.detectChanges();
      expect(queryFreqDays(el)).toBeTruthy();
      expect(queryThreshold(el)).toBeTruthy();
    });
  });

  // -------------------------------------------------------------------------
  // 3. Card grid vs empty-state — @if (policies().length > 0) @else
  // -------------------------------------------------------------------------
  describe('card grid vs empty-state', () => {
    it('when policies is empty, the empty-state paragraph is present and no cards are rendered', () => {
      fixture.componentRef.setInput('policies', []);
      fixture.detectChanges();
      expect(queryEmptyState(el)).toBeTruthy();
      expect(el.querySelectorAll('[data-testid^="activate-"],[data-testid^="active-"]').length).toBe(0);
    });

    it('when two policies are provided, two card buttons are rendered and no empty-state', () => {
      fixture.componentRef.setInput('policies', [
        makePolicy({ id: 'p1' }),
        makePolicy({ id: 'p2' }),
      ]);
      fixture.detectChanges();
      const buttons = el.querySelectorAll('[data-testid^="activate-"],[data-testid^="active-"]');
      expect(buttons.length).toBe(2);
      // empty-state paragraph is gone — the only <p> in the template is in the empty-state block
      const emptyParagraph = el.querySelector('.text-center p');
      expect(emptyParagraph).toBeNull();
    });
  });

  // -------------------------------------------------------------------------
  // 4. Active vs inactive card — [class] ring-2 and data-testid attrs
  // -------------------------------------------------------------------------
  describe('active vs inactive policy card', () => {
    it('when isActive is true, the active button carries data-testid="active-p1" and its card has ring-2', () => {
      fixture.componentRef.setInput('policies', [makePolicy({ id: 'p1', isActive: true })]);
      fixture.detectChanges();
      const activeBtn = el.querySelector('[data-testid="active-p1"]') as HTMLButtonElement;
      expect(activeBtn).toBeTruthy();
      expect(activeBtn.disabled).toBe(true);
      expect(activeBtn.parentElement!.classList.contains('ring-2')).toBe(true);
    });

    it('when isActive is false, the activate button carries data-testid="activate-p1" and its card lacks ring-2', () => {
      fixture.componentRef.setInput('policies', [makePolicy({ id: 'p1', isActive: false })]);
      fixture.detectChanges();
      const activateBtn = el.querySelector('[data-testid="activate-p1"]') as HTMLButtonElement;
      expect(activateBtn).toBeTruthy();
      expect(activateBtn.parentElement!.classList.contains('ring-2')).toBe(false);
    });
  });

  // -------------------------------------------------------------------------
  // 5. keyvalue @for — <dt>/<dd> rows inside a policy card
  // -------------------------------------------------------------------------
  describe('config keyvalue rows', () => {
    it('when config has two keys, two <dt> rows are rendered inside the card', () => {
      fixture.componentRef.setInput('policies', [
        makePolicy({ id: 'p1', config: { frequency_days: 63, threshold: 0.05 } }),
      ]);
      fixture.detectChanges();
      const dts = el.querySelectorAll('dl dt');
      expect(dts.length).toBeGreaterThanOrEqual(2);
    });
  });

  // -------------------------------------------------------------------------
  // 6. Create button [disabled] state — covers isFormValid() computed
  // -------------------------------------------------------------------------
  describe('Create policy button disabled state', () => {
    beforeEach(() => {
      comp.toggleCreate();
      fixture.detectChanges();
    });

    it('when name is empty (default form), the Create policy button is disabled', () => {
      const btn = queryCreatePolicyButton(el);
      expect(btn).toBeTruthy();
      expect(btn!.disabled).toBe(true);
    });

    it('when name is valid and policy_type is calendar with positive frequency, the button is enabled', () => {
      comp.updateField('name', 'Valid Name');
      // policy_type defaults to 'calendar', frequency_days defaults to 63 — both valid
      fixture.detectChanges();
      const btn = queryCreatePolicyButton(el);
      expect(btn!.disabled).toBe(false);
    });
  });
});
