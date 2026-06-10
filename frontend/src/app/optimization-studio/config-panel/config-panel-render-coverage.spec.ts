import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { ConfigPanelComponent } from './config-panel';
import type { StrategyType } from '../models/portfolio.model';

// ---------------------------------------------------------------------------
// DOM accessors — single-responsibility helpers, one query per concept
// ---------------------------------------------------------------------------
function queryRisk(el: HTMLElement): HTMLInputElement | null {
  return el.querySelector('input[formControlName="risk_aversion"]');
}

function queryCvar(el: HTMLElement): HTMLInputElement | null {
  return el.querySelector('input[formControlName="cvar_beta"]');
}

function submitBtn(el: HTMLElement): HTMLButtonElement {
  return el.querySelector('button[type="submit"]') as HTMLButtonElement;
}

// ---------------------------------------------------------------------------
// Render-coverage spec — mounts the full template to drive @if branches
// ---------------------------------------------------------------------------
describe('ConfigPanelComponent (render coverage)', () => {
  let fixture: ComponentFixture<ConfigPanelComponent>;
  let el: HTMLElement;

  beforeEach(async () => {
    await configureTestBed({ imports: [ConfigPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(ConfigPanelComponent);
    el = fixture.nativeElement as HTMLElement;
  });

  // -------------------------------------------------------------------------
  // 1. Control-visibility matrix — data-driven over @if branches
  // -------------------------------------------------------------------------
  describe('conditional control visibility', () => {
    interface VisibilityCase {
      strategy: StrategyType;
      riskPresent: boolean;
      cvarPresent: boolean;
    }

    const cases: VisibilityCase[] = [
      { strategy: 'max_utility',  riskPresent: true,  cvarPresent: false },
      { strategy: 'min_cvar',     riskPresent: false, cvarPresent: true  },
      { strategy: 'cvar_parity',  riskPresent: false, cvarPresent: true  },
      { strategy: 'min_variance', riskPresent: false, cvarPresent: false },
    ];

    cases.forEach(({ strategy, riskPresent, cvarPresent }) => {
      describe(`when strategy is ${strategy}`, () => {
        beforeEach(() => {
          fixture.componentRef.setInput('strategy', strategy);
          fixture.detectChanges();
        });

        it(
          `the risk-aversion control is ${riskPresent ? 'rendered' : 'absent'}`,
          () => {
            if (riskPresent) {
              expect(queryRisk(el)).toBeTruthy();
            } else {
              expect(queryRisk(el)).toBeNull();
            }
          },
        );

        it(
          `the cvar-beta control is ${cvarPresent ? 'rendered' : 'absent'}`,
          () => {
            if (cvarPresent) {
              expect(queryCvar(el)).toBeTruthy();
            } else {
              expect(queryCvar(el)).toBeNull();
            }
          },
        );
      });
    });
  });

  // -------------------------------------------------------------------------
  // 2. Submit-button disabled state — covers both operands of
  //    `!strategy() || disabled()`
  // -------------------------------------------------------------------------
  describe('submit button disabled state', () => {
    it('when strategy is null (default), the button is disabled', () => {
      fixture.detectChanges();
      expect(submitBtn(el).disabled).toBe(true);
    });

    it('when strategy is set and disabled input is false, the button is enabled', () => {
      fixture.componentRef.setInput('strategy', 'max_utility');
      fixture.detectChanges();
      expect(submitBtn(el).disabled).toBe(false);
    });

    it('when strategy is set but disabled input is true, the button is disabled', () => {
      fixture.componentRef.setInput('strategy', 'max_utility');
      fixture.componentRef.setInput('disabled', true);
      fixture.detectChanges();
      expect(submitBtn(el).disabled).toBe(true);
    });
  });
});
