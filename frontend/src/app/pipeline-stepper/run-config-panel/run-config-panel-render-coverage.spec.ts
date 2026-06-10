/**
 * Render-coverage spec for RunConfigPanelComponent (issue #900).
 *
 * Companion to run-config-panel.spec.ts, which asserts instance-level form
 * logic only. This file mounts the full template and drives every template
 * branch at the DOM:
 *   - [class.border-danger] binding on n_selected input
 *   - @if n_selected error paragraph
 *   - @if dateOrder error paragraph
 *   - [disabled] binding on the submit button
 *   - @for option lists for base_currency (3) and preset (4)
 *
 * STEP-SECTION NOTE: step-section is already fully covered by
 * step-section.spec.ts (all StepStatus dot/badge/label maps, expand/collapse,
 * continue/retry). No duplicate spec is created here.
 *
 * PATTERN ANCHOR: The build() harness + StatusCase loop below is the shape
 * that step-panel render-coverage specs copy.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { RunConfigPanelComponent } from './run-config-panel';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface PanelHarness {
  fixture: ComponentFixture<RunConfigPanelComponent>;
  comp: RunConfigPanelComponent;
  el: HTMLElement;
}

async function build(): Promise<PanelHarness> {
  await configureTestBed({ imports: [RunConfigPanelComponent], withHttp: false });
  const fixture = TestBed.createComponent(RunConfigPanelComponent);
  const comp = fixture.componentInstance;
  const el = fixture.nativeElement as HTMLElement;
  fixture.detectChanges();
  return { fixture, comp, el };
}

// ---------------------------------------------------------------------------
// Text-matching helper — scoped to <p class="...text-danger"> elements only
// ---------------------------------------------------------------------------

function dangerText(el: HTMLElement, needle: string): boolean {
  return Array.from(el.querySelectorAll('p.text-danger')).some((p) =>
    (p.textContent ?? '').includes(needle),
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('RunConfigPanelComponent (render coverage)', () => {
  // ---- 1. n_selected validator arm ----------------------------------------

  describe('n_selected @if error arm', () => {
    it('when n_selected is out of the 15–30 band and touched, the range error and border-danger render', async () => {
      const { fixture, comp, el } = await build();

      comp.form.controls.n_selected.setValue(99);
      comp.form.controls.n_selected.markAsTouched();
      fixture.detectChanges();

      expect(dangerText(el, 'between 15 and 30')).toBe(true);

      const input = el.querySelector<HTMLInputElement>(
        'input[formcontrolname="n_selected"]',
      )!;
      expect(input.classList.contains('border-danger')).toBe(true);
    });

    it('when n_selected is in band, no range error renders', async () => {
      const { el } = await build(); // default value is 20 — already in band

      expect(dangerText(el, 'between 15 and 30')).toBe(false);

      const input = el.querySelector<HTMLInputElement>(
        'input[formcontrolname="n_selected"]',
      )!;
      expect(input.classList.contains('border-danger')).toBe(false);
    });
  });

  // ---- 2. dateOrder validator arm -----------------------------------------

  describe('dateOrder @if error arm', () => {
    it('when end_date precedes start_date, the dateOrder error renders', async () => {
      const { fixture, comp, el } = await build();

      comp.form.patchValue({ start_date: '2024-06-01', end_date: '2024-01-01' });
      fixture.detectChanges();

      expect(dangerText(el, 'on or after start date')).toBe(true);
    });

    it('when dates are absent, no dateOrder error renders', async () => {
      const { el } = await build(); // default: both dates null

      expect(dangerText(el, 'on or after start date')).toBe(false);
    });
  });

  // ---- 3. Submit [disabled] — data-driven StatusCase loop -----------------

  interface StatusCase {
    label: string;
    setup: (comp: RunConfigPanelComponent) => void;
    expectedDisabled: boolean;
  }

  const submitCases: StatusCase[] = [
    {
      label: 'default-valid form',
      setup: () => { /* no-op — defaults are all valid */ },
      expectedDisabled: false,
    },
    {
      label: 'n_selected = 99 (out of band)',
      setup: (comp) => comp.form.controls.n_selected.setValue(99),
      expectedDisabled: true,
    },
  ];

  describe('submit button [disabled] binding', () => {
    submitCases.forEach(({ label, setup, expectedDisabled }) => {
      it(`when ${label}, submit button disabled === ${String(expectedDisabled)}`, async () => {
        const { fixture, comp, el } = await build();

        setup(comp);
        fixture.detectChanges();

        const btn = el.querySelector<HTMLButtonElement>('button[type="submit"]')!;
        expect(btn.disabled).toBe(expectedDisabled);
      });
    });
  });

  // ---- 4. @for option counts ----------------------------------------------

  describe('@for option lists', () => {
    it('when rendered, the base-currency select lists 3 options', async () => {
      const { el } = await build();

      const options = el
        .querySelector<HTMLSelectElement>('select[formcontrolname="base_currency"]')!
        .querySelectorAll('option');
      expect(options.length).toBe(3);
    });

    it('when rendered, the preset select lists 4 options', async () => {
      const { el } = await build();

      const options = el
        .querySelector<HTMLSelectElement>('select[formcontrolname="preset"]')!
        .querySelectorAll('option');
      expect(options.length).toBe(4);
    });
  });
});
