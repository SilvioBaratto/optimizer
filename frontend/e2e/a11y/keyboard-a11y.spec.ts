import AxeBuilder from '@axe-core/playwright';
import type { Page } from '@playwright/test';

import { test, expect } from '../support/fixtures';
import { stubLandingReads } from '../support/api-stubs';

// Acceptance item 10 — accessibility baseline run INSIDE the live Playwright
// browser so axe sees the fully-rendered, data-populated DOM (the static
// scripts/a11y.mjs audit on :4200 only sees skeletons). Complement, not
// replacement: a11y.mjs / a11y.yml stay as-is.
//
// All 14 routes (mirrors the #854 page-load smoke; INCLUDES the route the
// static a11y.mjs list omits, /portfolio-builder-legacy).
const ROUTES: readonly string[] = [
  '/',
  '/portfolio-builder',
  '/portfolio-builder-legacy',
  '/optimization-studio',
  '/backtesting',
  '/risk-center',
  '/factor-research',
  '/rebalancing',
  '/attribution',
  '/macro-intelligence',
  '/ai-control-room',
  '/settings',
  '/portfolio/e2e-portfolio',
  '/instrument/SM1',
];

const WCAG_TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'];

// Baseline policy (explicit — violations are surfaced, never swallowed):
// the FULL WCAG 2.1 A/AA name/role ruleset is asserted to ZERO on every route,
// with ONE documented deferral. `color-contrast` is deferred because the dark
// theme is incomplete; it is already tracked by the separate static
// scripts/a11y.mjs audit. When theming lands, delete this entry and the
// baseline tightens automatically — no code change to the assertion.
const DEFERRED_RULES = ['color-contrast'];

async function loadPage(page: Page, path: string): Promise<void> {
  await stubLandingReads(page);
  await page.goto(path);
  await expect(page.locator('main#main-content')).toBeVisible();
  await page.waitForLoadState('networkidle');
}

async function assertNoNameRoleViolations(page: Page): Promise<void> {
  const { violations } = await new AxeBuilder({ page })
    .withTags(WCAG_TAGS)
    .disableRules(DEFERRED_RULES)
    .analyze();
  const summary = violations.map((v) => `${v.id} (${v.nodes.length} node[s])`);
  expect(summary, 'WCAG 2.1 AA name/role violations').toEqual([]);
}

interface FocusState {
  readonly leftBody: boolean;
  readonly visibleIndicator: boolean;
  readonly tag: string;
}

function readFocusState(): FocusState {
  const el = document.activeElement as HTMLElement | null;
  if (!el || el === document.body) return { leftBody: false, visibleIndicator: false, tag: 'BODY' };
  const cs = getComputedStyle(el);
  const outline = cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth) > 0;
  const ring = cs.boxShadow !== 'none';
  return { leftBody: true, visibleIndicator: outline || ring, tag: el.tagName };
}

async function assertKeyboardReachable(page: Page): Promise<void> {
  await page.keyboard.press('Tab');
  const focus = await page.evaluate(readFocusState);
  expect(focus.leftBody, `Tab should move focus off BODY (got ${focus.tag})`).toBe(true);
  expect(focus.visibleIndicator, `focused ${focus.tag} must show a visible focus indicator`).toBe(true);
}

function collectUnnamedRoleButtons(els: Element[]): string[] {
  return els
    .filter((el) => {
      const name =
        el.getAttribute('aria-label') ??
        el.getAttribute('aria-labelledby') ??
        el.textContent ??
        '';
      return name.trim().length === 0;
    })
    .map((el) => el.outerHTML.slice(0, 100));
}

async function assertNamedRoleButtons(page: Page): Promise<void> {
  // A non-interactive div posing as a control: [role="button"] (not a native
  // <button>) with no accessible name.
  const unnamed = await page
    .locator('[role="button"]:not(button)')
    .evaluateAll(collectUnnamedRoleButtons);
  expect(unnamed, '[role="button"] surfaces must have an accessible name').toEqual([]);
}

test.describe('accessibility baseline (live stack)', () => {
  for (const path of ROUTES) {
    test(`${path}: axe name/role clean, keyboard-reachable, visible focus`, async ({
      authedPage: page,
    }) => {
      await loadPage(page, path);
      await assertNoNameRoleViolations(page);
      await assertKeyboardReachable(page);
      await assertNamedRoleButtons(page);
    });
  }
});
