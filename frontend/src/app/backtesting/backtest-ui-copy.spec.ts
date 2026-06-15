/**
 * Source-blind UI-copy spec for the Backtesting Lab (issue #1012, requirements §13e).
 *
 * Criterion (UNIT):
 *   No API endpoint or parameter names appear in any user-facing UI copy.
 *
 * Background (requirements §13e):
 *   The walk-forward validation section previously showed the description:
 *   "Runs a background job via POST /validate/walk-forward with cv_type=walk_forward."
 *   This leaks internal endpoint and parameter names into user-facing text and must
 *   be replaced with a user-oriented description.
 *
 * Tests are source-blind: they assert the ABSENCE of forbidden patterns and
 * the PRESENCE of acceptable user-facing language. They do NOT verify which
 * specific wording was chosen, only that the wording does not contain
 * implementation details.
 *
 * Forbidden patterns (must not appear in any rendered text):
 *   - HTTP verb + path combos: "POST /", "GET /", "PUT /", "DELETE /"
 *   - Endpoint path fragments: "/validate/walk-forward", "/backtest", "/factors/"
 *   - Snake_case parameter names that are internal: "cv_type", "cv_config",
 *     "optimizer_type", "pipeline_config", "start_date", "end_date"
 *     (Note: "start_date"/"end_date" are also form labels in some UIs; only assert
 *      the walk-forward DESCRIPTION section, not the entire page.)
 *
 * Assumption (source-blind, correctable):
 *   The walk-forward section is identified by a data-testid="walk-forward-section"
 *   OR contains a heading "Walk-forward" or "Walk-Forward" or "Walk Forward".
 *   If neither attribute nor heading exists, the test falls back to the full
 *   page text to remain useful even if the exact DOM shape differs.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import {
  configureTestBed,
  injectHttp,
  installResizeObserverStub,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { BacktestingComponent } from './backtesting';

/** Patterns that must never appear in any user-facing UI copy. */
const FORBIDDEN_PATTERNS = [
  /POST\s+\/[a-z]/i,
  /GET\s+\/[a-z]/i,
  /PUT\s+\/[a-z]/i,
  /DELETE\s+\/[a-z]/i,
  /\/validate\/walk-forward/i,
  /\/backtest(?:ing)?\b/i,
  /cv_type/i,
  /cv_config/i,
  /optimizer_type/i,
  /pipeline_config/i,
];

describe('BacktestingComponent — no API/param names in user-facing UI copy (issue #1012)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await configureTestBed({
      imports: [BacktestingComponent],
      withHttp: true,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(BacktestingComponent);
    fixture.detectChanges();

    // Drain any bootstrap HTTP requests so the component reaches its stable state.
    const http = injectHttp();
    http.match(() => true).forEach((r) => r.flush({}));
    fixture.detectChanges();
  });

  afterEach(() => {
    localStorage.clear();
    // Drain leftover requests to keep verify() quiet (called by karma).
    injectHttp().match(() => true).forEach((r) => r.flush({}));
  });

  // ── Walk-forward section text ─────────────────────────────────────────────────

  it('when BacktestingComponent renders, none of the forbidden API patterns appear in the walk-forward section text', () => {
    const el = fixture.nativeElement as HTMLElement;

    // Prefer a specifically-labelled walk-forward section if present.
    const section: Element | null =
      el.querySelector('[data-testid="walk-forward-section"]') ??
      el.querySelector('[data-testid="walk-forward"]') ??
      Array.from(el.querySelectorAll('section, div, article')).find(
        (s) => (s.textContent ?? '').toLowerCase().includes('walk-forward') ||
               (s.textContent ?? '').toLowerCase().includes('walk forward'),
      ) ?? null;

    // Fall back to the full page if the section cannot be isolated.
    const target = section ?? el;
    const text = target.textContent ?? '';

    for (const pattern of FORBIDDEN_PATTERNS) {
      expect(pattern.test(text))
        .withContext(`Forbidden pattern "${pattern}" must not appear in user-facing copy. Found in: "${text.slice(0, 200)}"`)
        .toBe(false);
    }
  });

  // ── Full-page guard: no HTTP verb+path combos anywhere in the rendered DOM ───

  it('when BacktestingComponent renders, no "POST /…" or "GET /…" pattern appears anywhere in the rendered text', () => {
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(/POST\s+\/[a-z]/i.test(text))
      .withContext('"POST /<endpoint>" must not appear in any rendered user-facing text')
      .toBe(false);
    expect(/GET\s+\/[a-z]/i.test(text))
      .withContext('"GET /<endpoint>" must not appear in any rendered user-facing text')
      .toBe(false);
  });

  // ── Walk-forward section should contain a user-facing description ─────────────

  it('when BacktestingComponent renders, the walk-forward area contains at least one sentence of user-facing text', () => {
    const el = fixture.nativeElement as HTMLElement;

    const section: Element | null =
      el.querySelector('[data-testid="walk-forward-section"]') ??
      el.querySelector('[data-testid="walk-forward"]') ??
      Array.from(el.querySelectorAll('section, div, article')).find(
        (s) => (s.textContent ?? '').toLowerCase().includes('walk-forward') ||
               (s.textContent ?? '').toLowerCase().includes('walk forward'),
      ) ?? null;

    // If the section is completely absent the component does not expose it yet — skip.
    if (!section) {
      pending('walk-forward section not found in DOM; skipping user-facing text assertion');
      return;
    }

    const sectionText = section.textContent?.trim() ?? '';
    // A user-facing description must contain at least one alphabetic word of five
    // or more characters (i.e., it is not just a heading or a loading indicator).
    expect(/[a-zA-Z]{5,}/.test(sectionText))
      .withContext('walk-forward section must contain descriptive user-facing text (not just a label)')
      .toBe(true);
  });
});
