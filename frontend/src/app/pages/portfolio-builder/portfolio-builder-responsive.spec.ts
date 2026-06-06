// Responsive breakpoint spec for the portfolio-builder shell.
//
// SCOPE LIMIT — JSDOM/Karma headless Chrome do not produce reliable, stable
// layout reflow across viewport widths in a unit-test context. Tailwind's
// responsive utilities are driven by CSS media queries, which are evaluated
// by the browser engine but not interacted with by unit specs. We therefore
// assert STRUCTURAL contracts only:
//   (a) the responsive class strings are present on the grid container,
//   (b) the three regions (left/center/right) are siblings under that same
//       grid parent.
//
// FOLLOW-UP (e2e) — Pixel-level verification belongs in a Playwright/Cypress
// suite. At viewport 1023x768 (one below Tailwind's `lg` breakpoint of 1024px)
// all three regions should share the same `offsetLeft` via
// `getBoundingClientRect()` because `max-lg:grid-cols-1` stacks them into a
// single column. At viewport 1024x768 and above they should align side-by-side
// because `lg:grid-cols-[320px_1fr_300px]` applies.
//
// Reviewer comment (issue #753) ratified this split: do not assert
// `getComputedStyle(...).gridTemplateColumns` here.

import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { NEVER } from 'rxjs';

import { PortfolioBuilderComponent } from './portfolio-builder';
import { JOB_POLL_TICK } from '../../shared/job-progress-tracker/job-progress-tracker';

function setup(): HTMLElement {
  TestBed.configureTestingModule({
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
      { provide: JOB_POLL_TICK, useValue: NEVER },
    ],
  });
  const fixture = TestBed.createComponent(PortfolioBuilderComponent);
  fixture.detectChanges();
  return fixture.nativeElement as HTMLElement;
}

function gridContainer(host: HTMLElement): HTMLElement {
  const left = host.querySelector('[data-region="left"]');
  if (!left) throw new Error('[data-region="left"] missing');
  const parent = left.parentElement;
  if (!parent) throw new Error('grid container missing');
  return parent;
}

describe('PortfolioBuilderComponent — responsive grid contract', () => {
  it('when rendered, the grid container carries the max-lg:grid-cols-1 utility (single-column stack below lg)', () => {
    const host = setup();
    const cls = gridContainer(host).className;
    expect(cls).toContain('max-lg:grid-cols-1');
  });

  it('when rendered, the grid container carries the lg:grid-cols-[320px_1fr_300px] utility (3-column above lg)', () => {
    const host = setup();
    const cls = gridContainer(host).className;
    expect(cls).toContain('lg:grid-cols-[320px_1fr_300px]');
  });

  it('when rendered, the grid container carries the base grid display utility', () => {
    const host = setup();
    const cls = gridContainer(host).className;
    expect(cls.split(/\s+/)).toContain('grid');
  });

  it('when rendered, left/center/right regions are siblings under the same grid parent', () => {
    const host = setup();
    const left = host.querySelector('[data-region="left"]');
    const center = host.querySelector('[data-region="center"]');
    const right = host.querySelector('[data-region="right"]');
    expect(left).not.toBeNull();
    expect(center).not.toBeNull();
    expect(right).not.toBeNull();

    const parent = gridContainer(host);
    expect(left!.parentElement).toBe(parent);
    expect(center!.parentElement).toBe(parent);
    expect(right!.parentElement).toBe(parent);
  });

  it('when rendered, left/center/right appear in DOM order: left, center, right', () => {
    const host = setup();
    const parent = gridContainer(host);
    const regionChildren = Array.from(parent.children)
      .filter((c) => c.getAttribute('data-region') !== null)
      .map((c) => c.getAttribute('data-region'));
    expect(regionChildren).toEqual(['left', 'center', 'right']);
  });

  it('when rendered, the action-bar region is a sibling of the grid container, not nested inside it', () => {
    const host = setup();
    const grid = gridContainer(host);
    const actionBar = host.querySelector('[data-region="action-bar"]');
    expect(actionBar).not.toBeNull();
    expect(actionBar!.parentElement).not.toBe(grid);
  });
});
