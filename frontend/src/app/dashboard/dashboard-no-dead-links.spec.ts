/**
 * Source-blind tests for issue #1037 — criterion 3.
 * Authored from acceptance criteria only; no implementation source was read.
 *
 * Criterion 3: dashboard.html quick-action cards no longer link to
 *   /rebalancing, /risk-center, or /ai-control-room — whether those links
 *   live in the template as static anchors, routerLink directives, or in a
 *   data-driven card array on the component instance.
 *
 * Import assumptions:
 *   - The dashboard component class is `DashboardComponent` exported from `./dashboard`.
 *     Adjust the import path if the filename or class name differs.
 *   - NO_ERRORS_SCHEMA suppresses errors from sub-components whose providers
 *     are not supplied here; it does not affect routerLink resolution.
 *
 * Additional providers may be required if DashboardComponent injects services
 * that cannot be resolved from its own imports. Add stubs as needed.
 */
import { NO_ERRORS_SCHEMA } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { RouterLink, provideRouter } from '@angular/router';
import { provideLocationMocks } from '@angular/common/testing';
import { DashboardComponent } from './dashboard';
import { ICON_PROVIDER } from '../icons';

// ---------------------------------------------------------------------------
// Dead link fragments under test
// ---------------------------------------------------------------------------

interface DeadLink {
  path: string;   // fragment to look for, e.g. 'rebalancing'
  label: string;  // human-readable label for error messages
}

const DEAD_LINKS: DeadLink[] = [
  { path: 'rebalancing',   label: '/rebalancing'   },
  { path: 'risk-center',   label: '/risk-center'   },
  { path: 'ai-control-room', label: '/ai-control-room' },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function staticAnchorHrefs(el: HTMLElement): string[] {
  return Array.from(el.querySelectorAll('a'))
    .map(a => a.getAttribute('href') ?? '');
}

function routerLinkTargets(fixture: ComponentFixture<unknown>): string[] {
  return fixture.debugElement
    .queryAll(By.directive(RouterLink))
    .map(de => {
      const rl = de.injector.get(RouterLink);
      // `href` is the serialised URL string produced by the RouterLink directive.
      return rl.href ?? '';
    });
}

// ---------------------------------------------------------------------------
// Criterion 3 — no dead quick-action links in the dashboard
// ---------------------------------------------------------------------------

describe('DashboardComponent — no dead quick-action links (criterion 3)', () => {
  let fixture: ComponentFixture<DashboardComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      providers: [
        provideRouter([]),
        provideLocationMocks(),
        ICON_PROVIDER,
      ],
      schemas: [NO_ERRORS_SCHEMA],
    }).compileComponents();

    fixture = TestBed.createComponent(DashboardComponent);
    fixture.detectChanges();
  });

  DEAD_LINKS.forEach(({ path, label }) => {
    it(`when the dashboard is rendered, no static <a> element links to "${label}"`, () => {
      const hrefs = staticAnchorHrefs(fixture.nativeElement);
      const offending = hrefs.filter(h => h === label || h === path || h.endsWith(`/${path}`));
      expect(offending.length)
        .withContext(`Found <a href="${label}"> in the dashboard template: ${JSON.stringify(offending)}`)
        .toBe(0);
    });

    it(`when the dashboard is rendered, no routerLink directive points to "${label}"`, () => {
      const targets = routerLinkTargets(fixture);
      const offending = targets.filter(t => t === label || t.endsWith(`/${path}`));
      expect(offending.length)
        .withContext(`Found [routerLink] targeting "${label}" in the dashboard: ${JSON.stringify(offending)}`)
        .toBe(0);
    });
  });

  it('when the dashboard component instance has a data-driven card list, no card routes to a removed path', () => {
    // If the card list is not data-driven, this test is vacuously satisfied.
    // The test guards the case described in criterion 3: "if the card list is
    // data-driven in dashboard.ts, prune the corresponding entries too."
    const comp = fixture.componentInstance as unknown as Record<string, unknown>;

    // Probe common property names for card arrays; extend as needed.
    const CARD_ARRAY_KEYS = ['cards', 'quickActions', 'actions', 'quickLinks', 'items'];
    const cardKey = CARD_ARRAY_KEYS.find(k => Array.isArray(comp[k]));

    if (!cardKey) {
      // No data-driven card array detected — template-only links are covered above.
      pending('No data-driven card array found on DashboardComponent; skipping data-driven check');
      return;
    }

    const cardItems = comp[cardKey] as Array<Record<string, unknown>>;
    const ROUTE_KEYS = ['route', 'link', 'path', 'href', 'routerLink', 'url'];
    const cardRoutes = cardItems.flatMap(card =>
      ROUTE_KEYS.map(k => String(card[k] ?? ''))
    );

    const DEAD_FRAGMENTS = DEAD_LINKS.map(d => d.path);
    const offending = cardRoutes.filter(r => DEAD_FRAGMENTS.some(f => r.endsWith(f)));
    expect(offending.length)
      .withContext(`Data-driven card list still contains routes to removed pages: ${JSON.stringify(offending)}`)
      .toBe(0);
  });
});
