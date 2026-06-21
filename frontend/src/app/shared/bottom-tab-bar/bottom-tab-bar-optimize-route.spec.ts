/**
 * Source-blind contract tests for issue #1042 — mobile bottom-tab-bar slice.
 *
 * Derived solely from the acceptance criterion:
 *   "shared/sidebar/nav-data.ts and mobile bottom-tab-bar point the
 *    Optimize item at /optimize."
 *
 * Strategy: the bottom-tab-bar either (a) consumes NAV_GROUPS from nav-data
 * directly, in which case the nav-data tests above already cover it, or
 * (b) hardcodes its own route strings, in which case its rendered output
 * must contain a link/href to '/optimize' and must NOT contain
 * '/optimization-studio'.
 *
 * These tests render the component in a minimal TestBed and check the DOM —
 * the most robust observable behaviour independent of internal structure.
 *
 * Tests MUST FAIL if the bottom-tab-bar still emits '/optimization-studio'
 * and PASS once it emits '/optimize'.
 */
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { BottomTabBarComponent } from './bottom-tab-bar';
import { ICON_PROVIDER } from '../../icons';

describe('BottomTabBarComponent – Optimize route (#1042)', () => {
  let fixture: ComponentFixture<BottomTabBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BottomTabBarComponent],
      providers: [provideRouter([]), ICON_PROVIDER],
    }).compileComponents();

    fixture = TestBed.createComponent(BottomTabBarComponent);
    fixture.detectChanges();
  });

  it('when rendered, then a link pointing to /optimize exists', () => {
    const anchors = Array.from<HTMLAnchorElement>(
      fixture.nativeElement.querySelectorAll('a'),
    );
    const hrefs = anchors.map((a) => a.getAttribute('href'));
    expect(hrefs).toContain('/optimize');
  });

  it('when rendered, then NO link points to /optimization-studio', () => {
    const anchors = Array.from<HTMLAnchorElement>(
      fixture.nativeElement.querySelectorAll('a'),
    );
    const hrefs = anchors.map((a) => a.getAttribute('href'));
    expect(hrefs).not.toContain('/optimization-studio');
  });
});
